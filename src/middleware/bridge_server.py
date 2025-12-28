"""
Middleware Bridge Server for Marduk-Commander
==============================================

This module implements the HTTP/WebSocket server that acts as the bridge between
the Marduk Cognitive Core and Total Commander on Android. It receives structured
JSON commands and translates them into Android Intent broadcasts.

The server can run on:
1. The Android device itself (via Termux or similar)
2. A connected computer (communicating via ADB)
3. A local network server (communicating via Android app)

Architecture:
    Marduk Cognitive Core --[HTTP/WS]--> Bridge Server --[Intent/ADB]--> Total Commander

Author: Manus AI
Date: December 28, 2025
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MardukBridge")


# ============================================================================
# Data Models
# ============================================================================

class ExecutionMode(str, Enum):
    """Mode of execution for Intent delivery."""
    ADB = "adb"           # Execute via ADB shell commands
    BROADCAST = "broadcast"  # Direct Android broadcast (on-device)
    TASKER = "tasker"     # Execute via Tasker integration


class OperationStatus(str, Enum):
    """Status of an operation."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IntentRequest(BaseModel):
    """Request model for Intent execution."""
    action: str
    package: str = "com.ghisler.android.TotalCommander"
    component: Optional[str] = None
    data_uri: Optional[str] = None
    mime_type: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list)


class FileOperationRequest(BaseModel):
    """Request model for file operations."""
    operation: str  # copy, move, delete, search, pack, unpack, navigate
    source_path: Optional[str] = None
    destination_path: Optional[str] = None
    pattern: Optional[str] = "*"
    recursive: bool = False
    confirm: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchOperationRequest(BaseModel):
    """Request model for batch operations."""
    operations: List[FileOperationRequest]
    sequential: bool = True  # Execute sequentially or in parallel
    stop_on_error: bool = True


class OperationResponse(BaseModel):
    """Response model for operations."""
    operation_id: str
    status: OperationStatus
    message: str
    result: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CognitiveCommand(BaseModel):
    """
    High-level command from Marduk Cognitive Core.
    
    This represents a natural language-derived command that has been
    parsed into a structured format by Marduk's inference engine.
    """
    command_id: str
    intent: str  # The parsed intent (e.g., "organize_files", "find_documents")
    entities: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    source: str = "marduk_cognitive_core"


# ============================================================================
# Intent Executor
# ============================================================================

class IntentExecutor:
    """
    Executes Android Intents through various methods.
    
    Supports:
    - ADB shell commands (for development/debugging)
    - Direct broadcast (when running on Android)
    - Tasker integration (for complex workflows)
    """
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.ADB):
        self.mode = mode
        self.adb_path = self._find_adb()
        self.execution_log: List[Dict[str, Any]] = []
        
    def _find_adb(self) -> str:
        """Find the ADB executable path."""
        # Check common locations
        paths = [
            "/usr/bin/adb",
            "/usr/local/bin/adb",
            os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
            "adb"  # Rely on PATH
        ]
        
        for path in paths:
            if os.path.exists(path) or path == "adb":
                return path
        
        return "adb"
    
    def _build_adb_command(self, intent: IntentRequest) -> List[str]:
        """Build ADB shell command for Intent execution."""
        cmd = [self.adb_path, "shell", "am", "start"]
        
        if intent.action:
            cmd.extend(["-a", intent.action])
        
        if intent.package and intent.component:
            cmd.extend(["-n", f"{intent.package}/{intent.component}"])
        elif intent.package:
            cmd.extend(["-n", f"{intent.package}/.TotalCommander"])
        
        if intent.data_uri:
            cmd.extend(["-d", intent.data_uri])
        
        if intent.mime_type:
            cmd.extend(["-t", intent.mime_type])
        
        for flag in intent.flags:
            cmd.extend(["-f", flag])
        
        for key, value in intent.extras.items():
            if isinstance(value, str):
                cmd.extend(["--es", key, value])
            elif isinstance(value, int):
                cmd.extend(["--ei", key, str(value)])
            elif isinstance(value, bool):
                cmd.extend(["--ez", key, str(value).lower()])
            elif isinstance(value, float):
                cmd.extend(["--ef", key, str(value)])
        
        return cmd
    
    async def execute_intent(self, intent: IntentRequest) -> Dict[str, Any]:
        """Execute a single Intent."""
        result = {
            "intent": intent.dict(),
            "status": "pending",
            "output": None,
            "error": None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            if self.mode == ExecutionMode.ADB:
                cmd = self._build_adb_command(intent)
                logger.info(f"Executing ADB command: {' '.join(cmd)}")
                
                # Execute asynchronously
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                result["output"] = stdout.decode() if stdout else None
                result["error"] = stderr.decode() if stderr else None
                result["status"] = "completed" if process.returncode == 0 else "failed"
                result["return_code"] = process.returncode
                
            elif self.mode == ExecutionMode.BROADCAST:
                # Direct broadcast mode (for on-device execution)
                result["status"] = "completed"
                result["output"] = f"Broadcast intent: {intent.action}"
                logger.info(f"Broadcasting intent: {intent.action}")
                
            elif self.mode == ExecutionMode.TASKER:
                # Tasker integration mode
                tasker_cmd = self._build_tasker_command(intent)
                result["tasker_command"] = tasker_cmd
                result["status"] = "completed"
                logger.info(f"Tasker command prepared: {tasker_cmd}")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Intent execution failed: {e}")
        
        self.execution_log.append(result)
        return result
    
    def _build_tasker_command(self, intent: IntentRequest) -> Dict[str, Any]:
        """Build a Tasker-compatible command structure."""
        return {
            "task": "TC_Execute_Intent",
            "parameters": {
                "action": intent.action,
                "package": intent.package,
                "extras": intent.extras
            }
        }
    
    async def execute_batch(
        self,
        intents: List[IntentRequest],
        sequential: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute multiple Intents."""
        results = []
        
        if sequential:
            for intent in intents:
                result = await self.execute_intent(intent)
                results.append(result)
                
                # Small delay between sequential operations
                await asyncio.sleep(0.1)
        else:
            # Parallel execution
            tasks = [self.execute_intent(intent) for intent in intents]
            results = await asyncio.gather(*tasks)
        
        return results


# ============================================================================
# Operation Translator
# ============================================================================

class OperationTranslator:
    """
    Translates high-level file operations into Android Intents.
    
    This class bridges the gap between Marduk's semantic understanding
    of file operations and the low-level Intent system of Android.
    """
    
    TC_PACKAGE = "com.ghisler.android.TotalCommander"
    ACTION_VIEW = "android.intent.action.VIEW"
    
    # Mapping of operation names to TC commands
    OPERATION_MAP = {
        "copy": "cm_Copy",
        "move": "cm_MoveOnly",
        "delete": "cm_Delete",
        "rename": "cm_RenameOnly",
        "mkdir": "cm_MkDir",
        "navigate": "cm_GoToDir",
        "search": "cm_SearchFor",
        "pack": "cm_PackFiles",
        "unpack": "cm_UnpackFiles",
        "view": "cm_View",
        "edit": "cm_Edit"
    }
    
    def translate(self, operation: FileOperationRequest) -> List[IntentRequest]:
        """Translate a file operation to Intent(s)."""
        intents = []
        
        op_name = operation.operation.lower()
        tc_command = self.OPERATION_MAP.get(op_name)
        
        if not tc_command:
            raise ValueError(f"Unknown operation: {operation.operation}")
        
        # Build the Intent based on operation type
        if op_name == "navigate":
            intents.append(IntentRequest(
                action=self.ACTION_VIEW,
                package=self.TC_PACKAGE,
                data_uri=f"file://{operation.destination_path or operation.source_path}",
                extras={
                    "com.ghisler.android.TotalCommander.Path": 
                        operation.destination_path or operation.source_path
                }
            ))
        
        elif op_name in ["copy", "move"]:
            intents.append(IntentRequest(
                action=self.ACTION_VIEW,
                package=self.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": tc_command,
                    "com.ghisler.android.TotalCommander.SourcePath": operation.source_path,
                    "com.ghisler.android.TotalCommander.TargetPath": operation.destination_path,
                    "com.ghisler.android.TotalCommander.Pattern": operation.pattern,
                    "com.ghisler.android.TotalCommander.Recursive": operation.recursive
                }
            ))
        
        elif op_name == "delete":
            intents.append(IntentRequest(
                action=self.ACTION_VIEW,
                package=self.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": tc_command,
                    "com.ghisler.android.TotalCommander.Path": operation.source_path,
                    "com.ghisler.android.TotalCommander.Confirm": operation.confirm
                }
            ))
        
        elif op_name == "search":
            intents.append(IntentRequest(
                action=self.ACTION_VIEW,
                package=self.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": tc_command,
                    "com.ghisler.android.TotalCommander.SearchPath": 
                        operation.source_path or "/sdcard",
                    "com.ghisler.android.TotalCommander.SearchPattern": operation.pattern
                }
            ))
        
        elif op_name in ["pack", "unpack"]:
            intents.append(IntentRequest(
                action=self.ACTION_VIEW,
                package=self.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": tc_command,
                    "com.ghisler.android.TotalCommander.SourcePath": operation.source_path,
                    "com.ghisler.android.TotalCommander.TargetPath": operation.destination_path
                }
            ))
        
        elif op_name == "mkdir":
            intents.append(IntentRequest(
                action=self.ACTION_VIEW,
                package=self.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": tc_command,
                    "com.ghisler.android.TotalCommander.Path": operation.destination_path
                }
            ))
        
        else:
            # Generic command
            intents.append(IntentRequest(
                action=self.ACTION_VIEW,
                package=self.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": tc_command,
                    "com.ghisler.android.TotalCommander.Path": 
                        operation.source_path or operation.destination_path
                }
            ))
        
        return intents


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time communication with Marduk."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.utcnow().isoformat()
        }
        logger.info(f"WebSocket connected: {self.connection_metadata[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            metadata = self.connection_metadata.pop(websocket, {})
            logger.info(f"WebSocket disconnected: {metadata.get('client_id', 'unknown')}")
    
    async def send_personal(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific connection."""
        await websocket.send_json(message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connections."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Marduk-Commander Bridge",
    description="Middleware bridge between Marduk Cognitive Core and Total Commander",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
executor = IntentExecutor(mode=ExecutionMode.ADB)
translator = OperationTranslator()
ws_manager = ConnectionManager()

# Operation tracking
operations: Dict[str, Dict[str, Any]] = {}
operation_counter = 0


def generate_operation_id() -> str:
    """Generate a unique operation ID."""
    global operation_counter
    operation_counter += 1
    return f"op_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{operation_counter:04d}"


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Marduk-Commander Bridge",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "execute_intent": "/api/v1/intent",
            "file_operation": "/api/v1/operation",
            "batch_operation": "/api/v1/batch",
            "cognitive_command": "/api/v1/cognitive",
            "websocket": "/ws"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "execution_mode": executor.mode.value,
        "active_websockets": len(ws_manager.active_connections),
        "pending_operations": sum(
            1 for op in operations.values() 
            if op.get("status") == OperationStatus.PENDING
        )
    }


@app.post("/api/v1/intent", response_model=OperationResponse)
async def execute_intent(intent: IntentRequest, background_tasks: BackgroundTasks):
    """Execute a raw Android Intent."""
    operation_id = generate_operation_id()
    
    operations[operation_id] = {
        "type": "intent",
        "status": OperationStatus.EXECUTING,
        "intent": intent.dict(),
        "started_at": datetime.utcnow().isoformat()
    }
    
    # Execute the intent
    result = await executor.execute_intent(intent)
    
    operations[operation_id]["status"] = (
        OperationStatus.COMPLETED if result["status"] == "completed" 
        else OperationStatus.FAILED
    )
    operations[operation_id]["result"] = result
    
    # Broadcast result to WebSocket clients
    background_tasks.add_task(
        ws_manager.broadcast,
        {"type": "operation_complete", "operation_id": operation_id, "result": result}
    )
    
    return OperationResponse(
        operation_id=operation_id,
        status=operations[operation_id]["status"],
        message=f"Intent executed: {intent.action}",
        result=result
    )


@app.post("/api/v1/operation", response_model=OperationResponse)
async def execute_file_operation(
    operation: FileOperationRequest,
    background_tasks: BackgroundTasks
):
    """Execute a file operation (translated to Intent)."""
    operation_id = generate_operation_id()
    
    try:
        # Translate operation to intents
        intents = translator.translate(operation)
        
        operations[operation_id] = {
            "type": "file_operation",
            "status": OperationStatus.EXECUTING,
            "operation": operation.dict(),
            "intents": [i.dict() for i in intents],
            "started_at": datetime.utcnow().isoformat()
        }
        
        # Execute all intents
        results = await executor.execute_batch(
            intents,
            sequential=True
        )
        
        # Determine overall status
        all_completed = all(r["status"] == "completed" for r in results)
        operations[operation_id]["status"] = (
            OperationStatus.COMPLETED if all_completed 
            else OperationStatus.FAILED
        )
        operations[operation_id]["results"] = results
        
        # Broadcast result
        background_tasks.add_task(
            ws_manager.broadcast,
            {
                "type": "operation_complete",
                "operation_id": operation_id,
                "operation": operation.operation,
                "results": results
            }
        )
        
        return OperationResponse(
            operation_id=operation_id,
            status=operations[operation_id]["status"],
            message=f"File operation '{operation.operation}' executed",
            result={"intents_executed": len(results), "results": results}
        )
        
    except ValueError as e:
        return OperationResponse(
            operation_id=operation_id,
            status=OperationStatus.FAILED,
            message=str(e),
            errors=[str(e)]
        )


@app.post("/api/v1/batch", response_model=OperationResponse)
async def execute_batch_operations(
    batch: BatchOperationRequest,
    background_tasks: BackgroundTasks
):
    """Execute multiple file operations in batch."""
    operation_id = generate_operation_id()
    
    operations[operation_id] = {
        "type": "batch",
        "status": OperationStatus.EXECUTING,
        "operations": [op.dict() for op in batch.operations],
        "started_at": datetime.utcnow().isoformat()
    }
    
    all_results = []
    errors = []
    
    for op in batch.operations:
        try:
            intents = translator.translate(op)
            results = await executor.execute_batch(intents, sequential=True)
            all_results.append({
                "operation": op.operation,
                "results": results,
                "success": all(r["status"] == "completed" for r in results)
            })
            
            if batch.stop_on_error and not all_results[-1]["success"]:
                break
                
        except Exception as e:
            errors.append(str(e))
            if batch.stop_on_error:
                break
    
    success_count = sum(1 for r in all_results if r.get("success"))
    operations[operation_id]["status"] = (
        OperationStatus.COMPLETED if success_count == len(batch.operations)
        else OperationStatus.FAILED
    )
    operations[operation_id]["results"] = all_results
    
    background_tasks.add_task(
        ws_manager.broadcast,
        {
            "type": "batch_complete",
            "operation_id": operation_id,
            "success_count": success_count,
            "total_count": len(batch.operations)
        }
    )
    
    return OperationResponse(
        operation_id=operation_id,
        status=operations[operation_id]["status"],
        message=f"Batch executed: {success_count}/{len(batch.operations)} successful",
        result={"results": all_results},
        errors=errors
    )


@app.post("/api/v1/cognitive", response_model=OperationResponse)
async def process_cognitive_command(
    command: CognitiveCommand,
    background_tasks: BackgroundTasks
):
    """
    Process a high-level cognitive command from Marduk.
    
    This endpoint receives parsed natural language commands and
    translates them into file operations.
    """
    operation_id = generate_operation_id()
    
    operations[operation_id] = {
        "type": "cognitive",
        "status": OperationStatus.EXECUTING,
        "command": command.dict(),
        "started_at": datetime.utcnow().isoformat()
    }
    
    try:
        # Map cognitive intent to file operation
        file_operation = _map_cognitive_to_operation(command)
        
        if file_operation:
            intents = translator.translate(file_operation)
            results = await executor.execute_batch(intents, sequential=True)
            
            operations[operation_id]["status"] = OperationStatus.COMPLETED
            operations[operation_id]["results"] = results
            
            # Send feedback to Marduk via WebSocket
            background_tasks.add_task(
                ws_manager.broadcast,
                {
                    "type": "cognitive_feedback",
                    "command_id": command.command_id,
                    "operation_id": operation_id,
                    "status": "completed",
                    "results": results
                }
            )
            
            return OperationResponse(
                operation_id=operation_id,
                status=OperationStatus.COMPLETED,
                message=f"Cognitive command '{command.intent}' executed",
                result={"file_operation": file_operation.dict(), "results": results}
            )
        else:
            raise ValueError(f"Could not map intent: {command.intent}")
            
    except Exception as e:
        operations[operation_id]["status"] = OperationStatus.FAILED
        return OperationResponse(
            operation_id=operation_id,
            status=OperationStatus.FAILED,
            message=f"Failed to process cognitive command: {e}",
            errors=[str(e)]
        )


def _map_cognitive_to_operation(command: CognitiveCommand) -> Optional[FileOperationRequest]:
    """Map a cognitive command to a file operation."""
    intent = command.intent.lower()
    entities = command.entities
    
    # Intent mapping
    intent_map = {
        "organize_files": "move",
        "find_documents": "search",
        "archive_files": "pack",
        "extract_archive": "unpack",
        "delete_files": "delete",
        "copy_files": "copy",
        "move_files": "move",
        "create_folder": "mkdir",
        "open_folder": "navigate"
    }
    
    operation_type = intent_map.get(intent)
    
    if not operation_type:
        return None
    
    return FileOperationRequest(
        operation=operation_type,
        source_path=entities.get("source_path") or entities.get("path"),
        destination_path=entities.get("destination_path") or entities.get("target"),
        pattern=entities.get("pattern", "*"),
        recursive=entities.get("recursive", False),
        metadata={"cognitive_command_id": command.command_id}
    )


@app.get("/api/v1/operations/{operation_id}")
async def get_operation_status(operation_id: str):
    """Get the status of an operation."""
    if operation_id not in operations:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    return operations[operation_id]


@app.get("/api/v1/operations")
async def list_operations(limit: int = 50, status: Optional[str] = None):
    """List recent operations."""
    ops = list(operations.items())[-limit:]
    
    if status:
        ops = [(k, v) for k, v in ops if v.get("status") == status]
    
    return {"operations": [{"id": k, **v} for k, v in ops]}


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication with Marduk Cognitive Core.
    
    Supports:
    - Real-time operation status updates
    - Bidirectional command/response flow
    - Event streaming
    """
    await ws_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from Marduk
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "ping":
                await ws_manager.send_personal(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    websocket
                )
            
            elif message_type == "cognitive_command":
                # Process cognitive command
                command = CognitiveCommand(**data.get("payload", {}))
                
                # Execute and send result
                file_operation = _map_cognitive_to_operation(command)
                if file_operation:
                    intents = translator.translate(file_operation)
                    results = await executor.execute_batch(intents)
                    
                    await ws_manager.send_personal(
                        {
                            "type": "cognitive_result",
                            "command_id": command.command_id,
                            "status": "completed",
                            "results": results
                        },
                        websocket
                    )
            
            elif message_type == "subscribe":
                # Subscribe to specific events
                events = data.get("events", [])
                ws_manager.connection_metadata[websocket]["subscriptions"] = events
                await ws_manager.send_personal(
                    {"type": "subscribed", "events": events},
                    websocket
                )
            
            else:
                await ws_manager.send_personal(
                    {"type": "error", "message": f"Unknown message type: {message_type}"},
                    websocket
                )
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "bridge_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
