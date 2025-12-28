"""
TC-Connector Module for Marduk-Commander
=========================================

This module provides the interface between Marduk Cognitive Core and Total Commander
for Android. It translates high-level file operations into Android Intents that
Total Commander can execute.

Architecture:
    Marduk Cognitive Core -> TC-Connector -> Middleware Bridge -> Total Commander

Author: Manus AI
Date: December 28, 2025
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TCConnector")


class TCAction(Enum):
    """Enumeration of Total Commander actions that can be triggered via Intents."""
    
    # File Operations
    COPY = "cm_Copy"
    MOVE = "cm_MoveOnly"
    DELETE = "cm_Delete"
    RENAME = "cm_RenameOnly"
    MKDIR = "cm_MkDir"
    
    # Navigation
    NAVIGATE = "cm_GoToDir"
    SWAP_PANELS = "cm_Exchange"
    REFRESH = "cm_RereadSource"
    
    # Selection
    SELECT_ALL = "cm_SelectAll"
    DESELECT_ALL = "cm_ClearAll"
    SELECT_BY_PATTERN = "cm_SelectGroup"
    
    # Archive Operations
    PACK = "cm_PackFiles"
    UNPACK = "cm_UnpackFiles"
    
    # Search
    SEARCH = "cm_SearchFor"
    
    # View/Edit
    VIEW = "cm_View"
    EDIT = "cm_Edit"
    
    # FTP/Network
    FTP_CONNECT = "cm_FtpConnect"
    FTP_DISCONNECT = "cm_FtpDisconnect"


@dataclass
class AndroidIntent:
    """Represents an Android Intent to be sent to Total Commander."""
    
    action: str
    package: str = "com.ghisler.android.TotalCommander"
    component: Optional[str] = None
    data_uri: Optional[str] = None
    mime_type: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Intent to dictionary for JSON serialization."""
        return {
            "action": self.action,
            "package": self.package,
            "component": self.component,
            "data": self.data_uri,
            "type": self.mime_type,
            "extras": self.extras,
            "flags": self.flags
        }
    
    def to_json(self) -> str:
        """Convert Intent to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_adb_command(self) -> str:
        """Generate ADB shell command to broadcast this Intent."""
        cmd_parts = ["adb", "shell", "am", "start"]
        
        if self.action:
            cmd_parts.extend(["-a", self.action])
        
        if self.package:
            cmd_parts.extend(["-n", f"{self.package}/.TotalCommander"])
        
        if self.data_uri:
            cmd_parts.extend(["-d", self.data_uri])
        
        if self.mime_type:
            cmd_parts.extend(["-t", self.mime_type])
        
        for key, value in self.extras.items():
            if isinstance(value, str):
                cmd_parts.extend(["--es", key, value])
            elif isinstance(value, int):
                cmd_parts.extend(["--ei", key, str(value)])
            elif isinstance(value, bool):
                cmd_parts.extend(["--ez", key, str(value).lower()])
        
        return " ".join(cmd_parts)


@dataclass
class FileOperation:
    """Represents a file operation request from Marduk."""
    
    action: TCAction
    source_path: Optional[str] = None
    destination_path: Optional[str] = None
    pattern: Optional[str] = None
    recursive: bool = False
    confirm: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the file operation parameters."""
        if self.action in [TCAction.COPY, TCAction.MOVE]:
            return bool(self.source_path and self.destination_path)
        elif self.action in [TCAction.DELETE, TCAction.VIEW, TCAction.EDIT]:
            return bool(self.source_path)
        elif self.action == TCAction.MKDIR:
            return bool(self.destination_path)
        elif self.action == TCAction.NAVIGATE:
            return bool(self.destination_path or self.source_path)
        return True


class IntentBuilder:
    """Builder class for constructing Android Intents for Total Commander."""
    
    TC_PACKAGE = "com.ghisler.android.TotalCommander"
    TC_MAIN_ACTIVITY = "com.ghisler.android.TotalCommander.TotalCommander"
    
    # Intent action constants
    ACTION_VIEW = "android.intent.action.VIEW"
    ACTION_SEND = "android.intent.action.SEND"
    ACTION_PICK = "android.intent.action.PICK"
    
    @classmethod
    def build_navigation_intent(cls, path: str) -> AndroidIntent:
        """Build an Intent to navigate Total Commander to a specific path."""
        return AndroidIntent(
            action=cls.ACTION_VIEW,
            package=cls.TC_PACKAGE,
            data_uri=f"file://{path}",
            extras={
                "com.ghisler.android.TotalCommander.Path": path
            }
        )
    
    @classmethod
    def build_file_operation_intent(
        cls,
        operation: FileOperation
    ) -> List[AndroidIntent]:
        """
        Build a sequence of Intents to perform a file operation.
        
        Complex operations may require multiple Intents executed in sequence.
        """
        intents = []
        
        if operation.action == TCAction.NAVIGATE:
            path = operation.destination_path or operation.source_path
            intents.append(cls.build_navigation_intent(path))
        
        elif operation.action == TCAction.COPY:
            # Step 1: Navigate to source
            intents.append(cls.build_navigation_intent(operation.source_path))
            
            # Step 2: Execute copy command with destination
            intents.append(AndroidIntent(
                action=cls.ACTION_VIEW,
                package=cls.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": "cm_Copy",
                    "com.ghisler.android.TotalCommander.SourcePath": operation.source_path,
                    "com.ghisler.android.TotalCommander.TargetPath": operation.destination_path,
                    "com.ghisler.android.TotalCommander.Pattern": operation.pattern or "*",
                    "com.ghisler.android.TotalCommander.Recursive": operation.recursive
                }
            ))
        
        elif operation.action == TCAction.MOVE:
            intents.append(AndroidIntent(
                action=cls.ACTION_VIEW,
                package=cls.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": "cm_MoveOnly",
                    "com.ghisler.android.TotalCommander.SourcePath": operation.source_path,
                    "com.ghisler.android.TotalCommander.TargetPath": operation.destination_path,
                    "com.ghisler.android.TotalCommander.Pattern": operation.pattern or "*"
                }
            ))
        
        elif operation.action == TCAction.DELETE:
            intents.append(AndroidIntent(
                action=cls.ACTION_VIEW,
                package=cls.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": "cm_Delete",
                    "com.ghisler.android.TotalCommander.Path": operation.source_path,
                    "com.ghisler.android.TotalCommander.Confirm": operation.confirm
                }
            ))
        
        elif operation.action == TCAction.MKDIR:
            intents.append(AndroidIntent(
                action=cls.ACTION_VIEW,
                package=cls.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": "cm_MkDir",
                    "com.ghisler.android.TotalCommander.Path": operation.destination_path
                }
            ))
        
        elif operation.action == TCAction.SEARCH:
            intents.append(AndroidIntent(
                action=cls.ACTION_VIEW,
                package=cls.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": "cm_SearchFor",
                    "com.ghisler.android.TotalCommander.SearchPath": operation.source_path or "/sdcard",
                    "com.ghisler.android.TotalCommander.SearchPattern": operation.pattern or "*"
                }
            ))
        
        elif operation.action == TCAction.PACK:
            intents.append(AndroidIntent(
                action=cls.ACTION_VIEW,
                package=cls.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": "cm_PackFiles",
                    "com.ghisler.android.TotalCommander.SourcePath": operation.source_path,
                    "com.ghisler.android.TotalCommander.ArchivePath": operation.destination_path
                }
            ))
        
        elif operation.action == TCAction.UNPACK:
            intents.append(AndroidIntent(
                action=cls.ACTION_VIEW,
                package=cls.TC_PACKAGE,
                extras={
                    "com.ghisler.android.TotalCommander.Command": "cm_UnpackFiles",
                    "com.ghisler.android.TotalCommander.ArchivePath": operation.source_path,
                    "com.ghisler.android.TotalCommander.TargetPath": operation.destination_path
                }
            ))
        
        return intents


class TCConnector:
    """
    Main connector class for Marduk-Total Commander integration.
    
    This class serves as the primary interface between the Marduk Cognitive Core
    and Total Commander. It handles:
    - Translation of high-level commands to Android Intents
    - Execution tracking and feedback
    - Error handling and recovery
    """
    
    def __init__(self, middleware_url: str = "http://localhost:8080"):
        """
        Initialize the TC-Connector.
        
        Args:
            middleware_url: URL of the Middleware Bridge service
        """
        self.middleware_url = middleware_url
        self.intent_builder = IntentBuilder()
        self.operation_history: List[Dict[str, Any]] = []
        self.callbacks: Dict[str, Callable] = {}
        
        logger.info(f"TCConnector initialized with middleware at {middleware_url}")
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for operation events."""
        self.callbacks[event] = callback
        logger.debug(f"Registered callback for event: {event}")
    
    def _emit_event(self, event: str, data: Any) -> None:
        """Emit an event to registered callbacks."""
        if event in self.callbacks:
            self.callbacks[event](data)
    
    def translate_operation(self, operation: FileOperation) -> List[AndroidIntent]:
        """
        Translate a FileOperation into Android Intents.
        
        Args:
            operation: The file operation to translate
            
        Returns:
            List of AndroidIntent objects to execute
        """
        if not operation.validate():
            raise ValueError(f"Invalid operation parameters for {operation.action}")
        
        intents = self.intent_builder.build_file_operation_intent(operation)
        
        logger.info(f"Translated {operation.action.value} to {len(intents)} intent(s)")
        return intents
    
    def execute_operation(self, operation: FileOperation) -> Dict[str, Any]:
        """
        Execute a file operation through the Middleware Bridge.
        
        Args:
            operation: The file operation to execute
            
        Returns:
            Execution result dictionary
        """
        result = {
            "operation": operation.action.value,
            "source": operation.source_path,
            "destination": operation.destination_path,
            "status": "pending",
            "intents": [],
            "errors": []
        }
        
        try:
            # Translate operation to intents
            intents = self.translate_operation(operation)
            result["intents"] = [intent.to_dict() for intent in intents]
            
            # Record in history
            self.operation_history.append({
                "operation": operation,
                "intents": intents,
                "result": result
            })
            
            # Emit pre-execution event
            self._emit_event("pre_execute", {"operation": operation, "intents": intents})
            
            # In a real implementation, this would send to the Middleware Bridge
            # For now, we generate the execution payload
            result["status"] = "ready"
            result["middleware_payload"] = {
                "type": "tc_operation",
                "intents": [intent.to_dict() for intent in intents],
                "metadata": operation.metadata
            }
            
            # Emit post-execution event
            self._emit_event("post_execute", result)
            
            logger.info(f"Operation {operation.action.value} prepared successfully")
            
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.error(f"Operation failed: {e}")
            self._emit_event("error", {"operation": operation, "error": e})
        
        return result
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get the history of executed operations."""
        return self.operation_history
    
    def clear_history(self) -> None:
        """Clear the operation history."""
        self.operation_history = []
        logger.info("Operation history cleared")


# Convenience functions for common operations
def copy_files(
    connector: TCConnector,
    source: str,
    destination: str,
    pattern: str = "*",
    recursive: bool = False
) -> Dict[str, Any]:
    """Copy files from source to destination."""
    operation = FileOperation(
        action=TCAction.COPY,
        source_path=source,
        destination_path=destination,
        pattern=pattern,
        recursive=recursive
    )
    return connector.execute_operation(operation)


def move_files(
    connector: TCConnector,
    source: str,
    destination: str,
    pattern: str = "*"
) -> Dict[str, Any]:
    """Move files from source to destination."""
    operation = FileOperation(
        action=TCAction.MOVE,
        source_path=source,
        destination_path=destination,
        pattern=pattern
    )
    return connector.execute_operation(operation)


def delete_files(
    connector: TCConnector,
    path: str,
    confirm: bool = True
) -> Dict[str, Any]:
    """Delete files at the specified path."""
    operation = FileOperation(
        action=TCAction.DELETE,
        source_path=path,
        confirm=confirm
    )
    return connector.execute_operation(operation)


def search_files(
    connector: TCConnector,
    search_path: str,
    pattern: str
) -> Dict[str, Any]:
    """Search for files matching a pattern."""
    operation = FileOperation(
        action=TCAction.SEARCH,
        source_path=search_path,
        pattern=pattern
    )
    return connector.execute_operation(operation)


def navigate_to(connector: TCConnector, path: str) -> Dict[str, Any]:
    """Navigate Total Commander to a specific path."""
    operation = FileOperation(
        action=TCAction.NAVIGATE,
        destination_path=path
    )
    return connector.execute_operation(operation)


def create_archive(
    connector: TCConnector,
    source: str,
    archive_path: str
) -> Dict[str, Any]:
    """Create an archive from source files."""
    operation = FileOperation(
        action=TCAction.PACK,
        source_path=source,
        destination_path=archive_path
    )
    return connector.execute_operation(operation)


def extract_archive(
    connector: TCConnector,
    archive_path: str,
    destination: str
) -> Dict[str, Any]:
    """Extract an archive to destination."""
    operation = FileOperation(
        action=TCAction.UNPACK,
        source_path=archive_path,
        destination_path=destination
    )
    return connector.execute_operation(operation)


if __name__ == "__main__":
    # Example usage
    connector = TCConnector()
    
    # Example: Copy PDFs from Downloads to Documents
    result = copy_files(
        connector,
        source="/sdcard/Download",
        destination="/sdcard/Documents/PDF_Archive",
        pattern="*.pdf"
    )
    
    print("Operation Result:")
    print(json.dumps(result, indent=2))
    
    # Example: Generate ADB commands
    operation = FileOperation(
        action=TCAction.NAVIGATE,
        destination_path="/sdcard/Documents"
    )
    intents = connector.translate_operation(operation)
    
    print("\nGenerated ADB Commands:")
    for intent in intents:
        print(intent.to_adb_command())
