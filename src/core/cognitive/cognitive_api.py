"""
Marduk Cognitive API Integration Layer
======================================

This module implements the cognitive API interface that connects Marduk's
higher-level reasoning capabilities with the TC-Connector system. It handles:

1. Natural language command parsing
2. Intent classification and entity extraction
3. Procedural memory for learned file operations
4. Episodic memory for operation history and context
5. Reflection and self-optimization

Architecture follows the Marduk Cognitive Core design:
    User Input -> Deliberation Engine -> Procedural Memory -> TC-Connector -> Total Commander

Author: Manus AI
Date: December 28, 2025
"""

import asyncio
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MardukCognitive")


# ============================================================================
# Memory Systems
# ============================================================================

class MemoryType(Enum):
    """Types of memory in the Marduk cognitive architecture."""
    DECLARATIVE = "declarative"    # Facts and concepts
    PROCEDURAL = "procedural"      # Skills and procedures
    EPISODIC = "episodic"          # Experiences and events
    SEMANTIC = "semantic"          # Meanings and relationships


@dataclass
class MemoryEntry:
    """A single entry in Marduk's memory system."""
    id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    access_count: int = 0
    relevance_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self) -> None:
        """Record an access to this memory entry."""
        self.access_count += 1
        self.metadata["last_accessed"] = datetime.utcnow().isoformat()


class MemorySystem:
    """
    Marduk's memory system implementation.
    
    Manages declarative, procedural, episodic, and semantic memories
    for the cognitive core.
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        self.memories: Dict[str, MemoryEntry] = {}
        self.indices: Dict[MemoryType, List[str]] = {
            mt: [] for mt in MemoryType
        }
        self.persistence_path = persistence_path
        
        if persistence_path:
            self._load_from_disk()
    
    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        self.memories[entry.id] = entry
        self.indices[entry.memory_type].append(entry.id)
        
        logger.debug(f"Stored memory: {entry.id} ({entry.memory_type.value})")
        return entry.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID."""
        entry = self.memories.get(memory_id)
        if entry:
            entry.access()
        return entry
    
    def query(
        self,
        memory_type: Optional[MemoryType] = None,
        filter_fn: Optional[Callable[[MemoryEntry], bool]] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Query memories with optional filtering."""
        if memory_type:
            candidate_ids = self.indices[memory_type]
        else:
            candidate_ids = list(self.memories.keys())
        
        candidates = [self.memories[mid] for mid in candidate_ids]
        
        if filter_fn:
            candidates = [c for c in candidates if filter_fn(c)]
        
        # Sort by relevance and recency
        candidates.sort(
            key=lambda x: (x.relevance_score, x.access_count),
            reverse=True
        )
        
        return candidates[:limit]
    
    def update_relevance(self, memory_id: str, delta: float) -> None:
        """Update the relevance score of a memory."""
        if memory_id in self.memories:
            self.memories[memory_id].relevance_score += delta
    
    def _load_from_disk(self) -> None:
        """Load memories from persistent storage."""
        if self.persistence_path and Path(self.persistence_path).exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    for entry_data in data.get("memories", []):
                        entry = MemoryEntry(
                            id=entry_data["id"],
                            memory_type=MemoryType(entry_data["memory_type"]),
                            content=entry_data["content"],
                            timestamp=entry_data.get("timestamp"),
                            access_count=entry_data.get("access_count", 0),
                            relevance_score=entry_data.get("relevance_score", 1.0),
                            metadata=entry_data.get("metadata", {})
                        )
                        self.store(entry)
                logger.info(f"Loaded {len(self.memories)} memories from disk")
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
    
    def save_to_disk(self) -> None:
        """Save memories to persistent storage."""
        if self.persistence_path:
            try:
                data = {
                    "memories": [
                        {
                            "id": m.id,
                            "memory_type": m.memory_type.value,
                            "content": m.content,
                            "timestamp": m.timestamp,
                            "access_count": m.access_count,
                            "relevance_score": m.relevance_score,
                            "metadata": m.metadata
                        }
                        for m in self.memories.values()
                    ]
                }
                with open(self.persistence_path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved {len(self.memories)} memories to disk")
            except Exception as e:
                logger.error(f"Failed to save memories: {e}")


# ============================================================================
# Intent Classification
# ============================================================================

class FileIntent(Enum):
    """Classified intents for file operations."""
    COPY = "copy"
    MOVE = "move"
    DELETE = "delete"
    SEARCH = "search"
    ORGANIZE = "organize"
    ARCHIVE = "archive"
    EXTRACT = "extract"
    NAVIGATE = "navigate"
    CREATE_FOLDER = "create_folder"
    RENAME = "rename"
    VIEW = "view"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Result of parsing a natural language command."""
    original_text: str
    intent: FileIntent
    entities: Dict[str, Any]
    confidence: float
    alternatives: List[Tuple[FileIntent, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "intent": self.intent.value,
            "entities": self.entities,
            "confidence": self.confidence,
            "alternatives": [(i.value, c) for i, c in self.alternatives]
        }


class CommandParser:
    """
    Natural language command parser for file operations.
    
    Uses pattern matching and keyword extraction to classify
    user intents and extract relevant entities (paths, patterns, etc.).
    """
    
    # Intent patterns (regex-based for initial implementation)
    INTENT_PATTERNS = {
        FileIntent.COPY: [
            r"copy\s+(.+?)\s+to\s+(.+)",
            r"duplicate\s+(.+?)\s+(?:to|in)\s+(.+)",
            r"make\s+a?\s*copy\s+of\s+(.+)",
        ],
        FileIntent.MOVE: [
            r"move\s+(.+?)\s+to\s+(.+)",
            r"transfer\s+(.+?)\s+to\s+(.+)",
            r"relocate\s+(.+?)\s+to\s+(.+)",
        ],
        FileIntent.DELETE: [
            r"delete\s+(.+)",
            r"remove\s+(.+)",
            r"trash\s+(.+)",
            r"get\s+rid\s+of\s+(.+)",
        ],
        FileIntent.SEARCH: [
            r"find\s+(.+)",
            r"search\s+(?:for\s+)?(.+)",
            r"locate\s+(.+)",
            r"where\s+(?:is|are)\s+(.+)",
        ],
        FileIntent.ORGANIZE: [
            r"organize\s+(.+)",
            r"sort\s+(.+)",
            r"arrange\s+(.+)",
            r"tidy\s+(?:up\s+)?(.+)",
        ],
        FileIntent.ARCHIVE: [
            r"archive\s+(.+)",
            r"compress\s+(.+)",
            r"zip\s+(.+)",
            r"pack\s+(.+)",
        ],
        FileIntent.EXTRACT: [
            r"extract\s+(.+)",
            r"unzip\s+(.+)",
            r"unpack\s+(.+)",
            r"decompress\s+(.+)",
        ],
        FileIntent.NAVIGATE: [
            r"(?:go\s+to|open|navigate\s+to)\s+(.+)",
            r"show\s+(?:me\s+)?(.+)",
        ],
        FileIntent.CREATE_FOLDER: [
            r"create\s+(?:a\s+)?(?:folder|directory)\s+(?:called\s+)?(.+)",
            r"make\s+(?:a\s+)?(?:folder|directory)\s+(?:called\s+)?(.+)",
            r"mkdir\s+(.+)",
        ],
        FileIntent.RENAME: [
            r"rename\s+(.+?)\s+to\s+(.+)",
            r"change\s+(?:the\s+)?name\s+of\s+(.+?)\s+to\s+(.+)",
        ],
    }
    
    # Entity extraction patterns
    PATH_PATTERN = r"(?:/[\w\-\.]+)+|(?:[\w\-\.]+/[\w\-\.]+)"
    FILE_PATTERN = r"\*\.?\w+|\w+\.\w+"
    
    # Common path aliases
    PATH_ALIASES = {
        "downloads": "/sdcard/Download",
        "documents": "/sdcard/Documents",
        "pictures": "/sdcard/Pictures",
        "photos": "/sdcard/DCIM",
        "music": "/sdcard/Music",
        "videos": "/sdcard/Movies",
        "desktop": "/sdcard",
        "home": "/sdcard",
        "root": "/",
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            self.compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def parse(self, text: str) -> ParsedCommand:
        """Parse a natural language command."""
        text = text.strip()
        best_match = None
        best_confidence = 0.0
        alternatives = []
        
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    confidence = self._calculate_confidence(text, match)
                    
                    if confidence > best_confidence:
                        if best_match:
                            alternatives.append((best_match[0], best_confidence))
                        best_match = (intent, match)
                        best_confidence = confidence
                    else:
                        alternatives.append((intent, confidence))
        
        if best_match:
            intent, match = best_match
            entities = self._extract_entities(text, intent, match)
        else:
            intent = FileIntent.UNKNOWN
            entities = {"raw_text": text}
        
        return ParsedCommand(
            original_text=text,
            intent=intent,
            entities=entities,
            confidence=best_confidence,
            alternatives=alternatives[:3]  # Top 3 alternatives
        )
    
    def _calculate_confidence(self, text: str, match: re.Match) -> float:
        """Calculate confidence score for a match."""
        # Base confidence from match coverage
        match_length = match.end() - match.start()
        coverage = match_length / len(text)
        
        # Adjust for position (prefer matches at start)
        position_factor = 1.0 - (match.start() / len(text)) * 0.2
        
        return min(coverage * position_factor, 1.0)
    
    def _extract_entities(
        self,
        text: str,
        intent: FileIntent,
        match: re.Match
    ) -> Dict[str, Any]:
        """Extract entities from the matched text."""
        entities = {}
        groups = match.groups()
        
        if intent in [FileIntent.COPY, FileIntent.MOVE, FileIntent.RENAME]:
            if len(groups) >= 2:
                entities["source"] = self._resolve_path(groups[0])
                entities["destination"] = self._resolve_path(groups[1])
            elif len(groups) >= 1:
                entities["source"] = self._resolve_path(groups[0])
        
        elif intent in [FileIntent.DELETE, FileIntent.SEARCH, FileIntent.ORGANIZE,
                        FileIntent.ARCHIVE, FileIntent.EXTRACT, FileIntent.NAVIGATE,
                        FileIntent.CREATE_FOLDER]:
            if groups:
                path_or_pattern = groups[0]
                entities["target"] = self._resolve_path(path_or_pattern)
        
        # Extract file patterns
        pattern_match = re.search(self.FILE_PATTERN, text)
        if pattern_match:
            entities["pattern"] = pattern_match.group()
        
        # Extract additional paths
        all_paths = re.findall(self.PATH_PATTERN, text)
        if all_paths:
            entities["paths"] = [self._resolve_path(p) for p in all_paths]
        
        return entities
    
    def _resolve_path(self, path_text: str) -> str:
        """Resolve path aliases and clean up path text."""
        path_text = path_text.strip().lower()
        
        # Check for aliases
        for alias, resolved in self.PATH_ALIASES.items():
            if alias in path_text:
                path_text = path_text.replace(alias, resolved)
                break
        
        # Clean up
        path_text = path_text.strip("'\"")
        
        # Ensure absolute path
        if not path_text.startswith("/"):
            path_text = f"/sdcard/{path_text}"
        
        return path_text


# ============================================================================
# Procedural Memory for File Operations
# ============================================================================

@dataclass
class Procedure:
    """A learned procedure for file operations."""
    id: str
    name: str
    description: str
    intent: FileIntent
    steps: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    success_rate: float = 1.0
    execution_count: int = 0
    
    def to_memory_entry(self) -> MemoryEntry:
        """Convert to a memory entry for storage."""
        return MemoryEntry(
            id=self.id,
            memory_type=MemoryType.PROCEDURAL,
            content={
                "name": self.name,
                "description": self.description,
                "intent": self.intent.value,
                "steps": self.steps,
                "parameters": self.parameters,
                "success_rate": self.success_rate,
                "execution_count": self.execution_count
            }
        )


class ProceduralMemory:
    """
    Manages procedural knowledge for file operations.
    
    Stores learned procedures that map intents to sequences
    of TC-Connector operations.
    """
    
    def __init__(self, memory_system: MemorySystem):
        self.memory = memory_system
        self._initialize_default_procedures()
    
    def _initialize_default_procedures(self) -> None:
        """Initialize default procedures for common operations."""
        default_procedures = [
            Procedure(
                id="proc_copy_files",
                name="Copy Files",
                description="Copy files from source to destination",
                intent=FileIntent.COPY,
                steps=[
                    {"action": "validate_paths", "params": ["source", "destination"]},
                    {"action": "tc_operation", "operation": "copy", "params": ["source", "destination", "pattern"]}
                ],
                parameters={"source": None, "destination": None, "pattern": "*"}
            ),
            Procedure(
                id="proc_move_files",
                name="Move Files",
                description="Move files from source to destination",
                intent=FileIntent.MOVE,
                steps=[
                    {"action": "validate_paths", "params": ["source", "destination"]},
                    {"action": "tc_operation", "operation": "move", "params": ["source", "destination", "pattern"]}
                ],
                parameters={"source": None, "destination": None, "pattern": "*"}
            ),
            Procedure(
                id="proc_delete_files",
                name="Delete Files",
                description="Delete files at specified path",
                intent=FileIntent.DELETE,
                steps=[
                    {"action": "validate_paths", "params": ["target"]},
                    {"action": "confirm_destructive", "message": "Confirm deletion?"},
                    {"action": "tc_operation", "operation": "delete", "params": ["target"]}
                ],
                parameters={"target": None, "confirm": True}
            ),
            Procedure(
                id="proc_search_files",
                name="Search Files",
                description="Search for files matching a pattern",
                intent=FileIntent.SEARCH,
                steps=[
                    {"action": "tc_operation", "operation": "search", "params": ["target", "pattern"]}
                ],
                parameters={"target": "/sdcard", "pattern": "*"}
            ),
            Procedure(
                id="proc_organize_files",
                name="Organize Files",
                description="Organize files by type into folders",
                intent=FileIntent.ORGANIZE,
                steps=[
                    {"action": "scan_directory", "params": ["target"]},
                    {"action": "classify_files", "params": ["files"]},
                    {"action": "create_folders", "params": ["categories"]},
                    {"action": "move_files_by_category", "params": ["files", "categories"]}
                ],
                parameters={"target": None, "categories": ["Documents", "Images", "Videos", "Music", "Archives", "Other"]}
            ),
            Procedure(
                id="proc_archive_files",
                name="Archive Files",
                description="Create an archive from files",
                intent=FileIntent.ARCHIVE,
                steps=[
                    {"action": "validate_paths", "params": ["target"]},
                    {"action": "tc_operation", "operation": "pack", "params": ["target", "archive_path"]}
                ],
                parameters={"target": None, "archive_path": None, "format": "zip"}
            ),
            Procedure(
                id="proc_extract_archive",
                name="Extract Archive",
                description="Extract files from an archive",
                intent=FileIntent.EXTRACT,
                steps=[
                    {"action": "validate_paths", "params": ["target"]},
                    {"action": "tc_operation", "operation": "unpack", "params": ["target", "destination"]}
                ],
                parameters={"target": None, "destination": None}
            ),
            Procedure(
                id="proc_navigate",
                name="Navigate to Path",
                description="Navigate Total Commander to a path",
                intent=FileIntent.NAVIGATE,
                steps=[
                    {"action": "tc_operation", "operation": "navigate", "params": ["target"]}
                ],
                parameters={"target": None}
            ),
            Procedure(
                id="proc_create_folder",
                name="Create Folder",
                description="Create a new folder",
                intent=FileIntent.CREATE_FOLDER,
                steps=[
                    {"action": "tc_operation", "operation": "mkdir", "params": ["target"]}
                ],
                parameters={"target": None}
            ),
        ]
        
        for proc in default_procedures:
            self.memory.store(proc.to_memory_entry())
        
        logger.info(f"Initialized {len(default_procedures)} default procedures")
    
    def get_procedure(self, intent: FileIntent) -> Optional[Procedure]:
        """Get the best procedure for an intent."""
        entries = self.memory.query(
            memory_type=MemoryType.PROCEDURAL,
            filter_fn=lambda e: e.content.get("intent") == intent.value,
            limit=1
        )
        
        if entries:
            entry = entries[0]
            return Procedure(
                id=entry.id,
                name=entry.content["name"],
                description=entry.content["description"],
                intent=FileIntent(entry.content["intent"]),
                steps=entry.content["steps"],
                parameters=entry.content["parameters"],
                success_rate=entry.content.get("success_rate", 1.0),
                execution_count=entry.content.get("execution_count", 0)
            )
        
        return None
    
    def record_execution(self, procedure_id: str, success: bool) -> None:
        """Record the result of a procedure execution."""
        entry = self.memory.retrieve(procedure_id)
        if entry:
            entry.content["execution_count"] = entry.content.get("execution_count", 0) + 1
            
            # Update success rate with exponential moving average
            current_rate = entry.content.get("success_rate", 1.0)
            alpha = 0.1  # Learning rate
            new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
            entry.content["success_rate"] = new_rate
            
            # Update relevance based on success
            self.memory.update_relevance(procedure_id, 0.1 if success else -0.05)


# ============================================================================
# Deliberation Engine
# ============================================================================

class DeliberationEngine:
    """
    The deliberation engine for Marduk's cognitive processing.
    
    Coordinates between:
    - Command parsing (understanding user intent)
    - Procedural memory (retrieving relevant procedures)
    - Execution planning (creating action sequences)
    - Reflection (learning from outcomes)
    """
    
    def __init__(
        self,
        memory_system: MemorySystem,
        procedural_memory: ProceduralMemory,
        command_parser: CommandParser
    ):
        self.memory = memory_system
        self.procedural = procedural_memory
        self.parser = command_parser
        self.execution_history: List[Dict[str, Any]] = []
    
    async def process_command(self, text: str) -> Dict[str, Any]:
        """
        Process a natural language command through the full cognitive pipeline.
        
        Returns an execution plan ready for the TC-Connector.
        """
        # Step 1: Parse the command
        parsed = self.parser.parse(text)
        logger.info(f"Parsed command: {parsed.intent.value} (confidence: {parsed.confidence:.2f})")
        
        if parsed.intent == FileIntent.UNKNOWN:
            return {
                "status": "error",
                "message": "Could not understand the command",
                "parsed": parsed.to_dict()
            }
        
        # Step 2: Retrieve relevant procedure
        procedure = self.procedural.get_procedure(parsed.intent)
        
        if not procedure:
            return {
                "status": "error",
                "message": f"No procedure found for intent: {parsed.intent.value}",
                "parsed": parsed.to_dict()
            }
        
        # Step 3: Build execution plan
        execution_plan = self._build_execution_plan(parsed, procedure)
        
        # Step 4: Store in episodic memory
        episode_id = self._record_episode(parsed, procedure, execution_plan)
        
        return {
            "status": "ready",
            "command_id": str(uuid.uuid4()),
            "episode_id": episode_id,
            "parsed": parsed.to_dict(),
            "procedure": {
                "id": procedure.id,
                "name": procedure.name,
                "success_rate": procedure.success_rate
            },
            "execution_plan": execution_plan
        }
    
    def _build_execution_plan(
        self,
        parsed: ParsedCommand,
        procedure: Procedure
    ) -> Dict[str, Any]:
        """Build an execution plan from parsed command and procedure."""
        # Merge parsed entities with procedure parameters
        parameters = procedure.parameters.copy()
        
        # Map parsed entities to procedure parameters
        entity_mapping = {
            "source": ["source", "target"],
            "destination": ["destination", "archive_path"],
            "target": ["target", "source"],
            "pattern": ["pattern"],
        }
        
        for entity_key, param_keys in entity_mapping.items():
            if entity_key in parsed.entities:
                for param_key in param_keys:
                    if param_key in parameters:
                        parameters[param_key] = parsed.entities[entity_key]
                        break
        
        # Build step-by-step plan
        steps = []
        for step in procedure.steps:
            step_plan = step.copy()
            
            # Resolve parameter references
            if "params" in step_plan:
                resolved_params = {}
                for param in step_plan["params"]:
                    resolved_params[param] = parameters.get(param)
                step_plan["resolved_params"] = resolved_params
            
            steps.append(step_plan)
        
        return {
            "procedure_id": procedure.id,
            "parameters": parameters,
            "steps": steps,
            "intent": parsed.intent.value
        }
    
    def _record_episode(
        self,
        parsed: ParsedCommand,
        procedure: Procedure,
        execution_plan: Dict[str, Any]
    ) -> str:
        """Record this interaction in episodic memory."""
        episode = MemoryEntry(
            id=f"episode_{uuid.uuid4().hex[:8]}",
            memory_type=MemoryType.EPISODIC,
            content={
                "command": parsed.original_text,
                "intent": parsed.intent.value,
                "entities": parsed.entities,
                "procedure_used": procedure.id,
                "execution_plan": execution_plan,
                "outcome": "pending"
            }
        )
        
        self.memory.store(episode)
        return episode.id
    
    def record_outcome(self, episode_id: str, success: bool, details: Dict[str, Any] = None) -> None:
        """Record the outcome of an execution."""
        episode = self.memory.retrieve(episode_id)
        if episode:
            episode.content["outcome"] = "success" if success else "failure"
            episode.content["outcome_details"] = details or {}
            
            # Update procedural memory
            procedure_id = episode.content.get("procedure_used")
            if procedure_id:
                self.procedural.record_execution(procedure_id, success)
        
        logger.info(f"Recorded outcome for episode {episode_id}: {'success' if success else 'failure'}")


# ============================================================================
# Cognitive API Interface
# ============================================================================

class CognitiveAPI:
    """
    Main API interface for the Marduk Cognitive Core.
    
    Provides high-level methods for:
    - Processing natural language commands
    - Executing file operations
    - Querying memory systems
    - Managing cognitive state
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        self.memory = MemorySystem(persistence_path)
        self.parser = CommandParser()
        self.procedural = ProceduralMemory(self.memory)
        self.deliberation = DeliberationEngine(
            self.memory,
            self.procedural,
            self.parser
        )
        
        logger.info("Marduk Cognitive API initialized")
    
    async def process(self, command: str) -> Dict[str, Any]:
        """Process a natural language command."""
        return await self.deliberation.process_command(command)
    
    def record_result(self, episode_id: str, success: bool, details: Dict[str, Any] = None) -> None:
        """Record the result of an execution."""
        self.deliberation.record_outcome(episode_id, success, details)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return {
            "total_memories": len(self.memory.memories),
            "by_type": {
                mt.value: len(self.memory.indices[mt])
                for mt in MemoryType
            }
        }
    
    def save_state(self) -> None:
        """Save cognitive state to disk."""
        self.memory.save_to_disk()
    
    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent episodic memories."""
        episodes = self.memory.query(
            memory_type=MemoryType.EPISODIC,
            limit=limit
        )
        return [e.content for e in episodes]


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of the Cognitive API."""
    api = CognitiveAPI()
    
    # Test commands
    test_commands = [
        "Move all PDFs from Downloads to Documents",
        "Find large files in my pictures folder",
        "Archive the logs folder",
        "Delete temporary files",
        "Organize my downloads folder",
        "Go to the music folder",
        "Create a folder called Projects",
    ]
    
    print("=" * 60)
    print("Marduk Cognitive API - Command Processing Demo")
    print("=" * 60)
    
    for cmd in test_commands:
        print(f"\nCommand: \"{cmd}\"")
        print("-" * 40)
        
        result = await api.process(cmd)
        
        print(f"Status: {result['status']}")
        if result['status'] == 'ready':
            print(f"Intent: {result['parsed']['intent']}")
            print(f"Confidence: {result['parsed']['confidence']:.2f}")
            print(f"Procedure: {result['procedure']['name']}")
            print(f"Parameters: {json.dumps(result['execution_plan']['parameters'], indent=2)}")
        else:
            print(f"Message: {result.get('message', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Memory Statistics:")
    print(json.dumps(api.get_memory_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
