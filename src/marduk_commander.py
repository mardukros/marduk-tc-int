"""
Marduk Commander - Main Application
====================================

This is the main entry point for the Marduk-Commander application, which
integrates the Marduk Cognitive Core with Total Commander for Android.

The application provides:
1. Natural language file management through voice or text
2. Content-aware file organization using on-device AI
3. Autonomous file operations with learning capabilities
4. Real-time feedback via WebSocket communication

Architecture:
    User Input -> Cognitive API -> Deliberation -> TC-Connector -> Bridge -> Total Commander
                      |                                                           |
                      +<-------------------- Feedback Loop ----------------------+

Author: Manus AI
Date: December 28, 2025
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.cognitive.cognitive_api import CognitiveAPI, FileIntent
from connectors.tc_connector import TCConnector, FileOperation, TCAction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('marduk_commander.log')
    ]
)
logger = logging.getLogger("MardukCommander")


class MardukCommander:
    """
    Main Marduk Commander application class.
    
    Orchestrates the cognitive processing pipeline and manages
    communication between all components.
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        middleware_url: str = "http://localhost:8080",
        persistence_path: Optional[str] = None
    ):
        """
        Initialize Marduk Commander.
        
        Args:
            middleware_url: URL of the Middleware Bridge server
            persistence_path: Path for persisting cognitive state
        """
        self.middleware_url = middleware_url
        self.persistence_path = persistence_path or str(
            Path.home() / ".marduk_commander" / "state.json"
        )
        
        # Ensure persistence directory exists
        Path(self.persistence_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cognitive = CognitiveAPI(self.persistence_path)
        self.tc_connector = TCConnector(middleware_url)
        
        # Register callbacks
        self._setup_callbacks()
        
        # State
        self.running = False
        self.session_stats = {
            "commands_processed": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "session_start": None
        }
        
        logger.info(f"Marduk Commander v{self.VERSION} initialized")
    
    def _setup_callbacks(self) -> None:
        """Set up event callbacks for the TC-Connector."""
        self.tc_connector.register_callback("pre_execute", self._on_pre_execute)
        self.tc_connector.register_callback("post_execute", self._on_post_execute)
        self.tc_connector.register_callback("error", self._on_error)
    
    def _on_pre_execute(self, data: Dict[str, Any]) -> None:
        """Callback before operation execution."""
        logger.info(f"Executing operation: {data['operation'].action.value}")
    
    def _on_post_execute(self, data: Dict[str, Any]) -> None:
        """Callback after operation execution."""
        status = data.get("status", "unknown")
        logger.info(f"Operation completed with status: {status}")
        
        if status == "ready":
            self.session_stats["successful_operations"] += 1
        else:
            self.session_stats["failed_operations"] += 1
    
    def _on_error(self, data: Dict[str, Any]) -> None:
        """Callback on operation error."""
        error = data.get("error", "Unknown error")
        logger.error(f"Operation error: {error}")
        self.session_stats["failed_operations"] += 1
    
    async def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command.
        
        This is the main entry point for user commands. It:
        1. Parses the command using the Cognitive API
        2. Translates to TC operations
        3. Executes via the TC-Connector
        4. Returns results and updates memory
        
        Args:
            command: Natural language command string
            
        Returns:
            Result dictionary with status and details
        """
        self.session_stats["commands_processed"] += 1
        
        result = {
            "command": command,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
            "cognitive_result": None,
            "execution_result": None,
            "errors": []
        }
        
        try:
            # Step 1: Cognitive processing
            logger.info(f"Processing command: {command}")
            cognitive_result = await self.cognitive.process(command)
            result["cognitive_result"] = cognitive_result
            
            if cognitive_result["status"] != "ready":
                result["status"] = "parse_error"
                result["errors"].append(cognitive_result.get("message", "Failed to parse command"))
                return result
            
            # Step 2: Map to TC operation
            execution_plan = cognitive_result["execution_plan"]
            tc_operation = self._map_to_tc_operation(execution_plan)
            
            if not tc_operation:
                result["status"] = "mapping_error"
                result["errors"].append("Could not map to TC operation")
                return result
            
            # Step 3: Execute via TC-Connector
            execution_result = self.tc_connector.execute_operation(tc_operation)
            result["execution_result"] = execution_result
            
            # Step 4: Record outcome
            success = execution_result.get("status") == "ready"
            self.cognitive.record_result(
                cognitive_result["episode_id"],
                success,
                {"execution_result": execution_result}
            )
            
            result["status"] = "success" if success else "execution_error"
            
        except Exception as e:
            logger.exception(f"Error processing command: {e}")
            result["status"] = "error"
            result["errors"].append(str(e))
        
        return result
    
    def _map_to_tc_operation(self, execution_plan: Dict[str, Any]) -> Optional[FileOperation]:
        """Map an execution plan to a TC operation."""
        intent = execution_plan.get("intent")
        params = execution_plan.get("parameters", {})
        
        # Intent to TCAction mapping
        intent_action_map = {
            "copy": TCAction.COPY,
            "move": TCAction.MOVE,
            "delete": TCAction.DELETE,
            "search": TCAction.SEARCH,
            "archive": TCAction.PACK,
            "extract": TCAction.UNPACK,
            "navigate": TCAction.NAVIGATE,
            "create_folder": TCAction.MKDIR,
            "organize": TCAction.MOVE,  # Organize uses move operations
            "rename": TCAction.RENAME,
        }
        
        action = intent_action_map.get(intent)
        if not action:
            logger.warning(f"No TC action mapping for intent: {intent}")
            return None
        
        return FileOperation(
            action=action,
            source_path=params.get("source") or params.get("target"),
            destination_path=params.get("destination") or params.get("archive_path"),
            pattern=params.get("pattern", "*"),
            recursive=params.get("recursive", False),
            confirm=params.get("confirm", True),
            metadata={"execution_plan": execution_plan}
        )
    
    async def interactive_session(self) -> None:
        """
        Run an interactive command session.
        
        Provides a REPL-style interface for entering commands.
        """
        self.running = True
        self.session_stats["session_start"] = datetime.utcnow().isoformat()
        
        print("\n" + "=" * 60)
        print(f"  Marduk Commander v{self.VERSION}")
        print("  Natural Language File Management for Total Commander")
        print("=" * 60)
        print("\nType 'help' for available commands, 'quit' to exit.\n")
        
        while self.running:
            try:
                # Get user input
                command = input("marduk> ").strip()
                
                if not command:
                    continue
                
                # Handle special commands
                if command.lower() == "quit" or command.lower() == "exit":
                    break
                elif command.lower() == "help":
                    self._print_help()
                    continue
                elif command.lower() == "stats":
                    self._print_stats()
                    continue
                elif command.lower() == "history":
                    self._print_history()
                    continue
                elif command.lower() == "save":
                    self.cognitive.save_state()
                    print("State saved.")
                    continue
                
                # Process the command
                result = await self.process_command(command)
                
                # Display result
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
            except EOFError:
                break
        
        # Cleanup
        self.running = False
        self.cognitive.save_state()
        print("\nSession ended. State saved.")
        self._print_stats()
    
    def _print_help(self) -> None:
        """Print help information."""
        help_text = """
Available Commands:
-------------------
File Operations:
  "Copy [files] from [source] to [destination]"
  "Move [files] to [destination]"
  "Delete [files/folder]"
  "Find [pattern] in [folder]"
  "Organize [folder]"
  "Archive [folder] to [archive.zip]"
  "Extract [archive] to [folder]"
  "Create folder [name]"
  "Go to [folder]"

System Commands:
  help     - Show this help message
  stats    - Show session statistics
  history  - Show recent command history
  save     - Save cognitive state
  quit     - Exit the application

Examples:
  "Move all PDFs from Downloads to Documents"
  "Find large images in Pictures"
  "Archive my project folder"
  "Organize Downloads by file type"
"""
        print(help_text)
    
    def _print_stats(self) -> None:
        """Print session statistics."""
        print("\nSession Statistics:")
        print("-" * 30)
        print(f"  Session Start: {self.session_stats['session_start']}")
        print(f"  Commands Processed: {self.session_stats['commands_processed']}")
        print(f"  Successful Operations: {self.session_stats['successful_operations']}")
        print(f"  Failed Operations: {self.session_stats['failed_operations']}")
        
        memory_stats = self.cognitive.get_memory_stats()
        print(f"\nMemory Statistics:")
        print(f"  Total Memories: {memory_stats['total_memories']}")
        for mem_type, count in memory_stats['by_type'].items():
            print(f"    {mem_type}: {count}")
        print()
    
    def _print_history(self) -> None:
        """Print recent command history."""
        episodes = self.cognitive.get_recent_episodes(10)
        
        print("\nRecent Commands:")
        print("-" * 40)
        
        if not episodes:
            print("  No commands in history.")
        else:
            for i, ep in enumerate(episodes, 1):
                outcome = ep.get("outcome", "pending")
                status_icon = "✓" if outcome == "success" else "✗" if outcome == "failure" else "○"
                print(f"  {i}. [{status_icon}] {ep.get('command', 'Unknown')}")
        print()
    
    def _display_result(self, result: Dict[str, Any]) -> None:
        """Display the result of a command."""
        status = result["status"]
        
        if status == "success":
            print(f"\n✓ Command executed successfully")
            
            cog = result.get("cognitive_result", {})
            if cog:
                print(f"  Intent: {cog.get('parsed', {}).get('intent', 'unknown')}")
                print(f"  Confidence: {cog.get('parsed', {}).get('confidence', 0):.1%}")
            
            exec_result = result.get("execution_result", {})
            if exec_result:
                intents = exec_result.get("intents", [])
                print(f"  Intents Generated: {len(intents)}")
        
        elif status == "parse_error":
            print(f"\n✗ Could not understand the command")
            for error in result.get("errors", []):
                print(f"  - {error}")
            
            cog = result.get("cognitive_result", {})
            alternatives = cog.get("parsed", {}).get("alternatives", [])
            if alternatives:
                print("\n  Did you mean one of these?")
                for intent, conf in alternatives[:3]:
                    print(f"    - {intent} ({conf:.1%})")
        
        elif status == "mapping_error":
            print(f"\n✗ Could not map command to file operation")
            for error in result.get("errors", []):
                print(f"  - {error}")
        
        else:
            print(f"\n✗ Command failed: {status}")
            for error in result.get("errors", []):
                print(f"  - {error}")
        
        print()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Marduk Commander - Natural Language File Management"
    )
    parser.add_argument(
        "--middleware", "-m",
        default="http://localhost:8080",
        help="URL of the Middleware Bridge server"
    )
    parser.add_argument(
        "--state", "-s",
        default=None,
        help="Path for state persistence"
    )
    parser.add_argument(
        "--command", "-c",
        default=None,
        help="Execute a single command and exit"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the Middleware Bridge server"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Marduk Commander v{MardukCommander.VERSION}"
    )
    
    args = parser.parse_args()
    
    if args.server:
        # Start the middleware server
        print("Starting Middleware Bridge server...")
        import uvicorn
        from middleware.bridge_server import app
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        # Initialize commander
        commander = MardukCommander(
            middleware_url=args.middleware,
            persistence_path=args.state
        )
        
        if args.command:
            # Single command mode
            result = asyncio.run(commander.process_command(args.command))
            print(json.dumps(result, indent=2))
        else:
            # Interactive mode
            asyncio.run(commander.interactive_session())


if __name__ == "__main__":
    main()
