"""
Tests for Marduk Cognitive API
==============================

Unit tests for command parsing, intent classification, and procedural memory.

Author: Manus AI
Date: December 28, 2025
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.cognitive.cognitive_api import (
    CognitiveAPI,
    CommandParser,
    FileIntent,
    MemorySystem,
    MemoryType,
    MemoryEntry,
    ProceduralMemory,
    DeliberationEngine,
    ParsedCommand,
)


class TestCommandParser:
    """Tests for the CommandParser class."""
    
    @pytest.fixture
    def parser(self):
        return CommandParser()
    
    def test_parse_copy_command(self, parser):
        """Test parsing copy commands."""
        result = parser.parse("Copy all PDFs from Downloads to Documents")
        
        assert result.intent == FileIntent.COPY
        assert result.confidence > 0.5
        assert "source" in result.entities or "destination" in result.entities
    
    def test_parse_move_command(self, parser):
        """Test parsing move commands."""
        result = parser.parse("Move my photos to the backup folder")
        
        assert result.intent == FileIntent.MOVE
        assert result.confidence > 0.5
    
    def test_parse_delete_command(self, parser):
        """Test parsing delete commands."""
        result = parser.parse("Delete temporary files")
        
        assert result.intent == FileIntent.DELETE
        assert result.confidence > 0.5
    
    def test_parse_search_command(self, parser):
        """Test parsing search commands."""
        result = parser.parse("Find all large images in Pictures")
        
        assert result.intent == FileIntent.SEARCH
        assert result.confidence > 0.5
    
    def test_parse_organize_command(self, parser):
        """Test parsing organize commands."""
        result = parser.parse("Organize my downloads folder")
        
        assert result.intent == FileIntent.ORGANIZE
        assert result.confidence > 0.5
    
    def test_parse_archive_command(self, parser):
        """Test parsing archive commands."""
        result = parser.parse("Archive the project folder")
        
        assert result.intent == FileIntent.ARCHIVE
        assert result.confidence > 0.5
    
    def test_parse_extract_command(self, parser):
        """Test parsing extract commands."""
        result = parser.parse("Extract backup.zip to Documents")
        
        assert result.intent == FileIntent.EXTRACT
        assert result.confidence > 0.5
    
    def test_parse_navigate_command(self, parser):
        """Test parsing navigation commands."""
        result = parser.parse("Go to the music folder")
        
        assert result.intent == FileIntent.NAVIGATE
        assert result.confidence > 0.5
    
    def test_parse_create_folder_command(self, parser):
        """Test parsing folder creation commands."""
        result = parser.parse("Create a folder called Projects")
        
        assert result.intent == FileIntent.CREATE_FOLDER
        assert result.confidence > 0.5
    
    def test_parse_unknown_command(self, parser):
        """Test parsing unknown commands."""
        result = parser.parse("xyzzy plugh")
        
        assert result.intent == FileIntent.UNKNOWN
    
    def test_path_alias_resolution(self, parser):
        """Test path alias resolution."""
        result = parser.parse("Go to downloads")
        
        assert result.intent == FileIntent.NAVIGATE
        # Check that 'downloads' was resolved
        target = result.entities.get("target", "")
        assert "/sdcard" in target.lower() or "download" in target.lower()
    
    def test_alternatives_provided(self, parser):
        """Test that alternatives are provided for ambiguous commands."""
        result = parser.parse("Move files")
        
        # Should have some alternatives
        assert isinstance(result.alternatives, list)


class TestMemorySystem:
    """Tests for the MemorySystem class."""
    
    @pytest.fixture
    def memory(self):
        return MemorySystem()
    
    def test_store_and_retrieve(self, memory):
        """Test storing and retrieving memories."""
        entry = MemoryEntry(
            id="test_001",
            memory_type=MemoryType.DECLARATIVE,
            content={"fact": "test fact"}
        )
        
        memory.store(entry)
        retrieved = memory.retrieve("test_001")
        
        assert retrieved is not None
        assert retrieved.content["fact"] == "test fact"
    
    def test_query_by_type(self, memory):
        """Test querying memories by type."""
        # Store different types
        memory.store(MemoryEntry(
            id="dec_001",
            memory_type=MemoryType.DECLARATIVE,
            content={"type": "declarative"}
        ))
        memory.store(MemoryEntry(
            id="proc_001",
            memory_type=MemoryType.PROCEDURAL,
            content={"type": "procedural"}
        ))
        
        declarative = memory.query(memory_type=MemoryType.DECLARATIVE)
        procedural = memory.query(memory_type=MemoryType.PROCEDURAL)
        
        assert len(declarative) == 1
        assert len(procedural) == 1
        assert declarative[0].content["type"] == "declarative"
    
    def test_access_count_increment(self, memory):
        """Test that access count increments on retrieval."""
        entry = MemoryEntry(
            id="test_002",
            memory_type=MemoryType.EPISODIC,
            content={"event": "test"}
        )
        
        memory.store(entry)
        
        # Access multiple times
        memory.retrieve("test_002")
        memory.retrieve("test_002")
        retrieved = memory.retrieve("test_002")
        
        assert retrieved.access_count == 3
    
    def test_relevance_update(self, memory):
        """Test relevance score updates."""
        entry = MemoryEntry(
            id="test_003",
            memory_type=MemoryType.SEMANTIC,
            content={"concept": "test"},
            relevance_score=1.0
        )
        
        memory.store(entry)
        memory.update_relevance("test_003", 0.5)
        
        retrieved = memory.retrieve("test_003")
        assert retrieved.relevance_score == 1.5


class TestProceduralMemory:
    """Tests for the ProceduralMemory class."""
    
    @pytest.fixture
    def procedural(self):
        memory = MemorySystem()
        return ProceduralMemory(memory)
    
    def test_default_procedures_initialized(self, procedural):
        """Test that default procedures are initialized."""
        # Should have procedures for common intents
        copy_proc = procedural.get_procedure(FileIntent.COPY)
        move_proc = procedural.get_procedure(FileIntent.MOVE)
        
        assert copy_proc is not None
        assert move_proc is not None
    
    def test_get_procedure_for_intent(self, procedural):
        """Test getting procedure for specific intent."""
        proc = procedural.get_procedure(FileIntent.SEARCH)
        
        assert proc is not None
        assert proc.intent == FileIntent.SEARCH
        assert len(proc.steps) > 0
    
    def test_record_execution_updates_stats(self, procedural):
        """Test that recording execution updates statistics."""
        proc = procedural.get_procedure(FileIntent.COPY)
        initial_count = proc.execution_count
        
        procedural.record_execution(proc.id, success=True)
        
        updated_proc = procedural.get_procedure(FileIntent.COPY)
        assert updated_proc.execution_count == initial_count + 1


class TestCognitiveAPI:
    """Tests for the CognitiveAPI class."""
    
    @pytest.fixture
    def api(self):
        return CognitiveAPI()
    
    @pytest.mark.asyncio
    async def test_process_valid_command(self, api):
        """Test processing a valid command."""
        result = await api.process("Copy files from Downloads to Documents")
        
        assert result["status"] == "ready"
        assert "execution_plan" in result
        assert "episode_id" in result
    
    @pytest.mark.asyncio
    async def test_process_invalid_command(self, api):
        """Test processing an invalid command."""
        result = await api.process("xyzzy plugh foo bar")
        
        assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_memory_stats(self, api):
        """Test getting memory statistics."""
        stats = api.get_memory_stats()
        
        assert "total_memories" in stats
        assert "by_type" in stats
        assert all(mt in stats["by_type"] for mt in ["declarative", "procedural", "episodic", "semantic"])
    
    @pytest.mark.asyncio
    async def test_recent_episodes(self, api):
        """Test getting recent episodes."""
        # Process a command to create an episode
        await api.process("Go to downloads")
        
        episodes = api.get_recent_episodes(10)
        
        assert isinstance(episodes, list)
        assert len(episodes) > 0


class TestIntegration:
    """Integration tests for the full cognitive pipeline."""
    
    @pytest.fixture
    def api(self):
        return CognitiveAPI()
    
    @pytest.mark.asyncio
    async def test_full_command_pipeline(self, api):
        """Test the full command processing pipeline."""
        commands = [
            "Move all PDFs from Downloads to Documents",
            "Find large files in Pictures",
            "Create a folder called Backup",
            "Archive my project folder",
        ]
        
        for cmd in commands:
            result = await api.process(cmd)
            
            # Should either succeed or fail gracefully
            assert result["status"] in ["ready", "error"]
            
            if result["status"] == "ready":
                # Should have valid execution plan
                assert "execution_plan" in result
                assert "steps" in result["execution_plan"]
    
    @pytest.mark.asyncio
    async def test_learning_from_outcomes(self, api):
        """Test that the system learns from execution outcomes."""
        # Process a command
        result = await api.process("Copy files to backup")
        
        if result["status"] == "ready":
            episode_id = result["episode_id"]
            
            # Record success
            api.record_result(episode_id, success=True)
            
            # Check that episode was updated
            episodes = api.get_recent_episodes(1)
            if episodes:
                assert episodes[0].get("outcome") == "success"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
