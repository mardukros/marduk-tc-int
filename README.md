# Marduk Commander

**Natural Language File Management for Total Commander (Android)**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/mardukros/marduk-tc-int)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

Marduk Commander is an intelligent orchestration layer that enables natural language control of [Total Commander](https://www.ghisler.com/android/) for Android. It combines cognitive AI capabilities with Android Intent automation to provide hands-free, conversational file management.

## Overview

Marduk Commander bridges the gap between human language and file system operations by implementing a three-layer architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│              (Voice / Text / Chat Interface)                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   Marduk Cognitive Core                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Command   │  │ Deliberation│  │    Procedural Memory    │  │
│  │   Parser    │──│   Engine    │──│  (Learned Operations)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    TC-Connector Module                           │
│         (Intent Translation & Operation Mapping)                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   Middleware Bridge                              │
│            (HTTP/WebSocket Server + ADB/Tasker)                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   Total Commander                                │
│                 (Android Application)                            │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Natural Language Processing
- **Intent Classification**: Understands file operation intents (copy, move, delete, search, etc.)
- **Entity Extraction**: Identifies paths, patterns, and parameters from natural language
- **Path Aliases**: Supports common folder names ("downloads", "documents", "pictures")
- **Confidence Scoring**: Provides alternatives when commands are ambiguous

### Cognitive Capabilities
- **Procedural Memory**: Learns and optimizes file operation procedures
- **Episodic Memory**: Tracks command history and outcomes
- **Self-Improvement**: Adjusts success rates based on execution feedback
- **Context Awareness**: Maintains session state for multi-step operations

### File Operations
| Operation | Example Commands |
|-----------|------------------|
| **Copy** | "Copy all PDFs from Downloads to Documents" |
| **Move** | "Move my photos to the backup folder" |
| **Delete** | "Delete temporary files" |
| **Search** | "Find large images in Pictures" |
| **Organize** | "Organize my downloads by file type" |
| **Archive** | "Archive the project folder" |
| **Extract** | "Unzip backup.zip to Documents" |
| **Navigate** | "Go to the music folder" |
| **Create Folder** | "Create a folder called Projects" |

### Integration Methods
- **ADB Mode**: Control via Android Debug Bridge (development)
- **Tasker Mode**: Integration with Tasker automation app
- **Direct Mode**: On-device execution (future)

## Installation

### Prerequisites
- Python 3.11 or higher
- Android device with Total Commander installed
- ADB (Android Debug Bridge) for development mode

### Quick Start

```bash
# Clone the repository
git clone https://github.com/mardukros/marduk-tc-int.git
cd marduk-tc-int

# Install dependencies
pip install -r requirements.txt

# Run interactive mode
python src/marduk_commander.py

# Or start the middleware server
python src/marduk_commander.py --server
```

### Configuration

Edit `config/default.json` to customize:
- Middleware server settings
- Path aliases
- File categorization rules
- Logging preferences

## Usage

### Interactive Mode

```bash
python src/marduk_commander.py
```

```
============================================================
  Marduk Commander v1.0.0
  Natural Language File Management for Total Commander
============================================================

Type 'help' for available commands, 'quit' to exit.

marduk> Move all PDFs from Downloads to Documents

✓ Command executed successfully
  Intent: move
  Confidence: 87%
  Intents Generated: 1

marduk> Find large files in Pictures

✓ Command executed successfully
  Intent: search
  Confidence: 92%
  Intents Generated: 1
```

### Single Command Mode

```bash
python src/marduk_commander.py -c "Archive my project folder"
```

### API Server Mode

Start the middleware bridge server:

```bash
python src/marduk_commander.py --server
```

Then send requests to the API:

```bash
# Execute a file operation
curl -X POST http://localhost:8080/api/v1/operation \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "move",
    "source_path": "/sdcard/Download/*.pdf",
    "destination_path": "/sdcard/Documents/PDFs"
  }'

# Process a cognitive command
curl -X POST http://localhost:8080/api/v1/cognitive \
  -H "Content-Type: application/json" \
  -d '{
    "command_id": "cmd_001",
    "intent": "organize_files",
    "entities": {"target": "/sdcard/Download"},
    "confidence": 0.95
  }'
```

### WebSocket Integration

Connect to `ws://localhost:8080/ws` for real-time communication:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'cognitive_command',
    payload: {
      command_id: 'cmd_001',
      intent: 'find_documents',
      entities: { pattern: '*.pdf' }
    }
  }));
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Result:', result);
};
```

## Project Structure

```
marduk-tc-integration/
├── src/
│   ├── core/
│   │   └── cognitive/
│   │       └── cognitive_api.py    # Cognitive processing engine
│   ├── connectors/
│   │   └── tc_connector.py         # Total Commander connector
│   ├── middleware/
│   │   └── bridge_server.py        # HTTP/WebSocket bridge
│   └── marduk_commander.py         # Main application
├── config/
│   └── default.json                # Configuration
├── tests/
│   └── test_cognitive_api.py       # Unit tests
├── libs/
│   └── arm64-v8a/                  # Native AI libraries
├── app/
│   └── src/main/                   # Total Commander assets
├── requirements.txt
└── README.md
```

## Native AI Libraries

The `libs/arm64-v8a/` directory contains pre-compiled native libraries for on-device AI capabilities:

| Library | Purpose |
|---------|---------|
| `libllama.so` | Local LLM inference (Llama models) |
| `libonnxruntime.so` | ONNX model inference |
| `libmediapipe_*.so` | Computer vision (image analysis) |
| `libkaldi-*.so` | Speech-to-text processing |
| `libpiper_*.so` | Text-to-speech synthesis |
| `libmmkv.so` | High-performance key-value storage |

These libraries enable future enhancements for:
- Voice-controlled file management
- Content-aware file organization (visual sorting)
- Semantic file search
- Offline natural language processing

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/v1/intent` | POST | Execute raw Android Intent |
| `/api/v1/operation` | POST | Execute file operation |
| `/api/v1/batch` | POST | Execute batch operations |
| `/api/v1/cognitive` | POST | Process cognitive command |
| `/api/v1/operations` | GET | List operations |
| `/api/v1/operations/{id}` | GET | Get operation status |

### WebSocket Events

| Event Type | Direction | Description |
|------------|-----------|-------------|
| `ping` | Client → Server | Keep-alive ping |
| `pong` | Server → Client | Keep-alive response |
| `cognitive_command` | Client → Server | Submit cognitive command |
| `cognitive_result` | Server → Client | Command execution result |
| `operation_complete` | Server → Client | Operation completion notification |
| `subscribe` | Client → Server | Subscribe to events |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Style

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] TC-Connector module with Intent mapping
- [x] Middleware Bridge server
- [x] Cognitive API with command parsing
- [x] Procedural memory system

### Phase 2: Enhanced Intelligence
- [ ] Integration with on-device LLM (libllama)
- [ ] Visual file sorting with MediaPipe
- [ ] Voice command support (STT/TTS)
- [ ] Semantic file search

### Phase 3: Advanced Features
- [ ] Multi-step workflow automation
- [ ] Cloud sync integration
- [ ] Plugin system for extensions
- [ ] Mobile companion app

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Total Commander](https://www.ghisler.com/) by Christian Ghisler
- [Marduk Cognitive Architecture](https://github.com/dtecho) concept
- The open-source AI/ML community

## References

1. Ghisler Software GmbH. *Total Commander for Android - Help*. https://www.ghisler.com/android/help.htm
2. Meta AI. *Llama Inference Engine*. https://github.com/facebookresearch/llama
3. Google. *MediaPipe Solutions*. https://developers.google.com/mediapipe

---

**Marduk Commander** - Bringing cognitive intelligence to file management.
