# CodeSmith Ollama Helper

A Flask-based API server that bridges CodeSmith with local Ollama LLMs for AI-assisted code editing and analysis.

## Features

- 🤖 **LLM Integration**: Seamless connection to local Ollama models
- ✅ **Startup Verification**: Ensures Ollama is running with automatic retry logic
- 📝 **Code Assistance**: Specialized endpoints for code analysis and patch generation
- 📦 **Snapshot Management**: Upload, validate, and manage CodeSmith project snapshots
- 🔒 **Security**: File validation, size limits, and input sanitization
- 🌊 **Streaming Support**: Real-time streaming responses for long-running queries
- 🎨 **Web UI**: Built-in browser interface for testing
- ⚡ **Auto-cleanup**: Automatic management of old snapshot files
- 🔧 **Smart Retries**: Exponential backoff for Ollama connection attempts

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│  CodeSmith  │─────▶│  Ollama Helper   │─────▶│   Ollama    │
│             │      │  (Flask API)     │      │   Server    │
└─────────────┘      └──────────────────┘      └─────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Snapshot Store │
                     │   (JSON files)  │
                     └─────────────────┘
```

### Module Structure

- **app.py**: Main Flask application with API routes
- **ollama_client.py**: HTTP client wrapper for Ollama API
- **config_ollama_helper.py**: Configuration and logging utilities
- **codesmith_patch_utils.py**: CodeSmith patch format handlers
- **ui_root.py**: Web-based testing interface

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- At least one Ollama model pulled (e.g., `ollama pull qwen2.5-coder:7b`)

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

4. **Ensure Ollama is running**:
   ```bash
   # Check if Ollama is accessible
   curl http://127.0.0.1:11434/api/tags
   ```

## Configuration

All configuration is done via environment variables. See `.env.example` for all options.

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Default model for requests |
| `OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |
| `REQUIRE_OLLAMA_ON_STARTUP` | `true` | Require Ollama at startup |
| `OLLAMA_STARTUP_RETRIES` | `5` | Connection retry attempts |
| `OLLAMA_STARTUP_RETRY_DELAY` | `2.0` | Initial retry delay (exponential backoff) |
| `HOST` | `0.0.0.0` | Flask bind address |
| `PORT` | `8070` | Flask port |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max snapshot upload size |
| `MAX_SNAPSHOTS` | `100` | Max snapshots to keep (0=unlimited) |

### Ollama Startup Verification

By default, the server **requires Ollama to be running** at startup and will:
- Automatically retry connection with exponential backoff (2s, 4s, 8s, 16s, 32s)
- Display helpful error messages with setup instructions if Ollama is not available
- Exit with error code 1 if Ollama cannot be reached after retries

**To bypass this check** (start server even without Ollama):
```bash
export REQUIRE_OLLAMA_ON_STARTUP=false
python app.py
```

**Customize retry behavior**:
```bash
# Try 10 times with 1 second initial delay
export OLLAMA_STARTUP_RETRIES=10
export OLLAMA_STARTUP_RETRY_DELAY=1.0
python app.py
```

## Usage

### Starting the Server

```bash
# Using Python directly
python app.py

# On Windows
Start.bat

# With custom configuration
OLLAMA_MODEL=deepseek-coder:33b PORT=8080 python app.py
```

The server will start on `http://localhost:8070` (or your configured port).

### Web Interface

Navigate to `http://localhost:8070` in your browser to access the testing UI.

## API Endpoints

### Health Check
```http
GET /api/health
```
Returns server status and Ollama connectivity.

**Response:**
```json
{
  "status": "ok",
  "ollama_connected": true,
  "ollama_host": "http://127.0.0.1:11434",
  "default_model": "qwen2.5-coder:7b",
  "heavy_model": "qwen2.5-coder:7b",
  "max_upload_mb": 50
}
```

### List Models
```http
GET /api/models
```
Lists all available Ollama models.

### Chat (Non-streaming)
```http
POST /api/chat
Content-Type: application/json

{
  "prompt": "Explain how Python decorators work",
  "model": "qwen2.5-coder:7b"  // optional
}
```

**Alternative format with message history:**
```json
{
  "messages": [
    {"role": "user", "content": "What is recursion?"},
    {"role": "assistant", "content": "Recursion is..."},
    {"role": "user", "content": "Show me an example"}
  ]
}
```

### Chat (Streaming)
```http
POST /api/chat/stream
Content-Type: application/json

{
  "prompt": "Write a binary search function",
  "model": "qwen2.5-coder:7b"
}
```

Returns newline-delimited JSON chunks as they arrive.

### Code Assistance
```http
POST /api/code-assist
Content-Type: application/json

{
  "prompt": "Find bugs in this code",
  "code": "def add(a, b):\n    return a + b",
  "filename": "math_utils.py",  // optional
  "model": "qwen2.5-coder:7b"   // optional
}
```

### Patch Normalization
```http
POST /api/patch/normalize
Content-Type: application/json

{
  "raw": "```json\n{\"edits\": [...]}\n```",
  "root_path": "/project/path",  // optional
  "file_hashes": {...}           // optional
}
```

Converts model output into valid CodeSmith patch format.

### Upload Snapshot
```http
POST /api/upload-snapshot
Content-Type: multipart/form-data

file: <snapshot.json>
```

Uploads and validates a CodeSmith snapshot. Automatically cleans up old snapshots.

### List Snapshots
```http
GET /api/snapshots
```

Returns all uploaded snapshots with metadata.

### Extract Snapshot Files
```http
POST /api/snapshot/files
Content-Type: application/json

{
  "filename": "snapshot-12345.json",
  "paths": ["app.py", "utils.py"],  // optional filter
  "max_total_chars": 16000,         // optional
  "max_file_chars": 4000,           // optional
  "include_source": true            // optional
}
```

Extracts file contents from a snapshot with smart truncation.

## CodeSmith Patch Format

The helper understands the `codesmith.patch.v1` schema:

```json
{
  "schema": "codesmith.patch.v1",
  "root_path": "/project/root",
  "transactional": true,
  "file_hashes": {
    "path/to/file.py": "sha256-hash"
  },
  "edits": [
    {
      "operation": "modify",
      "path": "src/app.py",
      "search": "old code",
      "replace": "new code"
    }
  ]
}
```

See `codesmith_patch_utils.py` for full schema documentation.

## Integration with CodeSmith

### Current Setup (Standalone)
Run this helper as a separate service that CodeSmith can call via HTTP.

### Future Integration
The modular design allows easy integration into CodeSmith's workflow:
1. Import the modules directly into CodeSmith
2. Use the patch utilities for code transformation
3. Leverage the Ollama client for LLM interactions

## Development

### Project Structure
```
Ollama_Assistant/
├── app.py                      # Main Flask application
├── ollama_client.py            # Ollama HTTP client
├── config_ollama_helper.py     # Configuration & logging
├── codesmith_patch_utils.py    # Patch format utilities
├── ui_root.py                  # Web UI
├── requirements.txt            # Python dependencies
├── .env.example                # Example configuration
├── Start.bat                   # Windows launcher
├── uploads/                    # Snapshot storage (auto-created)
└── README.md                   # This file
```

### Adding New Endpoints

1. Add route handler to `app.py`
2. Use `logger` for logging
3. Return JSON with proper error codes
4. Add validation for inputs
5. Update this README

### Testing

```bash
# Health check
curl http://localhost:8070/api/health

# Simple chat
curl -X POST http://localhost:8070/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!"}'

# Upload snapshot
curl -X POST http://localhost:8070/api/upload-snapshot \
  -F "file=@snapshot.json"
```

## Security Considerations

- ✅ File upload size limits enforced
- ✅ JSON validation before processing
- ✅ Filename sanitization
- ✅ Input validation on all endpoints
- ✅ Proper error handling without info leakage
- ⚠️ No authentication (add reverse proxy with auth if needed)
- ⚠️ No rate limiting (add nginx/middleware if exposed publicly)

## Troubleshooting

### "Cannot connect to Ollama"
The server now **automatically checks** for Ollama at startup with retry logic:

1. **If server exits immediately**: Ollama is not running
   - Start Ollama: `ollama serve` (macOS/Linux) or check system tray (Windows)
   - Verify: `ollama list`
   - Check configured host matches your setup: `echo $OLLAMA_HOST`

2. **To start server without Ollama** (for testing):
   ```bash
   REQUIRE_OLLAMA_ON_STARTUP=false python app.py
   ```

3. **If Ollama is on a different host/port**:
   ```bash
   export OLLAMA_HOST=http://your-server:11434
   python app.py
   ```

4. **Check firewall settings** if Ollama is remote

### "File too large" errors
- Increase `MAX_UPLOAD_SIZE_MB` in `.env`
- Check available disk space

### Slow responses
- Increase `OLLAMA_TIMEOUT`
- Use a smaller/faster model
- Check system resources

### Old snapshots accumulating
- Adjust `MAX_SNAPSHOTS` setting
- Manually clean `uploads/` directory

## Performance Tips

1. **Model Selection**: Smaller models (7B) are faster but less capable
2. **Streaming**: Use `/api/chat/stream` for better UX on long responses
3. **Snapshot Size**: Keep snapshots under 10MB for best performance
4. **Cleanup**: Set appropriate `MAX_SNAPSHOTS` to avoid disk bloat

## License

This project is provided as-is for use with CodeSmith.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Functions have docstrings
- Changes are logged appropriately
- README is updated for new features

## Support

For issues or questions:
1. Check this README
2. Review logs for error messages
3. Verify Ollama is working: `ollama list`
4. Test with the web UI first
