# Local LLM Pipeline

A simple, powerful pipeline for running Large Language Models (LLMs) locally with minimal setup. Inspired by Jan.ai's approach to local AI, this pipeline provides an OpenAI-compatible API server and CLI tools for easy LLM deployment and management.

## Features

✨ **One-Command Setup** - Get started in minutes with `./bin/setup.sh`
🚀 **OpenAI-Compatible API** - Drop-in replacement for OpenAI API endpoints
🔗 **MCP Integration** - Model Context Protocol support for universal AI app integration
🎯 **Model Management** - Download, manage, and switch between models easily
💻 **CLI Interface** - Powerful command-line tools for all operations
⚡ **Efficient Inference** - Built on llama.cpp for optimized performance
🛠️ **Configurable** - YAML-based configuration system
🌐 **Cross-Platform** - Works on macOS, Linux, and Windows

## Quick Start

### 1. Setup (One Command)

```bash
./bin/setup.sh
```

This will:
- Check Python 3.10+ requirement
- Create virtual environment
- Install all dependencies
- Setup directory structure
- Create default configurations

### 2. Download a Model

```bash
# Activate the environment (if not already)
source venv/bin/activate

# Download a small model for testing
./bin/llm-cli.sh model download tinyllama-1b

# Or download a more capable model
./bin/llm-cli.sh model download llama-2-7b-chat
```

### 3. Start the Server

```bash
# Start with default settings
./bin/server.sh start

# Or start with a specific model
./bin/server.sh start --model tinyllama-1b --port 8080
```

### 4. Use the API

```bash
# Test the health endpoint
curl http://localhost:8000/health

# OpenAI-compatible chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 100
  }'
```

### 5. Interactive Chat

```bash
# Start an interactive chat session
./bin/llm-cli.sh chat start --model tinyllama-1b
```

### 6. MCP Integration (Universal AI App Access)

```bash
# Start MCP server for Claude Desktop, VS Code, etc.
./bin/mcp-server.sh start

# Test MCP functionality
./bin/mcp-server.sh test

# List available MCP tools
./bin/llm-cli.sh mcp tools
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI/Client    │◄──►│   API Server     │◄──►│  Inference      │
│                 │    │  (FastAPI)       │    │  Engine         │
└─────────────────┘    └──────────────────┘    │  (llama.cpp)    │
                              │                 └─────────────────┘
                              ▼                           │
                       ┌──────────────────┐              │
                       │  Configuration   │              │
                       │  Manager         │              │
                       └──────────────────┘              │
                              │                          │
                              ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Model Manager   │◄──►│  Model Storage  │
                       │                  │    │  (.gguf files)  │
                       └──────────────────┘    └─────────────────┘
```

## Directory Structure

```
local-llm-pipeline/
├── bin/                    # Executable scripts
│   ├── setup.sh           # One-command setup
│   ├── server.sh          # Server management
│   ├── llm-cli.sh         # CLI wrapper
│   └── mcp-server.sh      # MCP server management
├── etc/                   # Configuration files
│   ├── pipeline.yaml      # Main configuration
│   ├── models.yaml        # Model definitions
│   ├── server.yaml        # Server settings
│   └── mcp.yaml           # MCP configuration
├── src/pipeline/          # Source code
│   ├── api/              # FastAPI server
│   ├── inference/        # Inference engine
│   ├── models/           # Model management
│   ├── config/           # Configuration system
│   ├── mcp/              # MCP integration
│   └── cli/              # CLI interface
├── models/               # Downloaded models
├── logs/                 # Application logs
└── venv/                 # Virtual environment
```

## CLI Commands

### Model Management

```bash
# List available models
./bin/llm-cli.sh model list

# Get model information
./bin/llm-cli.sh model info llama-2-7b-chat

# Download a model
./bin/llm-cli.sh model download tinyllama-1b

# Delete a model
./bin/llm-cli.sh model delete tinyllama-1b --force
```

### Server Management

```bash
# Start server
./bin/server.sh start --host 0.0.0.0 --port 8000

# Stop server
./bin/server.sh stop

# Restart server
./bin/server.sh restart

# Check status
./bin/server.sh status
```

### Chat Interface

```bash
# Start interactive chat
./bin/llm-cli.sh chat start --model llama-2-7b-chat

# With custom settings
./bin/llm-cli.sh chat start \
  --model llama-2-7b-chat \
  --temperature 0.8 \
  --max-tokens 1024 \
  --system "You are a helpful coding assistant"
```

### MCP Commands

```bash
# Start MCP server
./bin/mcp-server.sh start

# Start with HTTP transport
./bin/mcp-server.sh start --transport http --port 9000

# Test MCP functionality
./bin/mcp-server.sh test

# List MCP tools and resources
./bin/llm-cli.sh mcp tools
./bin/llm-cli.sh mcp resources

# Check MCP configuration
./bin/llm-cli.sh mcp config
```

### Python MCP Client

```bash
# Run example MCP client
python examples/mcp_client_example.py

# Interactive chat via MCP
python examples/mcp_client_example.py chat
```

## API Endpoints

### OpenAI-Compatible

- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion

### Custom Endpoints

- `GET /health` - Health check
- `GET /api/v1/models` - Detailed model information
- `POST /api/v1/model/load` - Load a specific model
- `POST /api/v1/model/unload` - Unload current model
- `GET /api/v1/system/status` - System status

## MCP Integration

The Local LLM Pipeline supports the **Model Context Protocol (MCP)**, enabling seamless integration with MCP-compatible applications like Claude Desktop, VS Code extensions, and custom applications.

### MCP Tools Available

- **`llm_chat`** - Chat with local LLM models
- **`llm_complete`** - Text completion with local LLM
- **`list_models`** - List available local models
- **`load_model`** - Load a specific model
- **`model_info`** - Get detailed model information
- **`system_status`** - Get system status and metrics

### MCP Resources Available

- **`model://available`** - List of available models
- **`config://models`** - Model configurations
- **`model://info/{model_id}`** - Individual model information

### Claude Desktop Integration

1. **Start the MCP server:**
   ```bash
   ./bin/mcp-server.sh start
   ```

2. **Configure Claude Desktop** (`claude_desktop_config.json`):
   ```json
   {
     "mcpServers": {
       "local-llm": {
         "command": "/absolute/path/to/local-llm-pipeline/bin/mcp-server.sh",
         "args": ["start"]
       }
     }
   }
   ```

3. **Restart Claude Desktop** and enjoy local LLM access!

### VS Code Integration

Use MCP-compatible VS Code extensions to access your local LLMs for code completion, documentation, and assistance.

### Custom Applications

See `examples/mcp_client_example.py` for a complete Python client example, or `examples/mcp_integration.md` for detailed integration guides.

## Configuration

### Main Configuration (`etc/pipeline.yaml`)

```yaml
# Application settings
app:
  name: "Local LLM Pipeline"
  version: "0.1.0"
  debug: false
  log_level: "INFO"

# Resource limits
resources:
  max_memory_gb: 8
  max_cpu_cores: 4
  gpu_enabled: false

# Model defaults
model:
  default_model: "llama-2-7b-chat"
  context_length: 4096
  max_tokens: 512
  temperature: 0.7
```

### Server Configuration (`etc/server.yaml`)

```yaml
# Server settings
server:
  host: "127.0.0.1"
  port: 8000
  workers: 1

# API settings
api:
  title: "Local LLM Pipeline API"
  docs_url: "/docs"

# CORS settings
cors:
  allow_origins:
    - "http://localhost:3000"
```

### MCP Configuration (`etc/mcp.yaml`)

```yaml
# MCP Server settings
server:
  enabled: true
  name: "Local LLM Pipeline"
  transport: "stdio"  # or "http"

  # Tools to expose
  tools:
    llm_chat:
      enabled: true
      max_tokens: 4096
    llm_complete:
      enabled: true
      max_tokens: 2048

  # Resources to expose
  resources:
    model_list:
      enabled: true
      uri: "model://available"

# Security settings
security:
  require_confirmation:
    - "load_model"
  rate_limit:
    enabled: true
    requests_per_minute: 30
```

## Supported Models

The pipeline comes pre-configured with popular GGUF models:

- **TinyLlama 1.1B** - Tiny model for testing (0.6 GB)
- **Llama 2 7B Chat** - Conversational model (4.1 GB)
- **Llama 2 13B Chat** - Larger conversational model (7.3 GB)
- **Code Llama 7B** - Code generation model (4.1 GB)
- **Mistral 7B Instruct** - Instruction-following model (4.1 GB)

Add custom models by editing `etc/models.yaml`.

## Development

### Requirements

- Python 3.10+
- 8GB+ RAM (for 7B models)
- 10GB+ free disk space

### Setup for Development

```bash
# Clone and setup
git clone <repository>
cd local-llm-pipeline
./bin/setup.sh

# Install development dependencies
source venv/bin/activate
pip install -e ".[dev]"

# Run with auto-reload
./bin/server.sh start --reload
```

### Running Tests

```bash
source venv/bin/activate
pytest tests/
```

## Troubleshooting

### Common Issues

**Model Download Fails**
```bash
# Check network connection and try again
./bin/llm-cli.sh model download tinyllama-1b
```

**Server Won't Start**
```bash
# Check if port is in use
lsof -i :8000

# Start on different port
./bin/server.sh start --port 8080
```

**Out of Memory**
```bash
# Use a smaller model
./bin/llm-cli.sh model download tinyllama-1b

# Or adjust memory settings in etc/pipeline.yaml
```

**Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Logs

- Application logs: `./logs/pipeline.log`
- Server PID: `./logs/server.pid`

## Performance Tips

1. **Use GPU acceleration** (if available):
   ```yaml
   # In etc/pipeline.yaml
   resources:
     gpu_enabled: true
     gpu_layers: 32
   ```

2. **Optimize for your hardware**:
   ```yaml
   inference:
     threads: 8  # Match your CPU cores
     use_mmap: true
     use_mlock: false
   ```

3. **Choose appropriate model size**:
   - 4GB RAM → TinyLlama (1B)
   - 8GB RAM → Llama 2 (7B)
   - 16GB+ RAM → Llama 2 (13B)

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Examples and Documentation

- **`examples/mcp_integration.md`** - Comprehensive MCP integration guide
- **`examples/mcp_client_example.py`** - Python MCP client example
- **`examples/claude_desktop_config.json`** - Claude Desktop configuration template

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient LLM inference
- [Jan.ai](https://jan.ai) - Inspiration for local AI tools
- [FastAPI](https://fastapi.tiangolo.com/) - Modern API framework
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Model Context Protocol](https://modelcontextprotocol.io/) - Standardized AI integration
- [Anthropic](https://www.anthropic.com/) - MCP specification and implementation