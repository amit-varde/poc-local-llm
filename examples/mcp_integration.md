# MCP Integration Examples

This document provides examples of how to integrate Local LLM Pipeline with MCP-compatible applications.

## Claude Desktop Integration

### Setup

1. **Start the Local LLM Pipeline MCP server:**
   ```bash
   # Start with stdio transport (recommended for Claude Desktop)
   ./bin/mcp-server.sh start
   ```

2. **Configure Claude Desktop:**

   Edit your Claude Desktop configuration file:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

   ```json
   {
     "mcpServers": {
       "local-llm": {
         "command": "/absolute/path/to/local-llm-pipeline/bin/mcp-server.sh",
         "args": ["start"],
         "env": {}
       }
     }
   }
   ```

3. **Restart Claude Desktop** to load the MCP server.

### Usage Examples

Once integrated, you can use these tools in Claude Desktop:

#### Chat with Local Models
```
Use the llm_chat tool to have a conversation with your local model:

{
  "messages": [
    {"role": "user", "content": "Hello! What can you help me with?"}
  ],
  "model": "llama-2-7b-chat",
  "temperature": 0.7,
  "max_tokens": 512
}
```

#### Text Completion
```
Use the llm_complete tool for text completion:

{
  "prompt": "The benefits of running LLMs locally include",
  "model": "llama-2-7b-chat",
  "max_tokens": 200
}
```

#### List Available Models
```
Use the list_models tool to see what models are available:

{
  "downloaded_only": true
}
```

#### Load a Different Model
```
Use the load_model tool to switch models:

{
  "model_id": "mistral-7b-instruct"
}
```

## VS Code Integration

### Setup with MCP Extension

1. **Install an MCP extension** for VS Code (if available)

2. **Configure the extension** to point to your Local LLM Pipeline:
   ```json
   {
     "mcp.servers": [
       {
         "name": "local-llm",
         "command": "/path/to/local-llm-pipeline/bin/mcp-server.sh",
         "args": ["start"],
         "transport": "stdio"
       }
     ]
   }
   ```

### Usage in VS Code

- **Code Completion**: Use local models for code suggestions
- **Documentation**: Generate documentation using local LLMs
- **Code Review**: Get code analysis from local models

## Custom MCP Client

### Python Client Example

```python
#!/usr/bin/env python3
"""Example MCP client for Local LLM Pipeline."""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Connect to Local LLM Pipeline MCP server
    server_params = StdioServerParameters(
        command="/path/to/local-llm-pipeline/bin/mcp-server.sh",
        args=["start"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools]}")

            # List available resources
            resources = await session.list_resources()
            print(f"Available resources: {[resource.name for resource in resources]}")

            # Call the llm_chat tool
            result = await session.call_tool(
                "llm_chat",
                {
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                    "model": "llama-2-7b-chat",
                    "max_tokens": 100
                }
            )

            print(f"Chat result: {result}")

            # Read a resource
            model_list = await session.read_resource("model://available")
            print(f"Available models: {model_list}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript/Node.js Client Example

```javascript
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');

async function main() {
    // Create stdio transport
    const transport = new StdioClientTransport({
        command: '/path/to/local-llm-pipeline/bin/mcp-server.sh',
        args: ['start']
    });

    // Create client
    const client = new Client({
        name: "local-llm-client",
        version: "1.0.0"
    }, {
        capabilities: {}
    });

    // Connect
    await client.connect(transport);

    try {
        // List tools
        const tools = await client.listTools();
        console.log('Available tools:', tools.tools.map(t => t.name));

        // Call llm_chat tool
        const chatResult = await client.callTool({
            name: "llm_chat",
            arguments: {
                messages: [
                    { role: "user", content: "Hello from Node.js!" }
                ],
                model: "llama-2-7b-chat",
                max_tokens: 100
            }
        });

        console.log('Chat result:', chatResult.content);

        // List available models
        const modelsResult = await client.callTool({
            name: "list_models",
            arguments: {
                downloaded_only: true
            }
        });

        console.log('Models:', modelsResult.content);

    } finally {
        await client.close();
    }
}

main().catch(console.error);
```

## Cline/Cursor Integration

### Setup for Cline (VS Code Extension)

1. **Install Cline extension** in VS Code

2. **Configure MCP in Cline settings:**
   ```json
   {
     "cline.mcp.servers": {
       "local-llm": {
         "command": "/path/to/local-llm-pipeline/bin/mcp-server.sh",
         "args": ["start"]
       }
     }
   }
   ```

### Usage with Cline

- Cline can use your local LLMs for code generation and editing
- Access through Cline's interface using MCP tools
- Switch between different local models as needed

## Troubleshooting

### Common Issues

1. **Server Not Starting:**
   ```bash
   # Test MCP server functionality
   ./bin/mcp-server.sh test

   # Check configuration
   ./bin/mcp-server.sh config
   ```

2. **Client Connection Issues:**
   ```bash
   # Verify server status
   ./bin/mcp-server.sh status

   # Check logs
   tail -f logs/mcp.log
   ```

3. **Model Loading Errors:**
   ```bash
   # List available models
   ./bin/llm-cli.sh model list

   # Download required model
   ./bin/llm-cli.sh model download llama-2-7b-chat
   ```

### Debugging

Enable verbose logging:
```bash
./bin/mcp-server.sh start --verbose
```

Check audit logs:
```bash
tail -f logs/mcp_audit.log
```

### Performance Tips

1. **Pre-load Models:**
   ```bash
   # Start main server with default model
   ./bin/server.sh start --model llama-2-7b-chat

   # Then start MCP server
   ./bin/mcp-server.sh start
   ```

2. **Optimize for Your Use Case:**
   ```yaml
   # In etc/mcp.yaml
   security:
     rate_limit:
       requests_per_minute: 60  # Adjust based on usage
   ```

3. **Monitor Resource Usage:**
   ```bash
   # Check system status
   ./bin/llm-cli.sh mcp server status
   ```

## Advanced Configuration

### Custom Tool Configurations

```yaml
# etc/mcp.yaml
server:
  tools:
    llm_chat:
      enabled: true
      description: "Custom chat tool for specific use case"
      max_tokens: 2048
      require_confirmation: false

    custom_completion:
      enabled: true
      description: "Specialized completion tool"
      max_tokens: 1024
```

### Security Settings

```yaml
# etc/mcp.yaml
security:
  require_confirmation:
    - "load_model"
    - "system_commands"

  allowed_operations:
    - "read"
    - "inference"
    - "model_management"

  rate_limit:
    enabled: true
    requests_per_minute: 30
    burst_size: 5
```

### Multiple Server Instances

You can run multiple MCP servers for different purposes:

```bash
# Server for general chat
./bin/mcp-server.sh start --transport http --port 9000

# Server for code assistance (different config)
MCP_CONFIG_DIR=./etc/mcp-code ./bin/mcp-server.sh start --transport http --port 9001
```