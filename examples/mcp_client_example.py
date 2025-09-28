#!/usr/bin/env python3
"""Example MCP client for Local LLM Pipeline."""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("MCP SDK not installed. Install with: pip install mcp")
    sys.exit(1)


async def test_local_llm_mcp():
    """Test Local LLM Pipeline MCP server functionality."""

    # Path to the MCP server script
    script_path = Path(__file__).parent.parent / "bin" / "mcp-server.sh"

    if not script_path.exists():
        print(f"MCP server script not found at {script_path}")
        return

    server_params = StdioServerParameters(
        command=str(script_path),
        args=["start"]
    )

    print("🚀 Connecting to Local LLM Pipeline MCP server...")

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                print("🔄 Initializing MCP session...")
                await session.initialize()

                # List available tools
                print("\n📋 Listing available tools...")
                tools = await session.list_tools()
                print(f"Found {len(tools)} tools:")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")

                # List available resources
                print("\n📦 Listing available resources...")
                resources = await session.list_resources()
                print(f"Found {len(resources)} resources:")
                for resource in resources:
                    print(f"  - {resource.name} ({resource.uri}): {resource.description}")

                # Test: List models
                print("\n🔍 Testing list_models tool...")
                try:
                    result = await session.call_tool(
                        "list_models",
                        {"downloaded_only": False}
                    )
                    models_data = json.loads(result.content[0].text)
                    print(f"Found {models_data.get('total_count', 0)} models")
                    print(f"Downloaded: {models_data.get('downloaded_count', 0)}")

                    if models_data.get('current_model'):
                        print(f"Current model: {models_data['current_model']}")

                except Exception as e:
                    print(f"Error listing models: {e}")

                # Test: Get system status
                print("\n⚡ Testing system_status tool...")
                try:
                    result = await session.call_tool("system_status", {})
                    status_data = json.loads(result.content[0].text)
                    engine_status = status_data.get('inference_engine', {})
                    print(f"Model loaded: {engine_status.get('model_loaded', False)}")

                    if engine_status.get('current_model'):
                        print(f"Current model: {engine_status['current_model']}")

                    storage = status_data.get('storage', {})
                    print(f"Storage used: {storage.get('total_size_gb', 0)} GB")

                except Exception as e:
                    print(f"Error getting system status: {e}")

                # Test: Chat (if model is available)
                print("\n💬 Testing llm_chat tool...")
                try:
                    result = await session.call_tool(
                        "llm_chat",
                        {
                            "messages": [
                                {"role": "user", "content": "Hello! Please respond with just 'Hello from Local LLM!'"}
                            ],
                            "max_tokens": 50,
                            "temperature": 0.1
                        }
                    )
                    chat_data = json.loads(result.content[0].text)
                    response_text = chat_data.get('response', 'No response')
                    print(f"Chat response: {response_text}")
                    print(f"Model used: {chat_data.get('model', 'Unknown')}")

                    usage = chat_data.get('usage', {})
                    print(f"Tokens used: {usage.get('total_tokens', 0)}")

                except Exception as e:
                    print(f"Error with chat: {e}")
                    print("Note: Make sure you have a model downloaded and loaded")

                # Test: Read a resource
                print("\n📖 Testing resource reading...")
                try:
                    content = await session.read_resource("model://available")
                    models_data = json.loads(content)
                    print(f"Resource data contains {len(models_data.get('models', {}))} models")

                except Exception as e:
                    print(f"Error reading resource: {e}")

                print("\n✅ MCP client test completed successfully!")

    except Exception as e:
        print(f"❌ Error connecting to MCP server: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run ./bin/setup.sh")
        print("2. Verify the MCP server script is executable")
        print("3. Check if required dependencies are installed")


async def interactive_chat():
    """Interactive chat session using MCP."""

    script_path = Path(__file__).parent.parent / "bin" / "mcp-server.sh"

    server_params = StdioServerParameters(
        command=str(script_path),
        args=["start"]
    )

    print("🚀 Starting interactive chat with Local LLM...")
    print("Type 'quit' to exit")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            messages = []

            while True:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

                try:
                    print("🤔 Thinking...")
                    result = await session.call_tool(
                        "llm_chat",
                        {
                            "messages": messages,
                            "max_tokens": 512,
                            "temperature": 0.7
                        }
                    )

                    chat_data = json.loads(result.content[0].text)
                    response_text = chat_data.get('response', 'No response')

                    print(f"Assistant: {response_text}")

                    # Add assistant response to history
                    messages.append({"role": "assistant", "content": response_text})

                    # Keep conversation history manageable
                    if len(messages) > 10:
                        messages = messages[-8:]  # Keep last 8 messages

                except Exception as e:
                    print(f"Error: {e}")


def main():
    """Main function with command-line options."""
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        print("Starting interactive chat mode...")
        asyncio.run(interactive_chat())
    else:
        print("Running MCP test suite...")
        asyncio.run(test_local_llm_mcp())


if __name__ == "__main__":
    main()