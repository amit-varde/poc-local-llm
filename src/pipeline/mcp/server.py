"""MCP server implementation for Local LLM Pipeline."""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence
from mcp import ClientSession, StdioServerParameters
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from mcp.types import (
    CallToolRequest,
    GetPromptRequest,
    ListPromptsRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
    Resource,
    TextContent,
    Tool,
    Prompt,
    PromptMessage,
    TextResourceContents
)

from ..inference import InferenceEngine
from .config import get_mcp_config
from .tools import MCPToolsHandler
from .resources import MCPResourcesHandler

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP server for Local LLM Pipeline."""

    def __init__(self, inference_engine: Optional[InferenceEngine] = None):
        """Initialize the MCP server.

        Args:
            inference_engine: Optional pre-initialized inference engine
        """
        self.mcp_config = get_mcp_config()
        self.config = self.mcp_config.load_config()
        self.inference_engine = inference_engine or InferenceEngine()

        # Initialize handlers
        self.tools_handler = MCPToolsHandler(self.inference_engine)
        self.resources_handler = MCPResourcesHandler(self.inference_engine)

        # Create MCP server
        self.server = Server(self.config.server.name)
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP server handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Handle list tools request."""
            return self.tools_handler.get_available_tools()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool call request."""
            return await self.tools_handler.handle_tool_call(name, arguments)

        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """Handle list resources request."""
            return self.resources_handler.get_available_resources()

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Handle read resource request."""
            content = await self.resources_handler.handle_resource_request(uri)
            return content.text

        @self.server.list_prompts()
        async def handle_list_prompts() -> list[Prompt]:
            """Handle list prompts request."""
            prompts = []
            enabled_prompts = self.mcp_config.get_enabled_prompts()

            for prompt_name, prompt_config in enabled_prompts.items():
                if prompt_name == "chat_template":
                    prompts.append(Prompt(
                        name="chat_template",
                        description="Template for chat conversations with system message",
                        arguments=[
                            {
                                "name": "system_message",
                                "description": "System message to set the assistant's behavior",
                                "required": False
                            },
                            {
                                "name": "model",
                                "description": "Model to use for the conversation",
                                "required": False
                            }
                        ]
                    ))
                elif prompt_name == "code_assistant":
                    prompts.append(Prompt(
                        name="code_assistant",
                        description="Template for code generation and assistance",
                        arguments=[
                            {
                                "name": "language",
                                "description": "Programming language",
                                "required": False
                            },
                            {
                                "name": "task",
                                "description": "Coding task description",
                                "required": True
                            }
                        ]
                    ))

            return prompts

        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict | None) -> list[PromptMessage]:
            """Handle get prompt request."""
            if name == "chat_template":
                system_message = arguments.get("system_message", "") if arguments else ""
                model = arguments.get("model", "") if arguments else ""

                messages = []
                if system_message:
                    messages.append(PromptMessage(
                        role="system",
                        content=TextContent(type="text", text=system_message)
                    ))

                if model:
                    messages.append(PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"Please use model: {model}")
                    ))

                if not messages:
                    messages.append(PromptMessage(
                        role="system",
                        content=TextContent(type="text", text="You are a helpful AI assistant.")
                    ))

                return messages

            elif name == "code_assistant":
                language = arguments.get("language", "") if arguments else ""
                task = arguments.get("task", "") if arguments else ""

                system_prompt = "You are an expert programmer. Help with coding tasks, provide clean, well-documented code, and explain your solutions."

                if language:
                    system_prompt += f" Focus on {language} programming."

                messages = [
                    PromptMessage(
                        role="system",
                        content=TextContent(type="text", text=system_prompt)
                    )
                ]

                if task:
                    messages.append(PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"Task: {task}")
                    ))

                return messages

            else:
                return [PromptMessage(
                    role="system",
                    content=TextContent(type="text", text=f"Unknown prompt: {name}")
                )]

    async def run_stdio(self):
        """Run the MCP server with stdio transport."""
        logger.info("Starting MCP server with stdio transport")

        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=self.config.server.name,
                    server_version=self.config.server.version,
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

    async def run_http(self, host: str = None, port: int = None):
        """Run the MCP server with HTTP transport."""
        host = host or self.config.server.http.host
        port = port or self.config.server.http.port

        logger.info(f"Starting MCP server with HTTP transport on {host}:{port}")

        # HTTP transport implementation would go here
        # This is a placeholder for future implementation
        raise NotImplementedError("HTTP transport not yet implemented")

    async def start(self, transport: str = None):
        """Start the MCP server.

        Args:
            transport: Transport type ("stdio" or "http"). Uses config default if not specified.
        """
        transport = transport or self.config.server.transport

        if not self.config.server.enabled:
            logger.info("MCP server is disabled in configuration")
            return

        # Ensure we have a default model loaded
        try:
            if not self.inference_engine.is_loaded:
                from ..config import get_config
                default_model = get_config().load_settings().model.default_model
                await self.inference_engine.load_model(default_model)
                logger.info(f"Loaded default model: {default_model}")
        except Exception as e:
            logger.warning(f"Could not load default model: {e}")

        if transport == "stdio":
            await self.run_stdio()
        elif transport == "http":
            await self.run_http()
        else:
            raise ValueError(f"Unknown transport type: {transport}")

    def stop(self):
        """Stop the MCP server."""
        logger.info("Stopping MCP server")
        # Cleanup would go here


def main():
    """Main entry point for standalone MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Local LLM Pipeline MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                        help="Transport type")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport")
    parser.add_argument("--port", type=int, default=9000, help="Port for HTTP transport")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and run server
    server = MCPServer()

    try:
        if args.transport == "stdio":
            asyncio.run(server.run_stdio())
        elif args.transport == "http":
            asyncio.run(server.run_http(args.host, args.port))
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()