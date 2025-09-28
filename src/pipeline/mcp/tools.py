"""MCP tools implementation for Local LLM Pipeline."""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions

from ..config import get_config
from ..models import ModelManager
from ..inference import InferenceEngine, ChatMessage, ChatRequest, GenerationRequest, MessageRole
from .config import get_mcp_config

logger = logging.getLogger(__name__)


class MCPToolsHandler:
    """Handles MCP tool implementations."""

    def __init__(self, inference_engine: Optional[InferenceEngine] = None):
        """Initialize the MCP tools handler.

        Args:
            inference_engine: Optional pre-initialized inference engine
        """
        self.config = get_config()
        self.mcp_config = get_mcp_config()
        self.model_manager = ModelManager()
        self.inference_engine = inference_engine or InferenceEngine()

    def get_available_tools(self) -> List[Tool]:
        """Get list of available MCP tools."""
        tools = []
        enabled_tools = self.mcp_config.get_enabled_tools()

        if "llm_chat" in enabled_tools:
            tools.append(self._get_llm_chat_tool())

        if "llm_complete" in enabled_tools:
            tools.append(self._get_llm_complete_tool())

        if "list_models" in enabled_tools:
            tools.append(self._get_list_models_tool())

        if "load_model" in enabled_tools:
            tools.append(self._get_load_model_tool())

        if "model_info" in enabled_tools:
            tools.append(self._get_model_info_tool())

        if "system_status" in enabled_tools:
            tools.append(self._get_system_status_tool())

        return tools

    def _get_llm_chat_tool(self) -> Tool:
        """Get the llm_chat tool definition."""
        tool_config = self.mcp_config.get_enabled_tools()["llm_chat"]

        return Tool(
            name="llm_chat",
            description=tool_config.description or "Chat with local LLM models",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["system", "user", "assistant"]
                                },
                                "content": {"type": "string"}
                            },
                            "required": ["role", "content"]
                        },
                        "description": "Array of chat messages"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (optional, uses current loaded model if not specified)"
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "default": 0.7,
                        "description": "Sampling temperature"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": tool_config.max_tokens or 4096,
                        "default": 512,
                        "description": "Maximum tokens to generate"
                    },
                    "top_p": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.9,
                        "description": "Top-p sampling parameter"
                    },
                    "stop": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stop sequences"
                    }
                },
                "required": ["messages"]
            }
        )

    def _get_llm_complete_tool(self) -> Tool:
        """Get the llm_complete tool definition."""
        tool_config = self.mcp_config.get_enabled_tools()["llm_complete"]

        return Tool(
            name="llm_complete",
            description=tool_config.description or "Text completion with local LLM",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text prompt for completion"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (optional)"
                    },
                    "temperature": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "default": 0.7,
                        "description": "Sampling temperature"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": tool_config.max_tokens or 2048,
                        "default": 512,
                        "description": "Maximum tokens to generate"
                    },
                    "top_p": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.9,
                        "description": "Top-p sampling parameter"
                    },
                    "stop": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stop sequences"
                    }
                },
                "required": ["prompt"]
            }
        )

    def _get_list_models_tool(self) -> Tool:
        """Get the list_models tool definition."""
        tool_config = self.mcp_config.get_enabled_tools()["list_models"]

        return Tool(
            name="list_models",
            description=tool_config.description or "List available local models",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter models by category (optional)"
                    },
                    "downloaded_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "Show only downloaded models"
                    }
                }
            }
        )

    def _get_load_model_tool(self) -> Tool:
        """Get the load_model tool definition."""
        tool_config = self.mcp_config.get_enabled_tools()["load_model"]

        return Tool(
            name="load_model",
            description=tool_config.description or "Load a specific model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to load"
                    },
                    "force": {
                        "type": "boolean",
                        "default": False,
                        "description": "Force load even if another model is loaded"
                    }
                },
                "required": ["model_id"]
            }
        )

    def _get_model_info_tool(self) -> Tool:
        """Get the model_info tool definition."""
        tool_config = self.mcp_config.get_enabled_tools()["model_info"]

        return Tool(
            name="model_info",
            description=tool_config.description or "Get detailed information about a model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "ID of the model to get info for"
                    }
                },
                "required": ["model_id"]
            }
        )

    def _get_system_status_tool(self) -> Tool:
        """Get the system_status tool definition."""
        tool_config = self.mcp_config.get_enabled_tools()["system_status"]

        return Tool(
            name="system_status",
            description=tool_config.description or "Get system status and performance metrics",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )

    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle a tool call and return the result."""
        try:
            if name == "llm_chat":
                return await self._handle_llm_chat(arguments)
            elif name == "llm_complete":
                return await self._handle_llm_complete(arguments)
            elif name == "list_models":
                return await self._handle_list_models(arguments)
            elif name == "load_model":
                return await self._handle_load_model(arguments)
            elif name == "model_info":
                return await self._handle_model_info(arguments)
            elif name == "system_status":
                return await self._handle_system_status(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.error(f"Error handling tool call {name}: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _handle_llm_chat(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle llm_chat tool call."""
        messages_data = arguments.get("messages", [])
        model_id = arguments.get("model")
        temperature = arguments.get("temperature", 0.7)
        max_tokens = arguments.get("max_tokens", 512)
        top_p = arguments.get("top_p", 0.9)
        stop = arguments.get("stop")

        # Convert messages to ChatMessage objects
        messages = []
        for msg_data in messages_data:
            role = MessageRole(msg_data["role"])
            content = msg_data["content"]
            messages.append(ChatMessage(role=role, content=content))

        # Load model if specified and different from current
        if model_id and self.inference_engine.current_model != model_id:
            if not self.inference_engine.is_loaded or self.inference_engine.current_model != model_id:
                await self.inference_engine.load_model(model_id)

        # Ensure we have a model loaded
        if not self.inference_engine.is_loaded:
            # Load default model
            default_model = self.config.load_settings().model.default_model
            await self.inference_engine.load_model(default_model)

        # Create chat request
        chat_request = ChatRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )

        # Generate response
        response = await self.inference_engine.chat(chat_request)

        # Format response
        result = {
            "response": response.message.content,
            "model": self.inference_engine.current_model,
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens
            },
            "finish_reason": response.finish_reason
        }

        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_llm_complete(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle llm_complete tool call."""
        prompt = arguments.get("prompt", "")
        model_id = arguments.get("model")
        temperature = arguments.get("temperature", 0.7)
        max_tokens = arguments.get("max_tokens", 512)
        top_p = arguments.get("top_p", 0.9)
        stop = arguments.get("stop")

        # Load model if specified
        if model_id and self.inference_engine.current_model != model_id:
            await self.inference_engine.load_model(model_id)

        # Ensure we have a model loaded
        if not self.inference_engine.is_loaded:
            default_model = self.config.load_settings().model.default_model
            await self.inference_engine.load_model(default_model)

        # Create generation request
        gen_request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )

        # Generate response
        response = await self.inference_engine.generate(gen_request)

        # Format response
        result = {
            "text": response.text,
            "model": self.inference_engine.current_model,
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens
            },
            "finish_reason": response.finish_reason
        }

        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_list_models(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle list_models tool call."""
        category = arguments.get("category")
        downloaded_only = arguments.get("downloaded_only", False)

        if category:
            models = self.model_manager.get_models_by_category(category)
        else:
            models = self.model_manager.list_available_models()

        if downloaded_only:
            models = {k: v for k, v in models.items() if v["downloaded"]}

        # Format response
        result = {
            "models": models,
            "current_model": self.inference_engine.current_model,
            "total_count": len(models),
            "downloaded_count": sum(1 for m in models.values() if m["downloaded"])
        }

        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_load_model(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle load_model tool call."""
        model_id = arguments.get("model_id")
        force = arguments.get("force", False)

        if not model_id:
            return [TextContent(type="text", text="Error: model_id is required")]

        # Check if model is already loaded
        if self.inference_engine.current_model == model_id and not force:
            return [TextContent(type="text", text=f"Model {model_id} is already loaded")]

        # Check if confirmation is required
        if self.mcp_config.requires_confirmation("load_model"):
            # In a real implementation, this would prompt the user
            # For now, we'll just log it
            logger.info(f"Loading model {model_id} (would require user confirmation in UI)")

        try:
            await self.inference_engine.load_model(model_id)
            result = {
                "success": True,
                "message": f"Successfully loaded model {model_id}",
                "model": model_id
            }
        except Exception as e:
            result = {
                "success": False,
                "message": f"Failed to load model {model_id}: {str(e)}",
                "error": str(e)
            }

        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_model_info(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle model_info tool call."""
        model_id = arguments.get("model_id")

        if not model_id:
            return [TextContent(type="text", text="Error: model_id is required")]

        model_info = self.model_manager.get_model_info(model_id)

        if not model_info:
            result = {
                "error": f"Model {model_id} not found"
            }
        else:
            result = {
                "model_info": model_info,
                "is_current": self.inference_engine.current_model == model_id
            }

        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_system_status(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle system_status tool call."""
        result = {
            "inference_engine": {
                "model_loaded": self.inference_engine.is_loaded,
                "current_model": self.inference_engine.current_model,
                "stats": self.inference_engine.stats.dict() if self.inference_engine.is_loaded else None
            },
            "storage": self.model_manager.get_storage_usage(),
            "available_models": len(self.model_manager.list_available_models()),
            "mcp_server": {
                "enabled": self.mcp_config.load_config().server.enabled,
                "transport": self.mcp_config.load_config().server.transport,
                "enabled_tools": list(self.mcp_config.get_enabled_tools().keys()),
                "enabled_resources": list(self.mcp_config.get_enabled_resources().keys())
            }
        }

        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]