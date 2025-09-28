"""MCP resources implementation for Local LLM Pipeline."""

import json
import logging
from typing import Any, Dict, List
from mcp.types import Resource, TextResourceContents

from ..config import get_config
from ..models import ModelManager
from ..inference import InferenceEngine
from .config import get_mcp_config

logger = logging.getLogger(__name__)


class MCPResourcesHandler:
    """Handles MCP resource implementations."""

    def __init__(self, inference_engine: InferenceEngine = None):
        """Initialize the MCP resources handler.

        Args:
            inference_engine: Optional pre-initialized inference engine
        """
        self.config = get_config()
        self.mcp_config = get_mcp_config()
        self.model_manager = ModelManager()
        self.inference_engine = inference_engine or InferenceEngine()

    def get_available_resources(self) -> List[Resource]:
        """Get list of available MCP resources."""
        resources = []
        enabled_resources = self.mcp_config.get_enabled_resources()

        for resource_name, resource_config in enabled_resources.items():
            resources.append(Resource(
                uri=resource_config.uri,
                name=resource_config.name,
                description=resource_config.description,
                mimeType="application/json"
            ))

        return resources

    async def handle_resource_request(self, uri: str) -> TextResourceContents:
        """Handle a resource request and return the content."""
        try:
            if uri == "model://available":
                return await self._get_model_list_resource()
            elif uri == "config://models":
                return await self._get_model_configs_resource()
            elif uri.startswith("model://info/"):
                model_id = uri.replace("model://info/", "")
                return await self._get_model_info_resource(model_id)
            else:
                return TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=json.dumps({"error": f"Unknown resource URI: {uri}"})
                )

        except Exception as e:
            logger.error(f"Error handling resource request {uri}: {e}")
            return TextResourceContents(
                uri=uri,
                mimeType="application/json",
                text=json.dumps({"error": str(e)})
            )

    async def _get_model_list_resource(self) -> TextResourceContents:
        """Get the model list resource."""
        models = self.model_manager.list_available_models()

        # Add current model status
        current_model = self.inference_engine.current_model

        resource_data = {
            "models": models,
            "current_model": current_model,
            "total_count": len(models),
            "downloaded_count": sum(1 for m in models.values() if m["downloaded"]),
            "categories": self._get_model_categories(),
            "last_updated": self._get_current_timestamp()
        }

        return TextResourceContents(
            uri="model://available",
            mimeType="application/json",
            text=json.dumps(resource_data, indent=2)
        )

    async def _get_model_configs_resource(self) -> TextResourceContents:
        """Get the model configurations resource."""
        registry = self.config.load_model_registry()

        # Extract model configurations
        model_configs = {}
        for model_id, model_def in registry.models.items():
            model_configs[model_id] = {
                "name": model_def.name,
                "type": model_def.type,
                "context_length": model_def.context_length,
                "quantization": model_def.quantization,
                "size_gb": model_def.size_gb,
                "url": model_def.url,
                "filename": model_def.filename
            }

        resource_data = {
            "model_configs": model_configs,
            "categories": registry.categories,
            "download_settings": registry.download.dict(),
            "last_updated": self._get_current_timestamp()
        }

        return TextResourceContents(
            uri="config://models",
            mimeType="application/json",
            text=json.dumps(resource_data, indent=2)
        )

    async def _get_model_info_resource(self, model_id: str) -> TextResourceContents:
        """Get detailed information about a specific model."""
        model_info = self.model_manager.get_model_info(model_id)

        if not model_info:
            resource_data = {
                "error": f"Model {model_id} not found"
            }
        else:
            # Add runtime information
            is_current = self.inference_engine.current_model == model_id
            stats = None

            if is_current and self.inference_engine.is_loaded:
                stats = self.inference_engine.stats.dict()

            resource_data = {
                "model_info": model_info,
                "runtime_status": {
                    "is_current_model": is_current,
                    "is_loaded": is_current and self.inference_engine.is_loaded,
                    "performance_stats": stats
                },
                "last_updated": self._get_current_timestamp()
            }

        return TextResourceContents(
            uri=f"model://info/{model_id}",
            mimeType="application/json",
            text=json.dumps(resource_data, indent=2)
        )

    def _get_model_categories(self) -> Dict[str, List[str]]:
        """Get model categories from the registry."""
        registry = self.config.load_model_registry()
        return registry.categories

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        import datetime
        return datetime.datetime.now().isoformat()

    def get_resource_templates(self) -> List[Resource]:
        """Get template resources that can be parameterized."""
        templates = []

        # Model info template - can be used with any model ID
        templates.append(Resource(
            uri="model://info/{model_id}",
            name="Model Information Template",
            description="Get detailed information about any model by replacing {model_id}",
            mimeType="application/json"
        ))

        return templates

    async def list_model_resources(self) -> List[Resource]:
        """Get individual resources for each available model."""
        resources = []
        models = self.model_manager.list_available_models()

        for model_id in models.keys():
            resources.append(Resource(
                uri=f"model://info/{model_id}",
                name=f"Model Info: {model_id}",
                description=f"Detailed information about model {model_id}",
                mimeType="application/json"
            ))

        return resources