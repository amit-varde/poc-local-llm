"""Configuration manager for loading and managing settings."""

import logging
from pathlib import Path
from typing import Optional
import yaml
from .settings import Settings, ModelRegistry

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and access."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Path to configuration directory. Defaults to ./etc
        """
        self.config_dir = config_dir or Path("./etc")
        self._settings: Optional[Settings] = None
        self._model_registry: Optional[ModelRegistry] = None

    def load_settings(self) -> Settings:
        """Load main pipeline settings from pipeline.yaml."""
        if self._settings is not None:
            return self._settings

        pipeline_config_path = self.config_dir / "pipeline.yaml"
        server_config_path = self.config_dir / "server.yaml"

        # Load pipeline config
        pipeline_config = {}
        if pipeline_config_path.exists():
            try:
                with open(pipeline_config_path, 'r') as f:
                    pipeline_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded pipeline config from {pipeline_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load pipeline config: {e}")

        # Load server config
        server_config = {}
        if server_config_path.exists():
            try:
                with open(server_config_path, 'r') as f:
                    server_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded server config from {server_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load server config: {e}")

        # Merge configs
        merged_config = {**pipeline_config, **server_config}

        # Create settings object
        self._settings = Settings(**merged_config)
        return self._settings

    def load_model_registry(self) -> ModelRegistry:
        """Load model registry from models.yaml."""
        if self._model_registry is not None:
            return self._model_registry

        models_config_path = self.config_dir / "models.yaml"

        models_config = {}
        if models_config_path.exists():
            try:
                with open(models_config_path, 'r') as f:
                    models_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded model registry from {models_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")

        # Create model registry object
        self._model_registry = ModelRegistry(**models_config)
        return self._model_registry

    def get_model_definition(self, model_id: str):
        """Get model definition by ID."""
        registry = self.load_model_registry()
        return registry.models.get(model_id)

    def list_available_models(self) -> list[str]:
        """List all available model IDs."""
        registry = self.load_model_registry()
        return list(registry.models.keys())

    def get_models_by_category(self, category: str) -> list[str]:
        """Get model IDs by category."""
        registry = self.load_model_registry()
        return registry.categories.get(category, [])

    def ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        settings = self.load_settings()

        # Create directories if they don't exist
        settings.paths.models.mkdir(parents=True, exist_ok=True)
        settings.paths.logs.mkdir(parents=True, exist_ok=True)
        settings.paths.cache.mkdir(parents=True, exist_ok=True)

        logger.info("Ensured all directories exist")


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager