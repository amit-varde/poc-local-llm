"""Model management and lifecycle."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

from ..config import get_config
from .downloader import ModelDownloader

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model lifecycle, metadata, and operations."""

    def __init__(self):
        """Initialize the model manager."""
        self.config = get_config()
        self.settings = self.config.load_settings()
        self.registry = self.config.load_model_registry()
        self.downloader = ModelDownloader()
        self._metadata_file = self.settings.paths.models / "metadata.json"
        self._metadata: Dict[str, Any] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load model metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    self._metadata = json.load(f)
                logger.debug("Loaded model metadata")
            except Exception as e:
                logger.warning(f"Failed to load model metadata: {e}")
                self._metadata = {}
        else:
            self._metadata = {}

    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        try:
            self._metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
            logger.debug("Saved model metadata")
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")

    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their status."""
        models = {}

        for model_id, model_def in self.registry.models.items():
            is_downloaded = self.downloader.is_model_downloaded(model_id)
            model_path = self.settings.paths.models / model_def.filename

            models[model_id] = {
                "name": model_def.name,
                "description": model_def.description,
                "type": model_def.type,
                "size_gb": model_def.size_gb,
                "context_length": model_def.context_length,
                "quantization": model_def.quantization,
                "downloaded": is_downloaded,
                "path": str(model_path) if is_downloaded else None,
                "last_used": self._metadata.get(model_id, {}).get("last_used"),
                "download_date": self._metadata.get(model_id, {}).get("download_date"),
            }

        return models

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        if model_id not in self.registry.models:
            return None

        model_def = self.registry.models[model_id]
        is_downloaded = self.downloader.is_model_downloaded(model_id)
        model_path = self.settings.paths.models / model_def.filename

        metadata = self._metadata.get(model_id, {})

        return {
            "id": model_id,
            "name": model_def.name,
            "description": model_def.description,
            "filename": model_def.filename,
            "type": model_def.type,
            "size_gb": model_def.size_gb,
            "context_length": model_def.context_length,
            "quantization": model_def.quantization,
            "url": model_def.url,
            "downloaded": is_downloaded,
            "path": str(model_path) if is_downloaded else None,
            "file_size_bytes": model_path.stat().st_size if is_downloaded and model_path.exists() else None,
            "last_used": metadata.get("last_used"),
            "download_date": metadata.get("download_date"),
            "usage_count": metadata.get("usage_count", 0),
        }

    def download_model(self, model_id: str, show_progress: bool = True) -> Path:
        """Download a model and update metadata."""
        if show_progress:
            model_path = self.downloader.download_model_with_progress(model_id)
        else:
            model_path = self.downloader.download_model_sync(model_id)

        # Update metadata
        import datetime
        self._metadata[model_id] = {
            "download_date": datetime.datetime.now().isoformat(),
            "usage_count": 0
        }
        self._save_metadata()

        logger.info(f"Downloaded model {model_id} to {model_path}")
        return model_path

    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its metadata."""
        success = self.downloader.delete_model(model_id)

        if success and model_id in self._metadata:
            del self._metadata[model_id]
            self._save_metadata()

        return success

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get the path to a model file if it exists."""
        model_def = self.config.get_model_definition(model_id)
        if not model_def:
            return None

        model_path = self.settings.paths.models / model_def.filename
        return model_path if model_path.exists() else None

    def mark_model_used(self, model_id: str) -> None:
        """Mark a model as recently used."""
        import datetime

        if model_id not in self._metadata:
            self._metadata[model_id] = {"usage_count": 0}

        self._metadata[model_id]["last_used"] = datetime.datetime.now().isoformat()
        self._metadata[model_id]["usage_count"] = self._metadata[model_id].get("usage_count", 0) + 1

        self._save_metadata()

    def get_default_model(self) -> str:
        """Get the default model ID."""
        return self.settings.model.default_model

    def validate_model(self, model_id: str) -> bool:
        """Validate that a model exists and is properly downloaded."""
        model_path = self.get_model_path(model_id)
        if not model_path:
            return False

        # Basic file validation
        try:
            if model_path.stat().st_size == 0:
                logger.warning(f"Model file {model_path} is empty")
                return False

            # Additional validation could go here (file format, etc.)
            return True

        except Exception as e:
            logger.error(f"Error validating model {model_id}: {e}")
            return False

    def get_models_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all models in a specific category."""
        all_models = self.list_available_models()
        category_models = self.config.get_models_by_category(category)

        return {
            model_id: all_models[model_id]
            for model_id in category_models
            if model_id in all_models
        }

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage information for models."""
        total_size = 0
        downloaded_models = 0
        models_dir = self.settings.paths.models

        if models_dir.exists():
            for file_path in models_dir.glob("*.gguf"):
                total_size += file_path.stat().st_size
                downloaded_models += 1

        return {
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024**3), 2),
            "downloaded_models": downloaded_models,
            "models_directory": str(models_dir),
        }