"""Model downloading utilities."""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional, Callable
import httpx
from rich.progress import Progress, DownloadColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from ..config import get_config

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Handles downloading models from remote sources."""

    def __init__(self):
        """Initialize the model downloader."""
        self.config = get_config()
        self.settings = self.config.load_settings()
        self.registry = self.config.load_model_registry()

    async def download_model(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """Download a model by ID.

        Args:
            model_id: The model identifier
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the downloaded model file

        Raises:
            ValueError: If model ID is not found
            Exception: If download fails
        """
        model_def = self.config.get_model_definition(model_id)
        if not model_def:
            raise ValueError(f"Model '{model_id}' not found in registry")

        model_path = self.settings.paths.models / model_def.filename

        # Check if model already exists
        if model_path.exists():
            logger.info(f"Model {model_id} already exists at {model_path}")
            return model_path

        # Ensure models directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {model_id} from {model_def.url}")

        # Download with progress tracking
        try:
            await self._download_file(
                model_def.url,
                model_path,
                progress_callback
            )

            logger.info(f"Successfully downloaded {model_id} to {model_path}")
            return model_path

        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise Exception(f"Failed to download {model_id}: {e}")

    async def _download_file(
        self,
        url: str,
        destination: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Download a file with progress tracking."""
        chunk_size = self.registry.download.chunk_size
        timeout = self.registry.download.timeout_seconds

        async with httpx.AsyncClient(timeout=timeout) as client:
            # Get file size first
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded_size = 0

                with open(destination, "wb") as file:
                    async for chunk in response.aiter_bytes(chunk_size):
                        if chunk:
                            file.write(chunk)
                            downloaded_size += len(chunk)

                            if progress_callback:
                                progress_callback(downloaded_size, total_size)

    def download_model_sync(self, model_id: str) -> Path:
        """Synchronous wrapper for downloading models."""
        return asyncio.run(self.download_model(model_id))

    def download_model_with_progress(self, model_id: str) -> Path:
        """Download model with rich progress bar."""
        model_def = self.config.get_model_definition(model_id)
        if not model_def:
            raise ValueError(f"Model '{model_id}' not found in registry")

        model_path = self.settings.paths.models / model_def.filename

        # Check if model already exists
        if model_path.exists():
            logger.info(f"Model {model_id} already exists")
            return model_path

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:

            task = progress.add_task(
                f"Downloading {model_def.name}",
                total=None
            )

            def update_progress(downloaded: int, total: int):
                if total > 0:
                    progress.update(task, total=total, completed=downloaded)

            # Run async download in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.download_model(model_id, update_progress)
                )
                return result
            finally:
                loop.close()

    def verify_model_checksum(self, model_path: Path, expected_hash: str) -> bool:
        """Verify model file checksum."""
        if not model_path.exists():
            return False

        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest() == expected_hash

    def get_model_size(self, model_id: str) -> float:
        """Get expected model size in GB."""
        model_def = self.config.get_model_definition(model_id)
        return model_def.size_gb if model_def else 0.0

    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded."""
        model_def = self.config.get_model_definition(model_id)
        if not model_def:
            return False

        model_path = self.settings.paths.models / model_def.filename
        return model_path.exists()

    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        model_def = self.config.get_model_definition(model_id)
        if not model_def:
            return False

        model_path = self.settings.paths.models / model_def.filename
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Deleted model {model_id}")
            return True

        return False