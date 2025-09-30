"""Inference engine implementation using llama-cpp-python."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, AsyncIterator, Dict, Any, List
from contextlib import asynccontextmanager

try:
    from llama_cpp import Llama, LlamaGrammar
except ImportError:
    Llama = None
    LlamaGrammar = None

from ..config import get_config
from ..models import ModelManager
from .types import (
    GenerationRequest,
    GenerationResponse,
    ChatRequest,
    ChatResponse,
    ChatMessage,
    MessageRole,
    StreamChunk,
    ModelStats,
    InferenceConfig
)

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Handles LLM inference using llama.cpp."""

    def __init__(self):
        """Initialize the inference engine."""
        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
            )

        self.config = get_config()
        self.settings = self.config.load_settings()
        self.model_manager = ModelManager()

        self._llama: Optional[Llama] = None
        self._current_model_id: Optional[str] = None
        self._stats = ModelStats(
            tokens_per_second=0.0,
            total_tokens_processed=0,
            average_response_time=0.0,
            memory_usage_mb=0.0,
            model_loaded=False
        )
        self._total_inference_time = 0.0
        self._inference_count = 0

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._llama is not None

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model ID."""
        return self._current_model_id

    @property
    def stats(self) -> ModelStats:
        """Get current performance statistics."""
        return self._stats

    async def load_model(self, model_id: str, config: Optional[InferenceConfig] = None) -> None:
        """Load a model for inference.

        Args:
            model_id: The model identifier
            config: Optional inference configuration

        Raises:
            ValueError: If model is not found or invalid
            Exception: If model loading fails
        """
        # Check if model is already loaded
        if self._current_model_id == model_id and self.is_loaded:
            logger.info(f"Model {model_id} is already loaded")
            return

        # Unload current model if any
        if self.is_loaded:
            await self.unload_model()

        # Get model path
        model_path = self.model_manager.get_model_path(model_id)
        if not model_path:
            # Try to download the model
            logger.info(f"Model {model_id} not found locally, downloading...")
            model_path = self.model_manager.download_model(model_id)

        if not self.model_manager.validate_model(model_id):
            raise ValueError(f"Model {model_id} is invalid or corrupted")

        # Use provided config or create default
        if config is None:
            model_def = self.config.get_model_definition(model_id)
            config = InferenceConfig(
                model_id=model_id,
                context_length=model_def.context_length if model_def else self.settings.model.context_length,
                threads=self.settings.inference.threads,
                gpu_layers=self.settings.resources.gpu_layers,
                use_mmap=self.settings.inference.use_mmap,
                use_mlock=self.settings.inference.use_mlock,
                low_vram=self.settings.inference.low_vram
            )

        logger.info(f"Loading model {model_id} from {model_path}")

        try:
            # Run model loading in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self._llama = await loop.run_in_executor(
                None,
                self._load_llama_model,
                model_path,
                config
            )

            self._current_model_id = model_id
            self._stats.model_loaded = True

            # Mark model as used
            self.model_manager.mark_model_used(model_id)

            logger.info(f"Successfully loaded model {model_id}")

        except Exception as e:
            self._llama = None
            self._current_model_id = None
            self._stats.model_loaded = False
            raise Exception(f"Failed to load model {model_id}: {e}")

    def _load_llama_model(self, model_path: Path, config: InferenceConfig) -> Llama:
        """Load llama model (runs in executor)."""
        return Llama(
            model_path=str(model_path),
            n_ctx=config.context_length,
            n_batch=config.batch_size,
            n_threads=config.threads if config.threads > 0 else None,
            n_gpu_layers=config.gpu_layers,
            use_mmap=config.use_mmap,
            use_mlock=config.use_mlock,
            low_vram=config.low_vram,
            verbose=config.verbose
        )

    async def unload_model(self) -> None:
        """Unload the currently loaded model."""
        if self._llama is not None:
            # llama-cpp-python doesn't have explicit cleanup, but we can release the reference
            self._llama = None
            self._current_model_id = None
            self._stats.model_loaded = False
            logger.info("Model unloaded")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text completion.

        Args:
            request: Generation request

        Returns:
            Generated text response

        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_loaded:
            raise RuntimeError("No model is loaded")

        start_time = time.time()

        try:
            # Run generation in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._generate_sync,
                request
            )

            # Update statistics
            end_time = time.time()
            inference_time = end_time - start_time
            self._update_stats(result.total_tokens, inference_time)

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _generate_sync(self, request: GenerationRequest) -> GenerationResponse:
        """Synchronous generation (runs in executor)."""
        output = self._llama(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            stop=request.stop or [],
            stream=False
        )

        choice = output["choices"][0]
        usage = output["usage"]

        return GenerationResponse(
            text=choice["text"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            finish_reason=choice["finish_reason"]
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate chat completion.

        Args:
            request: Chat request

        Returns:
            Chat response

        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_loaded:
            raise RuntimeError("No model is loaded")

        # Convert chat messages to prompt
        prompt = self._format_chat_prompt(request.messages)

        # Create generation request
        gen_request = GenerationRequest(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            stop=request.stop,
            stream=False
        )

        # Generate response
        gen_response = await self.generate(gen_request)

        # Create chat response
        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=gen_response.text.strip()
        )

        return ChatResponse(
            message=assistant_message,
            prompt_tokens=gen_response.prompt_tokens,
            completion_tokens=gen_response.completion_tokens,
            total_tokens=gen_response.total_tokens,
            finish_reason=gen_response.finish_reason
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream chat completion.

        Args:
            request: Chat request with stream=True

        Yields:
            Stream chunks

        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_loaded:
            raise RuntimeError("No model is loaded")

        # Convert chat messages to prompt
        prompt = self._format_chat_prompt(request.messages)

        async for chunk in self.stream_generate(GenerationRequest(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            stop=request.stop,
            stream=True
        )):
            yield chunk

    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Stream text generation.

        Args:
            request: Generation request with stream=True

        Yields:
            Stream chunks

        Raises:
            RuntimeError: If no model is loaded
        """
        if not self.is_loaded:
            raise RuntimeError("No model is loaded")

        start_time = time.time()
        total_tokens = 0

        try:
            # Create async generator
            loop = asyncio.get_event_loop()

            # Run streaming generation in executor
            def stream_sync():
                return self._llama(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repeat_penalty=request.repeat_penalty,
                    stop=request.stop or [],
                    stream=True
                )

            stream = await loop.run_in_executor(None, stream_sync)

            for output in stream:
                choice = output["choices"][0]
                delta = choice.get("text", "")
                finish_reason = choice.get("finish_reason")

                if delta:
                    total_tokens += 1

                yield StreamChunk(
                    delta=delta,
                    finish_reason=finish_reason
                )

                if finish_reason:
                    break

            # Update statistics
            end_time = time.time()
            inference_time = end_time - start_time
            self._update_stats(total_tokens, inference_time)

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a prompt string."""
        # Basic chat formatting - can be improved with model-specific templates
        prompt_parts = []

        for message in messages:
            if message.role == MessageRole.SYSTEM:
                prompt_parts.append(f"System: {message.content}")
            elif message.role == MessageRole.USER:
                prompt_parts.append(f"User: {message.content}")
            elif message.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {message.content}")

        prompt_parts.append("Assistant:")
        return "\\n\\n".join(prompt_parts)

    def _update_stats(self, tokens: int, inference_time: float) -> None:
        """Update performance statistics."""
        self._total_inference_time += inference_time
        self._inference_count += 1
        self._stats.total_tokens_processed += tokens

        # Calculate tokens per second
        if inference_time > 0:
            self._stats.tokens_per_second = tokens / inference_time

        # Calculate average response time
        self._stats.average_response_time = self._total_inference_time / self._inference_count

        # Update last inference time
        import datetime
        self._stats.last_inference_time = datetime.datetime.now().isoformat()

        logger.debug(f"Updated stats: {self._stats}")

    @asynccontextmanager
    async def model_context(self, model_id: str, config: Optional[InferenceConfig] = None):
        """Context manager for temporary model loading."""
        original_model = self._current_model_id
        try:
            await self.load_model(model_id, config)
            yield
        finally:
            if original_model and original_model != model_id:
                await self.load_model(original_model)
            elif not original_model:
                await self.unload_model()