"""Type definitions for inference operations."""

from enum import Enum
from typing import List, Optional, Dict, Any, AsyncIterator
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A chat message."""
    role: MessageRole
    content: str


class GenerationRequest(BaseModel):
    """Request for text generation."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    stream: bool = False


class GenerationResponse(BaseModel):
    """Response from text generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str  # "stop", "length", "error"


class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: Optional[List[str]] = None
    stream: bool = False


class ChatResponse(BaseModel):
    """Response from chat completion."""
    message: ChatMessage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str


class StreamChunk(BaseModel):
    """Streaming response chunk."""
    delta: str
    finish_reason: Optional[str] = None


class ModelStats(BaseModel):
    """Model performance statistics."""
    tokens_per_second: float
    total_tokens_processed: int
    average_response_time: float
    memory_usage_mb: float
    model_loaded: bool
    last_inference_time: Optional[str] = None


class InferenceConfig(BaseModel):
    """Configuration for inference operations."""
    model_id: str
    context_length: int = 4096
    batch_size: int = 1
    threads: int = -1  # -1 = auto
    gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = False
    low_vram: bool = False
    verbose: bool = False