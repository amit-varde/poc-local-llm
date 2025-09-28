"""API routes for the Local LLM Pipeline."""

import logging
import time
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
import json

from ..config import get_config
from ..models import ModelManager
from ..inference import (
    InferenceEngine,
    GenerationRequest,
    ChatRequest,
    ChatMessage,
    MessageRole
)
from .app import get_inference_engine

logger = logging.getLogger(__name__)

router = APIRouter()


# Dependency to get inference engine
def get_engine() -> InferenceEngine:
    """Get inference engine dependency."""
    return get_inference_engine()


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    config = get_config()
    settings = config.load_settings()

    try:
        engine = get_inference_engine()
        model_status = {
            "loaded": engine.is_loaded,
            "current_model": engine.current_model,
            "stats": engine.stats.dict() if settings.health.include_model_status else None
        }
    except Exception as e:
        model_status = {"error": str(e)}

    return {
        "status": "healthy",
        "version": settings.app.version,
        "model": model_status if settings.health.include_model_status else None
    }


# OpenAI-compatible endpoints
@router.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)."""
    model_manager = ModelManager()
    models = model_manager.list_available_models()

    openai_models = []
    for model_id, model_info in models.items():
        openai_models.append({
            "id": model_id,
            "object": "model",
            "created": 1677610602,  # Placeholder timestamp
            "owned_by": "local"
        })

    return {"object": "list", "data": openai_models}


@router.post("/v1/completions")
async def create_completion(
    request: Dict[str, Any],
    engine: InferenceEngine = Depends(get_engine)
):
    """Create completion (OpenAI compatible)."""
    try:
        # Extract parameters from OpenAI format
        gen_request = GenerationRequest(
            prompt=request.get("prompt", ""),
            max_tokens=request.get("max_tokens", 512),
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.9),
            stop=request.get("stop"),
            stream=request.get("stream", False)
        )

        if gen_request.stream:
            return StreamingResponse(
                stream_completion(gen_request, engine),
                media_type="text/plain"
            )

        # Generate response
        response = await engine.generate(gen_request)

        # Format as OpenAI response
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": engine.current_model,
            "choices": [
                {
                    "text": response.text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": response.finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens
            }
        }

    except Exception as e:
        logger.error(f"Completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Dict[str, Any],
    engine: InferenceEngine = Depends(get_engine)
):
    """Create chat completion (OpenAI compatible)."""
    try:
        # Convert OpenAI messages format
        messages = []
        for msg in request.get("messages", []):
            role = MessageRole(msg["role"])
            content = msg["content"]
            messages.append(ChatMessage(role=role, content=content))

        chat_request = ChatRequest(
            messages=messages,
            max_tokens=request.get("max_tokens", 512),
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.9),
            stop=request.get("stop"),
            stream=request.get("stream", False)
        )

        if chat_request.stream:
            return StreamingResponse(
                stream_chat_completion(chat_request, engine),
                media_type="text/plain"
            )

        # Generate response
        response = await engine.chat(chat_request)

        # Format as OpenAI response
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.current_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": response.message.role.value,
                        "content": response.message.content
                    },
                    "finish_reason": response.finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens
            }
        }

    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Custom API endpoints
@router.get("/api/v1/models")
async def get_models_detailed():
    """Get detailed model information."""
    model_manager = ModelManager()
    return model_manager.list_available_models()


@router.get("/api/v1/model/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    model_manager = ModelManager()
    model_info = model_manager.get_model_info(model_id)

    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return model_info


@router.post("/api/v1/model/load")
async def load_model(
    request: Dict[str, str],
    engine: InferenceEngine = Depends(get_engine)
):
    """Load a model for inference."""
    model_id = request.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    try:
        await engine.load_model(model_id)
        return {"message": f"Model {model_id} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/model/unload")
async def unload_model(engine: InferenceEngine = Depends(get_engine)):
    """Unload the current model."""
    try:
        await engine.unload_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/system/status")
async def get_system_status(engine: InferenceEngine = Depends(get_engine)):
    """Get system status and statistics."""
    model_manager = ModelManager()

    return {
        "inference_engine": {
            "model_loaded": engine.is_loaded,
            "current_model": engine.current_model,
            "stats": engine.stats.dict()
        },
        "storage": model_manager.get_storage_usage(),
        "available_models": len(model_manager.list_available_models())
    }


# Streaming helpers
async def stream_completion(request: GenerationRequest, engine: InferenceEngine):
    """Stream completion responses."""
    import time

    async for chunk in engine.stream_generate(request):
        # Format as OpenAI streaming response
        data = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": engine.current_model,
            "choices": [
                {
                    "text": chunk.delta,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": chunk.finish_reason
                }
            ]
        }

        yield f"data: {json.dumps(data)}\\n\\n"

        if chunk.finish_reason:
            yield "data: [DONE]\\n\\n"
            break


async def stream_chat_completion(request: ChatRequest, engine: InferenceEngine):
    """Stream chat completion responses."""
    import time

    async for chunk in engine.stream_chat(request):
        # Format as OpenAI streaming response
        data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": engine.current_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk.delta
                    } if chunk.delta else {},
                    "finish_reason": chunk.finish_reason
                }
            ]
        }

        yield f"data: {json.dumps(data)}\\n\\n"

        if chunk.finish_reason:
            yield "data: [DONE]\\n\\n"
            break