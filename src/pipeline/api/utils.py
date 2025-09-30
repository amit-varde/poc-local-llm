"""Utility functions for the API."""

from ..inference import InferenceEngine
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Global inference engine instance
inference_engine: Optional[InferenceEngine] = None

def get_inference_engine() -> InferenceEngine:
    """Get the global inference engine instance."""
    global inference_engine
    if inference_engine is None:
        raise RuntimeError("Inference engine not initialized")
    return inference_engine