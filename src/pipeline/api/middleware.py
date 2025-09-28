"""Custom middleware for the API server."""

import time
import logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next):
        """Process request and log details."""
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Calculate response time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            f"Response: {response.status_code} - {process_time:.3f}s"
        )

        # Add response time header
        response.headers["X-Process-Time"] = str(process_time)

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_times = {}

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting."""
        # Get client IP
        client_ip = request.client.host

        # Clean old requests (older than 1 minute)
        current_time = time.time()
        cutoff_time = current_time - 60

        if client_ip in self.request_times:
            self.request_times[client_ip] = [
                req_time for req_time in self.request_times[client_ip]
                if req_time > cutoff_time
            ]

        # Check rate limit
        if client_ip not in self.request_times:
            self.request_times[client_ip] = []

        if len(self.request_times[client_ip]) >= self.requests_per_minute:
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )

        # Record this request
        self.request_times[client_ip].append(current_time)

        # Process request
        response = await call_next(request)
        return response


def setup_middleware(app: FastAPI):
    """Setup all middleware for the application."""
    from ..config import get_config

    config = get_config()
    settings = config.load_settings()

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)

    # Add rate limiting if enabled
    if settings.rate_limit.enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit.requests_per_minute
        )