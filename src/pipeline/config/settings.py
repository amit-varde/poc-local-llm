"""Settings and configuration data models."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    """Application-level settings."""
    name: str = "Local LLM Pipeline"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"


class PathSettings(BaseModel):
    """Directory path settings."""
    models: Path = Path("./models")
    logs: Path = Path("./logs")
    cache: Path = Path("./cache")


class ResourceSettings(BaseModel):
    """Resource allocation settings."""
    max_memory_gb: int = 8
    max_cpu_cores: int = 4
    gpu_enabled: bool = False
    gpu_layers: int = 0


class ModelSettings(BaseModel):
    """Default model configuration."""
    default_model: str = "llama-2-7b-chat"
    context_length: int = 4096
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1


class InferenceSettings(BaseModel):
    """Inference engine settings."""
    batch_size: int = 1
    threads: int = -1  # -1 = auto-detect
    use_mmap: bool = True
    use_mlock: bool = False
    low_vram: bool = False


class LoggingSettings(BaseModel):
    """Logging configuration."""
    file: Path = Path("./logs/pipeline.log")
    max_size_mb: int = 100
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ServerSettings(BaseModel):
    """API server configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    access_log: bool = True


class APISettings(BaseModel):
    """API metadata settings."""
    title: str = "Local LLM Pipeline API"
    description: str = "OpenAI-compatible API for local LLM inference"
    version: str = "0.1.0"
    openapi_url: str = "/openapi.json"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


class CORSSettings(BaseModel):
    """CORS configuration."""
    allow_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    allow_credentials: bool = True
    allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: List[str] = ["*"]


class RateLimitSettings(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = False
    requests_per_minute: int = 60
    burst_size: int = 10


class AuthSettings(BaseModel):
    """Authentication settings."""
    enabled: bool = False
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None


class LimitSettings(BaseModel):
    """Request/Response limits."""
    max_request_size_mb: int = 10
    max_response_size_mb: int = 50
    request_timeout_seconds: int = 300


class HealthSettings(BaseModel):
    """Health check settings."""
    endpoint: str = "/health"
    include_model_status: bool = True
    include_system_stats: bool = True


class EndpointSettings(BaseModel):
    """API endpoint configuration."""
    # OpenAI Compatible
    chat_completions: str = "/v1/chat/completions"
    completions: str = "/v1/completions"
    models: str = "/v1/models"

    # Custom endpoints
    model_info: str = "/api/v1/model/info"
    model_load: str = "/api/v1/model/load"
    model_unload: str = "/api/v1/model/unload"
    system_status: str = "/api/v1/system/status"


class WebSocketSettings(BaseModel):
    """WebSocket configuration."""
    enabled: bool = True
    endpoint: str = "/ws"
    max_connections: int = 10
    heartbeat_interval: int = 30


class ModelDefinition(BaseModel):
    """Model definition from models.yaml."""
    name: str
    filename: str
    url: str
    size_gb: float
    context_length: int
    type: str
    quantization: str
    description: str


class DownloadSettings(BaseModel):
    """Model download settings."""
    chunk_size: int = 8192
    retry_attempts: int = 3
    timeout_seconds: int = 300
    verify_checksum: bool = True


class Settings(BaseModel):
    """Main settings container."""
    app: AppSettings = Field(default_factory=AppSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    resources: ResourceSettings = Field(default_factory=ResourceSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    api: APISettings = Field(default_factory=APISettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    health: HealthSettings = Field(default_factory=HealthSettings)
    endpoints: EndpointSettings = Field(default_factory=EndpointSettings)
    websocket: WebSocketSettings = Field(default_factory=WebSocketSettings)


class ModelRegistry(BaseModel):
    """Model registry from models.yaml."""
    models: Dict[str, ModelDefinition]
    categories: Dict[str, List[str]]
    download: DownloadSettings = Field(default_factory=DownloadSettings)