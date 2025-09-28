"""MCP configuration management."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import yaml

logger = logging.getLogger(__name__)


class MCPToolConfig(BaseModel):
    """Configuration for an MCP tool."""
    enabled: bool = True
    description: str = ""
    max_tokens: Optional[int] = None
    require_confirmation: bool = False


class MCPResourceConfig(BaseModel):
    """Configuration for an MCP resource."""
    enabled: bool = True
    uri: str
    name: str
    description: str


class MCPPromptConfig(BaseModel):
    """Configuration for an MCP prompt."""
    enabled: bool = True
    name: str
    description: str


class MCPHttpConfig(BaseModel):
    """HTTP transport configuration."""
    host: str = "127.0.0.1"
    port: int = 9000


class MCPServerConfig(BaseModel):
    """MCP server configuration."""
    enabled: bool = True
    name: str = "Local LLM Pipeline"
    version: str = "0.1.0"
    transport: str = "stdio"  # "stdio" or "http"
    http: MCPHttpConfig = Field(default_factory=MCPHttpConfig)
    tools: Dict[str, MCPToolConfig] = Field(default_factory=dict)
    resources: Dict[str, MCPResourceConfig] = Field(default_factory=dict)
    prompts: Dict[str, MCPPromptConfig] = Field(default_factory=dict)


class MCPExternalServer(BaseModel):
    """Configuration for external MCP server."""
    name: str
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)


class MCPClientConfig(BaseModel):
    """MCP client configuration."""
    enabled: bool = False
    servers: List[MCPExternalServer] = Field(default_factory=list)


class MCPRateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 30
    burst_size: int = 5


class MCPAuditLogConfig(BaseModel):
    """Audit logging configuration."""
    enabled: bool = True
    file: str = "./logs/mcp_audit.log"
    include_request_data: bool = False


class MCPSecurityConfig(BaseModel):
    """Security configuration."""
    require_confirmation: List[str] = Field(default_factory=lambda: ["load_model"])
    rate_limit: MCPRateLimitConfig = Field(default_factory=MCPRateLimitConfig)
    allowed_operations: List[str] = Field(default_factory=lambda: ["read", "inference", "model_management"])
    audit_log: MCPAuditLogConfig = Field(default_factory=MCPAuditLogConfig)


class MCPLoggingConfig(BaseModel):
    """MCP logging configuration."""
    level: str = "INFO"
    file: str = "./logs/mcp.log"
    max_size_mb: int = 50
    backup_count: int = 3


class MCPConfig(BaseModel):
    """Main MCP configuration."""
    server: MCPServerConfig = Field(default_factory=MCPServerConfig)
    client: MCPClientConfig = Field(default_factory=MCPClientConfig)
    security: MCPSecurityConfig = Field(default_factory=MCPSecurityConfig)
    logging: MCPLoggingConfig = Field(default_factory=MCPLoggingConfig)


class MCPConfigManager:
    """Manages MCP configuration loading and access."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize the MCP configuration manager.

        Args:
            config_dir: Path to configuration directory. Defaults to ./etc
        """
        self.config_dir = config_dir or Path("./etc")
        self._config: Optional[MCPConfig] = None

    def load_config(self) -> MCPConfig:
        """Load MCP configuration from mcp.yaml."""
        if self._config is not None:
            return self._config

        mcp_config_path = self.config_dir / "mcp.yaml"

        mcp_config = {}
        if mcp_config_path.exists():
            try:
                with open(mcp_config_path, 'r') as f:
                    mcp_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded MCP config from {mcp_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load MCP config: {e}")

        # Create MCP config object
        self._config = MCPConfig(**mcp_config)
        return self._config

    def get_enabled_tools(self) -> Dict[str, MCPToolConfig]:
        """Get all enabled tools."""
        config = self.load_config()
        return {
            name: tool_config
            for name, tool_config in config.server.tools.items()
            if tool_config.enabled
        }

    def get_enabled_resources(self) -> Dict[str, MCPResourceConfig]:
        """Get all enabled resources."""
        config = self.load_config()
        return {
            name: resource_config
            for name, resource_config in config.server.resources.items()
            if resource_config.enabled
        }

    def get_enabled_prompts(self) -> Dict[str, MCPPromptConfig]:
        """Get all enabled prompts."""
        config = self.load_config()
        return {
            name: prompt_config
            for name, prompt_config in config.server.prompts.items()
            if prompt_config.enabled
        }

    def is_operation_allowed(self, operation: str) -> bool:
        """Check if an operation is allowed."""
        config = self.load_config()
        return operation in config.security.allowed_operations

    def requires_confirmation(self, operation: str) -> bool:
        """Check if an operation requires user confirmation."""
        config = self.load_config()
        return operation in config.security.require_confirmation


# Global MCP config manager instance
_mcp_config_manager: Optional[MCPConfigManager] = None


def get_mcp_config() -> MCPConfigManager:
    """Get the global MCP configuration manager instance."""
    global _mcp_config_manager
    if _mcp_config_manager is None:
        _mcp_config_manager = MCPConfigManager()
    return _mcp_config_manager