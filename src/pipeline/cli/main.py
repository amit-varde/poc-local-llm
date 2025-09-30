"""Main CLI application using Typer."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rich_print
import warnings

from ..config import get_config
from ..models import ModelManager
from ..inference import InferenceEngine, ChatMessage, ChatRequest, MessageRole
from .server import ServerManager
from ..mcp.server import MCPServer
from ..mcp.config import get_mcp_config

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules after import of package.*")

# Setup console and app
console = Console()
app = typer.Typer(
    name="llm-pipeline",
    help="Local LLM Pipeline - Run LLMs locally with ease",
    no_args_is_help=True
)

# Subcommands
model_app = typer.Typer(name="model", help="Model management commands")
server_app = typer.Typer(name="server", help="Server management commands")
chat_app = typer.Typer(name="chat", help="Interactive chat commands")
mcp_app = typer.Typer(name="mcp", help="Model Context Protocol (MCP) commands")

app.add_typer(model_app, name="model")
app.add_typer(server_app, name="server")
app.add_typer(chat_app, name="chat")
app.add_typer(mcp_app, name="mcp")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    config_dir: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration directory")
):
    """Local LLM Pipeline CLI."""
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize config with custom directory if provided
    if config_dir:
        global _config_manager
        from ..config.config_manager import ConfigManager
        _config_manager = ConfigManager(config_dir)


# Model management commands
@model_app.command("list")
def list_models(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    downloaded_only: bool = typer.Option(False, "--downloaded", "-d", help="Show only downloaded models")
):
    """List available models."""
    model_manager = ModelManager()

    if category:
        models = model_manager.get_models_by_category(category)
        # Get full model info
        all_models = model_manager.list_available_models()
        models = {k: all_models[k] for k in models if k in all_models}
    else:
        models = model_manager.list_available_models()

    if downloaded_only:
        models = {k: v for k, v in models.items() if v["downloaded"]}

    if not models:
        rich_print("[yellow]No models found matching criteria[/yellow]")
        return

    # Create table
    table = Table(title="Available Models")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Size (GB)", justify="right", style="blue")
    table.add_column("Downloaded", justify="center")
    table.add_column("Last Used")

    for model_id, info in models.items():
        downloaded = "✅" if info["downloaded"] else "❌"
        last_used = info.get("last_used", "Never")
        if last_used and last_used != "Never":
            last_used = last_used.split("T")[0]  # Just the date part

        table.add_row(
            model_id,
            info["name"],
            info["type"],
            str(info["size_gb"]),
            downloaded,
            last_used
        )

    console.print(table)


@model_app.command("info")
def model_info(model_id: str = typer.Argument(..., help="Model ID")):
    """Get detailed information about a model."""
    model_manager = ModelManager()
    info = model_manager.get_model_info(model_id)

    if not info:
        rich_print(f"[red]Model '{model_id}' not found[/red]")
        raise typer.Exit(1)

    rich_print(f"[bold cyan]Model Information: {model_id}[/bold cyan]")
    rich_print(f"Name: {info['name']}")
    rich_print(f"Description: {info['description']}")
    rich_print(f"Type: {info['type']}")
    rich_print(f"Size: {info['size_gb']} GB")
    rich_print(f"Context Length: {info['context_length']}")
    rich_print(f"Quantization: {info['quantization']}")
    rich_print(f"Downloaded: {'Yes' if info['downloaded'] else 'No'}")

    if info['downloaded']:
        rich_print(f"Path: {info['path']}")
        if info['file_size_bytes']:
            file_size_gb = info['file_size_bytes'] / (1024**3)
            rich_print(f"Actual Size: {file_size_gb:.2f} GB")

    if info.get('last_used'):
        rich_print(f"Last Used: {info['last_used']}")
    if info.get('usage_count'):
        rich_print(f"Usage Count: {info['usage_count']}")


@model_app.command("download")
def download_model(model_id: str = typer.Argument(..., help="Model ID to download")):
    """Download a model."""
    model_manager = ModelManager()

    # Check if model exists
    if not model_manager.config.get_model_definition(model_id):
        rich_print(f"[red]Model '{model_id}' not found in registry[/red]")
        raise typer.Exit(1)

    # Check if already downloaded
    if model_manager.downloader.is_model_downloaded(model_id):
        rich_print(f"[yellow]Model '{model_id}' is already downloaded[/yellow]")
        return

    try:
        rich_print(f"[blue]Downloading model '{model_id}'...[/blue]")
        model_path = model_manager.download_model(model_id, show_progress=True)
        rich_print(f"[green]Successfully downloaded to: {model_path}[/green]")
    except Exception as e:
        rich_print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@model_app.command("delete")
def delete_model(
    model_id: str = typer.Argument(..., help="Model ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation")
):
    """Delete a downloaded model."""
    model_manager = ModelManager()

    if not model_manager.downloader.is_model_downloaded(model_id):
        rich_print(f"[yellow]Model '{model_id}' is not downloaded[/yellow]")
        return

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete model '{model_id}'?")
        if not confirm:
            rich_print("[yellow]Deletion cancelled[/yellow]")
            return

    try:
        if model_manager.delete_model(model_id):
            rich_print(f"[green]Successfully deleted model '{model_id}'[/green]")
        else:
            rich_print(f"[red]Failed to delete model '{model_id}'[/red]")
    except Exception as e:
        rich_print(f"[red]Error deleting model: {e}[/red]")
        raise typer.Exit(1)


# Server management commands
@server_app.command("start")
def start_server(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Host to bind to"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to load on startup")
):
    """Start the API server."""
    server_manager = ServerManager()
    server_manager.start(host=host, port=port, reload=reload, default_model=model)


@server_app.command("stop")
def stop_server():
    """Stop the API server."""
    server_manager = ServerManager()
    server_manager.stop()


@server_app.command("status")
def server_status():
    """Check server status."""
    server_manager = ServerManager()
    status = server_manager.status()

    if status["running"]:
        rich_print(f"[green]Server is running on {status['url']}[/green]")
        if status.get("model"):
            rich_print(f"Current model: {status['model']}")
    else:
        rich_print("[red]Server is not running[/red]")


# Chat commands
@chat_app.command("start")
def start_chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use for chat"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System message"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature")
):
    """Start an interactive chat session."""
    asyncio.run(_chat_session(model, system, max_tokens, temperature))


async def _chat_session(
    model_id: Optional[str],
    system_message: Optional[str],
    max_tokens: int,
    temperature: float
):
    """Run interactive chat session."""
    # Initialize engine
    engine = InferenceEngine()

    # Determine model to use
    if not model_id:
        config = get_config()
        settings = config.load_settings()
        model_id = settings.model.default_model

    try:
        # Load model
        rich_print(f"[blue]Loading model: {model_id}[/blue]")
        await engine.load_model(model_id)
        rich_print(f"[green]Model loaded successfully![/green]")
    except Exception as e:
        rich_print(f"[red]Failed to load model: {e}[/red]")
        return

    # Initialize chat history
    messages = []
    if system_message:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_message))

    rich_print("[cyan]Chat session started. Type 'quit' or 'exit' to end.[/cyan]")
    rich_print("[dim]Press Ctrl+C to force quit.[/dim]")

    try:
        while True:
            # Get user input
            user_input = typer.prompt("You", prompt_suffix=" > ")

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            # Add user message
            messages.append(ChatMessage(role=MessageRole.USER, content=user_input))

            # Create chat request
            chat_request = ChatRequest(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            try:
                # Generate response
                rich_print("[dim]Thinking...[/dim]")
                response = await engine.chat(chat_request)

                # Display response
                rich_print(f"[green]Assistant[/green] > {response.message.content}")

                # Add assistant message to history
                messages.append(response.message)

            except Exception as e:
                rich_print(f"[red]Error generating response: {e}[/red]")

    except KeyboardInterrupt:
        rich_print("\n[yellow]Chat session interrupted[/yellow]")
    finally:
        await engine.unload_model()
        rich_print("[cyan]Chat session ended[/cyan]")


# Utility commands
@app.command("init")
def init_project():
    """Initialize a new Local LLM Pipeline project."""
    config = get_config()

    rich_print("[blue]Initializing Local LLM Pipeline project...[/blue]")

    # Ensure directories exist
    config.ensure_directories()

    rich_print("[green]Project initialized successfully![/green]")
    rich_print("\nNext steps:")
    rich_print("1. Download a model: llm-pipeline model download tinyllama-1b")
    rich_print("2. Start the server: llm-pipeline server start")
    rich_print("3. Start a chat: llm-pipeline chat start")


@app.command("version")
def version():
    """Show version information."""
    from .. import __version__
    rich_print(f"Local LLM Pipeline v{__version__}")


# MCP commands
@mcp_app.command("server")
def mcp_server_command(
    action: str = typer.Argument(..., help="Action: start, stop, status"),
    transport: Optional[str] = typer.Option(None, "--transport", "-t", help="Transport type: stdio, http"),
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Host for HTTP transport"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port for HTTP transport"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging")
):
    """Manage MCP server."""
    if action == "start":
        _start_mcp_server(transport, host, port, verbose)
    elif action == "stop":
        rich_print("[yellow]MCP server stop not implemented yet[/yellow]")
    elif action == "status":
        _mcp_server_status()
    else:
        rich_print(f"[red]Unknown action: {action}[/red]")
        rich_print("Available actions: start, stop, status")
        raise typer.Exit(1)


@mcp_app.command("tools")
def list_mcp_tools():
    """List available MCP tools."""
    mcp_config = get_mcp_config()
    enabled_tools = mcp_config.get_enabled_tools()

    if not enabled_tools:
        rich_print("[yellow]No MCP tools are enabled[/yellow]")
        return

    # Create table
    table = Table(title="Available MCP Tools")
    table.add_column("Tool Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Max Tokens", justify="right", style="blue")
    table.add_column("Requires Confirmation", justify="center")

    for tool_name, tool_config in enabled_tools.items():
        max_tokens = str(tool_config.max_tokens) if tool_config.max_tokens else "N/A"
        confirmation = "✅" if tool_config.require_confirmation else "❌"

        table.add_row(
            tool_name,
            tool_config.description,
            max_tokens,
            confirmation
        )

    console.print(table)


@mcp_app.command("resources")
def list_mcp_resources():
    """List available MCP resources."""
    mcp_config = get_mcp_config()
    enabled_resources = mcp_config.get_enabled_resources()

    if not enabled_resources:
        rich_print("[yellow]No MCP resources are enabled[/yellow]")
        return

    # Create table
    table = Table(title="Available MCP Resources")
    table.add_column("Resource Name", style="cyan", no_wrap=True)
    table.add_column("URI", style="green")
    table.add_column("Description", style="magenta")

    for resource_name, resource_config in enabled_resources.items():
        table.add_row(
            resource_config.name,
            resource_config.uri,
            resource_config.description
        )

    console.print(table)


@mcp_app.command("test")
def test_mcp_connection():
    """Test MCP server functionality."""
    rich_print("[blue]Testing MCP server functionality...[/blue]")

    try:
        # Initialize MCP components
        from ..mcp.tools import MCPToolsHandler
        from ..mcp.resources import MCPResourcesHandler

        tools_handler = MCPToolsHandler()
        resources_handler = MCPResourcesHandler()

        # Test tools
        available_tools = tools_handler.get_available_tools()
        rich_print(f"[green]✓[/green] Found {len(available_tools)} available tools")

        # Test resources
        available_resources = resources_handler.get_available_resources()
        rich_print(f"[green]✓[/green] Found {len(available_resources)} available resources")

        # Test configuration
        mcp_config = get_mcp_config()
        config = mcp_config.load_config()
        rich_print(f"[green]✓[/green] MCP server enabled: {config.server.enabled}")
        rich_print(f"[green]✓[/green] Transport: {config.server.transport}")

        rich_print("[green]MCP server test completed successfully![/green]")

    except Exception as e:
        rich_print(f"[red]MCP test failed: {e}[/red]")
        raise typer.Exit(1)


@mcp_app.command("config")
def show_mcp_config():
    """Show MCP configuration."""
    mcp_config = get_mcp_config()
    config = mcp_config.load_config()

    rich_print("[bold cyan]MCP Configuration[/bold cyan]")
    rich_print(f"Server Enabled: {config.server.enabled}")
    rich_print(f"Server Name: {config.server.name}")
    rich_print(f"Transport: {config.server.transport}")

    if config.server.transport == "http":
        rich_print(f"HTTP Host: {config.server.http.host}")
        rich_print(f"HTTP Port: {config.server.http.port}")

    rich_print(f"\nEnabled Tools: {len(config.server.tools)}")
    for tool_name in config.server.tools:
        status = "enabled" if config.server.tools[tool_name].enabled else "disabled"
        rich_print(f"  - {tool_name}: {status}")

    rich_print(f"\nEnabled Resources: {len(config.server.resources)}")
    for resource_name in config.server.resources:
        status = "enabled" if config.server.resources[resource_name].enabled else "disabled"
        rich_print(f"  - {resource_name}: {status}")


def _start_mcp_server(transport: Optional[str], host: Optional[str], port: Optional[int], verbose: bool):
    """Start the MCP server."""
    # Setup logging
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    mcp_config = get_mcp_config()
    config = mcp_config.load_config()

    if not config.server.enabled:
        rich_print("[red]MCP server is disabled in configuration[/red]")
        rich_print("Enable it in etc/mcp.yaml")
        raise typer.Exit(1)

    transport = transport or config.server.transport

    try:
        rich_print(f"[blue]Starting MCP server with {transport} transport...[/blue]")

        # Create and start MCP server
        mcp_server = MCPServer()

        if transport == "stdio":
            rich_print("[green]MCP server started with stdio transport[/green]")
            rich_print("[dim]Server is running... Use Ctrl+C to stop[/dim]")
            asyncio.run(mcp_server.run_stdio())
        elif transport == "http":
            host = host or config.server.http.host
            port = port or config.server.http.port
            rich_print(f"[green]Starting MCP server on http://{host}:{port}[/green]")
            asyncio.run(mcp_server.run_http(host, port))
        else:
            rich_print(f"[red]Unknown transport: {transport}[/red]")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        rich_print("\n[yellow]MCP server stopped by user[/yellow]")
    except Exception as e:
        rich_print(f"[red]Failed to start MCP server: {e}[/red]")
        raise typer.Exit(1)


def _mcp_server_status():
    """Show MCP server status."""
    mcp_config = get_mcp_config()
    config = mcp_config.load_config()

    rich_print("[bold cyan]MCP Server Status[/bold cyan]")
    rich_print(f"Enabled: {config.server.enabled}")

    if not config.server.enabled:
        rich_print("[yellow]MCP server is disabled[/yellow]")
        return

    rich_print(f"Transport: {config.server.transport}")
    rich_print(f"Tools: {len(mcp_config.get_enabled_tools())} enabled")
    rich_print(f"Resources: {len(mcp_config.get_enabled_resources())} enabled")

    # Check if server process is running (simplified check)
    rich_print("[yellow]Note: Server status checking not fully implemented[/yellow]")


if __name__ == "__main__":
    app()