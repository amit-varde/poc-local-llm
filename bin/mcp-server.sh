#!/bin/bash

# Local LLM Pipeline - MCP Server Management Script
# Automatically manages Python virtual environment for MCP operations

set -euo pipefail

# Script directory and project root
BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$BIN_DIR")"
PROJECT_VENV="$PROJECT_ROOT/venv"

# Load environment utilities
bash_env_load=(logging python_env)
for _env in "${bash_env_load[@]}"; do
    _file="$PROJECT_ROOT/libs/envutils/bash/$_env"
    if [[ -r "$_file" ]]; then
        source "$_file"
    else
        printf 'Error: required bash helper not found: %s\n' "$_file" >&2
        exit 1
    fi
done

# Project utility functions
project_get_root() {
    echo "$PROJECT_ROOT"
}

project_get_venv_dir() {
    echo "$PROJECT_VENV"
}

# Find Python executable
if declare -F python_find_interpreter >/dev/null 2>&1; then
    py="$(python_find_interpreter)" || { err "python_find_interpreter failed"; exit 1; }
else
    py="$(command -v python3 || command -v python || true)"
    [[ -n "$py" ]] || { err "No python executable found (python3 or python)"; exit 1; }
fi

# Activate or create virtual environment
venv_activate() {
    if [[ -f "$PROJECT_VENV/bin/activate" ]]; then
        source "$PROJECT_VENV/bin/activate"
        return 0
    fi
    return 1
}

venv_create() {
    info "Creating virtual environment..."
    if python_create_venv 2>/dev/null || "$py" -m venv "$PROJECT_VENV"; then
        venv_activate
        # Install project in development mode
        "$PROJECT_VENV/bin/pip" install -e "$PROJECT_ROOT" >/dev/null 2>&1 || true
    else
        err "Failed to create virtual environment"
        exit 1
    fi
}

# Show help
show_help() {
    cat << EOF
Local LLM Pipeline - MCP Server Management
Usage: $0 [command] [options]

Commands:
  start      Start the MCP server
  stop       Stop the MCP server  
  status     Check MCP server status
  test       Test MCP server functionality
  config     Show MCP configuration
  tools      List available MCP tools
  resources  List available MCP resources
  help       Show this help message

Options:
  --transport, -t TRANSPORT  Transport type: stdio, http
  --host, -h HOST           Host for HTTP transport
  --port, -p PORT           Port for HTTP transport
  --verbose, -v             Enable verbose logging

Examples:
  $0 start                           # Start with default transport
  $0 start --transport http          # Start with HTTP transport
  $0 test                            # Test MCP functionality
  $0 tools                           # List available tools
EOF
}

# Main execution
main() {
    # Parse command
    local cmd="${1:-help}"
    shift || true
    
    # Show help for help command or no command
    if [[ "$cmd" == "help" || "$cmd" == "--help" || "$cmd" == "-h" ]]; then
        show_help
        return 0
    fi
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    [[ -d "$PROJECT_VENV" ]] && venv_activate || venv_create
    
    # Execute MCP command
    exec llm-pipeline mcp "$cmd" "$@"
}

# Run main function
main "$@"