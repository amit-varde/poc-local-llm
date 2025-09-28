"""Server management utilities for the CLI."""

import subprocess
import signal
import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import psutil

from ..config import get_config


class ServerManager:
    """Manages the API server lifecycle."""

    def __init__(self):
        self.config = get_config()
        self.settings = self.config.load_settings()
        self.pid_file = Path("./logs/server.pid")

    def start(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        reload: bool = False,
        default_model: Optional[str] = None
    ):
        """Start the API server."""
        # Check if server is already running
        if self.is_running():
            print("Server is already running")
            return

        # Use provided values or defaults from config
        host = host or self.settings.server.host
        port = port or self.settings.server.port

        # Ensure logs directory exists
        self.pid_file.parent.mkdir(exist_ok=True)

        # Build uvicorn command
        cmd = [
            "uvicorn",
            "pipeline.api.app:create_app",
            "--factory",
            "--host", host,
            "--port", str(port),
        ]

        if reload:
            cmd.append("--reload")

        # Set environment variables
        env = os.environ.copy()
        if default_model:
            env["DEFAULT_MODEL"] = default_model

        print(f"Starting server on {host}:{port}")

        try:
            # Start server process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))

            print(f"Server started with PID {process.pid}")
            print(f"API available at: http://{host}:{port}")
            print(f"Documentation: http://{host}:{port}/docs")

            # Wait a moment and check if server started successfully
            time.sleep(2)
            if process.poll() is not None:
                # Process has terminated
                stdout, stderr = process.communicate()
                print(f"Server failed to start: {stderr.decode()}")
                if self.pid_file.exists():
                    self.pid_file.unlink()
            else:
                print("Server is running successfully")

        except Exception as e:
            print(f"Failed to start server: {e}")
            if self.pid_file.exists():
                self.pid_file.unlink()

    def stop(self):
        """Stop the API server."""
        if not self.is_running():
            print("Server is not running")
            return

        try:
            # Read PID
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Check if process exists
            if psutil.pid_exists(pid):
                # Try graceful shutdown first
                os.killpg(os.getpgid(pid), signal.SIGTERM)

                # Wait for process to terminate
                for _ in range(10):  # Wait up to 10 seconds
                    if not psutil.pid_exists(pid):
                        break
                    time.sleep(1)

                # Force kill if still running
                if psutil.pid_exists(pid):
                    os.killpg(os.getpgid(pid), signal.SIGKILL)

                print(f"Server with PID {pid} stopped")
            else:
                print("Server process not found")

            # Remove PID file
            if self.pid_file.exists():
                self.pid_file.unlink()

        except Exception as e:
            print(f"Error stopping server: {e}")
            # Try to remove stale PID file
            if self.pid_file.exists():
                self.pid_file.unlink()

    def is_running(self) -> bool:
        """Check if the server is running."""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            return psutil.pid_exists(pid)
        except (ValueError, FileNotFoundError):
            return False

    def status(self) -> Dict[str, Any]:
        """Get server status."""
        running = self.is_running()
        status = {"running": running}

        if running:
            host = self.settings.server.host
            port = self.settings.server.port
            url = f"http://{host}:{port}"
            status["url"] = url

            # Try to get additional info from health endpoint
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    status["healthy"] = True
                    status["version"] = health_data.get("version")

                    model_info = health_data.get("model", {})
                    if model_info and model_info.get("current_model"):
                        status["model"] = model_info["current_model"]
                else:
                    status["healthy"] = False
            except requests.RequestException:
                status["healthy"] = False

        return status

    def restart(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        reload: bool = False,
        default_model: Optional[str] = None
    ):
        """Restart the server."""
        print("Restarting server...")
        self.stop()
        time.sleep(1)
        self.start(host, port, reload, default_model)