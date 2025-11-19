"""
Sandbox wrapper for Rust code evaluation.

This module provides secure isolation for running cargo commands on LLM-generated code.
Uses Docker containers for isolation, with fallback options for local development.
"""

import os
import subprocess
import tempfile
import json
from typing import Dict, Any, Optional
from pathlib import Path


class SandboxError(Exception):
    """Raised when sandbox operations fail."""
    pass


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_firejail_available() -> bool:
    """Check if Firejail is available."""
    try:
        result = subprocess.run(
            ["firejail", "--version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def build_docker_image() -> bool:
    """
    Build the Docker image for Rust evaluation.
    
    Returns:
        True if image was built successfully, False otherwise
    """
    # Get the directory containing this file
    script_dir = Path(__file__).parent
    dockerfile_path = script_dir / "Dockerfile.eval"
    
    if not dockerfile_path.exists():
        # Create Dockerfile if it doesn't exist
        create_dockerfile(dockerfile_path)
    
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "rust-eval-sandbox", "-f", str(dockerfile_path), str(script_dir)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max for build
        )
        if result.returncode != 0:
            print(f"Warning: Docker build failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("Warning: Docker build timed out")
        return False
    except Exception as e:
        print(f"Warning: Docker build error: {e}")
        return False


def create_dockerfile(dockerfile_path: Path):
    """Create the Dockerfile for the evaluation sandbox."""
    dockerfile_content = """# Rust evaluation sandbox
# This container provides a minimal, isolated environment for compiling Rust code

FROM rust:1.75-slim

# Install clippy and rustfmt
RUN rustup component add clippy rustfmt

# Create a non-root user for additional security
RUN useradd -m -u 1000 rustuser && \\
    mkdir -p /eval && \\
    chown -R rustuser:rustuser /eval

# Set working directory
WORKDIR /eval

# Switch to non-root user
USER rustuser

# Default command (can be overridden)
CMD ["/bin/bash"]
"""
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)


def run_cargo_in_docker(
    project_path: str,
    command: list[str],
    timeout: int = 30,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a cargo command inside a Docker container.
    
    Args:
        project_path: Path to the Rust project directory on host
        command: Cargo command to run (e.g., ["cargo", "check", "-q"])
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
    
    Returns:
        CompletedProcess with returncode, stdout, stderr
    """
    # Ensure Docker image exists
    check_result = subprocess.run(
        ["docker", "images", "-q", "rust-eval-sandbox"],
        capture_output=True,
        text=True
    )
    if not check_result.stdout.strip():
        print("Building Docker image for evaluation sandbox...")
        if not build_docker_image():
            raise SandboxError("Failed to build Docker image. Run with --no-sandbox for local dev.")
    
    # Convert host path to absolute
    project_path = os.path.abspath(project_path)
    project_name = os.path.basename(project_path)
    
    # Docker command: mount project directory as read-only, run command, remove container
    # Use /tmp/cargo-target for build artifacts (tmpfs) to avoid conflicts with read-only root
    docker_cmd = [
        "docker", "run",
        "--rm",  # Remove container after execution
        "--network=none",  # No network access
        "--memory=512m",  # Limit memory
        "--cpus=1",  # Limit CPU
        "--read-only",  # Read-only root filesystem
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=300m",  # Temporary writable space (increased for cargo target)
        "-v", f"{project_path}:/eval/{project_name}:ro",  # Mount project as read-only
        "-w", f"/eval/{project_name}",  # Working directory
        "-e", "CARGO_TARGET_DIR=/tmp/cargo-target",  # Set cargo to use tmpfs for build artifacts
        "rust-eval-sandbox",
    ] + command
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        
        # Check for Docker infrastructure errors (not compilation errors)
        if result.returncode != 0 and capture_output:
            error_output = (result.stderr or "") + (result.stdout or "")
            error_lower = error_output.lower()
            
            # Detect Docker daemon/infrastructure errors
            docker_error_patterns = [
                "docker: error response from daemon",
                "failed to create task for container",
                "failed to create shim task",
                "oci runtime create failed",
                "error mounting",
                "read-only file system",
                "cannot create directory",
                "permission denied",
                "no space left on device",
            ]
            
            if any(pattern in error_lower for pattern in docker_error_patterns):
                # This is a Docker infrastructure error, not a compilation error
                raise SandboxError(
                    f"Docker infrastructure error: {error_output[:500]}\n"
                    f"This indicates a problem with the Docker setup, not the code being evaluated.\n"
                    f"Check Docker daemon status, disk space, and permissions."
                )
        
        return result
    except subprocess.TimeoutExpired:
        # Return a CompletedProcess-like object with timeout error
        return subprocess.CompletedProcess(
            docker_cmd, returncode=124, stdout="", stderr="Command timed out"
        )


def run_cargo_with_firejail(
    project_path: str,
    command: list[str],
    timeout: int = 30,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a cargo command using Firejail for sandboxing.
    
    Args:
        project_path: Path to the Rust project directory
        command: Cargo command to run
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
    
    Returns:
        CompletedProcess with returncode, stdout, stderr
    """
    # Firejail command with restrictions
    firejail_cmd = [
        "firejail",
        "--quiet",
        "--net=none",  # No network
        "--private",  # Private filesystem
        "--private-cwd",  # Private working directory
        "--rlimit-as=512000000",  # 512MB memory limit
        "--timeout=30",  # Timeout
        "--cwd", project_path,
    ] + command
    
    try:
        result = subprocess.run(
            firejail_cmd,
            cwd=project_path,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            firejail_cmd, returncode=124, stdout="", stderr="Command timed out"
        )


def run_cargo_sandboxed(
    project_path: str,
    command: list[str],
    timeout: int = 30,
    capture_output: bool = True,
    sandbox_mode: Optional[str] = None
) -> subprocess.CompletedProcess:
    """
    Run a cargo command with sandboxing (Docker preferred, Firejail fallback).
    
    Args:
        project_path: Path to the Rust project directory
        command: Cargo command to run
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
        sandbox_mode: "docker", "firejail", "none", or None (auto-detect)
    
    Returns:
        CompletedProcess with returncode, stdout, stderr
    
    Raises:
        SandboxError: If sandboxing is required but unavailable
    """
    # Auto-detect sandbox mode if not specified
    if sandbox_mode is None:
        if check_docker_available():
            sandbox_mode = "docker"
        elif check_firejail_available():
            sandbox_mode = "firejail"
        else:
            sandbox_mode = "none"
    
    if sandbox_mode == "docker":
        return run_cargo_in_docker(project_path, command, timeout, capture_output)
    elif sandbox_mode == "firejail":
        return run_cargo_with_firejail(project_path, command, timeout, capture_output)
    elif sandbox_mode == "none":
        # No sandboxing - only for local development with trusted code
        return subprocess.run(
            command,
            cwd=project_path,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
    else:
        raise SandboxError(f"Unknown sandbox mode: {sandbox_mode}")

