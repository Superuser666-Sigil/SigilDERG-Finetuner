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
    # Import required crates from eval_template to keep them in sync
    try:
        from eval_template import REQUIRED_CRATES
    except ImportError:
        try:
            from .eval_template import REQUIRED_CRATES
        except ImportError:
            REQUIRED_CRATES = {"anyhow": "1.0", "thiserror": "1.0"}
    
    # Build dependencies section for Cargo.toml - each dependency needs its own echo with continuation
    # Handle crates with features properly
    deps_lines = ""
    deps_list = sorted(REQUIRED_CRATES.items())
    for i, (name, version) in enumerate(deps_list):
        # Format dependency line based on crate type
        if name == "serde":
            dep_line = f'    echo \'    {name} = {{ version = "{version}", features = ["derive"] }}\' >> Cargo.toml && \\'
        elif name == "chrono":
            dep_line = f'    echo \'    {name} = {{ version = "{version}", features = ["serde"] }}\' >> Cargo.toml && \\'
        elif name == "uuid":
            dep_line = f'    echo \'    {name} = {{ version = "{version}", features = ["v4", "serde"] }}\' >> Cargo.toml && \\'
        else:
            dep_line = f'    echo \'    {name} = "{version}"\' >> Cargo.toml && \\'
        
        deps_lines += dep_line + '\n'
    
    dockerfile_content = f"""# Rust evaluation sandbox
# This container provides a minimal, isolated environment for compiling Rust code

FROM rust:1.82-slim

# Install clippy and rustfmt
RUN rustup component add clippy rustfmt

# Create a non-root user for additional security
RUN useradd -m -u 1000 rustuser && \\
    mkdir -p /eval && \\
    chown -R rustuser:rustuser /eval

# Pre-download required dependencies for evaluation (so they work with --network=none)
# This allows the sandboxed container to compile code using these crates without network access
# Must be done as rustuser so dependencies are cached in the correct user's home directory
USER rustuser
RUN mkdir -p /tmp/deps_cache && \\
    cd /tmp/deps_cache && \\
    cargo init --name deps_cache && \\
    sed -i 's/^edition = .*/edition = "2021"/' Cargo.toml && \\
{deps_lines}    echo 'fn main() {{}}' > src/main.rs && \\
    cargo build --release && \\
    rm -rf /tmp/deps_cache

# Set working directory
WORKDIR /eval

# Default command (can be overridden)
CMD ["/bin/bash"]
"""
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)


def run_cargo_in_docker(
    project_path: str,
    command: list[str],
    timeout: int = 30,
    capture_output: bool = True,
    allow_network_fallback: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a cargo command inside a Docker container.
    
    Args:
        project_path: Path to the Rust project directory on host
        command: Cargo command to run (e.g., ["cargo", "check", "-q"])
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
        allow_network_fallback: If True, retry with network access if network error occurs
    
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
    
    # Base Docker options (security restrictions)
    # Project directory is mounted read-only for security (no host filesystem writes)
    # Cargo.lock is pre-generated in the template, and we use --frozen flag to prevent writes
    base_docker_opts = [
        "--rm",  # Remove container after execution
        "--memory=512m",  # Limit memory
        "--cpus=1",  # Limit CPU
        "--read-only",  # Read-only root filesystem
        "--tmpfs", "/tmp:rw,nosuid,size=300m",  # Temporary writable space for build artifacts
        "-v", f"{project_path}:/eval/{project_name}:ro",  # Mount project as read-only (SECURE - no host filesystem writes)
        "-w", f"/eval/{project_name}",  # Working directory
        "-e", "CARGO_TARGET_DIR=/tmp/cargo-target",  # Set cargo to use tmpfs for build artifacts
    ]
    
    # Network-related error patterns
    network_error_patterns = [
        "couldn't resolve host name",
        "could not resolve host",
        "failed to query replaced source registry",
        "failed to download from",
        "network is unreachable",
        "could not fetch",
    ]
    
    # First try with --network=none (most secure)
    docker_cmd = [
        "docker", "run",
    ] + base_docker_opts + [
        "--network=none",  # No network access (preferred for security)
        "rust-eval-sandbox",
    ] + command
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        
        # Check if we got a network error and should retry with network access
        if result.returncode != 0 and capture_output and allow_network_fallback:
            error_output = (result.stderr or "") + (result.stdout or "")
            error_lower = error_output.lower()
            
            # Check if this is a network-related error
            is_network_error = any(pattern in error_lower for pattern in network_error_patterns)
            
            if is_network_error:
                # Retry with network access enabled (still with other security restrictions)
                print("Network error detected, retrying with network access enabled...", file=__import__("sys").stderr)
                docker_cmd_with_network = [
                    "docker", "run",
                ] + base_docker_opts + [
                    # No --network flag = default network access
                    "rust-eval-sandbox",
                ] + command
                
                result = subprocess.run(
                    docker_cmd_with_network,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout
                )
                return result
        
        # Check for other Docker infrastructure errors (not compilation errors)
        if result.returncode != 0 and capture_output:
            error_output = (result.stderr or "") + (result.stdout or "")
            error_lower = error_output.lower()
            
            # Detect Docker daemon/infrastructure errors (excluding network errors we already handled)
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
            
            # Only raise SandboxError for non-network infrastructure errors
            if any(pattern in error_lower for pattern in docker_error_patterns):
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
    sandbox_mode: Optional[str] = None,
    allow_network_fallback: bool = True
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
        return run_cargo_in_docker(project_path, command, timeout, capture_output, allow_network_fallback)
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

