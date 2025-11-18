"""
Template project for Rust evaluation to avoid recreating Cargo projects.

This module provides a reusable template that can be copied for each evaluation,
avoiding the overhead of running `cargo new` for every sample.
"""

import os
import shutil
import tempfile
from pathlib import Path

_TEMPLATE_DIR = None


def get_template_project():
    """
    Get or create a template Cargo project for evaluation.
    
    Returns:
        Path to template project directory
    """
    global _TEMPLATE_DIR
    
    # Determine template location
    template_base = os.path.join(tempfile.gettempdir(), "rust_eval_template")
    os.makedirs(template_base, exist_ok=True)
    template_path = os.path.join(template_base, "template_app")
    
    # Check if template needs to be created or regenerated
    needs_regeneration = False
    if not os.path.exists(template_path):
        needs_regeneration = True
    else:
        # Check if Cargo.toml has required dependencies
        cargo_toml = os.path.join(template_path, "Cargo.toml")
        if os.path.exists(cargo_toml):
            with open(cargo_toml, "r") as f:
                cargo_content = f.read()
                # Check if it has the required dependencies
                if "anyhow" not in cargo_content or "thiserror" not in cargo_content:
                    needs_regeneration = True
        else:
            needs_regeneration = True
    
    if needs_regeneration:
        # Remove old template if it exists
        if os.path.exists(template_path):
            shutil.rmtree(template_path, ignore_errors=True)
        
        # Create template project
        import subprocess
        subprocess.run(
            ["cargo", "new", "--quiet", "template_app"],
            cwd=template_base,
            check=True,
            capture_output=True
        )
        
        # Write a Cargo.toml with common dependencies for evaluation
        # This includes crates commonly used in Rust code generation tasks
        cargo_toml = os.path.join(template_path, "Cargo.toml")
        with open(cargo_toml, "w") as f:
            f.write("""[package]
name = "app"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1.0"
thiserror = "1.0"
""")
    
    _TEMPLATE_DIR = template_path
    return _TEMPLATE_DIR


def create_eval_project(code: str) -> str:
    """
    Create a temporary evaluation project by copying the template.
    
    Args:
        code: Rust code to write to src/main.rs
    
    Returns:
        Path to the evaluation project directory
    """
    template = get_template_project()
    
    # Create temporary directory for this evaluation
    eval_dir = tempfile.mkdtemp(prefix="rust_eval_")
    project_dir = os.path.join(eval_dir, "app")
    
    # Copy template
    shutil.copytree(template, project_dir)
    
    # Write code
    main_rs = os.path.join(project_dir, "src", "main.rs")
    with open(main_rs, "w", encoding="utf-8") as f:
        f.write(code)
    
    return project_dir

