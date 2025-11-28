"""
Template project for Rust evaluation to avoid recreating Cargo projects.

This module provides a reusable template that can be copied for each evaluation,
avoiding the overhead of running `cargo new` for every sample.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.8.0
"""

import os
import shutil
import tempfile

_TEMPLATE_DIR = None

# Required crates for evaluation prompts
# Update this list when adding new prompts that require external crates
# Format: "crate_name": "version" or "crate_name": "version" with features in Cargo.toml
REQUIRED_CRATES = {
    "anyhow": "1.0",  # Used in prompt: anyhow error handling
    "thiserror": "1.0",  # Used in prompt: thiserror enum examples
    "serde": "1.0",  # Serialization (with derive feature)
    "serde_json": "1.0",  # JSON serialization
    "regex": "1.10",  # Regular expressions
    "chrono": "0.4",  # Date/time handling
    "uuid": "1.6",  # UUID generation
    "rand": "0.8",  # Random number generation
}


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
        # Check if Cargo.toml has all required dependencies
        cargo_toml = os.path.join(template_path, "Cargo.toml")
        cargo_lock = os.path.join(template_path, "Cargo.lock")
        if os.path.exists(cargo_toml):
            with open(cargo_toml, "r") as f:
                cargo_content = f.read()
                # Check if all required crates are present
                for crate_name in REQUIRED_CRATES.keys():
                    if crate_name not in cargo_content:
                        needs_regeneration = True
                        break
                # Also check if Cargo.lock exists (required for --frozen flag)
                if not os.path.exists(cargo_lock):
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
            capture_output=True,
        )

        # Read existing Cargo.toml and ensure edition is 2021 (not 2024)
        cargo_toml = os.path.join(template_path, "Cargo.toml")
        import re

        with open(cargo_toml, "r") as f:
            cargo_content = f.read()

        # Replace edition line if it exists, or ensure it's set to 2021
        if re.search(r"^edition\s*=", cargo_content, re.MULTILINE):
            cargo_content = re.sub(
                r"^edition\s*=.*", 'edition = "2021"', cargo_content, flags=re.MULTILINE
            )
        else:
            # Insert edition after [package]
            cargo_content = re.sub(r"(\[package\])", r'\1\nedition = "2021"', cargo_content)

        # Append dependencies section if it doesn't exist
        if "[dependencies]" not in cargo_content:
            cargo_content += "\n[dependencies]\n"

        with open(cargo_toml, "w") as f:
            f.write(cargo_content)
            # Add all required crates with appropriate features
            for crate_name, crate_version in sorted(REQUIRED_CRATES.items()):
                if crate_name == "serde":
                    f.write(
                        f'{crate_name} = {{ version = "{crate_version}", features = ["derive"] }}\n'
                    )
                elif crate_name == "chrono":
                    f.write(
                        f'{crate_name} = {{ version = "{crate_version}", features = ["serde"] }}\n'
                    )
                elif crate_name == "uuid":
                    f.write(
                        f'{crate_name} = {{ version = "{crate_version}", features = ["v4", "serde"] }}\n'
                    )
                else:
                    f.write(f'{crate_name} = "{crate_version}"\n')

        # Pre-generate Cargo.lock so we can use --frozen flag (prevents writes to read-only mount)
        # This is safe because Cargo.lock is just dependency metadata, not executable code
        import subprocess

        try:
            subprocess.run(
                ["cargo", "generate-lockfile"],
                cwd=template_path,
                check=True,
                capture_output=True,
                timeout=120,  # 2 minutes max for dependency resolution
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # If cargo is not available or fails, that's okay - we'll generate it on first use
            # The --frozen flag will fail gracefully if Cargo.lock doesn't exist
            pass

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
