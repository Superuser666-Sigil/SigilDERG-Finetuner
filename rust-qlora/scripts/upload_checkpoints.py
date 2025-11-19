#!/usr/bin/env python3

"""
Upload multiple checkpoints to HuggingFace Hub, preserving each in its own subdirectory.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi


def upload_checkpoint(api, repo_id, checkpoint_path, checkpoint_name):
    """Upload a checkpoint to a subdirectory in the repo."""
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return False

    print(f"📤 Uploading {checkpoint_name} from {checkpoint_path}...")

    try:
        # Upload to subdirectory: checkpoint-{step}/
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(checkpoint_dir),
            path_in_repo=checkpoint_name,  # Creates subdirectory in repo
            commit_message=f"Upload {checkpoint_name} - preserving checkpoint",
        )
        print(f"✅ Successfully uploaded {checkpoint_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload {checkpoint_name}: {e}")
        return False


def discover_checkpoints(base_path: str):
    """Auto-discover checkpoint directories under the base path."""
    base_dir = Path(base_path)
    if not base_dir.exists():
        print(f"❌ Error: Base path not found: {base_path}")
        sys.exit(1)

    checkpoints = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            step_str = entry.name.replace("checkpoint-", "")
            try:
                step_value = int(step_str)
            except ValueError:
                # Non-numeric suffixes go to the end but retain deterministic order
                step_value = float("inf")
            checkpoints.append((step_value, entry.name, str(entry)))

    if not checkpoints:
        print(f"⚠️  No checkpoints found under {base_path}")
        sys.exit(1)

    checkpoints.sort(key=lambda item: item[0])
    return [(name, path) for _, name, path in checkpoints]


def main():
    repo_id = "Superuser666-Sigil/Llama-3.1-8B-Instruct-Rust-QLora"
    base_path = "out/llama8b-rust-qlora-phase1"

    # Get token from environment
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ Error: HF_TOKEN environment variable not set")
        print("   Set it with: export HF_TOKEN='your_token_here'")
        sys.exit(1)

    # Initialize API
    api = HfApi(token=token)

    checkpoints = discover_checkpoints(base_path)

    print(f"🚀 Uploading checkpoints to {repo_id}")
    print(f"   Token: {'*' * 20}...{token[-4:] if len(token) > 4 else '****'}")
    print(f"   Found {len(checkpoints)} checkpoint(s) under {base_path}")
    print()

    results = []
    for checkpoint_name, checkpoint_path in checkpoints:
        success = upload_checkpoint(api, repo_id, checkpoint_path, checkpoint_name)
        results.append((checkpoint_name, success))
        print()

    # Summary
    print("=" * 60)
    print("Upload Summary:")
    print("=" * 60)
    for checkpoint_name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {checkpoint_name}: {status}")

    # Check if all succeeded
    all_success = all(success for _, success in results)
    if all_success:
        print(f"\n🎉 All checkpoints uploaded successfully!")
        print(f"   View at: https://huggingface.co/{repo_id}/tree/main")
    else:
        print(f"\n⚠️  Some uploads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

