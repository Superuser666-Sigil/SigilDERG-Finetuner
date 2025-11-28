"""
Unit tests for JSONL dataset loader.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.8.0
"""

from pathlib import Path

import pytest
from rust_qlora.dataset_utils.jsonl_loader import load_prompt_gen_jsonl


class TestLoadPromptGenJsonl:
    """Tests for load_prompt_gen_jsonl function."""

    def test_load_valid_jsonl(self, tmp_path: Path, mock_tokenizer):
        """Test loading valid JSONL file."""
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(
            '{"prompt": "fn add(a: i32, b: i32) -> i32 {", "gen": "    a + b\\n}\\n"}\n'
            '{"prompt": "fn sub(a: i32, b: i32) -> i32 {", "gen": "    a - b\\n}\\n"}\n'
        )

        generator = load_prompt_gen_jsonl(
            jsonl_path=str(jsonl_path),
            tokenizer=mock_tokenizer,
        )

        items = list(generator)
        assert len(items) == 2

    def test_load_empty_file(self, tmp_path: Path, mock_tokenizer):
        """Test loading empty JSONL file."""
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        generator = load_prompt_gen_jsonl(
            jsonl_path=str(jsonl_path),
            tokenizer=mock_tokenizer,
        )

        items = list(generator)
        assert len(items) == 0

    def test_load_with_metadata(self, tmp_path: Path, mock_tokenizer):
        """Test loading JSONL with metadata fields."""
        jsonl_path = tmp_path / "meta.jsonl"
        jsonl_path.write_text(
            '{"prompt": "fn test() {", "gen": "}", "_source": "test", "_split": "train"}\n'
        )

        generator = load_prompt_gen_jsonl(
            jsonl_path=str(jsonl_path),
            tokenizer=mock_tokenizer,
            remove_metadata=True,
        )

        items = list(generator)
        assert len(items) == 1
        # Metadata should be removed
        assert "_source" not in items[0]
        assert "_split" not in items[0]

    def test_load_with_task_weights(self, tmp_path: Path, mock_tokenizer):
        """Test loading with task weighting."""
        jsonl_path = tmp_path / "weighted.jsonl"
        jsonl_path.write_text(
            '{"prompt": "fn a() {", "gen": "}", "_task_type": "completion"}\n'
            '{"prompt": "fn b() {", "gen": "}", "_task_type": "infilling"}\n'
        )

        # Weight completion higher
        generator = load_prompt_gen_jsonl(
            jsonl_path=str(jsonl_path),
            tokenizer=mock_tokenizer,
            task_weights={"completion": 2.0, "infilling": 0.5},
        )

        items = list(generator)
        # Should still load both
        assert len(items) >= 1

    def test_file_not_found(self, mock_tokenizer):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            list(
                load_prompt_gen_jsonl(
                    jsonl_path="/nonexistent/path.jsonl",
                    tokenizer=mock_tokenizer,
                )
            )

    def test_invalid_json(self, tmp_path: Path, mock_tokenizer):
        """Test that invalid JSON is handled gracefully."""
        jsonl_path = tmp_path / "invalid.jsonl"
        jsonl_path.write_text(
            '{"prompt": "fn test() {", "gen": "}"}\n'
            "not valid json\n"
            '{"prompt": "fn test2() {", "gen": "}"}\n'
        )

        generator = load_prompt_gen_jsonl(
            jsonl_path=str(jsonl_path),
            tokenizer=mock_tokenizer,
        )

        # Should skip invalid line but continue
        items = list(generator)
        assert len(items) == 2

    def test_missing_required_fields(self, tmp_path: Path, mock_tokenizer):
        """Test that records missing prompt or gen are skipped."""
        jsonl_path = tmp_path / "missing.jsonl"
        jsonl_path.write_text(
            '{"prompt": "fn test() {"}\n'  # Missing gen
            '{"gen": "}"}\n'  # Missing prompt
            '{"prompt": "fn valid() {", "gen": "}"}\n'  # Valid
        )

        generator = load_prompt_gen_jsonl(
            jsonl_path=str(jsonl_path),
            tokenizer=mock_tokenizer,
        )

        items = list(generator)
        assert len(items) == 1
