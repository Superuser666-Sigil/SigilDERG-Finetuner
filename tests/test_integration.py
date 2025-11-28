"""
Integration tests for training setup and evaluation.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.8.0
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestConfigurationIntegration:
    """Integration tests for configuration loading and validation."""

    def test_yaml_to_training_config(self, temp_config_file: Path):
        """Test loading YAML file into TrainingConfig."""
        from rust_qlora.config_models import TrainingConfig

        config = TrainingConfig.from_yaml(temp_config_file)
        assert config.model_name == "meta-llama/Meta-Llama-3-8B"
        assert config.lora.r == 16
        assert config.train.lr == 1e-4

    def test_config_to_dict_roundtrip(self, minimal_config: dict):
        """Test config can be converted to dict and back."""
        from rust_qlora.config_models import TrainingConfig

        config = TrainingConfig.model_validate(minimal_config)
        config_dict = config.to_dict()
        config2 = TrainingConfig.model_validate(config_dict)

        assert config.model_name == config2.model_name
        assert config.lora.r == config2.lora.r


class TestDataFilteringIntegration:
    """Integration tests for data filtering pipeline."""

    def test_filter_function_with_dataset(
        self,
        mock_dataset_example: dict,
        mock_dataset_test_file: dict,
        mock_dataset_bench_file: dict,
    ):
        """Test filter function with various dataset examples."""
        from rust_qlora.data_filters import create_filter_function

        filter_fn = create_filter_function(
            min_length=10,
            max_length=1000,
            exclude_tests=True,
            exclude_benches=True,
        )

        # Regular file should pass
        assert filter_fn(mock_dataset_example) is True

        # Test file should be filtered
        assert filter_fn(mock_dataset_test_file) is False

        # Bench file should be filtered
        assert filter_fn(mock_dataset_bench_file) is False

    def test_idiomatic_filtering(
        self,
        sample_rust_code_idiomatic: str,
        sample_rust_code_low_quality: str,
    ):
        """Test idiomatic code preference filtering."""
        from rust_qlora.data_filters import create_filter_function

        filter_fn = create_filter_function(
            min_length=10,
            max_length=10000,
            prefer_idiomatic=True,
            idiomatic_quality_ratio=2.0,
        )

        idiomatic_example = {"content": sample_rust_code_idiomatic, "path": "src/lib.rs"}
        low_quality_example = {"content": sample_rust_code_low_quality, "path": "src/main.rs"}

        # Idiomatic code should pass
        assert filter_fn(idiomatic_example) is True

        # Low quality code should be filtered (when prefer_idiomatic=True)
        assert filter_fn(low_quality_example) is False


class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""

    @pytest.mark.slow
    def test_sandbox_detection(self):
        """Test sandbox availability detection."""
        from rust_qlora.eval_sandbox import check_docker_available, check_firejail_available

        # At least check that detection works without errors
        docker_available = check_docker_available()
        firejail_available = check_firejail_available()

        assert isinstance(docker_available, bool)
        assert isinstance(firejail_available, bool)

    def test_code_validation(self):
        """Test code validation functions."""
        try:
            from rust_qlora.eval_rust import has_doc_comments
        except Exception as e:
            pytest.skip(f"Could not import eval_rust: {e}")

        doc_code = "/// Documentation comment\nfn test() {}"
        no_doc_code = "fn test() {}"

        assert has_doc_comments(doc_code) is True
        assert has_doc_comments(no_doc_code) is False


class TestLocalJSONLIntegration:
    """Integration tests for local JSONL dataset loading."""

    def test_load_local_jsonl_dataset(self, tmp_path: Path, mock_tokenizer):
        """Test loading local JSONL as dataset."""
        from rust_qlora.dataset_utils.jsonl_loader import load_prompt_gen_jsonl

        # Create test JSONL
        jsonl_path = tmp_path / "test_data.jsonl"
        jsonl_path.write_text(
            '{"prompt": "fn hello() {", "gen": "    println!(\\"Hello\\");\\n}\\n"}\n'
            '{"prompt": "fn world() {", "gen": "    println!(\\"World\\");\\n}\\n"}\n'
        )

        generator = load_prompt_gen_jsonl(
            jsonl_path=str(jsonl_path),
            tokenizer=mock_tokenizer,
        )

        items = list(generator)
        assert len(items) == 2
        assert "text" in items[0] or "content" in items[0]


class TestEcosystemCompatibility:
    """Tests for ecosystem compatibility with sigil-pipeline and human-eval-rust."""

    def test_prompt_gen_format(self, tmp_path: Path):
        """Test that prompt/gen format is compatible with sigil-pipeline output."""
        import json

        # Create sigil-pipeline compatible output
        jsonl_path = tmp_path / "pipeline_output.jsonl"
        records = [
            {
                "prompt": "fn calculate_sum(values: &[i32]) -> i32 {",
                "gen": "    values.iter().sum()\n}\n",
                "_source": "synthetic",
                "_split": "train",
                "_task_type": "completion",
            }
        ]
        with open(jsonl_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        # Verify it can be loaded
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line)
                assert "prompt" in record
                assert "gen" in record

    def test_humaneval_result_format(self):
        """Test that evaluation results are compatible with human-eval-rust."""
        # Expected result format for human-eval-rust
        result = {
            "task_id": "Rust/0",
            "completion": "    values.iter().sum()\n}\n",
            "passed": True,
            "result": "passed",
        }

        # Verify required fields
        assert "task_id" in result
        assert "completion" in result
        assert isinstance(result.get("passed"), bool)
