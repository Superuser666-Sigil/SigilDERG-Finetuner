"""
Unit tests for Pydantic configuration models.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.8.0
"""

from pathlib import Path

import pytest
from pydantic import ValidationError
from rust_qlora.config_models import (
    BNB4BitConfig,
    DatasetConfig,
    LoRAConfig,
    MiscConfig,
    TrainConfig,
    TrainingConfig,
)


class TestDatasetConfig:
    """Tests for DatasetConfig validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DatasetConfig()
        assert config.names == ["ammarnasr/the-stack-rust-clean"]
        assert config.use_cache is True
        assert config.min_length == 64
        assert config.max_length == 200_000
        assert config.exclude_tests is True

    def test_valid_custom_values(self):
        """Test valid custom configuration."""
        config = DatasetConfig(
            names=["custom/dataset"],
            min_length=100,
            max_length=50_000,
            shuffle_seed=42,
        )
        assert config.names == ["custom/dataset"]
        assert config.min_length == 100
        assert config.shuffle_seed == 42

    def test_invalid_min_length_zero(self):
        """Test that min_length=0 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(min_length=0)
        assert "Length must be positive" in str(exc_info.value)

    def test_invalid_min_length_negative(self):
        """Test that negative min_length raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(min_length=-1)
        assert "Length must be positive" in str(exc_info.value)

    def test_max_less_than_min(self):
        """Test that max_length < min_length raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(min_length=1000, max_length=100)
        assert "max_length must be >= min_length" in str(exc_info.value)

    def test_invalid_quality_ratio(self):
        """Test that non-positive quality ratio raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(idiomatic_quality_ratio=0)
        assert "idiomatic_quality_ratio must be positive" in str(exc_info.value)

    def test_task_weights_validation_empty(self):
        """Test that empty task_weights raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(task_weights={})
        assert "task_weights cannot be empty" in str(exc_info.value)

    def test_task_weights_validation_negative(self):
        """Test that negative task weight raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(task_weights={"completion": -0.5})
        assert "must be positive" in str(exc_info.value)


class TestLoRAConfig:
    """Tests for LoRAConfig validation."""

    def test_default_values(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        assert config.r == 16
        assert config.alpha == 16
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules

    def test_custom_values(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(r=32, alpha=64, dropout=0.1)
        assert config.r == 32
        assert config.alpha == 64
        assert config.dropout == 0.1

    def test_invalid_rank_zero(self):
        """Test that r=0 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LoRAConfig(r=0)
        assert "Must be positive" in str(exc_info.value)

    def test_invalid_dropout_negative(self):
        """Test that negative dropout raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoRAConfig(dropout=-0.1)
        assert "Dropout must be between 0.0 and 1.0" in str(exc_info.value)

    def test_invalid_dropout_too_high(self):
        """Test that dropout > 1.0 raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LoRAConfig(dropout=1.5)
        assert "Dropout must be between 0.0 and 1.0" in str(exc_info.value)

    def test_target_modules_semicolon_format(self):
        """Test parsing semicolon-separated target_modules (legacy format)."""
        config = LoRAConfig(target_modules="q_proj;k_proj;v_proj")
        assert config.target_modules == ["q_proj", "k_proj", "v_proj"]

    def test_target_modules_list_format(self):
        """Test list format for target_modules."""
        config = LoRAConfig(target_modules=["q_proj", "k_proj"])
        assert config.target_modules == ["q_proj", "k_proj"]


class TestTrainConfig:
    """Tests for TrainConfig validation."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainConfig()
        assert config.micro_batch_size == 8
        assert config.lr == 1.0e-4
        assert config.bf16 is True
        assert config.grad_checkpointing is True

    def test_invalid_batch_size_zero(self):
        """Test that batch_size=0 raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(micro_batch_size=0)
        assert "Must be positive" in str(exc_info.value)

    def test_invalid_learning_rate_zero(self):
        """Test that lr=0 raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(lr=0)
        assert "Learning rate must be positive" in str(exc_info.value)

    def test_invalid_weight_decay_negative(self):
        """Test that negative weight_decay raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(weight_decay=-0.1)
        assert "Must be non-negative" in str(exc_info.value)


class TestBNB4BitConfig:
    """Tests for BitsAndBytes 4-bit config."""

    def test_default_values(self):
        """Test default BNB4Bit configuration."""
        config = BNB4BitConfig()
        assert config.quant_type == "nf4"
        assert config.compute_dtype == "bfloat16"
        assert config.use_double_quant is True

    def test_fp4_quant_type(self):
        """Test FP4 quantization type."""
        config = BNB4BitConfig(quant_type="fp4")
        assert config.quant_type == "fp4"


class TestMiscConfig:
    """Tests for MiscConfig validation."""

    def test_default_values(self):
        """Test default misc configuration."""
        config = MiscConfig()
        assert config.output_dir == "out/llama8b-rust-qlora-phase1"
        assert config.seed == 42

    def test_logging_dir_default(self):
        """Test that logging_dir defaults to output_dir/logs."""
        config = MiscConfig(output_dir="out/test")
        assert config.logging_dir == "out/test/logs"

    def test_invalid_seed_negative(self):
        """Test that negative seed raises error."""
        with pytest.raises(ValidationError) as exc_info:
            MiscConfig(seed=-1)
        assert "Seed must be non-negative" in str(exc_info.value)


class TestTrainingConfig:
    """Tests for complete TrainingConfig."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = TrainingConfig(model_name="meta-llama/Meta-Llama-3-8B")
        assert config.model_name == "meta-llama/Meta-Llama-3-8B"
        assert config.max_seq_len == 4096
        assert config.pack is True

    def test_missing_model_name(self):
        """Test that missing model_name raises error."""
        with pytest.raises(ValidationError):
            TrainingConfig()

    def test_empty_model_name(self):
        """Test that empty model_name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(model_name="")
        assert "model_name cannot be empty" in str(exc_info.value)

    def test_whitespace_model_name(self):
        """Test that whitespace-only model_name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(model_name="   ")
        assert "model_name cannot be empty" in str(exc_info.value)

    def test_from_yaml(self, tmp_path: Path):
        """Test loading configuration from YAML file."""
        import yaml

        config_data = {
            "model_name": "meta-llama/Meta-Llama-3-8B",
            "max_seq_len": 2048,
            "lora": {"r": 32, "alpha": 64},
        }
        config_path = tmp_path / "config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = TrainingConfig.from_yaml(config_path)
        assert config.model_name == "meta-llama/Meta-Llama-3-8B"
        assert config.max_seq_len == 2048
        assert config.lora.r == 32

    def test_from_yaml_missing_file(self, tmp_path: Path):
        """Test that missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TrainingConfig.from_yaml(tmp_path / "nonexistent.yml")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig(model_name="test/model")
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test/model"

    def test_get_method(self):
        """Test dictionary-like get access."""
        config = TrainingConfig(model_name="test/model")
        assert config.get("model_name") == "test/model"
        assert config.get("nonexistent", "default") == "default"
