"""
Pydantic models for training configuration validation.

This module provides type-safe configuration models that catch errors
before expensive training loops start.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    names: List[str] = Field(default_factory=lambda: ["ammarnasr/the-stack-rust-clean"])
    use_cache: bool = True
    min_length: int = 64
    max_length: int = 200_000
    exclude_tests: bool = True
    exclude_examples: bool = False
    exclude_benches: bool = True
    prefer_idiomatic: bool = False
    prefer_documented: bool = False
    shuffle_seed: Optional[int] = None
    interleave_mode: Literal["sequential", "round_robin", "weighted"] = "sequential"
    dataset_weights: Optional[dict[str, float]] = None
    cache_dir: Optional[str] = None
    
    @field_validator("min_length", "max_length")
    @classmethod
    def validate_lengths(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Length must be positive")
        return v
    
    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int, info) -> int:
        if "min_length" in info.data and v < info.data["min_length"]:
            raise ValueError("max_length must be >= min_length")
        return v


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""
    r: int = 16
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj; k_proj; v_proj; o_proj; up_proj; down_proj; gate_proj"]
    )
    
    @field_validator("r", "alpha")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be positive")
        return v
    
    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Dropout must be between 0.0 and 1.0")
        return v


class TrainConfig(BaseModel):
    """Training hyperparameters."""
    micro_batch_size: int = 8
    gradient_accumulation: int = 6
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    num_steps: int = 12000
    warmup_steps: int = 250
    logging_steps: int = 10
    save_every: int = 1000
    bf16: bool = True
    grad_checkpointing: bool = True
    max_grad_norm: float = 1.0
    log_backend: Literal["tensorboard", "wandb", "none"] = "tensorboard"
    lr_scheduler_type: str = "cosine"
    optimizer: str = "paged_adamw_8bit"
    save_total_limit: int = 3
    load_best_model_at_end: bool = False
    
    @field_validator("micro_batch_size", "gradient_accumulation", "num_steps", "warmup_steps", "logging_steps", "save_every")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be positive")
        return v
    
    @field_validator("lr")
    @classmethod
    def validate_lr(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v
    
    @field_validator("weight_decay", "max_grad_norm")
    @classmethod
    def validate_non_negative_float(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Must be non-negative")
        return v


class BNB4BitConfig(BaseModel):
    """BitsAndBytes 4-bit quantization configuration."""
    quant_type: Literal["nf4", "fp4"] = "nf4"
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    use_double_quant: bool = True


class MiscConfig(BaseModel):
    """Miscellaneous configuration."""
    output_dir: str = "out/llama8b-rust-qlora-phase1"
    logging_dir: Optional[str] = None
    seed: int = 42
    load_from: Optional[str] = None
    log_file: Optional[str] = None
    
    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Seed must be non-negative")
        return v
    
    @model_validator(mode='after')
    def set_logging_dir_default(self):
        """Set logging_dir default if not provided."""
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"
        return self


class TrainingConfig(BaseModel):
    """Complete training configuration."""
    model_name: str
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    max_seq_len: int = 4096
    pack: bool = True
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    bnb_4bit: BNB4BitConfig = Field(default_factory=BNB4BitConfig)
    misc: MiscConfig = Field(default_factory=MiscConfig)
    
    @field_validator("max_seq_len")
    @classmethod
    def validate_max_seq_len(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_seq_len must be positive")
        return v
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Validated TrainingConfig instance
        """
        import yaml
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.model_validate(data)
    
    def to_dict(self) -> dict:
        """Convert to dictionary (for backward compatibility)."""
        return self.model_dump(exclude_none=True, mode="json")
    
    def get(self, key: str, default=None):
        """Dictionary-like access for backward compatibility."""
        return getattr(self, key, default)

