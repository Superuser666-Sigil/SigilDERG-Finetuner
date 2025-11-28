"""
Shared pytest fixtures for sigilderg-finetuner tests.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.9.0
"""

from pathlib import Path

import pytest

# -----------------------------------------------------------------------------
# Configuration Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def minimal_config() -> dict:
    """Minimal valid configuration for testing."""
    return {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "max_seq_len": 4096,
        "pack": True,
        "dataset": {
            "names": ["ammarnasr/the-stack-rust-clean"],
            "use_cache": True,
            "min_length": 64,
            "max_length": 200_000,
        },
        "lora": {
            "r": 16,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
        },
        "train": {
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
            "lr": 1e-4,
            "num_steps": 10,
        },
        "misc": {
            "output_dir": "out/test",
            "seed": 42,
        },
    }


@pytest.fixture
def invalid_config_no_model() -> dict:
    """Configuration missing required model_name."""
    return {
        "max_seq_len": 4096,
        "pack": True,
    }


@pytest.fixture
def config_with_local_jsonl(tmp_path: Path) -> dict:
    """Configuration using local JSONL dataset."""
    jsonl_path = tmp_path / "test_data.jsonl"
    jsonl_path.write_text(
        '{"prompt": "fn test() {", "gen": "    println!(\\"Hello\\");\\n}\\n"}\n'
        '{"prompt": "fn add(a: i32, b: i32) -> i32 {", "gen": "    a + b\\n}\\n"}\n'
    )
    return {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "max_seq_len": 4096,
        "dataset": {
            "names": [f"local:{jsonl_path}"],
            "use_cache": True,
        },
    }


# -----------------------------------------------------------------------------
# Sample Code Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_rust_code_valid() -> str:
    """Valid Rust code sample."""
    return """
/// Returns the sum of two numbers.
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = add(2, 3);
    println!("Result: {}", result);
}
"""


@pytest.fixture
def sample_rust_code_with_tests() -> str:
    """Rust code with test module."""
    return """
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 2), 4);
    }
}
"""


@pytest.fixture
def sample_rust_code_idiomatic() -> str:
    """Idiomatic Rust code with Result handling."""
    return """
use std::io;

/// Reads a number from stdin.
pub fn read_number() -> Result<i32, io::Error> {
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    input.trim()
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
"""


@pytest.fixture
def sample_rust_code_low_quality() -> str:
    """Low quality Rust code with warnings."""
    return """
// TODO: fix this
fn main() {
    unsafe {
        // HACK: temporary workaround
        #[allow(unused)]
        let x = 42;
        println!("{}", x);
    }
}
"""


# -----------------------------------------------------------------------------
# Dataset Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_dataset_example() -> dict:
    """Single dataset example with content and path."""
    return {
        "content": 'fn main() { println!("Hello, world!"); }',
        "path": "src/main.rs",
    }


@pytest.fixture
def mock_dataset_test_file() -> dict:
    """Dataset example from test directory."""
    return {
        "content": "#[test] fn test_it() { assert!(true); }",
        "path": "/tests/test_main.rs",
    }


@pytest.fixture
def mock_dataset_bench_file() -> dict:
    """Dataset example from bench directory."""
    return {
        "content": "#[bench] fn bench_it(b: &mut Bencher) {}",
        "path": "/benches/bench_main.rs",
    }


# -----------------------------------------------------------------------------
# Temporary Directory Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for training outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Temporary directory for dataset cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_config_file(tmp_path: Path, minimal_config: dict) -> Path:
    """Temporary YAML config file."""
    import yaml

    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(minimal_config, f)
    return config_path


# -----------------------------------------------------------------------------
# Mock Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer(mocker):
    """Mock tokenizer for testing without loading real models."""
    tokenizer = mocker.MagicMock()
    tokenizer.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]],
    }
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.vocab_size = 32000
    return tokenizer


@pytest.fixture
def mock_model(mocker):
    """Mock model for testing without loading real models."""
    model = mocker.MagicMock()
    model.config.vocab_size = 32000
    model.config.hidden_size = 4096
    return model
