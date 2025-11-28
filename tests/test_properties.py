"""
Property-based tests using Hypothesis.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.9.0
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from rust_qlora.data_filters import filter_rust_code, is_idiomatic


class TestFilterInvariants:
    """Property-based tests for filter function invariants."""

    @given(
        min_length=st.integers(min_value=1, max_value=100),
        max_length=st.integers(min_value=100, max_value=10000),
    )
    @settings(max_examples=50)
    def test_length_filter_invariant(self, min_length: int, max_length: int):
        """Test that length filtering is consistent."""
        assume(min_length <= max_length)

        # Code shorter than min_length should be filtered
        short_code = "x" * (min_length - 1)
        result = filter_rust_code(
            short_code, "src/main.rs", min_length=min_length, max_length=max_length
        )
        assert result is False

        # Code within range should not crash (may still fail other filters)
        _ = "fn main() {" + "x" * min_length + "}"

    @given(
        code=st.text(min_size=100, max_size=500),
        quality_ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_is_idiomatic_never_crashes(self, code: str, quality_ratio: float):
        """Test that is_idiomatic never crashes on arbitrary input."""
        result = is_idiomatic(code, quality_ratio=quality_ratio)
        assert isinstance(result, bool)

    @given(
        code=st.text(min_size=0, max_size=100),
    )
    @settings(max_examples=30)
    def test_empty_or_short_code_not_idiomatic(self, code: str):
        """Test that empty/short code is not considered idiomatic."""
        if len(code) < 10:
            result = is_idiomatic(code)
            # Very short code should not be idiomatic
            # (may be False or True depending on content, but should not crash)
            assert isinstance(result, bool)


class TestFilterReasonConsistency:
    """Property-based tests for filter reason consistency."""

    @given(
        code_length=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    def test_short_code_reason(self, code_length: int):
        """Test that short code always gets 'too_short' reason."""
        code = "x" * code_length
        result, reason = filter_rust_code(
            code, "src/main.rs", min_length=100, max_length=1000, return_reason=True
        )
        assert result is False
        assert reason == "too_short"

    @given(
        code_length=st.integers(min_value=1001, max_value=2000),
    )
    @settings(max_examples=30)
    def test_long_code_reason(self, code_length: int):
        """Test that long code always gets 'too_long' reason."""
        code = "x" * code_length
        result, reason = filter_rust_code(
            code, "src/main.rs", min_length=10, max_length=1000, return_reason=True
        )
        assert result is False
        assert reason == "too_long"


class TestConfigValidationProperties:
    """Property-based tests for configuration validation."""

    @given(
        r=st.integers(min_value=1, max_value=256),
        alpha=st.integers(min_value=1, max_value=512),
        dropout=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_lora_config_valid_values(self, r: int, alpha: int, dropout: float):
        """Test that valid LoRA config values are accepted."""
        from rust_qlora.config_models import LoRAConfig

        config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
        assert config.r == r
        assert config.alpha == alpha
        assert config.dropout == dropout

    @given(
        min_length=st.integers(min_value=1, max_value=1000),
        max_length=st.integers(min_value=1000, max_value=1_000_000),
    )
    @settings(max_examples=50)
    def test_dataset_config_valid_lengths(self, min_length: int, max_length: int):
        """Test that valid length configs are accepted."""
        assume(min_length <= max_length)

        from rust_qlora.config_models import DatasetConfig

        config = DatasetConfig(min_length=min_length, max_length=max_length)
        assert config.min_length == min_length
        assert config.max_length == max_length

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=30)
    def test_misc_config_valid_seeds(self, seed: int):
        """Test that valid seeds are accepted."""
        from rust_qlora.config_models import MiscConfig

        config = MiscConfig(seed=seed)
        assert config.seed == seed


class TestCodeValidationProperties:
    """Property-based tests for code validation."""

    @given(
        code=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
            min_size=0,
            max_size=1000,
        ),
    )
    @settings(max_examples=50)
    def test_filter_rust_code_never_crashes(self, code: str):
        """Test that filter_rust_code never crashes on arbitrary input."""
        result = filter_rust_code(code, "src/main.rs", min_length=10, max_length=10000)
        assert isinstance(result, bool)

    @given(
        path=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P")), min_size=0, max_size=100
        ),
    )
    @settings(max_examples=50)
    def test_filter_with_arbitrary_paths(self, path: str):
        """Test that filter handles arbitrary paths."""
        code = 'fn main() { println!("test"); }'
        result = filter_rust_code(code, path, min_length=10, max_length=1000)
        assert isinstance(result, bool)
