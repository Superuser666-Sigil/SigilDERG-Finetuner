"""
Unit tests for data filtering utilities.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.9.0
"""

from rust_qlora.data_filters import (
    BENCH_PAT,
    EXAMPLE_PAT,
    LOCK_PAT,
    TEST_PAT,
    VENDOR_PAT,
    create_filter_function,
    filter_rust_code,
    is_idiomatic,
)


class TestIsIdiomatic:
    """Tests for is_idiomatic function."""

    def test_empty_code(self):
        """Test that empty code is not idiomatic."""
        assert is_idiomatic("") is False

    def test_result_handling(self):
        """Test that Result handling is idiomatic."""
        code = 'fn foo() -> Result<String, Error> { Ok("test".to_string()) }'
        assert is_idiomatic(code) is True

    def test_option_handling(self):
        """Test that Option handling is idiomatic."""
        code = "fn foo() -> Option<i32> { Some(42) }"
        assert is_idiomatic(code) is True

    def test_method_chains(self):
        """Test that method chaining is idiomatic."""
        code = "let x = values.iter().map(|v| v * 2).and_then(|v| Some(v));"
        assert is_idiomatic(code) is True

    def test_derive_macros(self):
        """Test that derive macros are idiomatic."""
        code = "#[derive(Debug, Clone)] struct Foo { x: i32 }"
        assert is_idiomatic(code) is True

    def test_trait_impl(self):
        """Test that trait implementations are idiomatic."""
        code = "impl Display for Foo { fn fmt(&self) -> Result<(), Error> {} }"
        assert is_idiomatic(code) is True

    def test_public_api(self):
        """Test that public API is idiomatic."""
        code = "pub fn foo() {}"
        assert is_idiomatic(code) is True

    def test_low_quality_todos(self):
        """Test that TODO comments are low quality."""
        code = "// TODO: fix this\nfn foo() {}"
        # Only TODO without other idiomatic patterns
        assert is_idiomatic(code) is False

    def test_low_quality_debug_prints(self):
        """Test that debug prints alone are low quality."""
        code = 'fn main() { println!("debug"); dbg!(x); }'
        assert is_idiomatic(code) is False

    def test_low_quality_unsafe(self):
        """Test that unsafe blocks alone are low quality."""
        code = "fn main() { unsafe { } }"
        assert is_idiomatic(code) is False

    def test_quality_ratio(self):
        """Test quality ratio affects classification."""
        # Code with 1 idiomatic pattern and 1 low-quality marker
        code = 'pub fn foo() { println!("debug"); }'
        # With ratio 2.0, needs 2x more idiomatic than low-quality
        assert is_idiomatic(code, quality_ratio=2.0) is False
        # With ratio 1.0, equal is enough
        assert is_idiomatic(code, quality_ratio=1.0) is True


class TestFilterRustCode:
    """Tests for filter_rust_code function."""

    def test_valid_code(self):
        """Test that valid code passes filter."""
        code = 'fn main() {\n    println!("Hello");\n}\n'
        result = filter_rust_code(code, "src/main.rs", min_length=10, max_length=1000)
        assert result is True

    def test_too_short(self):
        """Test that short code is filtered."""
        code = "fn x() {}"
        result = filter_rust_code(code, "src/main.rs", min_length=100, max_length=1000)
        assert result is False

    def test_too_long(self):
        """Test that long code is filtered."""
        code = "x" * 1001
        result = filter_rust_code(code, "src/main.rs", min_length=10, max_length=1000)
        assert result is False

    def test_vendor_path(self):
        """Test that vendor paths are filtered."""
        code = 'fn main() { println!("test"); }'
        result = filter_rust_code(code, "/vendor/crate/src/lib.rs", min_length=10, max_length=1000)
        assert result is False

    def test_node_modules_path(self):
        """Test that node_modules paths are filtered."""
        code = 'fn main() { println!("test"); }'
        result = filter_rust_code(
            code, "/node_modules/x/src/lib.rs", min_length=10, max_length=1000
        )
        assert result is False

    def test_cargo_lock(self):
        """Test that Cargo.lock is filtered."""
        code = '[[package]]\nname = "test"'
        result = filter_rust_code(code, "Cargo.lock", min_length=10, max_length=1000)
        assert result is False

    def test_exclude_tests(self):
        """Test that test files are filtered when exclude_tests=True."""
        code = 'fn main() { println!("test"); }'
        result = filter_rust_code(
            code, "/tests/test_main.rs", min_length=10, max_length=1000, exclude_tests=True
        )
        assert result is False

    def test_include_tests(self):
        """Test that test files are included when exclude_tests=False."""
        code = 'fn main() { println!("test"); }'
        result = filter_rust_code(
            code, "/tests/test_main.rs", min_length=10, max_length=1000, exclude_tests=False
        )
        assert result is True

    def test_exclude_examples(self):
        """Test that example files are filtered when exclude_examples=True."""
        code = 'fn main() { println!("example"); }'
        result = filter_rust_code(
            code, "/examples/demo.rs", min_length=10, max_length=1000, exclude_examples=True
        )
        assert result is False

    def test_exclude_benches(self):
        """Test that bench files are filtered when exclude_benches=True."""
        code = 'fn main() { println!("bench"); }'
        result = filter_rust_code(
            code, "/benches/bench_main.rs", min_length=10, max_length=1000, exclude_benches=True
        )
        assert result is False

    def test_return_reason_short(self):
        """Test return_reason=True returns reason for short code."""
        code = "x"
        result, reason = filter_rust_code(
            code, "src/main.rs", min_length=100, max_length=1000, return_reason=True
        )
        assert result is False
        assert reason == "too_short"

    def test_return_reason_long(self):
        """Test return_reason=True returns reason for long code."""
        code = "x" * 1001
        result, reason = filter_rust_code(
            code, "src/main.rs", min_length=10, max_length=1000, return_reason=True
        )
        assert result is False
        assert reason == "too_long"


class TestCreateFilterFunction:
    """Tests for create_filter_function factory."""

    def test_creates_callable(self):
        """Test that create_filter_function returns a callable."""
        filter_fn = create_filter_function()
        assert callable(filter_fn)

    def test_filter_fn_accepts_example(self):
        """Test that filter function accepts example dict."""
        filter_fn = create_filter_function(min_length=10, max_length=1000)
        example = {"content": 'fn main() { println!("Hello"); }', "path": "src/main.rs"}
        result = filter_fn(example)
        assert isinstance(result, bool)

    def test_filter_fn_filters_short(self):
        """Test that filter function filters short code."""
        filter_fn = create_filter_function(min_length=1000, max_length=10000)
        example = {"content": "fn x() {}", "path": "src/main.rs"}
        assert filter_fn(example) is False

    def test_filter_fn_passes_valid(self):
        """Test that filter function passes valid code."""
        filter_fn = create_filter_function(min_length=10, max_length=10000)
        example = {"content": 'fn main() {\n    println!("Hello");\n}', "path": "src/main.rs"}
        assert filter_fn(example) is True


class TestPatterns:
    """Tests for regex patterns."""

    def test_vendor_pattern(self):
        """Test vendor directory pattern."""
        assert VENDOR_PAT.search("/target/debug/deps/") is not None
        assert VENDOR_PAT.search("/vendor/crate/src/") is not None
        assert VENDOR_PAT.search("/node_modules/pkg/") is not None
        assert VENDOR_PAT.search("/src/main.rs") is None

    def test_lock_pattern(self):
        """Test lock file pattern."""
        assert LOCK_PAT.search("Cargo.lock") is not None
        assert LOCK_PAT.search("Cargo.toml") is None

    def test_test_pattern(self):
        """Test test directory pattern."""
        assert TEST_PAT.search("/tests/test_main.rs") is not None
        assert TEST_PAT.search("/test/main.rs") is not None
        assert TEST_PAT.search("/src/main.rs") is None

    def test_bench_pattern(self):
        """Test bench directory pattern."""
        assert BENCH_PAT.search("/benches/bench.rs") is not None
        # Note: Pattern is /benches?/ which only matches /benches/ not /bench/
        assert BENCH_PAT.search("/src/main.rs") is None

    def test_example_pattern(self):
        """Test example directory pattern."""
        assert EXAMPLE_PAT.search("/examples/demo.rs") is not None
        assert EXAMPLE_PAT.search("/src/main.rs") is None
