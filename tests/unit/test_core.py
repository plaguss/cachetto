import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from dfcache.core import _create_cache_key, _get_func_name, _make_hashable, dfcache


# Función global
def sample_function():
    pass


# Clase de ejemplo
class ExampleClass:
    def method(self):
        pass

    @classmethod
    def class_method(cls):
        pass

    @staticmethod
    def static_method():
        pass


class TestGetFuncName:
    def test_global_function(self):
        assert (
            _get_func_name(sample_function)
            == f"{sample_function.__module__}_sample_function"
        )

    def test_class_method(self):
        assert (
            _get_func_name(ExampleClass.class_method)
            == f"{ExampleClass.class_method.__module__}_ExampleClass_class_method"
        )

    def test_instance_method(self):
        assert (
            _get_func_name(ExampleClass().method)
            == f"{ExampleClass.method.__module__}_ExampleClass_method"
        )

    def test_static_method(self):
        assert (
            _get_func_name(ExampleClass.static_method)
            == f"{ExampleClass.static_method.__module__}_ExampleClass_static_method"
        )

    def test_nested_function(self):
        def outer():
            def inner():
                pass

            return inner

        inner_func = outer()
        expected = "test_core_TestGetFuncName_test_nested_function_<locals>_outer_<locals>_inner"
        assert _get_func_name(inner_func) == expected

    def test_lambda_function(self):
        f = lambda x: x  # noqa: E731
        expected = "test_core_TestGetFuncName_test_lambda_function_<locals>_<lambda>"
        assert _get_func_name(f) == expected


class TestMakeHashable:
    @pytest.mark.parametrize(
        "input_obj,expected",
        [
            (42, 42),
            ("hi", "hi"),
            (3.14, 3.14),
            (True, True),
            (None, None),
            ([1, 2, 3], (1, 2, 3)),
            ((4, 5, 6), (4, 5, 6)),
            ({3, 1, 2}, (1, 2, 3)),
            ({"x": 10, "y": [1, 2]}, (("x", 10), ("y", (1, 2)))),
            (
                {"a": [1, {2, 3}], "b": (None, 5)},
                (("a", (1, (2, 3))), ("b", (None, 5))),
            ),
        ],
    )
    def test_make_hashable_parametrized(self, input_obj: Any, expected: Any) -> None:
        result = _make_hashable(input_obj)
        assert result == expected

    def test_make_hashable_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _make_hashable(df)
        assert isinstance(result, dict)
        assert result["type"] == "DataFrame"
        assert result["shape"] == (2, 2)
        assert result["columns"] == ("a", "b")
        assert result["dtypes"] == (df.dtypes["a"], df.dtypes["b"])
        assert isinstance(result["hash"], str)
        assert len(result["hash"]) == 32

    @pytest.mark.parametrize(
        "df_data1,df_data2,should_be_equal",
        [
            ({"a": [1, 2]}, {"a": [1, 2]}, True),
            ({"a": [1, 2]}, {"a": [2, 1]}, False),
        ],
    )
    def test_make_hashable_dataframe_hash(
        self, df_data1: pd.DataFrame, df_data2: pd.DataFrame, should_be_equal: bool
    ) -> None:
        df1 = pd.DataFrame(df_data1)
        df2 = pd.DataFrame(df_data2)
        hash1 = _make_hashable(df1)["hash"]
        hash2 = _make_hashable(df2)["hash"]
        if should_be_equal:
            assert hash1 == hash2
        else:
            assert hash1 != hash2


class TestCreateCacheKey:
    def test_simple_function_args(self) -> None:
        def foo(a, b):
            return a + b

        key = _create_cache_key(foo, (1, 2), {})

        # Verify the hash for the string
        expected_data = {
            "function": f"{foo.__module__}.{foo.__qualname__}",
            "args": (("a", 1), ("b", 2)),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_function_with_kwargs(self) -> None:
        def bar(x, y=10):
            return x * y

        key = _create_cache_key(bar, (5,), {"y": 20})

        expected_data = {
            "function": f"{bar.__module__}.{bar.__qualname__}",
            "args": (("x", 5), ("y", 20)),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_method_self_ignored(self) -> None:
        class Baz:
            def method(self, x):
                return x

        baz = Baz()

        key = _create_cache_key(Baz.method, (baz, 42), {})

        expected_data = {
            "function": f"{Baz.method.__module__}.{Baz.method.__qualname__}",
            "args": (("x", 42),),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_complex_args(self, monkeypatch) -> None:
        def func(a, b):
            return a + b

        # Simulate _make_hashable for lists and dicts
        monkeypatch.setattr("dfcache.core._make_hashable", lambda x: str(x))

        key = _create_cache_key(func, ([1, 2, 3], {"foo": "bar"}), {})

        expected_data = {
            "function": f"{func.__module__}.{func.__qualname__}",
            "args": str({"a": [1, 2, 3], "b": {"foo": "bar"}}),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_func_no_args(self) -> None:
        def foo():
            return 42

        key = _create_cache_key(foo, (), {})
        expected_data = {
            "function": f"{foo.__module__}.{foo.__qualname__}",
            "args": (),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_func_only_kwargs(self) -> None:
        def foo(a=1, b=2):
            return a + b

        key = _create_cache_key(foo, (), {"a": 3, "b": 4})
        expected_data = {
            "function": f"{foo.__module__}.{foo.__qualname__}",
            "args": (("a", 3), ("b", 4)),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_func_args_and_kwargs(self) -> None:
        def foo(a, b=2):
            return a + b

        key = _create_cache_key(foo, (5,), {"b": 7})
        expected_data = {
            "function": f"{foo.__module__}.{foo.__qualname__}",
            "args": (("a", 5), ("b", 7)),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_kwargs_order_independence(self) -> None:
        def foo(a, b):
            return a + b

        key1 = _create_cache_key(foo, (), {"a": 1, "b": 2})
        key2 = _create_cache_key(foo, (), {"b": 2, "a": 1})
        assert key1 == key2

    def test_mutable_args(self) -> None:
        def foo(a):
            return sum(a)

        key = _create_cache_key(foo, ([1, 2, 3],), {})
        assert key == "39cbee696452b6105968b2c6995019dc"

    def test_empty_args_kwargs(self) -> None:
        def foo():
            return 1

        key = _create_cache_key(foo, (), {})
        expected_data = {
            "function": f"{foo.__module__}.{foo.__qualname__}",
            "args": (),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_nonstandard_types(self) -> None:
        def foo(a):
            return a

        class Custom:
            def __repr__(self):
                return "CustomInstance"

        obj = Custom()
        key = _create_cache_key(foo, (obj,), {})
        expected_data = {
            "function": f"{foo.__module__}.{foo.__qualname__}",
            "args": (("a", obj),),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash

    def test_args_with_none(self) -> None:
        def foo(a, b=None):
            return a if b is None else b

        key = _create_cache_key(foo, (1,), {})
        expected_data = {
            "function": f"{foo.__module__}.{foo.__qualname__}",
            "args": (("a", 1), ("b", None)),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash


class TestDfcache:
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for caching tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1.1, 2.2, 3.3]}
        )

    @pytest.mark.xfail(
        reason=(
            "This test is failing as the default cache dir already "
            "contains the cached data, the cache_dir should be moved to temp_cache_dir"
        )
    )
    def test_decorator_without_parentheses(
        self, temp_cache_dir, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test @dfcache usage without parentheses."""
        call_count = 0

        @dfcache
        def get_data():
            nonlocal call_count
            call_count += 1
            return sample_dataframe.copy()

        # First call should execute function
        result1 = get_data()
        assert call_count == 1
        pd.testing.assert_frame_equal(result1, sample_dataframe)

        # Second call should use cache (function not called again)
        result2 = get_data()
        assert call_count == 1  # Should still be 1
        pd.testing.assert_frame_equal(result2, sample_dataframe)

    def test_decorator_with_cache_dir(
        self, temp_cache_dir, sample_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """Test @dfcache(cache_dir="path") usage."""
        call_count = 0

        @dfcache(cache_dir=temp_cache_dir)
        def get_data():
            nonlocal call_count
            call_count += 1
            return sample_dataframe.copy()

        # Verify cache directory is set correctly
        assert str(get_data._cache_dir) == temp_cache_dir

        # Test caching behavior
        result1 = get_data()
        assert call_count == 1

        result2 = get_data()
        assert call_count == 1
        pd.testing.assert_frame_equal(result1, result2)

    def test_cache_with_function_args(
        self, temp_cache_dir, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test caching with function arguments."""
        call_count = 0

        @dfcache(cache_dir=temp_cache_dir)
        def get_filtered_data(filter_val):
            nonlocal call_count
            call_count += 1
            return sample_dataframe[sample_dataframe["A"] > filter_val].copy()

        # Different arguments should create different cache entries
        result1 = get_filtered_data(1)
        assert call_count == 1

        get_filtered_data(2)
        assert call_count == 2  # Different argument, new call

        # Same argument should use cache
        result3 = get_filtered_data(1)
        assert call_count == 2  # Should still be 2

        pd.testing.assert_frame_equal(result1, result3)

    def test_cache_with_kwargs(
        self, temp_cache_dir, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test caching with keyword arguments."""
        call_count = 0

        @dfcache(cache_dir=temp_cache_dir)
        def get_data(columns=None, multiplier=1):
            nonlocal call_count
            call_count += 1
            df = sample_dataframe.copy()
            if columns:
                df = df[columns]
            df = df * multiplier
            return df

        # Different kwargs should create different cache entries
        result1 = get_data(columns=["A"], multiplier=2)
        assert call_count == 1

        get_data(columns=["B"], multiplier=2)
        assert call_count == 2

        # Same kwargs should use cache
        result3 = get_data(columns=["A"], multiplier=2)
        assert call_count == 2

        pd.testing.assert_frame_equal(result1, result3)

    def test_non_dataframe_return_not_cached(self, temp_cache_dir) -> None:
        call_count = 0

        @dfcache(cache_dir=temp_cache_dir)
        def get_string():
            nonlocal call_count
            call_count += 1
            return "not a dataframe"

        result1 = get_string()
        result2 = get_string()

        # Function should be called both times since result isn't cached
        assert call_count == 2
        assert result1 == "not a dataframe"
        assert result2 == "not a dataframe"

        # Verify no cache files were created
        cache_dir = Path(temp_cache_dir)
        cache_files = list(cache_dir.glob("*.parquet"))
        assert len(cache_files) == 0

    def test_clear_cache_method(
        self, temp_cache_dir, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test the clear_cache method."""
        call_count = 0

        @dfcache(cache_dir=temp_cache_dir)
        def get_data():
            nonlocal call_count
            call_count += 1
            return sample_dataframe.copy()

        # Create cache
        get_data()
        assert call_count == 1

        # Verify cache exists
        cache_dir = Path(temp_cache_dir)
        cache_files = list(cache_dir.glob("*.parquet"))
        assert len(cache_files) > 0

        # Clear cache
        get_data.clear_cache()

        # Verify cache is cleared
        cache_files = list(cache_dir.glob("*.parquet"))
        assert len(cache_files) == 0

        # Next call should execute function again
        get_data()
        assert call_count == 2

    def test_caching_failure_handling(
        self, temp_cache_dir, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test handling of caching failures."""

        @dfcache(cache_dir=temp_cache_dir)
        def get_data():
            return sample_dataframe.copy()

        # Mock to_parquet to raise an exception
        with patch.object(
            pd.DataFrame, "to_parquet", side_effect=Exception("Caching failed")
        ):
            with patch("builtins.print") as mock_print:
                result = get_data()
                mock_print.assert_called_once()  # Should print error message

        # Should still return the DataFrame despite caching failure
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directory is created if it doesn't exist."""
        cache_path = Path(temp_cache_dir) / "nested" / "cache"

        @dfcache(cache_dir=str(cache_path))
        def get_data():
            return pd.DataFrame({"A": [1, 2, 3]})

        # Directory should be created when decorator is applied
        assert cache_path.exists()
        assert cache_path.is_dir()

    def test_method_decoration(
        self, temp_cache_dir, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test decorator works with class methods."""

        class DataProcessor:
            def __init__(self):
                self.call_count = 0

            @dfcache(cache_dir=temp_cache_dir)
            def process_data(self, multiplier=1):
                self.call_count += 1
                return sample_dataframe * multiplier

        processor = DataProcessor()

        # First call
        result1 = processor.process_data(2)
        assert processor.call_count == 1

        # Second call should use cache
        result2 = processor.process_data(2)
        assert processor.call_count == 1

        pd.testing.assert_frame_equal(result1, result2)

    def test_function_wrapping_preservation(self, temp_cache_dir):
        """Test that function metadata is preserved."""

        @dfcache(cache_dir=temp_cache_dir)
        def documented_function():
            """This function has documentation."""
            return pd.DataFrame({"A": [1, 2, 3]})

        # Function name and docstring should be preserved
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function has documentation."

    def test_default_cache_dir(self, sample_dataframe):
        """Test using default cache directory."""

        @dfcache
        def get_data():
            return sample_dataframe.copy()

        from dfcache.core import DEFAULT_CACHE_DIR

        # Should use DEFAULT_CACHE_DIR
        assert get_data._cache_dir == DEFAULT_CACHE_DIR

        # Clean up
        get_data.clear_cache()

    def test_cache_key_generation_consistency(self, temp_cache_dir, sample_dataframe):
        """Test that cache keys are generated consistently."""

        @dfcache(cache_dir=temp_cache_dir)
        def get_data(a, b=None):
            return sample_dataframe.copy()

        # Same arguments should produce same cache key
        get_data(1, b=2)
        cache_files1 = list(Path(temp_cache_dir).glob("*.parquet"))

        get_data.clear_cache()

        get_data(1, b=2)
        cache_files2 = list(Path(temp_cache_dir).glob("*.parquet"))

        # Should have same cache file names
        assert [f.name for f in cache_files1] == [f.name for f in cache_files2]

    # Edgy cases

    def test_empty_dataframe_caching(self, temp_cache_dir):
        """Test caching of empty DataFrames."""

        @dfcache(cache_dir=temp_cache_dir)
        def get_empty_df():
            return pd.DataFrame()

        result1 = get_empty_df()
        result2 = get_empty_df()

        pd.testing.assert_frame_equal(result1, result2)
        assert len(result1) == 0

    def test_large_dataframe_args(self, temp_cache_dir):
        """Test with large arguments that might affect cache key generation."""
        large_list = list(range(1000))

        @dfcache(cache_dir=temp_cache_dir)
        def process_large_args(data):
            return pd.DataFrame({"sum": [sum(data)]})

        result1 = process_large_args(large_list)
        result2 = process_large_args(large_list)

        pd.testing.assert_frame_equal(result1, result2)

    def test_special_characters_in_args(self, temp_cache_dir):
        """Test with special characters in arguments."""

        @dfcache(cache_dir=temp_cache_dir)
        def process_text(text):
            return pd.DataFrame({"text": [text], "length": [len(text)]})

        special_text = "Hello, 世界! @#$%^&*()"
        result1 = process_text(special_text)
        result2 = process_text(special_text)

        pd.testing.assert_frame_equal(result1, result2)
