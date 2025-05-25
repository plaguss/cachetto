import hashlib
import functools
from pathlib import Path
from typing import Callable, Any
import inspect

import pandas as pd


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "dfcache"


def dfcache(
    func: Callable | None = None,
    *,
    cache_dir: str | None = None,
) -> None:
    """
    A decorator for caching pandas DataFrame results.

    Can be used as:
    - @dfcache (with default settings)
    - @dfcache(cache_dir="path", max_size=100, ignore_args=["arg1"])

    Args:
        func: The function to decorate (used when called without parentheses)
        cache_dir: Directory to store cache files (default: ".cache")

    Works with both functions and class methods.
    """

    def decorator(f: Callable) -> Callable:
        # Create cache directory
        cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache_path.mkdir(parents=True, exist_ok=True)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name, args, and kwargs
            cache_key = _create_cache_key(f, args, kwargs)
            cache_file = cache_path / f"{cache_key}.parquet"

            # Try to load from cache
            if cache_file.exists():
                try:
                    return pd.read_parquet(cache_file)
                except Exception as e:  # TODO: What errors can happen here?
                    # If cache is corrupted, remove it and continue
                    print(f"Unhandled exception while loading from cache:\n{e}")
                    cache_file.unlink(missing_ok=True)

            # Execute function and cache result
            result = f(*args, **kwargs)

            # Only cache if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                try:
                    result.to_parquet()

                except Exception as e:  # TODO: What errors can happen here?
                    # If caching fails, continue without caching
                    print(f"Unhandled exception while caching data:\n{e}")
                    pass

            return result

        # Add cache management methods
        def clear_cache():
            """Clear all cached results for this function."""
            func_prefix = f.__name__
            for cache_file in cache_path.glob(f"*{func_prefix}*.parquet"):
                cache_file.unlink(missing_ok=True)

        wrapper.clear_cache = clear_cache
        wrapper._cache_dir = cache_path

        return wrapper

    # Handle both @dfcache and @dfcache(...) usage
    if func is None:
        # Called with arguments: @dfcache(...)
        return decorator
    else:
        # Called without arguments: @dfcache
        return decorator(func)


def _create_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Create a unique cache key for the function call."""

    # Get function signature to map args to parameter names
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    args = dict(bound_args.arguments.items())

    # Handle 'self' parameter for methods (ignore it for cache key)
    if "self" in args:
        args.pop("self")

    # Create hashable representation
    key_data = {
        "function": f"{func.__module__}.{func.__qualname__}",
        "args": _make_hashable(args),
    }

    # Create hash
    key_str = str(key_data)
    return hashlib.md5(key_str.encode()).hexdigest()


def _make_hashable(obj: Any) -> Any:
    """Convert objects to hashable representation for cache key creation."""
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": obj.shape,
            "columns": tuple(obj.columns.tolist()),
            "dtypes": tuple(obj.dtypes.tolist()),
            "hash": hashlib.md5(
                pd.util.hash_pandas_object(obj).values.tobytes()
            ).hexdigest(),
        }
    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(_make_hashable(item) for item in obj))
    elif obj is None:
        return None
    else:
        return obj


if __name__ == "__main__":
    # Test the decorator with both usage patterns

    @dfcache()
    def load_data(file_id: int, filter_col: str = "default") -> pd.DataFrame:
        """Simulate loading data."""
        print(f"Loading data for file_id={file_id}, filter_col={filter_col}")
        return pd.DataFrame(
            {
                "id": range(file_id * 10),
                "value": range(file_id * 10, file_id * 20),
                "filter": [filter_col] * (file_id * 10),
            }
        )

    @dfcache(cache_dir="custom_cache")
    def process_data(df: pd.DataFrame, multiplier: int = 1) -> pd.DataFrame:
        """Simulate data processing."""
        print(f"Processing data with multiplier={multiplier}")
        result = df.copy()
        result["value"] = result["value"] * multiplier
        return result

    class DataProcessor:
        @dfcache
        def transform(self, data: pd.DataFrame, operation: str = "sum") -> pd.DataFrame:
            """Transform data based on operation."""
            print(f"Transforming data with operation={operation}")
            if operation == "sum":
                return data.groupby("filter").sum().reset_index()
            return data

    # Test usage
    print("Testing dfcache decorator:")

    # First call - will execute and cache
    df1 = load_data(5, "test")
    print(f"Result shape: {df1.shape}")

    # Second call - will load from cache
    df2 = load_data(5, "test")
    print(f"Result shape: {df2.shape}")

    # Test with class method
    processor = DataProcessor()
    result = processor.transform(df1, "sum")
    print(f"Transformed result shape: {result.shape}")
