import datetime as dt
import functools
import hashlib
import inspect
import re
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

T = TypeVar("T")

DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(d|h|m|w|s|)$")


def parse_duration(duration_str: str) -> dt.timedelta:
    """Parse a duration string like '1d', '2h', '30m', '1w' into a timedelta.

    Args:
        duration_str (str): The string representing the units to check.
        Supported units:
        - 'd': days
        - 'h': hours
        - 'm': minutes
        - 'w': weeks
        - 's': seconds

    Returns:
        time (dt.timedelta): Timedelta for the duration.
    """
    if not duration_str:
        raise ValueError("Duration string cannot be empty")

    match = re.match(DURATION_PATTERN, duration_str.lower().strip())

    if not match:
        raise ValueError(
            f"Invalid duration format: '{duration_str}'. Use formats like '1d', '2h', '30m', '1w'"
        )

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "d":
        return dt.timedelta(days=value)
    elif unit == "h":
        return dt.timedelta(hours=value)
    elif unit == "m":
        return dt.timedelta(minutes=value)
    elif unit == "w":
        return dt.timedelta(weeks=value)
    elif unit == "s":
        return dt.timedelta(seconds=value)


def is_cache_invalid(cache_timestamp: dt.datetime, invalid_after: str | None) -> bool:
    """Check if cache is invalid based on the invalid_after duration.

    Args:
        cache_timestamp (dt.datetime): When the cache was created.
        invalid_after (str | None): Duration string like '1d', '2h', etc.
            None means never invalid.

    Returns:
        valid (bool): True if cache is invalid (expired), False otherwise.
    """
    if invalid_after is None:
        return False

    try:
        duration = parse_duration(invalid_after)
        expiry_time = cache_timestamp + duration
        return dt.datetime.now() > expiry_time
    except ValueError as e:
        # If parsing fails, consider cache invalid to be safe
        print(f"Warning: {e}. Treating cache as invalid.")
        return True


def _read_cached_file(file: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(file)
    except Exception as e:  # TODO: What errors can happen here?
        # If cache is corrupted, remove it and continue
        print(f"Unhandled exception while loading from cache:\n{e}")
        file.unlink(missing_ok=True)


def dfcache(
    func: Callable | None = None,
    *,
    cache_dir: str | None = None,
    caching_enabled: bool = True,
    invalid_after: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator for caching pandas DataFrame results.

    Can be used as:
    - @dfcache (with default settings)
    - @dfcache(cache_dir="path")

    Args:
        func: The function to decorate (used when called without parentheses)
        cache_dir: Directory to store cache files (default: ".cache")

    Works with both functions and class methods.
    """
    from dfcache.config import cfg

    def decorator(f: Callable) -> Callable:
        # Create cache directory
        cache_path = Path(cache_dir) if cache_dir else cfg.cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name, args, and kwargs
            cache_key = _create_cache_key(f, args, kwargs)
            cache_file = cache_path / f"{_get_func_name(f)}_{cache_key}.parquet"

            # TODO: caching_enabled has to be implemented
            # TODO: This is a placeholder, the cache_timestamp must be obtained from the file itself
            cache_timestamp = dt.datetime.now()
            # Try to load from cache
            if cache_file.exists() and not is_cache_invalid(
                cache_timestamp, invalid_after
            ):
                return _read_cached_file(cache_file)

            # Execute function and cache result
            result = f(*args, **kwargs)

            # Only cache if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                try:
                    result.to_parquet(cache_file)
                except Exception as e:  # TODO: What errors can happen here?
                    # If caching fails, continue without caching
                    print(f"Unhandled exception while caching data:\n{e}")
                    cache_file.unlink(missing_ok=True)

            return result

        # Add cache management methods
        def clear_cache():
            """Clear all cached results for this function."""
            func_prefix = _get_func_name(f)
            for cache_file in cache_path.glob(f"*{func_prefix}*.parquet"):
                cache_file.unlink(missing_ok=True)

        wrapper.clear_cache = clear_cache
        wrapper.cache_dir = cache_path

        return wrapper

    # Handle both @dfcache and @dfcache(...) usage
    if func is None:
        # Called with arguments: @dfcache(...)
        return decorator
    else:
        # Called without arguments: @dfcache
        return decorator(func)


def _get_func_name(func: Any) -> str:
    return f"{func.__module__}_{func.__qualname__.replace('.', '_').replace('<', '_').replace('>', '_')}"


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
