import datetime as dt
import functools
import hashlib
import inspect
import re
from pathlib import Path
from typing import Any, Callable, TypedDict, TypeVar

import pandas as pd

from .config import Config

T = TypeVar("T")

DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(d|h|m|w|s|)$")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


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
        func: The function to decorate
        cache_dir: Directory to store cache files
        caching_enabled (bool): ...
        invalid_after (str | None): ...

    Works with both functions and class methods.
    """
    from dfcache.config import get_config

    func_args = _get_defaults_from_config(
        get_config(),
        cache_dir=cache_dir,
        caching_enabled=caching_enabled,
        invalid_after=invalid_after,
    )

    def decorator(f: Callable) -> Callable:
        cache_path = Path(func_args["cache_dir"])
        cache_path.mkdir(parents=True, exist_ok=True)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            if not func_args["caching_enabled"]:
                return result

            cache_key = _create_cache_key(f, args, kwargs)
            cache_filename_info = _get_cache_filename(
                cache_path, func_name=_get_func_name(f), cache_key=cache_key
            )

            (success, data) = _try_load_from_cache(
                cache_path,
                cache_filename_info["filename_start"],
                func_args["invalid_after"],
            )
            if success:
                return data

            # Only cache if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                _save_to_file(result, cache_filename_info["filename"])

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
        return decorator
    else:
        return decorator(func)


def _get_defaults_from_config(config: Config, **kwargs) -> dict:
    """Replaces uninformed values with the corresponding one from the config."""
    result = {}
    for key, value in kwargs.items():
        if value is None:
            result[key] = getattr(config, key)
        else:
            result[key] = value
    return result


def _parse_duration(duration_str: str) -> dt.timedelta:
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
    else:
        raise ValueError("The unit has to be informed.")


def _is_cache_invalid(cache_timestamp: dt.datetime, invalid_after: str | None) -> bool:
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
        duration = _parse_duration(invalid_after)
        expiry_time = cache_timestamp + duration
        return dt.datetime.now() > expiry_time
    except ValueError as e:
        # If parsing fails, consider cache invalid to be safe
        print(f"Warning: {e}. Treating cache as invalid.")
        return True


def _read_cached_file(filename: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(filename)
    except Exception as e:  # TODO: What errors can happen here?
        # If cache is corrupted, remove it and continue
        print(f"Unhandled exception while loading from cache:\n{e}")
        filename.unlink(missing_ok=True)


def _save_to_file(result: pd.DataFrame, filename: Path) -> None:
    try:
        result.to_parquet(filename)
    except Exception as e:  # TODO: What errors can happen here?
        # If caching fails, continue without caching
        print(f"Unhandled exception while caching data:\n{e}")
        filename.unlink(missing_ok=True)


def _try_load_from_cache(
    cache_path: Path, filename_start: str, invalid_after: str | None
) -> tuple[bool, pd.DataFrame | None]:
    # Check if there's any file that looks like our cached file:
    candidates: list[Path] = []
    for file in cache_path.iterdir():
        if file.stem.startswith(filename_start):
            candidates.append(file)
    if candidates:
        cache_file = sorted(candidates)[-1]  # Just checking the last created is enough
        cache_timestamp = _parse_timestamp_from_filename(cache_file)
        if not _is_cache_invalid(cache_timestamp, invalid_after):
            return (True, _read_cached_file(cache_file))
    return (False, None)


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


def _get_timestamp() -> str:
    return dt.datetime.now().strftime(TIMESTAMP_FORMAT)


def _parse_timestamp_from_filename(filename: Path) -> dt.datetime:
    """Check _get_cache_filename to see the reasoning for this function.

    Args:
        filename (Path): _description_

    Returns:
        dt.datetime: _description_
    """
    timestamp = "_".join(filename.stem.split("_")[-2:])
    return dt.datetime.strptime(timestamp, TIMESTAMP_FORMAT)


class FilenameInfo(TypedDict):
    filename: Path
    timestamp: str
    filename_start: str


def _get_cache_filename(
    cache_path: Path, func_name: str, cache_key: str, extension: str = "parquet"
) -> FilenameInfo:
    """Generates the filename info for the cached result.

    It contains the full filename to the cached file, the timestamp and the start
    of the filename to find it in case there's more than one with different timestamps.

    Args:
        cache_path (Path): Folder where the file will be saved.
        func_name (str): Name of the function.
        cache_key (str): Cached key from the function and args/kwargs.
        extension (str, optional): File extension. Defaults to "parquet".

    Returns:
        filename (str): Name of the file.
    """
    timestamp = _get_timestamp()
    return {
        "filename": cache_path / f"{func_name}_{cache_key}_{timestamp}.{extension}",
        "timestamp": timestamp,
        "filename_start": f"{func_name}_{cache_key}",
    }


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
