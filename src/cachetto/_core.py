import functools
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar, overload

import pandas as pd

from ._config import Config
from ._files import FILE_EXTENSION, get_cache_filename, read_cached_file, save_to_file
from ._hashing import create_cache_key
from ._utils import get_func_name, is_cache_invalid, parse_timestamp_from_filename

F = TypeVar("F", bound=Callable[..., Any])


class CachedFunction(Protocol):
    """Protocol for functions decorated with @cached"""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def clear_cache(self) -> None: ...
    @property
    def cache_dir(self) -> str | None: ...


@overload
def cached(
    func: F,
) -> CachedFunction: ...


@overload
def cached(
    func: None = None,
    *,
    cache_dir: str | None = None,
    caching_enabled: bool = True,
    invalid_after: str | None = None,
) -> Callable[[F], CachedFunction]: ...


def cached(
    func: F | None = None,
    *,
    cache_dir: str | None = None,
    caching_enabled: bool = True,
    invalid_after: str | None = None,
    verbose: bool = False,
) -> CachedFunction | Callable[[F], CachedFunction]:
    """Decorator for caching python functions and class' methods to disk.

    This decorator caches the output of a function that work with python builtin
    objects, and pandas dataframes storing them in a specified cache directory
    as a pickle file. If the cache exists and is still valid (based on
    `invalid_after`), it will be loaded instead of recomputing the function.

    Examples:
        ```py
        Can be used with or without parentheses:
        @cached
        def my_function(...): ...

        @cached(cache_dir="path/to/cache", invalid_after="7d")
        def my_function(...): ...
        ```

    Args:
        func (Callable | None): The function to decorate.
            If None, the decorator is being used with parameters.
        cache_dir (str | None): Directory where cache files will be stored.
            If None, the default from config is used.
        caching_enabled (bool): Whether caching is enabled.
            If False, the function will always execute without caching.
        invalid_after (str | None): Duration string (e.g. '1d',
            '6h') specifying how long the cache remains valid.
            If None, the cache is considered always valid.

    Returns:
        Callable: A decorated function that caches its output.

    Raises:
        ValueError: If `invalid_after` is not a valid duration format.

    Attributes:
        clear_cache (Callable[[], None]): Method to clear all cached files for
            this function.
        cache_dir (Path): Path object representing the cache directory in use.

    Note:
        - Only functions that return builtin python objects, including
        `pandas.DataFrame` will be cached.
        - Cached files are stored in `.pickle` format.
        - Caching works for both functions and class instance methods.
    """
    from cachetto._config import get_config

    func_args = _get_defaults_from_config(
        get_config(),
        cache_dir=cache_dir,
        caching_enabled=caching_enabled,
        invalid_after=invalid_after,
    )

    def decorator(f: F) -> CachedFunction:
        cache_path = Path(func_args["cache_dir"])
        cache_path.mkdir(parents=True, exist_ok=True)

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not func_args["caching_enabled"]:
                return f(*args, **kwargs)

            cache_key = create_cache_key(f, args, kwargs)
            cache_filename_info = get_cache_filename(
                cache_path, func_name=get_func_name(f), cache_key=cache_key
            )

            (cached_filename, data) = _try_load_from_cache(
                cache_path,
                cache_filename_info["filename_start"],
                func_args["invalid_after"],
            )

            if cached_filename:
                log = print if verbose else lambda x: None
                log(f"Cache hit: '{cached_filename}'")
                return data

            result = f(*args, **kwargs)

            save_to_file(result, cache_filename_info["filename"])

            return result

        # Add cache management methods
        def clear_cache():
            """Clear all cached results for this function."""
            func_prefix = get_func_name(f)
            for cache_file in cache_path.glob(f"*{func_prefix}*.{FILE_EXTENSION}"):
                cache_file.unlink(missing_ok=True)

        wrapper.clear_cache = clear_cache  # type: ignore
        wrapper.cache_dir = cache_path  # type: ignore

        return wrapper  # type: ignore

    # Handle both @cached and @cached(...) usage
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


def _try_load_from_cache(
    cache_path: Path, filename_start: str, invalid_after: str | None
) -> tuple[Path | None, pd.DataFrame | None]:
    """Checks if there's a file that can be loaded from cache.

    Args:
        cache_path (Path): Folder to look for the data.
        filename_start (str): Beginning of the file name to track it.
        invalid_after (str | None): Pattern to determine if a file is
            valid.

    Returns:
        tuple[Path | None, pd.DataFrame | None]: If found, it will
            return a tuple with the Path of the file and the data,
            otherwise a tuple of None and None.
    """
    # Check if there's any file that looks like our cached file:
    candidates: list[Path] = []
    for file in cache_path.iterdir():
        if file.stem.startswith(filename_start):
            candidates.append(file)
    if candidates:
        cache_file = sorted(candidates)[-1]  # Just checking the last created is enough
        cache_timestamp = parse_timestamp_from_filename(cache_file)
        if not is_cache_invalid(cache_timestamp, invalid_after):
            return (cache_file, read_cached_file(cache_file))
    return (None, None)
