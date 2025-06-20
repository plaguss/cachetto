import pickle
from pathlib import Path
from typing import Any, Final, TypedDict

from ._utils import get_timestamp

FILE_EXTENSION: Final[str] = "pickle"


def read_cached_file(filename: Path) -> Any | None:
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (
        pickle.UnpicklingError,
        EOFError,
        FileNotFoundError,
        PermissionError,
        AttributeError,
        ModuleNotFoundError,
        OSError,
    ) as e:
        # If cache is corrupted, remove it and continue
        print(f"Unhandled exception while loading from cache:\n{e}")
        filename.unlink(missing_ok=True)
        return None


def save_to_file(result: Any, filename: Path) -> None:
    try:
        with open(filename, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except (
        pickle.PicklingError,
        TypeError,
        PermissionError,
        FileNotFoundError,
        IsADirectoryError,
        OSError,
        AttributeError,
    ) as e:
        # If caching fails, continue without caching
        print(f"Unhandled exception while caching data:\n{e}")
        filename.unlink(missing_ok=True)


class FilenameInfo(TypedDict):
    filename: Path
    timestamp: str
    filename_start: str


def get_cache_filename(
    cache_path: Path, func_name: str, cache_key: str, extension: str = FILE_EXTENSION
) -> FilenameInfo:
    """Generates the filename info for the cached result.

    It contains the full filename to the cached file, the timestamp and the start
    of the filename to find it in case there's more than one with different timestamps.

    Args:
        cache_path (Path): Folder where the file will be saved.
        func_name (str): Name of the function.
        cache_key (str): Cached key from the function and args/kwargs.
        extension (str, optional): File extension. Defaults to "pickle".

    Returns:
        filename (str): Name of the file.
    """
    timestamp = get_timestamp()
    return {
        "filename": cache_path / f"{func_name}_{cache_key}_{timestamp}.{extension}",
        "timestamp": timestamp,
        "filename_start": f"{func_name}_{cache_key}",
    }
