from pathlib import Path
from typing import TypedDict

import pandas as pd

from ._utils import get_timestamp


def read_cached_file(filename: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(filename)
    except Exception as e:  # TODO: What errors can happen here?
        # If cache is corrupted, remove it and continue
        print(f"Unhandled exception while loading from cache:\n{e}")
        filename.unlink(missing_ok=True)
        return None


def save_to_file(result: pd.DataFrame, filename: Path) -> None:
    try:
        result.to_parquet(filename)
    except Exception as e:  # TODO: What errors can happen here?
        # If caching fails, continue without caching
        print(f"Unhandled exception while caching data:\n{e}")
        filename.unlink(missing_ok=True)


class FilenameInfo(TypedDict):
    filename: Path
    timestamp: str
    filename_start: str


def get_cache_filename(
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
    timestamp = get_timestamp()
    return {
        "filename": cache_path / f"{func_name}_{cache_key}_{timestamp}.{extension}",
        "timestamp": timestamp,
        "filename_start": f"{func_name}_{cache_key}",
    }
