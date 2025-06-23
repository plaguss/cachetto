import datetime as dt
import re
from pathlib import Path
from typing import Any

DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*(d|h|m|w|s|)$")
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def get_func_name(func: Any) -> str:
    return f"{func.__module__.replace('.', '_')}_{func.__qualname__.replace('.', '_').replace('<', '_').replace('>', '_')}"


def get_timestamp() -> str:
    return dt.datetime.now().strftime(TIMESTAMP_FORMAT)


def parse_timestamp_from_filename(filename: Path) -> dt.datetime:
    """Check _get_cache_filename to see the reasoning for this function.

    Args:
        filename (Path): _description_

    Returns:
        dt.datetime: _description_
    """
    timestamp = "_".join(filename.stem.split("_")[-2:])
    return dt.datetime.strptime(timestamp, TIMESTAMP_FORMAT)


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
    else:
        raise ValueError("The unit has to be informed.")


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
