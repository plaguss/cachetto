from typing import Callable, Any
import inspect
import hashlib

import pandas as pd


def create_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Create a unique cache key for the function call."""

    # Get function signature to map args to parameter names
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    args_ = dict(bound_args.arguments.items())

    # Handle 'self' parameter for methods (ignore it for cache key)
    if "self" in args_:
        args_.pop("self")

    # Create hashable representation
    key_data = {
        "function": f"{func.__module__}.{func.__qualname__}",
        "args": make_hashable(args_),
    }

    # Create hash
    key_str = str(key_data)
    return hashlib.md5(key_str.encode()).hexdigest()


def make_hashable(obj: Any) -> Any:
    """Convert objects to hashable representation for cache key creation."""
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": obj.shape,
            "columns": tuple(obj.columns.tolist()),
            "dtypes": tuple(obj.dtypes.tolist()),
            "hash": hashlib.md5(
                pd.util.hash_pandas_object(obj).to_numpy().tobytes()
            ).hexdigest(),
        }
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(make_hashable(item) for item in obj))
    elif obj is None:
        return None
    else:
        return obj
