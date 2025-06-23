import hashlib
import inspect
from typing import Any, Callable

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
        df_to_hash = _cast_unhashable_columns_to_str(obj)
        return {
            "type": "DataFrame",
            "shape": obj.shape,
            "columns": tuple(obj.columns.tolist()),
            "dtypes": tuple(obj.dtypes.tolist()),
            "hash": hashlib.md5(
                pd.util.hash_pandas_object(df_to_hash).to_numpy().tobytes()
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


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
        return True
    except TypeError:
        return False


def _cast_unhashable_columns_to_str(
    df: pd.DataFrame, sample_size: int = 100, seed: int = 667
) -> pd.DataFrame:
    """Cast columns that cannot be hashed to be strings.

    This is a helper function to hash the dataframes.

    Args:
        df (pd.DataFrame): DataFrame to check for unhashable columns.
        sample_size (int): Sample size to sample from the dataframe. Defaults to 100.
        seed (int): Random state to sample the dataframe. Defaults to 667.

    Returns:
        data (pd.DataFrame): DataFrame with casted columns.
    """
    unhashable_cols = []
    for col in df.columns:
        sample = (
            df[col]
            .dropna()
            .sample(n=min(sample_size, len(df[col].dropna())), random_state=seed)
        )
        if not sample.map(_is_hashable).all():
            unhashable_cols.append(col)
    if unhashable_cols:
        df_copy = df.copy()
        for col in unhashable_cols:
            df_copy[col] = df_copy[col].astype(str)
        return df_copy
    return df
