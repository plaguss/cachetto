import hashlib
from typing import Any

import pandas as pd
import pytest

from cachetto._hashing import (
    _cast_unhashable_columns_to_str,
    create_cache_key,
    make_hashable,
)


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
    def testmake_hashable_parametrized(self, input_obj: Any, expected: Any) -> None:
        result = make_hashable(input_obj)
        assert result == expected

    def testmake_hashable_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = make_hashable(df)
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
    def testmake_hashable_dataframe_hash(
        self, df_data1: pd.DataFrame, df_data2: pd.DataFrame, should_be_equal: bool
    ) -> None:
        df1 = pd.DataFrame(df_data1)
        df2 = pd.DataFrame(df_data2)
        hash1 = make_hashable(df1)["hash"]
        hash2 = make_hashable(df2)["hash"]
        if should_be_equal:
            assert hash1 == hash2
        else:
            assert hash1 != hash2


class TestCreateCacheKey:
    def test_simple_function_args(self) -> None:
        def foo(a, b):
            return a + b

        key = create_cache_key(foo, (1, 2), {})

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

        key = create_cache_key(bar, (5,), {"y": 20})

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

        key = create_cache_key(Baz.method, (baz, 42), {})

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

        # Simulate make_hashable for lists and dicts
        monkeypatch.setattr("cachetto._hashing.make_hashable", lambda x: str(x))

        key = create_cache_key(func, ([1, 2, 3], {"foo": "bar"}), {})

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

        key = create_cache_key(foo, (), {})
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

        key = create_cache_key(foo, (), {"a": 3, "b": 4})
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

        key = create_cache_key(foo, (5,), {"b": 7})
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

        key1 = create_cache_key(foo, (), {"a": 1, "b": 2})
        key2 = create_cache_key(foo, (), {"b": 2, "a": 1})
        assert key1 == key2

    def test_mutable_args(self) -> None:
        def foo(a):
            return sum(a)

        key = create_cache_key(foo, ([1, 2, 3],), {})
        assert key == "a08b9bdbea0ba9b8273cd7449b62271e"

    def test_empty_args_kwargs(self) -> None:
        def foo():
            return 1

        key = create_cache_key(foo, (), {})
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
        key = create_cache_key(foo, (obj,), {})
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

        key = create_cache_key(foo, (1,), {})
        expected_data = {
            "function": f"{foo.__module__}.{foo.__qualname__}",
            "args": (("a", 1), ("b", None)),
        }
        expected_str = str(expected_data)
        expected_hash = hashlib.md5(expected_str.encode()).hexdigest()
        assert key == expected_hash


class UnhashableObject:
    def __init__(self, value):
        self.value = value

    # Deliberately disable __hash__ to trigger TypeError
    __hash__ = None


@pytest.mark.parametrize(
    "input_df, expected_unhashable_cols",
    [
        (pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}), []),
        (pd.DataFrame({"a": [1, 2, 3], "b": [[1], [2], [3]]}), ["b"]),
        (pd.DataFrame({"a": [[1], [2], [3]], "b": [[4], [5], [6]]}), ["a", "b"]),
        (pd.DataFrame({"a": [None, None], "b": [[1], [2]]}), ["b"]),
        (pd.DataFrame({"a": [], "b": []}), []),
        (pd.DataFrame({"a": [{"x": 1}, {"y": 2}], "b": [1, 2]}), ["a"]),
        (
            pd.DataFrame(
                {"a": [UnhashableObject(1), UnhashableObject(2)], "b": ["ok", "fine"]}
            ),
            ["a"],
        ),
    ],
)
def test_cast_unhashable_columns_to_str(input_df, expected_unhashable_cols):
    result_df = _cast_unhashable_columns_to_str(input_df)

    for col in input_df.columns:
        if col in expected_unhashable_cols:
            assert result_df[col].apply(type).eq(str).all(), (
                f"Column '{col}' should be cast to str"
            )
        else:
            pd.testing.assert_series_equal(
                result_df[col], input_df[col], check_names=False
            )
