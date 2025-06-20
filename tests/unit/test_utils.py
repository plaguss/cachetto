import datetime as dt

import pytest

from cachetto._utils import get_func_name, is_cache_invalid, parse_duration


def sample_function():
    pass


class ExampleClass:
    def method(self):
        pass

    @classmethod
    def class_method(cls):
        pass

    @staticmethod
    def static_method():
        pass


class TestGetFuncName:
    def test_global_function(self):
        assert (
            get_func_name(sample_function)
            == f"{sample_function.__module__}_sample_function"
        )

    def test_class_method(self):
        assert (
            get_func_name(ExampleClass.class_method)
            == f"{ExampleClass.class_method.__module__}_ExampleClass_class_method"
        )

    def test_instance_method(self):
        assert (
            get_func_name(ExampleClass().method)
            == f"{ExampleClass.method.__module__}_ExampleClass_method"
        )

    def test_static_method(self):
        assert (
            get_func_name(ExampleClass.static_method)
            == f"{ExampleClass.static_method.__module__}_ExampleClass_static_method"
        )

    def test_nested_function(self):
        def outer():
            def inner():
                pass

            return inner

        inner_func = outer()
        expected = "test_utils_TestGetFuncName_test_nested_function__locals__outer__locals__inner"
        assert get_func_name(inner_func) == expected

    def test_lambda_function(self):
        f = lambda x: x  # noqa: E731
        expected = "test_utils_TestGetFuncName_test_lambda_function__locals___lambda_"
        assert get_func_name(f) == expected


class TestParseDuration:
    @pytest.mark.parametrize(
        "duration_str,expected",
        [
            ("1d", dt.timedelta(days=1)),
            ("2h", dt.timedelta(hours=2)),
            ("30m", dt.timedelta(minutes=30)),
            ("1w", dt.timedelta(weeks=1)),
            ("45s", dt.timedelta(seconds=45)),
            ("1.5h", dt.timedelta(hours=1.5)),
            ("  2d  ", dt.timedelta(days=2)),  # with extra spaces
            ("3.25m", dt.timedelta(minutes=3.25)),
        ],
    )
    def test_parse_duration_valid(self, duration_str: str, expected: dt.timedelta):
        assert parse_duration(duration_str) == expected

    @pytest.mark.parametrize(
        "invalid_input", ["", "10x", "abc", "1", "h", "10dd", "5hours", "12hm"]
    )
    def test_parse_duration_invalid(self, invalid_input: str) -> None:
        with pytest.raises(ValueError):
            parse_duration(invalid_input)


class TestIsCacheInvalid:
    def test_is_cache_invalid_none(self) -> None:
        assert is_cache_invalid(dt.datetime.now(), None) is False

    def test_is_cache_invalid_expired(self, monkeypatch) -> None:
        past_time = dt.datetime.now() - dt.timedelta(hours=2)
        assert is_cache_invalid(past_time, "1h") is True

    def test_is_cache_invalid_not_expired(self, monkeypatch) -> None:
        now = dt.datetime.now()
        monkeypatch.setattr("cachetto._utils.dt", dt)
        assert is_cache_invalid(now, "1d") is False

    def test_is_cache_invalid_invalid_duration(self, monkeypatch) -> None:
        now = dt.datetime.now()
        monkeypatch.setattr("cachetto._utils.dt", dt)
        assert is_cache_invalid(now, "badformat") is True
