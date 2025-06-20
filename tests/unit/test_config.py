from dataclasses import replace
from pathlib import Path

import pytest

from cachetto import _config


@pytest.fixture(autouse=True)
def reset_config():
    # Save original and restore after each test
    original = _config._cfg
    _config._cfg = replace(original)
    yield
    _config._cfg = original


def test_get_config() -> None:
    cfg = _config.get_config()
    assert cfg == _config._cfg


def test_set_config():
    new_cache_dir = Path("/tmp/test_cache")
    _config.set_config(cache_dir=new_cache_dir)

    assert _config._cfg.cache_dir == new_cache_dir


def test_set_config_ignores_invalid_keys():
    original_cfg = _config._cfg
    _config.set_config(unknown_param=True)

    assert _config._cfg == original_cfg


def test_disable_caching():
    _config.enable_caching()
    _config.disable_caching()
    assert _config._cfg.caching_enabled is False


def test_enable_caching():
    _config.disable_caching()
    _config.enable_caching()
    assert _config._cfg.caching_enabled is True
