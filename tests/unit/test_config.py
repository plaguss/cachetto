from dataclasses import replace
from pathlib import Path

import pytest

from dfcache import config


@pytest.fixture(autouse=True)
def reset_config():
    # Save original and restore after each test
    original = config.cfg
    config.cfg = replace(original)
    yield
    config.cfg = original


def test_get_config() -> None:
    cfg = config.get_config()
    assert cfg == config.cfg


def test_set_config():
    new_cache_dir = Path("/tmp/test_cache")
    config.set_config(cache_dir=new_cache_dir)

    assert config.cfg.cache_dir == new_cache_dir


def test_set_config_ignores_invalid_keys():
    original_cfg = config.cfg
    config.set_config(unknown_param=True)

    assert config.cfg == original_cfg


def test_disable_caching():
    config.enable_caching()
    config.disable_caching()
    assert config.cfg.caching_enabled is False


def test_enable_caching():
    config.disable_caching()
    config.enable_caching()
    assert config.cfg.caching_enabled is True
