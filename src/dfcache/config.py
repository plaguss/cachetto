from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class Config:
    cache_dir: Path = Path.home() / ".cache" / "dfcache"
    caching_enabled: bool = True
    invalid_after: str | None = None


cfg = Config()


def set_config(**params: Mapping) -> None:
    """Configures global configuration."""
    import dfcache.config

    valid_params = {k: v for k, v in params.items() if hasattr(dfcache.config.cfg, k)}
    dfcache.config.cfg = replace(
        dfcache.config.cfg,
        **valid_params,
    )


def get_config() -> Config:
    """Get the global config."""
    import dfcache.config

    return dfcache.config.cfg


def enable_caching():
    """Enable caching globally."""
    import dfcache.config

    dfcache.config.cfg.caching_enabled = True


def disable_caching():
    """Disable caching globally."""
    import dfcache.config

    dfcache.config.cfg.caching_enabled = False
