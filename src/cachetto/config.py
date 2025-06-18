from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class Config:
    cache_dir: Path = Path.home() / ".cache" / "cachetto"
    caching_enabled: bool = True
    invalid_after: str | None = None


_cfg = Config()


def set_config(**params: Mapping) -> None:
    """Configures global configuration."""
    import cachetto.config

    valid_params = {k: v for k, v in params.items() if hasattr(cachetto.config._cfg, k)}
    cachetto.config._cfg = replace(
        cachetto.config._cfg,
        **valid_params,
    )


def get_config() -> Config:
    """Get the global config."""
    import cachetto.config

    return cachetto.config._cfg


def enable_caching():
    """Enable caching globally."""
    import cachetto.config

    cachetto.config._cfg.caching_enabled = True


def disable_caching():
    """Disable caching globally."""
    import cachetto.config

    cachetto.config._cfg.caching_enabled = False
