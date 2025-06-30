from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass
class Config:
    cache_dir: Path = Path.home() / ".cache" / "cachetto"
    caching_enabled: bool = True
    invalid_after: str | None = None
    verbose: bool = True


_cfg = Config()


def set_config(**params: Any) -> None:
    """Configures global configuration."""
    import cachetto._config

    valid_params = {
        k: v for k, v in params.items() if hasattr(cachetto._config._cfg, k)
    }
    cachetto._config._cfg = replace(
        cachetto._config._cfg,
        **valid_params,
    )


def get_config() -> Config:
    """Get the global config."""
    import cachetto._config

    return cachetto._config._cfg


def enable_caching():
    """Enable caching globally."""
    import cachetto._config

    cachetto._config._cfg.caching_enabled = True


def disable_caching():
    """Disable caching globally."""
    import cachetto._config

    cachetto._config._cfg.caching_enabled = False
