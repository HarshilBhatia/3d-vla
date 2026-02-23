"""Compose Hydra config and return an args-like object for existing code.

Hydra returns OmegaConf; we convert to SimpleNamespace so that vars(args) and
getattr(args, k) work everywhere (e.g. wandb config=vars(args), printing).
We also convert path strings to pathlib.Path and resolve relative paths to
the project root; Hydra does not do that by default.
"""

from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def get_config_path() -> Path:
    """Return the project config directory (absolute). Use from any entry point so config path is consistent."""
    return Path(__file__).resolve().parent.parent / "config"


# Paths that get resolved relative to project root (absolute). exp_log_dir and run_log_dir
# are kept as relative so log_dir = base_log_dir / exp_log_dir / run_log_dir works.
_PATH_KEYS = frozenset({
    "train_data_dir", "eval_data_dir", "train_instructions", "val_instructions",
    "base_log_dir",
    "checkpoint",
    "data_dir", "output_file",
})


def _resolve_relative_paths(args: SimpleNamespace, base: Path) -> None:
    for k in _PATH_KEYS:
        v = getattr(args, k, None)
        if v is not None and v != "" and isinstance(v, Path) and not v.is_absolute():
            setattr(args, k, (base / v).resolve())


def _cfg_to_args(cfg, base_dir: Path = None) -> SimpleNamespace:
    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(cfg, resolve=True)
    out = {}
    for k, v in raw.items():
        if k in _PATH_KEYS and v is not None and v != "":
            out[k] = Path(v) if not isinstance(v, Path) else v
        else:
            out[k] = v
    args = SimpleNamespace(**out)
    if base_dir is not None:
        _resolve_relative_paths(args, base_dir)
    return args


# Config groups that use @_global_ in defaults; CLI override "group=option" must be passed as "group@_global_=option"
_GLOBAL_GROUP_OVERRIDES = ("data", "rope_mode", "experiment")


def _normalize_overrides(overrides):
    """Convert data=x, rope_mode=y, experiment=z to data@_global_=x etc. so Hydra accepts them."""
    out = []
    for s in overrides or []:
        if "=" not in s:
            out.append(s)
            continue
        key, _, value = s.partition("=")
        if key in _GLOBAL_GROUP_OVERRIDES and "@" not in key:
            out.append(f"{key}@_global_={value}")
        else:
            out.append(s)
    return out


def get_config(
    overrides=None,
    config_name: str = "config",
    config_path: Path = None,
):
    if config_path is None:
        raise ValueError("config_path must be set (e.g. Path(__file__).parent / 'config')")
    config_path = Path(config_path).resolve()
    if not config_path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_path}")

    normalized = _normalize_overrides(overrides)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_path)):
        cfg = compose(config_name=config_name, overrides=normalized)
    base_dir = config_path.resolve().parent
    return _cfg_to_args(cfg, base_dir=base_dir)
