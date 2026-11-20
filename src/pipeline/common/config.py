"""Config loading: params.yaml plus an optional per-model overlay.

params.yaml holds everything the models share (data selection, sharding, the training
recipe) and, as it always has, the settings of the XLS-R run. A model that differs
ships configs/models/<name>.yaml with only the keys it changes; it is deep-merged over
params.yaml, so what the overlay does not mention is the shared recipe by construction.
"""

import argparse
import copy
from pathlib import Path

import yaml

from pipeline.schema import Params

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_DIR = Path("configs") / "models"


def deep_merge(base: dict, overlay: dict) -> dict:
    """overlay's keys win; nested dicts merge recursively. Returns a new dict."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def model_overlay_path(model: str) -> Path:
    """configs/models/<model>.yaml, looked up relative to the working directory first
    (as params.yaml is) and then to the repository root."""
    candidates = [
        MODEL_CONFIG_DIR / f"{model}.yaml",
        REPO_ROOT / MODEL_CONFIG_DIR / f"{model}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no model config for {model!r}; looked in {[str(c) for c in candidates]}"
    )


def load_raw(
    config: str = "params.yaml", model: str | None = None, overlay: str | None = None
) -> dict:
    """params.yaml, then configs/models/<model>.yaml, then an extra overlay file."""
    with open(config) as f:
        raw = yaml.safe_load(f)
    for path in ([model_overlay_path(model)] if model else []) + (
        [Path(overlay)] if overlay else []
    ):
        with open(path) as f:
            raw = deep_merge(raw, yaml.safe_load(f) or {})
    return raw


def load_params(
    config: str = "params.yaml", model: str | None = None, overlay: str | None = None
) -> Params:
    return Params(**load_raw(config, model, overlay))


def add_config_args(
    parser: argparse.ArgumentParser, model_choices: list[str] | None = None
) -> None:
    """The flags every stage that reads the config shares."""
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument(
        "--model",
        default=None,
        choices=model_choices,
        help="model whose configs/models/<model>.yaml is merged over --config "
        "(default: params.yaml as it is, the XLS-R run)",
    )
    parser.add_argument(
        "--overlay", default=None, help="extra YAML of overrides merged on top of everything"
    )


def params_from_args(args: argparse.Namespace) -> Params:
    return load_params(args.config, args.model, args.overlay)
