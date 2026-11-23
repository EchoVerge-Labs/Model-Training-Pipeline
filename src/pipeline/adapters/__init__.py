"""Training method for continued pre-training: full CoPT, or a PE-CoPT template.

Only `full_copt` (the default) does anything today. residual_adapter, lora and gc_lora are
placeholder templates -- each module says what it will do and raises NotImplementedError.
They get implemented after the full-CoPT refactor is merged and tested.
"""

import importlib

from pipeline.adapters.config import PECoPTConfig

_METHOD_MODULES = {
    "residual_adapter": "pipeline.adapters.residual_adapter",
    "lora": "pipeline.adapters.lora",
    "gc_lora": "pipeline.adapters.gc_lora",
}


def inject_adapters(model, config: PECoPTConfig) -> int:
    """Adds the configured PE-CoPT modules to `model` in place; full_copt adds nothing."""
    if not config.is_parameter_efficient:
        return 0
    return importlib.import_module(_METHOD_MODULES[config.method]).inject_into_model(model, config)


__all__ = ["PECoPTConfig", "inject_adapters"]
