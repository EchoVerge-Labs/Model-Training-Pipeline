"""PE-CoPT (parameter-efficient continued pre-training): config and method templates.

Only `full_copt` (the default) does anything today. residual_adapter, lora and gc_lora are
placeholder templates -- each module says what it will do and raises NotImplementedError.
They get implemented after the full-CoPT refactor is merged and tested.
"""

import importlib

from pipeline.pecopt.config import PECoPTConfig

_METHOD_MODULES = {
    "residual_adapter": "pipeline.pecopt.residual_adapter",
    "lora": "pipeline.pecopt.lora",
    "gc_lora": "pipeline.pecopt.gc_lora",
}


def apply_pecopt(model, config: PECoPTConfig) -> int:
    """Adds the configured PE-CoPT modules to `model` in place; full_copt adds nothing."""
    if not config.is_parameter_efficient:
        return 0
    return importlib.import_module(_METHOD_MODULES[config.method]).inject_into_model(model, config)


__all__ = ["PECoPTConfig", "apply_pecopt"]
