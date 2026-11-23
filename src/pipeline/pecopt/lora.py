"""LoRA (template -- not implemented).

Plan: wrap each targeted attention projection (config.lora_target_modules, default q and v)
as y = W x + (alpha / rank) * B A x, W frozen, B initialised to zero so the model starts
unchanged. Trained with the model's own SSL loss.

Papers: https://arxiv.org/pdf/2407.00756v1
"""
# TODO: validate against paper


class LoRALinear:
    """Template: a frozen nn.Linear plus low-rank A and B."""

    def __init__(self, original, rank: int, alpha: float):
        raise NotImplementedError("LoRA: template only")


def inject_into_model(model, config) -> int:
    """Template: replace config.lora_target_modules in every encoder layer with LoRALinear."""
    raise NotImplementedError("LoRA: template only")
