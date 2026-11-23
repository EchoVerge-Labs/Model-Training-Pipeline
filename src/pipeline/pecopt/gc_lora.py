"""GC-LoRA, gated convolutional LoRA (template -- not implemented).

Plan: LoRA on the attention output projection only, with a gated depthwise-separable
convolution between A and B: pointwise -> GLU -> depthwise (kernel config.gc_lora_kernel_size)
-> GroupNorm -> Swish -> pointwise, added back to the low-rank features, then B.

Paper: https://arxiv.org/html/2606.10464v1
"""
# TODO: validate against paper


class GCLoRAModule:
    """Template: LoRA whose low-rank path includes the gated convolution."""

    def __init__(self, original, rank: int, alpha: float, kernel_size: int):
        raise NotImplementedError("GC-LoRA: template only")


def inject_into_model(model, config) -> int:
    """Template: wrap the attention output projection of every encoder layer."""
    raise NotImplementedError("GC-LoRA: template only")
