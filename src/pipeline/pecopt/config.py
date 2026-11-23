"""How to train: full continued pre-training (CoPT) or one of three PE-CoPT methods.

`method` names the training approach. `full_copt` is the default and today's behaviour:
every weight trains (except the CNN, per pretrain.freeze_feature_encoder). The other three
are parameter-efficient templates, not implemented yet (see the stub modules).

Pure pydantic, so the schema can use it without importing torch.
"""

from typing import Literal

from pydantic import BaseModel, Field

TrainingMethod = Literal["full_copt", "residual_adapter", "lora", "gc_lora"]
# The DRAFT paper says adapters go "after each encoder block" without saying where, so the
# placement is a setting to compare rather than a fixed choice.
AdapterPlacement = Literal["after_attention", "after_ffn", "after_layer"]


class PECoPTConfig(BaseModel):
    method: TrainingMethod = "full_copt"

    # Residual adapter (DRAFT)
    adapter_bottleneck_dim: int = Field(default=256, gt=0)
    adapter_placement: AdapterPlacement = "after_layer"

    # LoRA, on these attention projections (q_proj / k_proj / v_proj / out_proj)
    lora_rank: int = Field(default=8, gt=0)
    lora_alpha: float = Field(default=16.0, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    # GC-LoRA: LoRA on the attention output projection with a gated convolution inside
    gc_lora_rank: int = Field(default=8, gt=0)
    gc_lora_kernel_size: int = Field(default=31, gt=0)

    @property
    def is_parameter_efficient(self) -> bool:
        return self.method != "full_copt"
