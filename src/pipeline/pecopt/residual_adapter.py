"""Residual adapters, DRAFT-style (template -- not implemented).

Plan: a bottleneck module after each encoder block, h' = h + W_up(act(W_down(LN(h)))),
with W_up initialised to zero so the model starts unchanged. Trained with the model's own
SSL loss while the backbone stays frozen. Where in the block it goes is
config.adapter_placement; the DRAFT paper does not say.

Paper: https://ar5iv.labs.arxiv.org/html/2206.07931
"""
# TODO: validate against paper


class ResidualAdapter:
    """Template: hidden_size -> adapter_bottleneck_dim -> hidden_size, with a residual."""

    def __init__(self, hidden_size: int, bottleneck_dim: int):
        raise NotImplementedError("residual adapter: template only")


def inject_into_model(model, config) -> int:
    """Template: add a ResidualAdapter at config.adapter_placement of every encoder layer."""
    raise NotImplementedError("residual adapter: template only")
