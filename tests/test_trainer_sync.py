"""The wavlm and hubert trainers are deliberately separate files, but they are the same loop.

This keeps them from drifting apart: after the model names are normalised, the only code
that may differ is WavLM's utterance mixing. A fix made to one of them and not the other
(resume, logging, checkpointing, the loss) fails here.
"""

import ast
import inspect
import re

from pipeline.models.hubert import trainer as hubert
from pipeline.models.wavlm import trainer as wavlm

SHARED = ("compute_mask", "save_checkpoint", "prepare_curves", "train")


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = visit_ClassDef = visit_Module = _strip


def _normalised(module, name: str) -> list[str]:
    """The function's code, without docstrings, comments or formatting, with the model's
    name replaced by a placeholder and WavLM's utterance mixing removed."""
    tree = _StripDocstrings().visit(ast.parse(inspect.getsource(getattr(module, name))))
    text = ast.unparse(tree)
    text = re.sub(r"wavlm|hubert", "MODEL", text, flags=re.IGNORECASE)
    # utterance mixing: one keyword argument to the data loader, one logged parameter
    text = text.replace(", utterance_mix_prob=cfg.utterance_mix_prob", "")
    text = text.replace(", 'utterance_mix_prob': cfg.utterance_mix_prob", "")
    return text.splitlines()


def test_wavlm_and_hubert_trainers_differ_only_in_their_model_and_utterance_mixing():
    for name in SHARED:
        assert _normalised(wavlm, name) == _normalised(hubert, name), (
            f"{name}() differs between the wavlm and hubert trainers beyond the model name "
            f"and utterance mixing -- a change was made to one and not the other"
        )


def test_the_curve_columns_are_the_same():
    assert wavlm.CURVES_HEADER == hubert.CURVES_HEADER


def test_the_models_wrappers_differ_only_by_class_and_attribute_name():
    from pipeline.models.hubert import model as hubert_model
    from pipeline.models.wavlm import model as wavlm_model

    def code(module):
        text = ast.unparse(_StripDocstrings().visit(ast.parse(inspect.getsource(module))))
        return re.sub(r"wavlm|hubert", "MODEL", text, flags=re.IGNORECASE)

    assert code(wavlm_model) == code(hubert_model)
