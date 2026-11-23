"""PE-CoPT config and templates."""

import sys
import types

import pytest
from pydantic import ValidationError

from pipeline.common.schema import Params, load_raw_params
from pipeline.pecopt import PECoPTConfig, apply_pecopt


def test_full_copt_is_the_default_in_every_params_file():
    for path in (
        "params-mhubert147.yaml",
        "params-hubert.yaml",
        "params-wavlm.yaml",
        "params-xlsr.yaml",
    ):
        cfg = Params.from_yaml(path).pretrain.pecopt
        assert cfg.method == "full_copt"
        assert not cfg.is_parameter_efficient


def test_a_params_file_can_pick_a_method():

    raw = load_raw_params("params-wavlm.yaml")
    raw["pretrain"]["pecopt"] = {"method": "lora", "lora_rank": 4}
    cfg = Params(**raw).pretrain.pecopt
    assert (cfg.method, cfg.lora_rank, cfg.is_parameter_efficient) == ("lora", 4, True)


@pytest.mark.parametrize(
    "bad", [{"method": "prefix_tuning"}, {"adapter_placement": "before_layer"}, {"lora_rank": 0}]
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValidationError):
        PECoPTConfig(**bad)


def test_full_copt_injects_nothing():
    assert apply_pecopt(object(), PECoPTConfig()) == 0


def test_inject_dispatches_to_the_method_module(monkeypatch):
    calls = []
    fake = types.ModuleType("pipeline.pecopt.lora")
    fake.inject_into_model = lambda model, config: calls.append((model, config)) or 24
    monkeypatch.setitem(sys.modules, "pipeline.pecopt.lora", fake)
    model, config = object(), PECoPTConfig(method="lora")
    assert apply_pecopt(model, config) == 24
    assert calls == [(model, config)]
