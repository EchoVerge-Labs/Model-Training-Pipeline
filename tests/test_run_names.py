"""A run is <model>-<experiment>; everything it writes carries that name, so two
experiments (or two models) never share a folder, report or benchmark directory."""

import pytest

from pipeline.common.schema import Params, load_raw_params
from pipeline.eval.benchmark import bench_name, build_command


def test_default_run_name_is_model_prefix_plus_experiment():
    assert Params.from_yaml("params-wavlm.yaml").run == "wavlm-large-si-ta-200h"
    assert Params.from_yaml("params-mhubert147.yaml").run == "mhubert147-si-ta-200h"


def test_experiment_flag_renames_every_output_path():
    p = Params.from_yaml("params-wavlm.yaml", "si-100h")
    assert p.run == "wavlm-large-si-100h"
    assert p.pretrain.output_dir == "models/wavlm-large-si-100h"
    raw = load_raw_params("params-wavlm.yaml", "si-100h")
    assert raw["checkpoint_selection"]["output_dir"] == "models/selected-wavlm-large-si-100h"
    assert "wavlm-large-si-100h" in raw["benchmark"]["upstreams"]
    assert "{run}" not in str(raw)


def test_two_experiments_and_two_models_share_no_output_path():
    a = Params.from_yaml("params-wavlm.yaml", "si-ta-200h")
    b = Params.from_yaml("params-wavlm.yaml", "si-100h")
    c = Params.from_yaml("params-hubert.yaml", "si-100h")
    assert len({a.pretrain.output_dir, b.pretrain.output_dir, c.pretrain.output_dir}) == 3


@pytest.mark.parametrize("bad", ["a/b", "x{run}"])
def test_experiment_must_be_a_plain_name(bad):
    with pytest.raises(ValueError, match="plain name"):
        load_raw_params("params-wavlm.yaml", bad)


def test_bench_defaults_to_the_runs_selected_checkpoint_and_own_folder():
    cmd = build_command(load_raw_params("params-wavlm.yaml", "si-100h"))
    assert cmd[2] == "models/selected-wavlm-large-si-100h"
    assert cmd[5] == "reports/bench/wavlm-large-si-100h"
    assert cmd[-1] == "model=wavlm,experiment=si-100h" and cmd[-2] == "wavlm-large-si-100h"


def test_bench_of_any_other_checkpoint_gets_its_own_folder():
    raw = load_raw_params("params-wavlm.yaml")
    cmd = build_command(raw, "models/wavlm-large-si-ta-200h/checkpoint-9000")
    assert cmd[5] == "reports/bench/wavlm-large-si-ta-200h-checkpoint-9000"
    assert bench_name("microsoft/wavlm-large") == "microsoft-wavlm-large"
