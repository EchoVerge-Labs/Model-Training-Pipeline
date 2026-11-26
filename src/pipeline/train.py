"""Entry point for continued pre-training of any of the five models.

The model's objective family decides which trainer runs: masked prediction on k-means
cluster ids (pipeline.training.masked_prediction) for mHuBERT-147, HuBERT Large and WavLM
Large, wav2vec 2.0 contrastive learning (pipeline.training.contrastive) for XLS-R and
wav2vec2 Large. The
model comes from `model_key` in the params file, or from --model, which selects that
model's params file from the registry.

Usage:
  torchrun --nproc_per_node=1 -m pipeline.train --model wavlm
  torchrun --nproc_per_node=1 -m pipeline.train --config params-xlsr.yaml
  torchrun ... -m pipeline.train --config params-xlsr.yaml --resume-from models/.../checkpoint-3000

Every other argument is handed to the family trainer unchanged, so a family's own
options (e.g. the contrastive trainer's --resume-from / --no-resume) work as before.
"""

import argparse
import signal
import sys

from pipeline.common.schema import Params
from pipeline.common.tracking import end_active_run
from pipeline.models.registry import MODEL_REGISTRY, get_model_spec


def resolve_config(model: str | None, config: str | None) -> str:
    """The params file to train from; --model alone means that model's own file."""
    if config is not None:
        return config
    return get_model_spec(model).params_file if model else "params-mhubert147.yaml"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="params file (default: --model's own, else params-mhubert147.yaml)",
    )
    args, rest = parser.parse_known_args(argv)

    config = resolve_config(args.model, args.config)
    params = Params.from_yaml(config)
    if args.model and params.model_key != args.model:
        raise SystemExit(f"--model {args.model} but {config} is for model_key={params.model_key}")

    family = get_model_spec(params.model_key).family
    if family == "masked_prediction":
        from pipeline.training.masked_prediction import main as family_main
    else:
        from pipeline.training.contrastive import main as family_main

    sys.argv = [sys.argv[0], "--config", config, *rest]
    # torchrun stops a worker with SIGTERM: turn it into an exception so the run's MLflow run
    # is ended KILLED (a crash ends it FAILED) instead of being left RUNNING, or being marked
    # FINISHED by MLflow's exit hook. pipeline.common.tracking ends a completed run FINISHED.
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        family_main()
    except (KeyboardInterrupt, SystemExit) as e:
        if not (isinstance(e, SystemExit) and e.code in (0, None)):
            end_active_run("KILLED")
        raise
    except BaseException:
        # A SIGTERM reaches the DataLoader workers too, so the error that surfaces can be
        # theirs ("worker killed by signal") rather than our SystemExit: still a kill.
        end_active_run("KILLED" if _signalled else "FAILED")
        raise


_signalled = False


def _raise_on_sigterm(signum, frame):
    global _signalled
    _signalled = True
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    main()
