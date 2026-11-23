"""Entry point for continued pre-training of any of the five models.

The model's objective family decides which trainer runs: masked prediction on k-means
cluster ids (train_masked_prediction) for mHuBERT-147, HuBERT Large and WavLM Large,
wav2vec 2.0 contrastive learning (train_contrastive) for XLS-R and wav2vec2 Large. The
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
import sys

from pipeline.model_registry import MODEL_REGISTRY, get_model_spec
from pipeline.schema import Params


def resolve_config(model: str | None, config: str | None) -> str:
    """The params file to train from; --model alone means that model's own file."""
    if config is not None:
        return config
    return get_model_spec(model).params_file if model else "params.yaml"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument(
        "--config", default=None, help="params file (default: --model's own, else params.yaml)"
    )
    args, rest = parser.parse_known_args(argv)

    config = resolve_config(args.model, args.config)
    params = Params.from_yaml(config)
    if args.model and params.model_key != args.model:
        raise SystemExit(f"--model {args.model} but {config} is for model_key={params.model_key}")

    family = get_model_spec(params.model_key).family
    if family == "masked_prediction":
        from pipeline.train_masked_prediction import main as family_main
    else:
        from pipeline.train_contrastive import main as family_main

    sys.argv = [sys.argv[0], "--config", config, *rest]
    family_main()


if __name__ == "__main__":
    main()
