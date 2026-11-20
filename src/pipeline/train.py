"""Continued pre-training entry point: runs the trainer of the chosen model.

  # XLS-R (what params.yaml is)
  torchrun --nproc_per_node=1 -m pipeline.train --config params.yaml

  # any other model: its configs/models/<name>.yaml is merged over params.yaml
  torchrun --nproc_per_node=1 -m pipeline.train --model wavlm

Models are xlsr, wav2vec2, wavlm, hubert and mhubert (see pipeline.models.registry);
the trainer is the model config's `model.trainer`. Each trainer lives in
pipeline/models/<trainer>/trainer.py; what they share lives in pipeline/common/.

Resuming: relaunching the SAME command after a crash resumes automatically from the
newest complete checkpoint in pretrain.output_dir; pass --resume-from CKPT_DIR to pick
a specific one, or --no-resume to start over.
"""

import argparse

from pipeline.common.config import add_config_args, params_from_args
from pipeline.models.registry import MODELS, trainer_module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_config_args(parser, model_choices=sorted(MODELS))
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume-from",
        metavar="CKPT_DIR",
        default=None,
        help="resume from this checkpoint directory (e.g. models/.../checkpoint-3000)",
    )
    resume.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore checkpoints already in pretrain.output_dir and start from the base model "
        "(by default the newest complete checkpoint there is resumed automatically)",
    )
    args = parser.parse_args(argv)

    params = params_from_args(args)
    trainer = trainer_module(params.model.trainer)
    return trainer.train(
        params,
        resume_from=args.resume_from,
        auto_resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
