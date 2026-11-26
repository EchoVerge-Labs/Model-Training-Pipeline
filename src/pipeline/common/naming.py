"""One name per run, derived from its config.

    <model>-<method>-<languages>-<hours>h[-cnn][-p<N>][-homog][-<tag>]

  model      the model's name on the SLSB leaderboard (wavlm-large, xlsr300m, ...)
  method     copt | adapter | lora | gclora                    (pretrain.pecopt.method)
  languages  si-ta                                            (select.language_mix)
  hours      hours of audio the run trains on                  (select.target_hours)
  cnn        the CNN feature encoder trains                    (freeze_feature_encoder: false)
  p<N>       passes over the data (whole passes), only when not the standard 30
  homog      language-homogeneous updates                      (pretrain.language_batching)
  tag        free text for anything the tokens above do not show (e.g. smoke)

The name is the run's folder under models/, reports/ and logs/, its ssl-pretraining run on
DagsHub and its SLSB leaderboard row. Data shared between runs is named by its data id,
`<languages>-<hours>h[-<tag>]`. Nothing here is typed by hand, so a name cannot claim a
setting the config does not have: an overlay that changes a setting no token shows needs a
`tag` (pipeline.common.schema.load_raw_params), and run.json records the config the name
stands for (pipeline.common.tracking.check_run_folder).
"""

import hashlib
import json
import re

# Order is the order in names (si-ta, never ta-si).
LANGUAGE_CODES = {"sinhala": "si", "tamil": "ta"}
METHOD_TOKENS = {
    "full_copt": "copt",
    "residual_adapter": "adapter",
    "lora": "lora",
    "gc_lora": "gclora",
}
STANDARD_PASSES = 30
TAG_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def hours_str(hours: float) -> str:
    hours = round(float(hours), 3)
    return str(int(hours)) if hours.is_integer() else f"{hours:g}"


def check_tag(tag: str | None) -> str | None:
    if tag is not None and not TAG_RE.fullmatch(tag):
        raise ValueError(f"tag {tag!r} must be lower-case letters, digits, '.' and '-'")
    return tag


def languages_token(languages) -> str:
    unknown = sorted(set(languages) - set(LANGUAGE_CODES))
    if unknown:
        raise ValueError(f"no language code for {unknown}; known: {sorted(LANGUAGE_CODES)}")
    return "-".join(code for lang, code in LANGUAGE_CODES.items() if lang in languages)


def data_id(languages, hours: float, data_tag: str | None = None) -> str:
    """Name of a selection (manifests, shards, labels): si-ta-200h, si-ta-1h-smoke."""
    return "-".join(
        [languages_token(languages), f"{hours_str(hours)}h"] + ([data_tag] if data_tag else [])
    )


def passes(max_updates: int, target_batch_seconds: float, hours: float) -> float:
    """Passes over the training audio: updates x seconds per update / seconds of audio."""
    return max_updates * target_batch_seconds / (hours * 3600)


def passes_token(n_passes: float) -> str | None:
    """p<N> in whole passes (one decimal below 1), or None at the standard 30 -- so 68 updates
    over 1 h (30.2 passes) is a standard run, not p30.2."""
    value = round(n_passes) if n_passes >= 1 else round(n_passes, 1)
    if value == STANDARD_PASSES:
        return None
    return f"p{value:g}"


def run_name(
    model: str,
    method: str,
    languages,
    hours: float,
    cnn_trains: bool,
    n_passes: float,
    homogeneous: bool,
    tag: str | None = None,
) -> str:
    tokens = [model, METHOD_TOKENS[method], languages_token(languages), f"{hours_str(hours)}h"]
    if cnn_trains:
        tokens.append("cnn")
    p = passes_token(n_passes)
    if p:
        tokens.append(p)
    if homogeneous:
        tokens.append("homog")
    if tag:
        tokens.append(tag)
    return "-".join(tokens)


def fingerprint(config: dict) -> str:
    """Stable short hash of a JSON-able config."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def changed_keys(before, after, prefix: str = "") -> list[str]:
    """Dotted keys whose value differs between two nested dicts."""
    if isinstance(before, dict) and isinstance(after, dict):
        out = []
        for key in sorted(set(before) | set(after)):
            out += changed_keys(before.get(key), after.get(key), f"{prefix}{key}.")
        return out
    return [] if before == after else [prefix.rstrip(".")]


def main(argv=None) -> None:
    """Print a run's name (or with --frozen, its model's frozen baseline name), for make and
    tmux: python -m pipeline.common.naming --config params-wavlm.yaml --variant ... --tag ..."""
    import argparse

    from pipeline.common.schema import add_run_args, load_params, variants_arg

    parser = argparse.ArgumentParser(description=main.__doc__)
    add_run_args(parser)
    parser.add_argument("--frozen", action="store_true", help="the frozen baseline's name")
    parser.add_argument("--data-id", action="store_true", help="the run's data id instead")
    args = parser.parse_args(argv)
    _, params = load_params(args.config, variants_arg(args.variant), args.tag)
    if args.frozen:
        from pipeline.eval.benchmark import frozen_name

        print(frozen_name(params.model_key))
    elif args.data_id:
        print(params.data_id)
    else:
        print(params.run)


if __name__ == "__main__":
    main()
