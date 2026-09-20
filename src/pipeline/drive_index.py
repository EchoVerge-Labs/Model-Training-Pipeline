"""Index every wav under the Drive mount: relative path + size, one TSV.

The catalog identifies a clip by (language, genre, name_in_drive, size), but
`name_in_drive` alone is NOT unique on the Drive -- the same <hash>_NNN.wav
name recurs across different source videos. The relative path
(<Lang>/<genre>/<source-dir>/<file>) is unique by construction (the
filesystem enforces it), so select.py joins catalog rows to this index to get
each clip's real path. Indexing is a separate, slow, mount-dependent step so
that selection itself stays pure and testable.
"""
import argparse
import os
from pathlib import Path

from pipeline.schema import Params

LANGUAGE_DIRS = ("Sinhala", "Tamil")
HEADER = "relpath\tsize"


def build_index(root: Path) -> list[tuple[str, int]]:
    """(relpath, size_bytes) for every .wav under root/<Sinhala|Tamil>/, sorted
    by relpath so the output is deterministic. relpath uses '/' separators and
    is relative to root."""
    root = Path(root)
    entries: list[tuple[str, int]] = []
    for lang_dir in LANGUAGE_DIRS:
        top = root / lang_dir
        if not top.is_dir():
            raise FileNotFoundError(f"expected language folder not found: {top}")
        stack = [top]
        while stack:
            current = stack.pop()
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.name.lower().endswith(".wav") and entry.is_file(follow_symlinks=False):
                        rel = os.path.relpath(entry.path, root).replace(os.sep, "/")
                        if "\t" in rel or "\n" in rel:
                            raise ValueError(f"tab/newline in path, can't index as TSV: {rel!r}")
                        entries.append((rel, entry.stat().st_size))
    entries.sort()
    return entries


def write_index(entries: list[tuple[str, int]], out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for rel, size in entries:
            f.write(f"{rel}\t{size}\n")


def load_index(path: Path) -> list[tuple[str, int]]:
    entries = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != HEADER:
            raise ValueError(f"{path}: unexpected header {header!r} (expected {HEADER!r})")
        for line in f:
            rel, size = line.rstrip("\n").split("\t")
            entries.append((rel, int(size)))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Index wav files on the Drive mount")
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    cfg = Params.from_yaml(args.config).select
    root = Path(os.path.expanduser(cfg.drive_root))
    entries = build_index(root)
    write_index(entries, Path(cfg.drive_index_path))

    total_gb = sum(s for _, s in entries) / 1e9
    per_lang = {ld: sum(1 for r, _ in entries if r.startswith(ld + "/")) for ld in LANGUAGE_DIRS}
    print(f"Indexed {len(entries):,} wavs ({total_gb:.1f} GB) under {root}: {per_lang}")
    print(f"Wrote {cfg.drive_index_path}")


if __name__ == "__main__":
    main()
