"""Copy the manifest's clips from the Drive mount to local disk (data/raw/).

The manifests list each clip's path relative to the Drive root
(<Lang>/<genre>/<source-dir>/<file>); this copies each one to the same
relative path under data/raw/, so nothing can collide and shard.py finds it
at data/raw/<filename>. The Drive is already mounted locally, so this is a
plain file copy -- no rclone API calls.

Idempotent and resumable: a destination whose size already matches the source
is skipped, and copies land via a .part file so a crash never leaves a
truncated file that looks complete. Exits non-zero if any file is missing or
fails, since the paths come from an index of this same mount.
"""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.schema import Params

MANIFESTS = ("data/manifests/train.jsonl", "data/manifests/dev.jsonl")


def read_filenames(manifest_paths) -> list[str]:
    """Sorted, de-duplicated manifest `filename` values (paths under drive_root)."""
    names = set()
    for manifest in manifest_paths:
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                names.add(json.loads(line)["filename"])
    return sorted(names)


def copy_one(src: Path, dst: Path) -> tuple[str, int]:
    """Returns (status, bytes): copied | skipped | missing. Raises on a bad copy."""
    if not src.exists():
        return "missing", 0
    size = src.stat().st_size
    if dst.exists() and dst.stat().st_size == size:
        return "skipped", size
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, part)
    if part.stat().st_size != size:
        part.unlink()
        raise OSError(f"size mismatch after copying {src}")
    os.replace(part, dst)
    return "copied", size


def materialise(filenames: list[str], drive_root: Path, raw_dir: Path, workers: int = 16) -> dict:
    counts = {"copied": 0, "skipped": 0, "missing": 0, "failed": 0}
    bytes_by = {"copied": 0, "skipped": 0}
    missing, failed = [], []

    def work(name: str):
        try:
            return name, *copy_one(drive_root / name, raw_dir / name)
        except OSError as e:  # keep going; every failure is reported at the end
            return name, "failed", 0, str(e)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(work, filenames), start=1):
            name, status, size, *err = result
            counts[status] += 1
            if status in bytes_by:
                bytes_by[status] += size
            elif status == "missing":
                missing.append(name)
            else:
                failed.append({"file": name, "error": err[0]})
            if i % 2000 == 0 or i == len(filenames):
                done_gb = (bytes_by["copied"] + bytes_by["skipped"]) / 1e9
                print(f"  {i:,}/{len(filenames):,} files ({done_gb:.1f} GB)  {counts}", flush=True)

    return {
        "expected_files": len(filenames),
        **counts,
        "copied_gb": round(bytes_by["copied"] / 1e9, 2),
        "total_gb_on_disk": round((bytes_by["copied"] + bytes_by["skipped"]) / 1e9, 2),
        "missing_files": missing,
        "failed_files": failed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--report", default="reports/materialise.json")
    args = parser.parse_args()

    cfg = Params.from_yaml(args.config).select
    drive_root = Path(os.path.expanduser(cfg.drive_root))
    filenames = read_filenames(MANIFESTS)
    print(f"Copying {len(filenames):,} files from {drive_root} to {args.raw_dir}/ ...")

    report = materialise(filenames, drive_root, Path(args.raw_dir), args.workers)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"Copied {report['copied']:,} ({report['copied_gb']} GB), skipped {report['skipped']:,} "
        f"already-present, missing {report['missing']}, failed {report['failed']}. "
        f"{report['total_gb_on_disk']} GB now under {args.raw_dir}/. Report: {args.report}"
    )
    if report["missing"] or report["failed"]:
        print("ERROR: some files were not copied -- see the report; re-run to resume.")
        sys.exit(1)


if __name__ == "__main__":
    main()
