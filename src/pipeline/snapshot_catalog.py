"""Snapshot the Google Sheet catalog to an immutable CSV."""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Optional: use gspread if available, otherwise fall back to --local-csv
try:
    import gspread

    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False


# Columns of the actual "Sinhala X Tamil Voice Dataset" sheet tabs, minus
# `language`: every row's `language` is set from its tab/file (see
# _prepare_rows), so it isn't required of the sheet itself -- but if the sheet
# does carry one, _prepare_rows refuses rows that contradict it.
REQUIRED_COLUMNS = [
    "source_id",
    "source_url",
    "title",
    "genre",
    "speaking_style",
    "speaker_count",
    "speaker_gender",
    "acoustic_condition",
    "language_formality",
    "accent_or_region",
    "code_switching",
    "duration_minutes",
    "uploaded_date",
    "name_in_drive",
    "size",
    "stored_date",
    "to_train",
    "tagged",
]

TABS = ["Pre-Processed-Sinhala", "Pre-Processed-Tamil"]

DEFAULT_AUTHORIZED_USER_JSON = "~/.config/gspread/authorized_user.json"


def snapshot_with_gspread(sheet_id: str, out_dir: Path) -> Path:
    """Pull both tabs via gspread (OAuth installed-app flow) and merge into one CSV.

    GOOGLE_CLIENT_SECRET_JSON: the OAuth client_secret file downloaded from the
    Google Cloud console (required).
    GOOGLE_AUTH_USER_JSON: where the authorized-user token is cached after the
    first interactive sign-in (default: ~/.config/gspread/authorized_user.json).
    """
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET_JSON")
    if not client_secret:
        print("ERROR: Set GOOGLE_CLIENT_SECRET_JSON env var to the OAuth client_secret file path")
        sys.exit(1)
    authorized_user = os.path.expanduser(
        os.environ.get("GOOGLE_AUTH_USER_JSON", DEFAULT_AUTHORIZED_USER_JSON)
    )

    gc = gspread.oauth(
        scopes=gspread.auth.READONLY_SCOPES,
        credentials_filename=os.path.expanduser(client_secret),
        authorized_user_filename=authorized_user,
    )
    spreadsheet = gc.open_by_key(sheet_id)

    all_rows = []
    for tab_name in TABS:
        ws = spreadsheet.worksheet(tab_name)
        records = ws.get_all_records()
        all_rows.extend(_prepare_rows(records, _language_from_name(tab_name), tab_name))

    return _write_csv(all_rows, out_dir)


def snapshot_from_local_csv(csv_paths: list[str], out_dir: Path) -> Path:
    """Fallback: merge manually exported CSVs."""
    all_rows = []
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        all_rows.extend(_prepare_rows(records, _language_from_name(Path(csv_path).stem), csv_path))
    return _write_csv(all_rows, out_dir)


def _language_from_name(name: str) -> str:
    """Language of a tab name or exported-CSV filename stem: the LAST of
    "sinhala"/"tamil" in it. Exports are named like
    "Sinhala X Tamil Voice Dataset - Pre-Processed-Tamil", which contains both
    -- the tab it came from is the final one."""
    found = re.findall(r"sinhala|tamil", name.lower())
    if not found:
        raise ValueError(f"cannot tell the language of {name!r}: no 'sinhala' or 'tamil' in it")
    return found[-1]


def _prepare_rows(records: list[dict], lang: str, source: str) -> list[dict]:
    """Normalise one tab's rows: drop blank-named columns (the Tamil export has
    a trailing headerless one, which would make the merged CSV writer crash),
    set `language`/`source_tab`, and refuse rows whose own `language` column
    disagrees with the language we derived for the tab."""
    out, contradicted = [], Counter()
    for row in records:
        row = {k: v for k, v in row.items() if k}
        sheet_lang = str(row.get("language", "")).strip().lower()
        if sheet_lang and sheet_lang != lang:
            contradicted[sheet_lang] += 1
        row["language"] = lang
        row["source_tab"] = source
        out.append(row)
    if contradicted:
        raise ValueError(
            f"{source}: derived language {lang!r} but {sum(contradicted.values())} rows say "
            f"{dict(contradicted)} in their own `language` column"
        )
    return out


def _write_csv(rows: list[dict], out_dir: Path) -> Path:
    """Write merged CSV with timestamp and validation."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate required columns
    if rows:
        missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            print(f"ERROR: Missing required columns: {missing}")
            print(f"Available columns: {list(rows[0].keys())}")
            sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dated_path = out_dir / f"catalog_{timestamp}.csv"
    latest_path = out_dir / "catalog_latest.csv"

    fieldnames = list(rows[0].keys()) if rows else []
    with open(dated_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Atomic-ish overwrite for the latest pointer
    import shutil

    shutil.copy2(dated_path, latest_path)

    # Write metadata
    meta = {
        "timestamp": timestamp,
        "total_rows": len(rows),
        "sinhala_rows": sum(1 for r in rows if r.get("language") == "sinhala"),
        "tamil_rows": sum(1 for r in rows if r.get("language") == "tamil"),
        "to_train_rows": sum(1 for r in rows if r.get("to_train", "").lower() == "yes"),
    }
    with open(out_dir / "catalog_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Snapshot: {len(rows)} rows → {dated_path}")
    print(f"  Sinhala: {meta['sinhala_rows']}, Tamil: {meta['tamil_rows']}")
    print(f"  Marked to_train=yes: {meta['to_train_rows']}")
    return latest_path


def main():
    parser = argparse.ArgumentParser(description="Snapshot Google Sheet catalog to CSV")
    parser.add_argument("--sheet-id", help="Google Sheet ID")
    parser.add_argument("--out", default="data/catalog", help="Output directory")
    parser.add_argument(
        "--local-csv", nargs="*", help="Fallback: paths to manually exported CSVs instead of API"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.local_csv:
        snapshot_from_local_csv(args.local_csv, out_dir)
    elif args.sheet_id and HAS_GSPREAD:
        snapshot_with_gspread(args.sheet_id, out_dir)
    else:
        print("ERROR: Provide --sheet-id (with gspread installed) or --local-csv paths")
        print("Quickest path: export both tabs as CSV from Google Sheets, then:")
        print(
            "  python -m pipeline.snapshot_catalog --local-csv sinhala.csv tamil.csv --out data/catalog"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
