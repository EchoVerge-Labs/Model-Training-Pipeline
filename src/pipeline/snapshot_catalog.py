"""Snapshot the Google Sheet catalog to an immutable CSV."""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Optional: use gspread if available, otherwise fall back to CSV export URL
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False


REQUIRED_COLUMNS = [
    "genre", "speaking_style", "language_form", "accent_or_region",
    "code_switch", "duration_minutes", "name_in_drive", "size",
    "to_train", "tagged",
]

TABS = ["Pre-Processed-Sinhala", "Pre-Processed-Tamil"]


def snapshot_with_gspread(sheet_id: str, out_dir: Path) -> Path:
    """Pull both tabs via gspread and merge into one CSV."""
    creds_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_path:
        print("ERROR: Set GOOGLE_SERVICE_ACCOUNT_JSON env var to the service account key path")
        sys.exit(1)

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_key(sheet_id)

    all_rows = []
    for tab_name in TABS:
        ws = spreadsheet.worksheet(tab_name)
        records = ws.get_all_records()
        lang = "sinhala" if "sinhala" in tab_name.lower() else "tamil"
        for row in records:
            row["language"] = lang
            row["source_tab"] = tab_name
            all_rows.append(row)

    return _write_csv(all_rows, out_dir)


def snapshot_from_local_csv(csv_paths: list[str], out_dir: Path) -> Path:
    """Fallback: merge manually exported CSVs."""
    all_rows = []
    for csv_path in csv_paths:
        lang = "sinhala" if "sinhala" in csv_path.lower() else "tamil"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["language"] = lang
                row["source_tab"] = csv_path
                all_rows.append(row)
    return _write_csv(all_rows, out_dir)


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
    parser.add_argument("--local-csv", nargs="*",
                        help="Fallback: paths to manually exported CSVs instead of API")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.local_csv:
        snapshot_from_local_csv(args.local_csv, out_dir)
    elif args.sheet_id and HAS_GSPREAD:
        snapshot_with_gspread(args.sheet_id, out_dir)
    else:
        print("ERROR: Provide --sheet-id (with gspread installed) or --local-csv paths")
        print("Quickest path: export both tabs as CSV from Google Sheets, then:")
        print("  python -m pipeline.snapshot_catalog --local-csv sinhala.csv tamil.csv --out data/catalog")
        sys.exit(1)


if __name__ == "__main__":
    main()
