#!/usr/bin/env bash
set -euo pipefail

# Pull selected files from Google Drive to local disk
# Reads the train + dev manifests and extracts filenames

MANIFEST_DIR="data/manifests"
RAW_DIR="data/raw"
DRIVE_REMOTE="gdrive:Pre Processed Data"

mkdir -p "$RAW_DIR" logs

echo "Extracting file list from manifests..."
# Extract filenames from JSONL manifests
python3 -c "
import json, sys
files = set()
for manifest in ['${MANIFEST_DIR}/train.jsonl', '${MANIFEST_DIR}/dev.jsonl']:
    with open(manifest) as f:
        for line in f:
            entry = json.loads(line)
            files.add(entry['filename'])
for f in sorted(files):
    print(f)
" > /tmp/rclone_filelist.txt

FILE_COUNT=$(wc -l < /tmp/rclone_filelist.txt)
echo "Pulling $FILE_COUNT files from Drive..."

# Pull Sinhala files
rclone copy "$DRIVE_REMOTE/Sinhala" "$RAW_DIR/" \
    --files-from /tmp/rclone_filelist.txt \
    --transfers 16 --checkers 32 --fast-list \
    --stats 30s --stats-one-line \
    --log-file logs/rclone_sinhala.log \
    2>&1 || true

# Pull Tamil files
rclone copy "$DRIVE_REMOTE/Tamil" "$RAW_DIR/" \
    --files-from /tmp/rclone_filelist.txt \
    --transfers 16 --checkers 32 --fast-list \
    --stats 30s --stats-one-line \
    --log-file logs/rclone_tamil.log \
    2>&1 || true

# Verify
PULLED=$(find "$RAW_DIR" -name "*.wav" | wc -l)
echo "Pulled $PULLED files (expected $FILE_COUNT)"

if [ "$PULLED" -lt "$((FILE_COUNT * 95 / 100))" ]; then
    echo "ERROR: Missing >5% of files. Check rclone logs."
    exit 1
fi

echo "Materialisation complete: $PULLED files in $RAW_DIR"
