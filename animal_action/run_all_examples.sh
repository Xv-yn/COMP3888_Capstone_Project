#!/usr/bin/env bash
# Run tools/top_down_video_det.py on every image in ./examples

set -euo pipefail

# Go to the repo root (folder containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Make sure Python can import the local 'yolox' package if needed
export PYTHONPATH="$(pwd)"

# Accept common image extensions
shopt -s nullglob
files=(examples/*.{png,jpg,jpeg,bmp,webp})

if ((${#files[@]} == 0)); then
  echo "No images found in ./examples"
  exit 0
fi

for img in "${files[@]}"; do
  echo "[INFO] Processing: $img"
  python3 tools/top_down_video_det.py "$img"
done

echo "[INFO] Done."

