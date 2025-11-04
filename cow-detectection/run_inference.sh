
#!/usr/bin/env bash
# run_inference.sh
# Usage:
#   ./run_inference.sh <stage: 1|2|3> <path/to/image> [--show-skeleton|--no-show-skeleton]

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <stage: 1|2|3> <path/to/image> [--show-skeleton|--no-show-skeleton]"
  exit 1
fi

STAGE="$1"
IMG_INPUT="$2"

# Optional skeleton flag (must be the last arg if present)
SKELETON_FLAG=""
if [[ $# -eq 3 ]]; then
  case "$3" in
    --show-skeleton|--no-show-skeleton)
      SKELETON_FLAG="$3"
      ;;
    *)
      echo "Error: third argument must be --show-skeleton or --no-show-skeleton"
      exit 1
      ;;
  esac
fi

# Use venv's python if active; otherwise system python
PYBIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYBIN="${PYBIN:-python}"

# Resolve image path to an absolute path BEFORE changing directories
# Works on Linux; if `realpath` isn't available, fall back to Python.
if command -v realpath >/dev/null 2>&1; then
  IMG_ABS="$(realpath "$IMG_INPUT")"
else
  IMG_ABS="$("$PYBIN" - <<PY
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
"$IMG_INPUT")"
fi

# Auto-select device (prefer CUDA)
DEVICE="$("$PYBIN" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"

# Repo root (this script's directory)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run exactly like the working example — inside modeling/
cd "$ROOT/cow_detectection/modeling"

# Exec the script with absolute image path so cv2.imread always finds it
if [[ -n "$SKELETON_FLAG" ]]; then
  exec "$PYBIN" predict.py \
    --option "$STAGE" \
    --image-path "$IMG_ABS" \
    --device "$DEVICE" \
    "$SKELETON_FLAG"
else
  exec "$PYBIN" predict.py \
    --option "$STAGE" \
    --image-path "$IMG_ABS" \
    --device "$DEVICE"
fi

