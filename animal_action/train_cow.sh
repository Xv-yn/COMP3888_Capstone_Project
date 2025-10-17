#!/usr/bin/env bash
# Run YOLOX cow training from the repo root with the venv.
# Usage: ./train_cow.sh [--ckpt path] [--exp exps/default/yolox_s_cow.py] [--batch 8] [--devices 1]
# You can also override via env vars: CKPT, EXP_FILE, BATCH, DEVICES.

set -euo pipefail

# --- configurable defaults ---
EXP_FILE="${EXP_FILE:-exps/default/yolox_s_cow.py}"
CKPT="${CKPT:-YOLOX_outputs/cow_finetune/best_ckpt.pth}"
BATCH="${BATCH:-8}"
DEVICES="${DEVICES:-1}"

# --- parse simple flags (optional) ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)    CKPT="$2"; shift 2 ;;
    --exp)     EXP_FILE="$2"; shift 2 ;;
    --batch)   BATCH="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    -h|--help)
      grep -E '^# (Usage|Run YOLOX)' "$0" | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# --- move to repo root (folder containing this script) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- ensure we’re in the right project directory ---
if [[ ! -f tools/train.py ]]; then
  echo "Error: tools/train.py not found. Run this from the repo root." >&2
  exit 1
fi

# --- activate venv if present ---
if [[ -d "venv" && -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

# --- show versions (helpful for debugging) ---
echo "[INFO] Python: $(command -v python3)"
python3 -V
pip -V || true

# --- set PYTHONPATH to repo root so 'yolox' is importable ---
export PYTHONPATH="$(pwd)"

# --- run training ---
echo "[INFO] Starting training..."
python3 tools/train.py \
  -f "$EXP_FILE" \
  -d "$DEVICES" \
  -b "$BATCH" \
  --fp16 \
  -o \
  -c "$CKPT"

