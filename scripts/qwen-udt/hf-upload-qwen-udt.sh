#!/usr/bin/env bash
# Upload release GGUFs to Hugging Face (AtomicChat org). Requires: huggingface-cli login.
#
# Usage:
#   export HF_TOKEN=...   # or rely on cached login
#   bash scripts/qwen-udt/hf-upload-qwen-udt.sh /path/to/local/quants/dir
#
# Repos (create empty repos + model cards in the HF UI first if needed):
#   AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF
#   AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${1:-${ROOT}/.scratch/qwen-udt-quants}"

if [[ ! -d "$DIR" ]]; then
  echo "error: not a directory: $DIR" >&2
  exit 1
fi

upload_one() {
  local f="$1"
  local base
  base=$(basename "$f")
  if [[ "$base" == Qwen3.6-27B-* ]]; then
    huggingface-cli upload AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF "$f" --repo-type model --commit-message "Add ${base}"
  elif [[ "$base" == Qwen3.6-35B-A3B-* ]]; then
    huggingface-cli upload AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF "$f" --repo-type model --commit-message "Add ${base}"
  else
    echo "skip: $base"
  fi
}

shopt -s nullglob
for f in "$DIR"/Qwen3.6-27B-UDT-*.gguf "$DIR"/Qwen3.6-35B-A3B-UDT-*.gguf; do
  [[ -f "$f" ]] || continue
  upload_one "$f"
done
shopt -u nullglob

echo "ok: upload pass complete (large files may take hours)."
