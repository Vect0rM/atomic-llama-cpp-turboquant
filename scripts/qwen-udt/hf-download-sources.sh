#!/usr/bin/env bash
# Download BF16 shards + imatrix + reference UD-Q4_K_XL.gguf from Unsloth HF repos.
#
# Usage:
#   bash scripts/qwen-udt/hf-download-sources.sh [DEST_DIR]
#
# Default DEST_DIR: ${REPO}/.scratch/qwen-ud-sources with per-model subdirs 27b/ and 35a3b/
#
# Requires: huggingface-cli (pip install -U "huggingface_hub[cli]") and HF read access.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-${ROOT}/.scratch/qwen-ud-sources}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "error: huggingface-cli not found (pip install -U \"huggingface_hub[cli]\")" >&2
  exit 1
fi

mkdir -p "$DEST/27b" "$DEST/35a3b"

echo "info: 27B — imatrix..."
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF imatrix_unsloth.gguf_file \
  --local-dir "$DEST/27b" --local-dir-use-symlinks False
echo "info: 27B — reference quant..."
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --local-dir "$DEST/27b" --local-dir-use-symlinks False
echo "info: 27B — BF16 shards..."
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF --include "BF16/*" \
  --local-dir "$DEST/27b" --local-dir-use-symlinks False

echo "info: 35B-A3B — imatrix..."
huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF imatrix_unsloth.gguf_file \
  --local-dir "$DEST/35a3b" --local-dir-use-symlinks False
echo "info: 35B-A3B — reference quant..."
huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --local-dir "$DEST/35a3b" --local-dir-use-symlinks False
echo "info: 35B-A3B — BF16 shards..."
huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF --include "BF16/*" \
  --local-dir "$DEST/35a3b" --local-dir-use-symlinks False

echo "ok: sources under $DEST/{27b,35a3b}"
