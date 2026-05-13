#!/usr/bin/env bash
# Run scripts/bench-matrix-qwen.sh for each UDT GGUF in a directory (combined MTP files).
#
# Usage:
#   QWEN_UDT_BENCH_DIR=/path/to/quants ./scripts/bench-qwen-udt-matrix-local.sh
# Env:
#   QWEN_UDT_BENCH_DIR   directory containing *.gguf (default: ${ROOT}/.scratch/qwen-udt-quants)
#   BENCH_MATRIX_MD      append all summaries to this markdown file
#   QWEN_UDT_BENCH_TAG   optional filter substring (e.g. "-V1" or "Q4_K_XL")

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${QWEN_UDT_BENCH_DIR:-${ROOT}/.scratch/qwen-udt-quants}"
TAG="${QWEN_UDT_BENCH_TAG:-}"
OUT_MD="${BENCH_MATRIX_MD:-}"

if [[ ! -d "$DIR" ]]; then
  echo "error: directory not found: $DIR" >&2
  exit 1
fi

shopt -s nullglob
mapfile -t FILES < <(find "$DIR" -maxdepth 1 -name '*.gguf' -print | sort)
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "error: no .gguf under $DIR" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  base=$(basename "$f")
  [[ -n "$TAG" && "$base" != *"$TAG"* ]] && continue
  echo "" >&2
  echo "===== bench: $base =====" >&2
  if [[ "$base" == *"35B-A3B"* ]] || [[ "$base" == *"35B-A3b"* ]]; then
    export BENCH_QWEN_MODELS=35
    export QWEN35_MTP="$f" QWEN35_BASE="$f"
  elif [[ "$base" == *"27B"* ]]; then
    export BENCH_QWEN_MODELS=27
    export QWEN27_MTP="$f" QWEN27_BASE="$f"
  else
    echo "skip (unknown model prefix): $base" >&2
    continue
  fi
  export QWEN_BENCH_COMBINED_GGUF_ONLY=1
  export BENCH_LABEL="### UDT bench: ${base}"
  if [[ -n "$OUT_MD" ]]; then
    export BENCH_MATRIX_MD="$OUT_MD"
  else
    unset BENCH_MATRIX_MD || true
  fi
  bash "${ROOT}/scripts/bench-matrix-qwen.sh"
done
