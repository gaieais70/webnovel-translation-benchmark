#!/usr/bin/env bash
# Usage: lane.sh NAME model [model ...]
# Up to three passes. Completed chapters are skipped on later passes, so
# passes 2 and 3 only retry chapters that failed.
cd "E:/Artifical_translation_index" || exit 1
name="$1"; shift
args=()
for m in "$@"; do args+=(--model "$m"); done
for pass in 1 2 3; do
  echo "=== $name pass $pass $(date -Iseconds)"
  python -u -m ati.cli run --run 2026-09 "${args[@]}"
done
echo "=== $name done $(date -Iseconds)"
