#!/usr/bin/env bash
# Finish the Command Code models after the 5-hour usage window reset:
# inkling-small (re-run with max_tokens) in parallel with everything else that
# is still incomplete. Completed chapters are skipped. Writes "=== ccA done"
# only when both halves finish, which is what the judge driver waits for.
cd "E:/Artifical_translation_index" || exit 1
CC="laguna-s-2.1 minimax-m3 muse-spark-1.2-contributor mimo-v2.5 qwen3.8-flash glm-5.3-flash hy3 mimo-v2.5-pro gpt-5.6-luna deepseek-v4-pro nemotron-3-ultra hy4-preview qwen3.8-27b deepseek-v4-flash glm-5.2"
args=(); for m in $CC; do args+=(--model "$m"); done
echo "=== ccA finish $(date -Iseconds)"
bash runs/logs-2026-09/lane.sh ccA-ink2 inkling-small > runs/logs-2026-09/ccA-ink2.log 2>&1 &
for pass in 1 2 3; do
  echo "=== ccA finish pass $pass $(date -Iseconds)"
  python -u -m ati.cli run --run 2026-09 "${args[@]}"
done
wait
echo "=== ccA done $(date -Iseconds)"
