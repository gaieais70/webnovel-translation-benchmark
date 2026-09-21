@echo off
REM Middle tier for run 2026-09 (glm-5.3, kimi-k3), end to end: translate,
REM rubric-judge, compare head to head, report.
REM
REM Command Code's WEEKLY limit gates the translations (resets 2026-09-21
REM 21:40 local). Registered as a Scheduled Task rather than a detached
REM process: run 2026-09 launched the waiter with Start-Process twice and both
REM times it was gone by morning.
REM
REM Every Gemini stage is scoped. judge: --model on the two new rows, which
REM only works since judge_run honours it. pairwise: the pool is the 23
REM published subjects plus the two new rows, generated from the roster -
REM without it, the unjudged Claude-column rows are also "least compared" and
REM would take budget. select_pairs divides by prior comparisons, so pairs
REM involving a new row come first. +80 comparisons (525 total) gives each new
REM row about forty, in line with its neighbours (34-44), so its interval is
REM comparable. Every agy call still goes through CallGate, and --wait waits
REM out a stated reset rather than knocking.
cd /d E:\Artifical_translation_index
set LOG=runs\logs-2026-09\middle-tier.log

echo ==== translations %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli run --run 2026-09 --passes 3 --wait --model glm-5.3 --model kimi-k3 >> %LOG% 2>&1

echo ==== rubric judging %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli judge --run 2026-09 --wait --model glm-5.3 --model kimi-k3 >> %LOG% 2>&1

echo ==== head to head %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli pairwise --run 2026-09 --wait --target 525 --max-comparisons 10 --model laguna-s-2.1 --model minimax-m3 --model muse-spark-1.2-contributor --model muse-spark-1.3-contributor --model deepseek-v4.1-flash --model gemini-3.8-flash --model mimo-v2.5 --model qwen3.8-flash --model glm-5.3-flash --model hy3 --model mimo-v2.5-pro --model gpt-5.6-luna --model inkling-small --model deepseek-v4-pro --model nemotron-3-ultra --model hy4-preview --model qwen3.8-27b --model deepseek-v4-flash --model glm-5.2 --model glm-5.3 --model solar-pro4 --model nemotron-3-super-120b --model nemotron-3.5-lightning --model dots-3-note-preview --model kimi-k3 >> %LOG% 2>&1

echo ==== report %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli report --run 2026-09 >> %LOG% 2>&1

echo ==== done %DATE% %TIME% ==== >> %LOG%
