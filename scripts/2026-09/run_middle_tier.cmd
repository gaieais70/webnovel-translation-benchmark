@echo off
REM Middle-tier translations for run 2026-09: glm-5.3 and kimi-k3.
REM Command Code's WEEKLY limit gates these (resets 2026-09-21 21:40 local).
REM Registered as a Scheduled Task rather than a detached process: run
REM 2026-09 launched the waiter with Start-Process twice and both times it
REM was gone by morning, so nothing ran when the window opened.
cd /d E:\Artifical_translation_index
python -u -m ati.cli run --run 2026-09 --passes 3 --wait --model glm-5.3 --model kimi-k3 >> runs\logs-2026-09\middle-tier.log 2>&1
