@echo off
REM Middle tier for run 2026-09 (glm-5.3, kimi-k3), end to end: translate,
REM rubric-judge, compare head to head, report.
REM
REM Command Code's WEEKLY limit gates the translations (resets 2026-09-21
REM 21:40 local). Registered as a Scheduled Task rather than a detached
REM process: run 2026-09 launched the waiter with Start-Process twice and both
REM times it was gone by morning.
REM
REM The task launches this through a hidden PowerShell host (-WindowStyle
REM Hidden), not directly. Run directly, a .cmd opens a visible console in the
REM operator's session, and closing that window ends the run: the first launch
REM died that way at 23:47 on 2026-09-21 (0xC000013A), seven chapters in. To
REM stop it on purpose:  schtasks /End /TN ATI-middle-tier-2026-09
REM
REM Each step runs only if the one before exited 0. On 2026-09-22 at 02:05 the
REM translation step was killed mid-sleep when the session ended (0x40010004),
REM and without this the script went straight on to judging, head to head and
REM the report. They died with it, so nothing was spent - but in an ordinary
REM failure they would have put Gemini quota into a half-translated pool. The
REM test compares with 0 rather than `if errorlevel 1`, which is false for the
REM negative codes a killed process leaves (0xC000013A is -1073741510).
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
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (set STEP=translations& goto :stopped)

echo ==== rubric judging %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli judge --run 2026-09 --wait --model glm-5.3 --model kimi-k3 >> %LOG% 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (set STEP=rubric judging& goto :stopped)

echo ==== head to head %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli pairwise --run 2026-09 --wait --target 525 --max-comparisons 10 --model laguna-s-2.1 --model minimax-m3 --model muse-spark-1.2-contributor --model muse-spark-1.3-contributor --model deepseek-v4.1-flash --model gemini-3.8-flash --model mimo-v2.5 --model qwen3.8-flash --model glm-5.3-flash --model hy3 --model mimo-v2.5-pro --model gpt-5.6-luna --model inkling-small --model deepseek-v4-pro --model nemotron-3-ultra --model hy4-preview --model qwen3.8-27b --model deepseek-v4-flash --model glm-5.2 --model glm-5.3 --model solar-pro4 --model nemotron-3-super-120b --model nemotron-3.5-lightning --model dots-3-note-preview --model kimi-k3 >> %LOG% 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (set STEP=head to head& goto :stopped)

echo ==== report %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli report --run 2026-09 >> %LOG% 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (set STEP=report& goto :stopped)

echo ==== done %DATE% %TIME% ==== >> %LOG%
exit /b 0

:stopped
echo ==== stopped: %STEP% exited %RC%, later steps not run %DATE% %TIME% ==== >> %LOG%
exit /b 1
