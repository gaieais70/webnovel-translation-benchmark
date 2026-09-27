@echo off
REM Follow-up to run_middle_tier.cmd for run 2026-09: put the Claude column
REM (opus-5, sonnet-5, haiku-4.5) into the head-to-head pool, then rebuild the
REM report. Their rubric judgments run separately (`ati judge --model ...`).
REM
REM Waits for the middle-tier task first and runs only if it succeeded: both
REM spend Gemini quota, and a failed middle tier needs looking at before
REM anything else is added. Not appended to that script because it was already
REM running, and cmd.exe reads a batch file from disk as it goes.
REM
REM +120 comparisons (525 -> 645) gives each Claude row about forty, in line
REM with the rest; select_pairs puts the least-compared rows first. These rows
REM ran through the local Claude Code CLI, a different route from every other
REM row, so they are not directly comparable to the API rows (run notes §14).
cd /d E:\Artifical_translation_index
set LOG=runs\logs-2026-09\claude-pairs.log

echo ==== waiting for middle tier %DATE% %TIME% ==== >> %LOG%
REM Start-Sleep, not timeout: launched from git-bash, PATH finds coreutils
REM `timeout`, which rejects /t at once - on 2026-09-26 the wait spun for two
REM hours and wrote 5.5 MB of usage errors instead of sleeping. Windows'
REM timeout.exe by full path is no better: with no console on stdin it exits
REM at once ("Input redirection is not supported").
:wait
schtasks /Query /TN ATI-middle-tier-2026-09 /FO LIST | findstr /C:"Running" >nul
if not errorlevel 1 (powershell -NoProfile -Command "Start-Sleep -Seconds 300" & goto :wait)
REM The task's own result, not the log: middle-tier.log already holds a
REM "==== done" line from an earlier attempt that stopped at translations.
powershell -NoProfile -Command "exit (Get-ScheduledTaskInfo ATI-middle-tier-2026-09).LastTaskResult"
if not "%ERRORLEVEL%"=="0" (set STEP=middle tier did not succeed& set RC=1& goto :stopped)

echo ==== head to head %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli pairwise --run 2026-09 --wait --target 645 --max-comparisons 10 --model laguna-s-2.1 --model minimax-m3 --model muse-spark-1.2-contributor --model muse-spark-1.3-contributor --model deepseek-v4.1-flash --model gemini-3.8-flash --model mimo-v2.5 --model qwen3.8-flash --model glm-5.3-flash --model hy3 --model mimo-v2.5-pro --model gpt-5.6-luna --model inkling-small --model deepseek-v4-pro --model nemotron-3-ultra --model hy4-preview --model qwen3.8-27b --model deepseek-v4-flash --model glm-5.2 --model glm-5.3 --model solar-pro4 --model nemotron-3-super-120b --model nemotron-3.5-lightning --model dots-3-note-preview --model kimi-k3 --model opus-5 --model sonnet-5 --model haiku-4.5 >> %LOG% 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (set STEP=head to head& goto :stopped)

echo ==== report %DATE% %TIME% ==== >> %LOG%
python -u -m ati.cli report --run 2026-09 >> %LOG% 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (set STEP=report& goto :stopped)

echo ==== done %DATE% %TIME% ==== >> %LOG%
exit /b 0

:stopped
echo ==== stopped: %STEP% exited %RC% %DATE% %TIME% ==== >> %LOG%
exit /b 1
