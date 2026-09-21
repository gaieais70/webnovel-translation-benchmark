@echo off
REM Middle-tier translations for run 2026-09: glm-5.3 and kimi-k3, then their
REM judging.
REM
REM Command Code's WEEKLY limit gates the translations (resets 2026-09-21
REM 21:40 local). Registered as a Scheduled Task rather than a detached
REM process: run 2026-09 launched the waiter with Start-Process twice and both
REM times it was gone by morning, so nothing ran when the window opened.
REM
REM The judge pass is scoped with --model to these two rows. That scoping only
REM works because judge_run honours `models`; before that fix --model was
REM accepted and ignored here, and an unscoped pass would have spent the
REM operator's own Gemini quota judging every unjudged output in the run
REM (eighteen Claude-column outputs it was never asked to touch). Every agy
REM call still goes through CallGate, and --wait waits out a stated reset
REM rather than knocking.
cd /d E:\Artifical_translation_index

echo ==== translations %DATE% %TIME% ==== >> runs\logs-2026-09\middle-tier.log
python -u -m ati.cli run --run 2026-09 --passes 3 --wait --model glm-5.3 --model kimi-k3 >> runs\logs-2026-09\middle-tier.log 2>&1

echo ==== judging %DATE% %TIME% ==== >> runs\logs-2026-09\middle-tier.log
python -u -m ati.cli judge --run 2026-09 --wait --model glm-5.3 --model kimi-k3 >> runs\logs-2026-09\middle-tier.log 2>&1

echo ==== done %DATE% %TIME% ==== >> runs\logs-2026-09\middle-tier.log
