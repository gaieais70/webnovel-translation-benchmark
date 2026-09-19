"""Apply the run-2026-09 output guards to records written before they existed.

Only records that match a guard are touched; each change is printed.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
from ati.runstage import left_untranslated, provider_blocked

changed = 0
for f in sorted(Path("runs/2026-09").rglob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    tag = f"{d['model_id']} {d['passage_id']}"
    if "blocked_by_provider" in d:
        continue  # written by the new code already
    if provider_blocked(d.get("text") or "", d.get("error")):
        d.update(text="", refused=True, blocked_by_provider=True, error=None)
        print(f"refused (provider block): {tag}")
    elif not d.get("error") and d.get("text") and left_untranslated(d["text"]):
        d.update(text="", error="output is mostly in the source script; not a translation")
        print(f"failed (untranslated): {tag}")
    else:
        continue
    f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    changed += 1
print(f"{changed} records reclassified")
