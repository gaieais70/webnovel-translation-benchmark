"""Per-model progress for run 2026-09: done / failed / pending, plus errors."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
RUN = Path("runs/2026-09")
CHAPTERS = 11
rows = defaultdict(lambda: {"ok": 0, "ref": 0, "err": [], "lat": [], "ratio": [], "route": set()})
for f in sorted(RUN.rglob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    r = rows[d["model_id"]]
    ch = d["passage_id"].rsplit("-", 1)[-1]
    if d.get("error") or d.get("unavailable"):
        r["err"].append(f"{ch}: {str(d.get('error'))[:110]}")
    elif d.get("refused"):
        r["ref"] += 1
    else:
        r["ok"] += 1
        r["lat"].append(d.get("latency_s") or 0)
        if d.get("source_chars"):
            r["ratio"].append(len(d.get("text") or "") / d["source_chars"])
    if d.get("route"):
        r["route"].add(d["route"])
for m, r in sorted(rows.items()):
    lat = sum(r["lat"]) / len(r["lat"]) if r["lat"] else 0
    ratio = sum(r["ratio"]) / len(r["ratio"]) if r["ratio"] else 0
    print(f"{m:28} ok {r['ok']:2}/{CHAPTERS}  ref {r['ref']}  err {len(r['err']):2}  "
          f"avg {lat:6.0f}s  x{ratio:.2f}  {','.join(sorted(r['route']))}")
    for e in r["err"]:
        print(f"    {e}")
