"""Cross-judge comparison for run 2026-09: Gemini vs grok vs jev.

Method, fixed here so the published figures are reproducible: the weighted
mean of the rubric axes, each score normalised so 1 maps to 0 and 5 maps to
100. Run 2026-09's first cross-judge table was computed in a session
scratchpad that no longer exists; this reproduces its Gemini column exactly
(six axes) and is what the page now cites.

Terminology is excluded by default. jev returns no term table, and comparing
judges on axes one of them cannot score is not a comparison - so all three
judges are put on the same five-axis basis.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = "2026-09"
FIVE = ["accuracy", "register", "fluency", "fidelity", "sanitization"]
WEIGHTS = {
    "accuracy": .25, "terminology": .20, "register": .15,
    "fluency": .15, "fidelity": .15, "sanitization": .10,
}
JUDGES = {
    "gemini": "judgments",
    "grok": "judgments/secondary",
    "jev": "judgments/jev",
}


def load(root, axes):
    out = defaultdict(dict)
    directory = REPO / root / RUN
    if not directory.exists():
        return out
    for path in directory.glob("*/*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("skipped") or data.get("refused") or not data.get("scores"):
            continue
        if not all(a in data["scores"] for a in axes):
            continue
        out[data["model_id"]][f"{data['passage_id']}.{data['iteration']}"] = data
    return out


def composite(judgment, axes):
    """Weighted mean of the axes, each normalised 1->0, 5->100."""
    total = sum(WEIGHTS[a] for a in axes)
    return sum(
        WEIGHTS[a] * (float(judgment["scores"][a]) - 1) / 4 for a in axes
    ) / total * 100


def table(axes=FIVE):
    loaded = {name: load(root, axes) for name, root in JUDGES.items()}
    rows = []
    for model in sorted(loaded["gemini"]):
        row = {"model": model}
        for name, data in loaded.items():
            if model not in data:
                continue
            # Only instances both this judge and Gemini scored, so a judge
            # that covered fewer passages is not compared on a different set.
            shared = sorted(set(loaded["gemini"][model]) & set(data[model]))
            if not shared:
                continue
            row[name] = st.mean(composite(data[model][k], axes) for k in shared)
            row[f"{name}_n"] = len(shared)
        rows.append(row)
    return rows


def ceiling_share(root, axes=FIVE):
    """Share of axis scores at the top of the scale - the compression measure."""
    data = load(root, axes)
    vals = [
        float(j["scores"][a]) for m in data for j in data[m].values() for a in axes
    ]
    at_top = sum(1 for v in vals if round(v) >= 5)
    return st.mean(vals), st.pstdev(vals), at_top / len(vals) * 100, len(vals)


if __name__ == "__main__":
    rows = table()
    print(f"{'model':28s} {'gemini':>7s} {'grok':>7s} {'jev':>7s} {'jev-gem':>8s}")
    for r in sorted(rows, key=lambda r: -r.get("gemini", 0)):
        g = r.get("gemini"); k = r.get("grok"); j = r.get("jev")
        print(
            f"{r['model']:28s} {g:7.1f} "
            f"{(f'{k:7.1f}' if k is not None else '      -')} "
            f"{(f'{j:7.1f}' if j is not None else '      -')} "
            f"{(f'{j-g:+8.1f}' if j is not None else '       -')}"
        )
    print()
    for name, root in JUDGES.items():
        try:
            mu, sd, top, n = ceiling_share(root)
            print(f"{name:7s} mean {mu:.2f}  sd {sd:.2f}  at ceiling {top:5.1f}%  (n={n})")
        except (ZeroDivisionError, StatisticsError := Exception):
            print(f"{name:7s} no judgments")
