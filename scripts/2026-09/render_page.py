"""Render the leaderboard page's data tables from a run's report output.

The published page (claude.ai artifact "Mount Hua Translation Index") carries
two tables built from data: the leaderboard (rubric composite plus the
Bradley-Terry head-to-head rating) and the per-axis heat grid. Adding a model
refits every rating, so the tables are regenerated whole rather than patched
row by row.

Ratings come from the same calls `ati report` makes - fit, to_elo anchored on
the weak anchor, bootstrap_ci with its fixed seed, pairwise_records - so the
page and results/leaderboard.md cannot disagree.

    python scripts/2026-09/render_page.py --check PAGE
        Render from the archived run (archive/2026-09-hwasan) and compare with
        the tables in PAGE. They must match: this is the test that the
        renderer reproduces what is already published before it is trusted
        with new data.

    python scripts/2026-09/render_page.py --apply PAGE [--new ID ...]
        Render from the live results/ and pairwise/ and write the tables into
        PAGE. --new marks rows added since the last publication.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ati.bradleyterry import bootstrap_ci, fit, to_elo  # noqa: E402
from ati.config import load_models  # noqa: E402
from ati.pairwise import load_comparisons  # noqa: E402
from ati.report import pairwise_records  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_judge  # noqa: E402

RUN = "2026-09"
AXES = ["accuracy", "terminology", "register", "fluency", "fidelity", "sanitization"]
# Scored by a judge of its own family; see the page's callout.
FLAGGED = {"gemini-3.8-flash"}
DASH = "–"
APOS = "’"
PAD_ROW = " " * 10
PAD_CELL = " " * 12


def ratings_for(pairwise_root: Path, models_path: Path):
    comparisons = load_comparisons(pairwise_root, RUN)
    anchor = next((m.id for m in load_models(models_path) if m.anchor == "weak"), None)
    ratings = to_elo(fit(comparisons), anchor_id=anchor)
    intervals = bootstrap_ci(comparisons, anchor_id=anchor)
    records = pairwise_records(comparisons)
    pairs = len({tuple(sorted((a, b))) for a, b, _ in comparisons})
    return ratings, intervals, records, anchor, len(comparisons), pairs


def ranked(results: list[dict]) -> list[dict]:
    """Rows the leaderboard ranks: judged, and not moved to Incomplete."""
    rows = [r for r in results if r.get("composite") is not None and not r.get("incomplete")]
    return sorted(rows, key=lambda r: -r["composite"])


def scored(r: dict) -> int:
    return r["total_tasks"] - r["failed_tasks"] - r["refused_tasks"]


def heat_class(value: float) -> str:
    # Bands on the displayed (one-decimal) value, so a cell's colour always
    # agrees with the number printed in it.
    v = round(value, 1)
    if v >= 4.5:
        return "h5"
    if v >= 3.5:
        return "h4"
    if v >= 3.0:
        return "h3"
    if v >= 2.0:
        return "h2"
    return "h1"


def judged(value) -> str:
    return f"{value:.1f}" if value is not None else "&mdash;"


def leaderboard_body(rows, ratings, intervals, records, anchor, new_ids, cj) -> str:
    out = []
    for rank, r in enumerate(rows, 1):
        mid = r["model_id"]
        partial = scored(r) < r["total_tasks"]
        classes = []
        if mid in FLAGGED:
            classes.append("flagged-row")
        if mid == anchor:
            classes.append("anchor-row")
        if partial:
            classes.append("partial")
        tr = f'<tr class="{" ".join(classes)}">' if classes else "<tr>"

        name = mid
        if partial:
            name = f'<span class="flag" aria-hidden="true">*</span>{name}'
        if mid in FLAGGED:
            name += f'<span class="badge warn">judge{APOS}s own family</span>'
        if mid == anchor:
            name += '<span class="badge">weak anchor</span>'
        if mid in new_ids:
            name += '<span class="badge new">new</span>'

        comp = f"{r['composite']:.1f}"
        if mid in ratings:
            if mid == anchor:
                rating = f'{ratings[mid]:.0f} <span class="interval">pinned</span>'
            else:
                lo, hi = intervals[mid]
                rating = (
                    f'{ratings[mid]:.0f} <span class="interval">'
                    f"{lo:.0f} {DASH} {hi:.0f}</span>"
                )
            w, l, t = records.get(mid, (0, 0, 0))
            wlt = f"{w}{DASH}{l}{DASH}{t}"
        else:
            rating, wlt = "&mdash;", "&mdash;"

        out.append(f"{PAD_ROW}{tr}")
        out.append(f'{PAD_CELL}<td class="num">{rank}</td>')
        out.append(f'{PAD_CELL}<th scope="row">{name}</th>')
        out.append(
            f'{PAD_CELL}<td class="composite"><span class="val">{comp}</span>'
            f'<span class="cbar" aria-hidden="true"><i style="--w:{comp}%"></i></span></td>'
        )
        out.append(f'{PAD_CELL}<td class="num">{rating}</td>')
        out.append(f'{PAD_CELL}<td class="num">{wlt}</td>')
        out.append(f'{PAD_CELL}<td class="num">{scored(r)}/{r["total_tasks"]}</td>')
        out.append(f'{PAD_CELL}<td class="num">{r["refusal_rate"]:.1f}</td>')
        # Five judged axes, normalised 0-100, for Gemini and jev alike: the
        # only basis on which the two judges can be read side by side.
        c = cj.get(mid, {})
        out.append(f'{PAD_CELL}<td class="num judged">{judged(c.get("gemini"))}</td>')
        out.append(f'{PAD_CELL}<td class="num judged">{judged(c.get("jev"))}</td>')
        out.append(f"{PAD_ROW}</tr>")
    return "\n".join(out)


def heat_body(rows, anchor, new_ids) -> str:
    out = []
    for r in rows:
        mid = r["model_id"]
        name = mid
        if mid == anchor:
            name += '<span class="badge">weak anchor</span>'
        if mid in new_ids:
            name += '<span class="badge new">new</span>'
        tr = '<tr class="anchor-row">' if mid == anchor else "<tr>"
        cells = "".join(
            f'<td class="h {heat_class(r["axis_means"][a])} num">'
            f'{r["axis_means"][a]:.1f}<small>±{r["axis_stdevs"][a]:.1f}</small></td>'
            for a in AXES
        )
        out.append(f'{PAD_ROW}{tr}<th scope="row">{name}</th>')
        out.append(f"{PAD_CELL}{cells}</tr>")
    return "\n".join(out)


def facts(rows, n_comp, n_pairs, rated) -> dict[str, str]:
    """The fact strip's derived entries, over the ranked rows."""
    total = sum(r["total_tasks"] for r in rows)
    ok = sum(scored(r) for r in rows)
    refused = sum(r["refused_tasks"] for r in rows)
    failed = sum(r["failed_tasks"] for r in rows)
    possible = rated * (rated - 1) // 2
    cover = (
        f"all {possible} model pairs" if n_pairs == possible
        else f"{n_pairs} of {possible} model pairs"
    )
    return {
        "Models": str(len(rows)),
        "Scored": f"{ok} of {total} translations {chr(0x2014)} {refused} refused, {failed} failed",
        "Head to head": f"{n_comp} comparisons, {cover}",
    }


FACT_RX = r"(<div><dt>{key}</dt><dd>)(.*?)(</dd></div>)"


SECTION_TBODY = {
    "leaderboard": re.compile(
        r'(<section id="leaderboard">.*?<tbody>\n)(.*?)(\n\s*</tbody>)', re.S
    ),
    "axes": re.compile(r'(<section id="axes">.*?<tbody>\n)(.*?)(\n\s*</tbody>)', re.S),
}


def render(results_path: Path, pairwise_root: Path, models_path: Path, new_ids=()):
    results = json.loads(results_path.read_text(encoding="utf-8"))["results"]
    ratings, intervals, records, anchor, n_comp, n_pairs = ratings_for(
        pairwise_root, models_path
    )
    rows = ranked(results)
    cj = {row["model"]: row for row in cross_judge.table()}
    return {
        "leaderboard": leaderboard_body(
            rows, ratings, intervals, records, anchor, set(new_ids), cj
        ),
        "axes": heat_body(rows, anchor, set(new_ids)),
        "rows": rows,
        "comparisons": n_comp,
        "pairs": n_pairs,
        "rated": len(ratings),
        "facts": facts(rows, n_comp, n_pairs, len(ratings)),
    }


def current_tables(page: str) -> dict[str, str]:
    return {k: rx.search(page).group(2) for k, rx in SECTION_TBODY.items()}


def normalise(html: str) -> list[str]:
    return [line.strip() for line in html.strip().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", metavar="PAGE", type=Path)
    mode.add_argument("--apply", metavar="PAGE", type=Path)
    ap.add_argument("--new", action="append", default=[], metavar="ID")
    ap.add_argument(
        "--source", choices=("live", "archive"), default="live",
        help="data for --apply: live results/ and pairwise/, or the archived run",
    )
    args = ap.parse_args()

    if args.check:
        arch = REPO / "archive" / "2026-09-hwasan"
        got = render(
            arch / "results" / "2026-09.json",
            arch / "pairwise",
            arch / "config" / "models.yaml",
        )
        page = args.check.read_text(encoding="utf-8")
        live = current_tables(page)
        ok = True
        for key in ("leaderboard", "axes"):
            a, b = normalise(live[key]), normalise(got[key])
            # The heat grid's cell line is split differently in the page; the
            # comparison is on content, so join each table to one string.
            if "".join(a) != "".join(b):
                ok = False
                print(f"MISMATCH in {key}")
                for i, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        print(f"  first difference at line {i}:\n    page: {x[:150]}\n    ours: {y[:150]}")
                        break
                else:
                    print(f"  lengths differ: page {len(a)} lines, ours {len(b)}")
            else:
                print(f"match: {key} ({len(got['rows'])} rows)")
        for key, want in got["facts"].items():
            m = re.search(FACT_RX.format(key=re.escape(key)), page)
            have = m.group(2) if m else None
            if have != want:
                ok = False
                print(f"MISMATCH in fact {key!r}: page {have!r}, ours {want!r}")
            else:
                print(f"match: fact {key!r}")
        print(f"{got['comparisons']} comparisons over {got['pairs']} pairs, {got['rated']} rated")
        return 0 if ok else 1

    if args.source == "archive":
        arch = REPO / "archive" / "2026-09-hwasan"
        paths = (arch / "results" / "2026-09.json", arch / "pairwise", arch / "config" / "models.yaml")
    else:
        paths = (REPO / "results" / "2026-09.json", REPO / "pairwise", REPO / "config" / "models.yaml")
    got = render(*paths, new_ids=args.new)
    page = args.apply.read_text(encoding="utf-8")
    for key, rx in SECTION_TBODY.items():
        page, n = rx.subn(lambda m, k=key: m.group(1) + got[k] + m.group(3), page, count=1)
        assert n == 1, f"could not find the {key} table"
    for key, value in got["facts"].items():
        page, n = re.subn(
            FACT_RX.format(key=re.escape(key)),
            lambda m, v=value: m.group(1) + v + m.group(3),
            page, count=1,
        )
        assert n == 1, f"could not find the {key!r} fact"
    args.apply.write_text(page, encoding="utf-8")
    print(
        f"wrote {len(got['rows'])} ranked rows; {got['comparisons']} comparisons "
        f"over {got['pairs']} of {got['rated'] * (got['rated'] - 1) // 2} pairs"
    )
    for r in got["rows"]:
        print(f"  {r['model_id']:28s} {r['composite']:5.1f}  {scored(r)}/{r['total_tasks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
