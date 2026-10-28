"""Build the public site (GitHub Pages) from an archived run.

    python scripts/build_site.py [--run 2026-09] [--archive archive/2026-09-hwasan]

Writes site/: the leaderboard page, one page per model, sitemap.xml,
robots.txt and 404.html. Everything on the page comes from the archived
results and comparisons, through the same calls `ati report` makes, so the
site and the leaderboard cannot disagree. Rebuild after archiving a run.

The page answers one question - which AI to use to translate a web novel, at
what cost - and is written to be found by people asking it. No source or
translation text is read or published.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ati.bradleyterry import bootstrap_ci, fit, to_elo  # noqa: E402
from ati.config import load_models  # noqa: E402
from ati.pairwise import load_comparisons  # noqa: E402

SITE_URL = "https://gaieais70.github.io/webnovel-translation-benchmark/"
REPO_URL = "https://github.com/gaieais70/webnovel-translation-benchmark"
TITLE = "Webnovel Translation Benchmark"
CHAPTER_CHARS = 7056  # mean source length of run 2026-09's chapters, Korean characters

# Display names and makers. Ids are the roster's; names are how people search.
MODELS = {
    "gemini-3.8-flash": ("Gemini 3.8 Flash", "Google"),
    "gemini-4-argon": ("Gemini 4 Argon", "Google"),
    "opus-5": ("Claude Opus 5", "Anthropic"),
    "sonnet-5": ("Claude Sonnet 5", "Anthropic"),
    "haiku-4.5": ("Claude Haiku 4.5", "Anthropic"),
    "opus-5.5": ("Claude Opus 5.5", "Anthropic"),
    "sonnet-5.5": ("Claude Sonnet 5.5", "Anthropic"),
    "haiku-5.5": ("Claude Haiku 5.5", "Anthropic"),
    "gpt-5.6-luna": ("GPT-5.6 Luna", "OpenAI"),
    "gpt-6.1-sol": ("GPT-6.1 Sol", "OpenAI"),
    "gpt-6-astra": ("GPT-6 Astra", "OpenAI"),
    "gpt-6-luna": ("GPT-6 Luna", "OpenAI"),
    "kimi-k3": ("Kimi K3", "Moonshot AI"),
    "glm-5.3": ("GLM-5.3", "Z.ai"),
    "glm-5.3-flash": ("GLM-5.3 Flash", "Z.ai"),
    "glm-5.2": ("GLM-5.2", "Z.ai"),
    "muse-spark-1.2-contributor": ("Muse Spark 1.2", "Meta"),
    "muse-spark-1.3-contributor": ("Muse Spark 1.3", "Meta"),
    "deepseek-v4-pro": ("DeepSeek V4 Pro", "DeepSeek"),
    "deepseek-v4.1-flash": ("DeepSeek V4.1 Flash", "DeepSeek"),
    "deepseek-v4-flash": ("DeepSeek V4 Flash", "DeepSeek"),
    "qwen3.8-flash": ("Qwen3.8 Flash", "Alibaba"),
    "qwen3.8-27b": ("Qwen3.8 27B", "Alibaba"),
    "hy4-preview": ("Hunyuan HY4 Preview", "Tencent"),
    "hy3": ("Hunyuan HY3", "Tencent"),
    "nemotron-3-ultra": ("Nemotron 3 Ultra", "NVIDIA"),
    "nemotron-3-super-120b": ("Nemotron 3 Super 120B", "NVIDIA"),
    "nemotron-3.5-lightning": ("Nemotron 3.5 Lightning", "NVIDIA"),
    "mimo-v2.5-pro": ("MiMo V2.5 Pro", "Xiaomi"),
    "mimo-v2.5": ("MiMo V2.5", "Xiaomi"),
    "minimax-m3": ("MiniMax M3", "MiniMax"),
    "solar-pro4": ("Solar Pro 4", "Upstage"),
    "inkling-small": ("Inkling Small", "Thinking Machines"),
    "dots-3-note-preview": ("dots.3 note preview", "dots studio"),
    "laguna-s-2.1": ("Laguna S 2.1", "Poolside"),
}
IN_TESTING = ["gemini-4-argon"]
# Models that are also on the judging panel; each is scored by a judge of its own family.
JUDGES = {"haiku-judge": ("Claude Haiku 5.5", "haiku-5.5"),
          "gemini-3.8-judge": ("Gemini 3.8 Flash", "gemini-3.8-flash")}
SAME_FAMILY_AS_JUDGE = {m for _, m in JUDGES.values()}
AXES = ["accuracy", "terminology", "register", "fluency", "fidelity", "sanitization"]
AXIS_HELP = {
    "accuracy": "meaning carried over correctly",
    "terminology": "names, titles and martial-arts terms kept consistent across chapters",
    "register": "tone and speech levels (honorifics, banter, formality)",
    "fluency": "reads as natural English prose",
    "fidelity": "nothing added, dropped or summarised",
    "sanitization": "violence and crude language kept, not softened",
}


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def name(model_id: str) -> str:
    return MODELS.get(model_id, (model_id, ""))[0]


def maker(model_id: str) -> str:
    return MODELS.get(model_id, ("", ""))[1]


def slug(model_id: str) -> str:
    return model_id.replace(".", "-")


def money(value: float | None, places: int = 3) -> str:
    if value is None:
        return "not measured"
    if value == 0:
        return "free"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.{places}f}".rstrip("0").rstrip(".") if places > 2 else f"${value:.2f}"


def per_100(value: float | None) -> str:
    if value is None:
        return "not measured"
    if value == 0:
        return "free"
    return f"${value * 100:,.2f}"


# --- data --------------------------------------------------------------------


def load(run: str, archive: Path):
    data = json.loads((archive / "results" / f"{run}.json").read_text(encoding="utf-8"))
    rows = [r for r in data["results"] if not r.get("incomplete")]
    roster = {m.id: m for m in load_models(REPO / "config" / "models.yaml")}
    comparisons = pooled_comparisons(archive, run, data)
    anchor = next((m.id for m in roster.values() if m.anchor == "weak"), None)
    ratings = to_elo(fit(comparisons), anchor_id=anchor) if comparisons else {}
    intervals = bootstrap_ci(comparisons, anchor_id=anchor) if comparisons else {}
    for r in rows:
        r["rating"] = ratings.get(r["model_id"])
        r["interval"] = intervals.get(r["model_id"])
        r["scored"] = r["total_tasks"] - r["failed_tasks"] - r.get("refused_tasks", 0)
    rows.sort(key=lambda r: -(r["composite"] or 0))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    mark_frontier(rows)
    return data, rows, len(comparisons)


def pooled_comparisons(archive: Path, run: str, data: dict):
    """Every panel judge's verdicts in one fit, as `ati report --panel` pools them."""
    judges = [j for j in data["metadata"].get("judge_id", "").split(" + ") if j]
    root = archive / "pairwise"
    if len(judges) > 1:
        return [c for j in judges for c in load_comparisons(root / j, run)]
    return load_comparisons(root, run)


def judge_agreement(rows) -> tuple[float, tuple, tuple] | None:
    """(correlation, haiku range, gemini range) of the two judges' composites."""
    names = list(JUDGES)
    pairs = [(r["judge_composites"][names[0]], r["judge_composites"][names[1]]) for r in rows
             if all((r.get("judge_composites") or {}).get(n) is not None for n in names)]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    corr = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return corr, (min(xs), max(xs)), (min(ys), max(ys))


def mark_frontier(rows) -> None:
    """Best quality at its price: no model scores higher for the same or less."""
    priced = sorted(
        (r for r in rows if r.get("list_cost_per_chapter") is not None),
        key=lambda r: (r["list_cost_per_chapter"], -(r["composite"] or 0)),
    )
    best = -1.0
    for r in priced:
        r["frontier"] = (r["composite"] or 0) > best
        best = max(best, r["composite"] or 0)


# --- chart -------------------------------------------------------------------

W, H = 960, 480
PAD_L, PAD_R, PAD_T, PAD_B = 64, 150, 24, 56
X_MIN, X_MAX = 0.001, 1.0  # dollars per chapter, log scale
Y_MIN, Y_MAX = 10, 95


def x_of(cost: float) -> float:
    cost = max(cost, X_MIN)
    span = math.log10(X_MAX) - math.log10(X_MIN)
    return PAD_L + (math.log10(cost) - math.log10(X_MIN)) / span * (W - PAD_L - PAD_R)


def y_of(score: float) -> float:
    return PAD_T + (Y_MAX - score) / (Y_MAX - Y_MIN) * (H - PAD_T - PAD_B)


def chart(rows) -> str:
    pts = [r for r in rows if r.get("list_cost_per_chapter") is not None]
    out = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="chart-title chart-desc" '
        f'class="chart">',
        '<title id="chart-title">Translation quality against cost per chapter</title>',
        '<desc id="chart-desc">Each dot is one AI model: its quality score (higher is '
        'better) against what one chapter costs at list price (left is cheaper, log '
        'scale). Labelled dots give the best quality available at their price. The '
        'same data is in the table below.</desc>',
    ]
    for score in range(20, 100, 10):
        y = y_of(score)
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{PAD_L - 10}" y="{y + 4:.1f}" text-anchor="end">{score}</text>')
    for cost, label in [(0.001, "free–$0.001"), (0.01, "$0.01"), (0.1, "$0.10"), (1.0, "$1")]:
        x = x_of(cost)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{H - PAD_B}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{H - PAD_B + 20}" text-anchor="middle">{label}</text>')
    out.append(f'<text class="axis" x="{PAD_L}" y="{H - 10}">Cost per chapter at list price →</text>')
    out.append(f'<text class="axis" x="14" y="{PAD_T + 4}" transform="rotate(-90 14 {PAD_T + 4})" '
               f'text-anchor="end">Quality score</text>')

    frontier = sorted((r for r in pts if r.get("frontier")), key=lambda r: r["list_cost_per_chapter"])
    if len(frontier) > 1:
        path = []
        for i, r in enumerate(frontier):
            x, y = x_of(r["list_cost_per_chapter"]), y_of(r["composite"])
            if i:
                path.append(f"H{x:.1f}V{y:.1f}")
            else:
                path.append(f"M{x:.1f},{y:.1f}")
        out.append(f'<path class="frontier-line" d="{"".join(path)}"/>')

    for r in sorted(pts, key=lambda r: bool(r.get("frontier"))):
        x, y = x_of(r["list_cost_per_chapter"]), y_of(r["composite"])
        cls = "dot frontier" if r.get("frontier") else "dot"
        tip = (f'{name(r["model_id"])} · score {r["composite"]:.1f} · '
               f'{money(r["list_cost_per_chapter"])} per chapter')
        out.append(
            f'<a href="models/{slug(r["model_id"])}/" class="dot-link" '
            f'data-tip="{esc(tip)}" aria-label="{esc(tip)}">'
            f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>'
            f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{6 if r.get("frontier") else 5}"/></a>'
        )
        if r.get("frontier"):
            out.append(f'<text class="dot-label" x="{x + 10:.1f}" y="{y - 8:.1f}">{esc(name(r["model_id"]))}</text>')
    out.append("</svg>")
    return "\n".join(out)


# --- pages -------------------------------------------------------------------

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700'
    '&family=Public+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">'
)

CSS = """
:root{--paper:#F6F7F9;--ink:#231A2C;--muted:#6A6476;--rule:#DCD8E0;--plum:#B23A6B;
--dot:#9A95A3;--panel:#FFFFFF;--plum-mark:#B23A6B;--focus:#3A5BD9}
@media (prefers-color-scheme:dark){:root{--paper:#17121D;--ink:#EEE9F2;--muted:#A79FB3;
--rule:#352C3F;--plum:#E07CA6;--dot:#625D6E;--panel:#1F1827;--plum-mark:#D0608F;--focus:#8EA6FF}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:400 17px/1.6 "Public Sans",system-ui,sans-serif;
font-variant-numeric:tabular-nums}
a{color:var(--plum)}
a:focus-visible,button:focus-visible,th button:focus-visible{outline:3px solid var(--focus);outline-offset:2px;border-radius:2px}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px}
.prose{max-width:68ch}
header.site{padding:28px 0 0}
header.site a{color:var(--ink);text-decoration:none;font-family:"Gowun Batang",serif;font-weight:700;font-size:19px}
h1,h2,h3{font-family:"Gowun Batang",serif;font-weight:700;line-height:1.2;letter-spacing:-.01em}
h1{font-size:clamp(34px,5.2vw,58px);margin:40px 0 16px;max-width:18ch}
h2{font-size:30px;margin:64px 0 12px}
h3{font-size:21px;margin:32px 0 8px}
.lede{font-size:20px;color:var(--muted);max-width:60ch;margin:0 0 28px}
.answer{font-size:19px;max-width:64ch}
.answer strong{font-weight:600}
figure{margin:24px 0 8px;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px 12px 8px;position:relative}
figcaption{font-size:15px;color:var(--muted);padding:4px 8px 6px;max-width:72ch}
.chart-scroll{overflow-x:auto}
.chart{width:100%;height:auto;display:block}
.table-note{font-size:15px;color:var(--muted);max-width:68ch;margin:20px 0 0}
.visually-hidden{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:13px}
.chart .axis{fill:var(--muted);font-size:13px;font-weight:600}
.chart .dot{fill:var(--dot);stroke:var(--panel);stroke-width:2}
.chart .dot.frontier{fill:var(--plum-mark)}
.chart .hit{fill:transparent}
.chart .frontier-line{fill:none;stroke:var(--plum-mark);stroke-width:2;stroke-dasharray:5 4;opacity:.7}
.chart .dot-label{fill:var(--ink);font-size:14px;font-weight:600;paint-order:stroke;stroke:var(--panel);stroke-width:4px}
.chart a:focus-visible .dot{stroke:var(--focus);stroke-width:3}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--paper);font-size:14px;
padding:6px 10px;border-radius:6px;white-space:nowrap;opacity:0;transform:translate(-50%,-130%)}
.tip.on{opacity:1}
.legend{display:flex;gap:22px;flex-wrap:wrap;font-size:15px;color:var(--muted);padding:0 8px 6px}
.legend span::before{content:"";display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px;vertical-align:-1px;background:var(--dot)}
.legend .f::before{background:var(--plum-mark)}
.table-wrap{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--panel);margin-top:16px}
table{border-collapse:collapse;width:100%;font-size:15.5px}
caption{text-align:left;padding:14px 16px 4px;color:var(--muted);font-size:15px}
th,td{padding:10px 14px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule)}
th:nth-child(2),td:nth-child(2){text-align:left}
tbody tr:last-child td{border-bottom:0}
thead th{font-weight:600;color:var(--muted);vertical-align:bottom}
th button{all:unset;cursor:pointer}
th button[aria-sort]::after{content:" ↓"}
th button[aria-sort="ascending"]::after{content:" ↑"}
td.model{white-space:normal;min-width:11em}
td.model a{color:var(--ink);font-weight:600;text-decoration:none}
td.model a:hover{text-decoration:underline}
td.model small{display:block;color:var(--muted);font-weight:400}
.best-price{color:var(--plum);font-weight:600}
.flag{color:var(--muted);font-size:13px;display:block}
.testing{display:flex;flex-wrap:wrap;gap:8px 18px;padding:0;margin:12px 0;list-style:none}
.testing li{border:1px dashed var(--rule);border-radius:999px;padding:4px 14px;font-size:15px}
details{border-top:1px solid var(--rule);padding:14px 0}
details:last-of-type{border-bottom:1px solid var(--rule)}
summary{cursor:pointer;font-weight:600;font-size:18px}
dl.axes{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px}
dl.axes dt{font-weight:600}
dl.axes dd{margin:0;color:var(--muted)}
.stat{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px;margin:20px 0}
.stat div{border-left:3px solid var(--plum);padding:2px 0 2px 14px}
.stat b{display:block;font-size:26px;font-family:"Gowun Batang",serif}
.stat span{color:var(--muted);font-size:15px}
footer{margin:80px 0 40px;color:var(--muted);font-size:15px;border-top:1px solid var(--rule);padding-top:20px}
@media (max-width:640px){body{font-size:16px}th,td{padding:8px 10px}.hide-sm{display:none}
.chart{min-width:720px}figure{padding:12px 4px 6px}h2{font-size:26px;margin-top:48px}
table{font-size:14.5px}th,td{padding:8px 6px}td.model{min-width:0}.wrap{padding:0 16px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
(function(){
  var fig=document.querySelector('figure.plot');
  if(fig){var tip=document.createElement('div');tip.className='tip';fig.appendChild(tip);
    fig.querySelectorAll('.dot-link').forEach(function(a){
      function show(){var c=a.querySelector('.dot'),b=c.getBoundingClientRect(),f=fig.getBoundingClientRect();
        tip.textContent=a.getAttribute('data-tip');tip.style.left=(b.left-f.left+b.width/2)+'px';
        tip.style.top=(b.top-f.top)+'px';tip.classList.add('on');}
      function hide(){tip.classList.remove('on');}
      a.addEventListener('mouseenter',show);a.addEventListener('focus',show);
      a.addEventListener('mouseleave',hide);a.addEventListener('blur',hide);});}
  var table=document.querySelector('table.board');
  if(table){var body=table.tBodies[0];
    table.querySelectorAll('th button').forEach(function(btn){
      btn.addEventListener('click',function(){
        var col=+btn.dataset.col,asc=btn.getAttribute('aria-sort')==='descending';
        table.querySelectorAll('th button').forEach(function(b){b.removeAttribute('aria-sort');});
        btn.setAttribute('aria-sort',asc?'ascending':'descending');
        var rows=[].slice.call(body.rows);
        rows.sort(function(r1,r2){
          var a=parseFloat(r1.cells[col].dataset.v),b=parseFloat(r2.cells[col].dataset.v);
          if(isNaN(a))return 1;if(isNaN(b))return -1;return asc?a-b:b-a;});
        rows.forEach(function(r){body.appendChild(r);});});});}
})();
"""


def page(title: str, description: str, path: str, body: str, jsonld: list[dict]) -> str:
    url = SITE_URL + path
    scripts = "".join(
        f'<script type="application/ld+json">{json.dumps(d, ensure_ascii=False)}</script>'
        for d in jsonld
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="canonical" href="{url}">
<meta name="robots" content="index,follow">
<meta property="og:type" content="website">
<meta property="og:site_name" content="{TITLE}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:url" content="{url}">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{esc(title)}">
<meta name="twitter:description" content="{esc(description)}">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cg fill='%23B23A6B'%3E%3Ccircle cx='16' cy='8' r='6'/%3E%3Ccircle cx='24' cy='14' r='6'/%3E%3Ccircle cx='21' cy='24' r='6'/%3E%3Ccircle cx='11' cy='24' r='6'/%3E%3Ccircle cx='8' cy='14' r='6'/%3E%3C/g%3E%3Ccircle cx='16' cy='16' r='3' fill='%23F6F7F9'/%3E%3C/svg%3E">
{FONTS}
<style>{CSS}</style>
{scripts}
</head>
<body>
<header class="site"><div class="wrap"><a href="{SITE_URL}">{TITLE}</a></div></header>
<main class="wrap">
{body}
</main>
<footer><div class="wrap prose">
<p>{TITLE} ranks AI models on translating web novels into English. It publishes
scores, costs and methods, never source or translation text. Code, method and
every judged comparison are on <a href="{REPO_URL}">GitHub</a>.</p>
</div></footer>
<script>{JS}</script>
</body>
</html>
"""


def leaderboard_table(rows) -> str:
    head = [
        ("#", 0, ""), ("Model", None, ""), ("Quality", 2, ""), ("Cost / chapter", 3, ""),
        ("Per 100 chapters", 4, "hide-sm"), ("Head-to-head", 5, "hide-sm"),
        ("Output tokens / ch.", 6, "hide-sm"), ("Chapters scored", 7, "hide-sm"),
        ("Haiku 5.5 judge", 8, "hide-sm"), ("Gemini 3.8 judge", 9, "hide-sm"),
    ]
    th = []
    for label, col, cls in head:
        c = f' class="{cls}"' if cls else ""
        if col is None:
            th.append(f"<th scope=\"col\"{c}>{label}</th>")
        else:
            sort = ' aria-sort="descending"' if col == 2 else ""
            th.append(f'<th scope="col"{c}><button data-col="{col}"{sort}>{label}</button></th>')
    trs = []
    for r in rows:
        mid = r["model_id"]
        cost = r.get("list_cost_per_chapter")
        flag = (' <span class="flag" title="This model is also one of the two judges">also a judge</span>'
                if mid in SAME_FAMILY_AS_JUDGE else "")
        jc = r.get("judge_composites") or {}
        judge_cells = "".join(
            f'<td class="hide-sm" data-v="{jc[j] if jc.get(j) is not None else ""}">'
            f'{f"{jc[j]:.1f}" if jc.get(j) is not None else "–"}</td>'
            for j in JUDGES
        )
        cost_cell = money(cost) if cost is not None else '<span title="not measured">n/a</span>'
        if r.get("frontier"):
            cost_cell = f'<span class="best-price" title="Best quality at this price">{cost_cell}</span>'
        tok = r.get("tokens_out_per_chapter")
        trs.append(
            "<tr>"
            f'<td data-v="{r["rank"]}">{r["rank"]}</td>'
            f'<td class="model"><a href="models/{slug(mid)}/">{esc(name(mid))}</a>{flag}'
            f"<small>{esc(maker(mid))}</small></td>"
            f'<td data-v="{r["composite"]:.2f}">{r["composite"]:.1f}</td>'
            f'<td data-v="{cost if cost is not None else ""}">{cost_cell}</td>'
            f'<td class="hide-sm" data-v="{cost if cost is not None else ""}">{per_100(cost) if cost is not None else "n/a"}</td>'
            f'<td class="hide-sm" data-v="{r["rating"] or ""}">{f"{r["rating"]:.0f}" if r["rating"] else "–"}</td>'
            f'<td class="hide-sm" data-v="{tok if tok is not None else ""}">{f"{tok:,.0f}" if tok is not None else "n/a"}</td>'
            f'<td class="hide-sm" data-v="{r["scored"]}">{r["scored"]} of {r["total_tasks"]}</td>'
            f"{judge_cells}</tr>"
        )
    return (
        '<p class="table-note">Select a column heading to sort. Cost is what one chapter '
        "(about 7,000 Korean characters) costs at the provider's list price, reasoning "
        "included. Quality is the average of two judges; each judge's own score is at the right.</p>"
        '<div class="table-wrap"><table class="board">'
        "<caption class=\"visually-hidden\">All models, ranked by quality</caption>"
        f"<thead><tr>{''.join(th)}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
    )


def faq(rows, n_comparisons: int) -> list[tuple[str, str]]:
    top = rows[0]
    priced = [r for r in rows if r.get("list_cost_per_chapter") is not None]
    frontier = sorted((r for r in priced if r.get("frontier") and (r["composite"] or 0) >= 70),
                      key=lambda r: r["list_cost_per_chapter"])
    cheap_good = frontier[0] if frontier else None
    h2h = sorted((r for r in rows if r.get("rating")), key=lambda r: -r["rating"])
    by_maker = {}
    for r in rows:
        by_maker.setdefault(maker(r["model_id"]), r)

    def best_of(vendor):
        r = by_maker.get(vendor)
        return f"{name(r['model_id'])} ({r['composite']:.1f})" if r else "not tested"

    items = [
        (
            "What is the best AI for translating Korean web novels?",
            f"In this benchmark, {name(top['model_id'])} scored highest ({top['composite']:.1f} "
            f"of 100) translating eleven chapters of Return of the Mount Hua Sect (화산귀환) from "
            f"Korean to English, averaging two AI judges. "
            + ", ".join(f"{name(r['model_id'])} ({r['composite']:.1f})" for r in rows[1:4])
            + f" follow. In blind side-by-side comparisons the order differs: "
            f"{name(h2h[0]['model_id'])} rated highest, so treat the top few as a close group.",
        ),
        (
            "What is the cheapest AI that translates web novels well?",
            (f"{name(cheap_good['model_id'])} scored {cheap_good['composite']:.1f} at "
             f"{money(cheap_good['list_cost_per_chapter'])} per chapter, about "
             f"{per_100(cheap_good['list_cost_per_chapter'])} for 100 chapters at list price. "
             f"No model scored higher for less. The chart's labelled models are the best "
             f"available at each price.") if cheap_good else "Not yet measured.",
        ),
        (
            "Is Claude or ChatGPT better for translating web novels?",
            f"Best Claude model tested: {best_of('Anthropic')}. Best OpenAI model: "
            f"{best_of('OpenAI')}. Best Google model: {best_of('Google')}. Gemini 4 Argon "
            f"has not been tested yet.",
        ),
        (
            "How much does it cost to translate a web novel chapter with AI?",
            f"From {money(min(r['list_cost_per_chapter'] for r in priced if r['list_cost_per_chapter']))} to "
            f"{money(max(r['list_cost_per_chapter'] for r in priced))} per chapter of roughly "
            "7,000 Korean characters at list price, depending on the model. Reasoning "
            "models spend most of that on thinking tokens before they write: compare the "
            "output-token column. Subscriptions change what you pay, not how many tokens "
            "a model uses.",
        ),
        (
            "How are the translations scored?",
            f"Two AI judges, Claude Haiku 5.5 and Gemini 3.8 Flash, each score every chapter "
            f"from 1 to 5 on six qualities (below) and compare translations two at a time, "
            f"in both orders, blind to which model wrote them ({n_comparisons} comparisons "
            f"pooled). The quality score is their average. Every model gets the same "
            f"prompt, the same chapters and its maximum reasoning setting.",
        ),
    ]
    return items


def build_index(data, rows, n_comparisons, out: Path) -> None:
    run_date = data["metadata"].get("date", "")
    top = rows[0]
    priced = [r for r in rows if r.get("list_cost_per_chapter") is not None]
    frontier = sorted((r for r in priced if r.get("frontier")), key=lambda r: r["list_cost_per_chapter"])
    good_cheap = next((r for r in frontier if (r["composite"] or 0) >= 75), None)
    questions = faq(rows, n_comparisons)

    answer = (
        f'<p class="answer">For Korean web novels, <strong>{esc(name(top["model_id"]))}</strong> '
        f"scored highest in this test, at {money(top.get('list_cost_per_chapter'))} a chapter"
        + (' (judged by its own family: <a href="#limits">see the limits</a>). '
           if top["model_id"] in SAME_FAMILY_AS_JUDGE else ". ")
    )
    if good_cheap and good_cheap is not top:
        answer += (
            f"<strong>{esc(name(good_cheap['model_id']))}</strong> came within "
            f"{top['composite'] - good_cheap['composite']:.0f} points for "
            f"{money(good_cheap['list_cost_per_chapter'])} a chapter: "
            f"{per_100(good_cheap['list_cost_per_chapter'])} per hundred chapters"
            + (" (it is also one of the judges: <a href=\"#limits\">see the limits</a>)."
               if good_cheap["model_id"] in SAME_FAMILY_AS_JUDGE else ".")
        )
    answer += "</p>"

    agree = judge_agreement(rows)
    agreement = (
        f" The two judges' scores correlate at {agree[0]:.2f}, but Gemini spreads them "
        f"wider ({agree[2][0]:.0f} to {agree[2][1]:.0f}, against {agree[1][0]:.0f} to "
        f"{agree[1][1]:.0f} for Claude Haiku 5.5)."
        if agree else ""
    )
    body = f"""
<h1>Which AI should translate your web novel?</h1>
<p class="lede">{len(rows)} AI models translated the same eleven chapters of a Korean
murim novel into English. Here is how good each one was, and what a chapter costs.</p>
{answer}
<figure class="plot">
<div class="legend"><span class="f">Best quality at its price</span><span>Other models</span></div>
<div class="chart-scroll">{chart(rows)}</div>
<figcaption>Quality score against cost per chapter at list price, log scale. Models
whose cost was not recorded are in the table only. Hover or tab to a dot for details;
select it for the model's page.</figcaption>
</figure>

<h2>Full ranking</h2>
<p class="prose">Quality is the judged score out of 100. Head-to-head is a rating from
blind side-by-side comparisons, where 400 points is 10-to-1 odds. Cost is per chapter
at the provider's list price, whatever route the benchmark used.</p>
{leaderboard_table(rows)}

<h2>Now testing</h2>
<p class="prose">Not ranked yet, to be run on the same chapters with the same prompt:</p>
<ul class="testing">{"".join(f"<li>{esc(name(m))}</li>" for m in IN_TESTING)}</ul>
<p class="prose">Earlier editions were judged by Gemini 3.7 Flash, which is no longer
available. Every model here was re-judged by two judges from different families, and
each judge's own score is shown beside the average.</p>

<h2>Questions</h2>
<div class="prose">
{"".join(f"<details><summary>{esc(q)}</summary><p>{esc(a)}</p></details>" for q, a in questions)}
</div>

<h2>What the score measures</h2>
<div class="prose">
<dl class="axes">{"".join(f"<dt>{a.capitalize()}</dt><dd>{AXIS_HELP[a]}</dd>" for a in AXES)}</dl>
<h3 id="limits">Read it with these limits</h3>
<ul>
<li>One novel, one language pair: eleven chapters of 화산귀환 (Return of the Mount Hua
Sect), Korean to English. Chinese and Japanese works are next.</li>
<li>The judges are AI models, and judges can favour their own family. Claude Haiku 5.5
and Gemini 3.8 Flash are judges and are also ranked; compare their score from the other
judge in the table.{agreement}</li>
<li>The ordering is close at the top. The side-by-side comparisons rank the leading
models differently from the quality score, and their rating ranges overlap.</li>
<li>Claude and GPT models ran through their command-line tools with every tool
switched off, Gemini through Google's Antigravity tool, and the rest through their
providers' APIs. The text sent was the same.</li>
<li>Models change under a stable name. These are the versions tested in run
2026-09, reported {esc(run_date)}.</li>
</ul>
<p>Method, rubric and every comparison: <a href="{REPO_URL}">the repository</a>.</p>
</div>
"""
    title = "Best AI for Web Novel Translation (Korean to English) – Ranked by Quality and Cost"
    description = (
        f"{len(rows)} AI models, including Gemini, Claude, GPT, DeepSeek, Kimi and Qwen, "
        f"ranked on translating Korean web novels to English: quality scores and the cost "
        f"per chapter for each."
    )
    jsonld = [
        {
            "@context": "https://schema.org", "@type": "Dataset", "name": TITLE,
            "description": description, "url": SITE_URL, "sameAs": REPO_URL,
            "keywords": ["web novel translation", "AI translation benchmark", "Korean web novel",
                         "machine translation", "LLM leaderboard", "murim", "manhwa novel"],
            "dateModified": date.today().isoformat(),
            "variableMeasured": ["translation quality score", "cost per chapter (USD)",
                                 "head-to-head rating", "output tokens per chapter"],
            "isAccessibleForFree": True,
        },
        {
            "@context": "https://schema.org", "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": q,
                 "acceptedAnswer": {"@type": "Answer", "text": a}}
                for q, a in questions
            ],
        },
        {
            "@context": "https://schema.org", "@type": "ItemList",
            "name": "AI models ranked for web novel translation",
            "itemListElement": [
                {"@type": "ListItem", "position": r["rank"], "name": name(r["model_id"]),
                 "url": f"{SITE_URL}models/{slug(r['model_id'])}/"}
                for r in rows
            ],
        },
    ]
    (out / "index.html").write_text(page(title, description, "", body, jsonld), encoding="utf-8")


def build_model_page(r, rows, out: Path) -> None:
    mid = r["model_id"]
    n, mk = name(mid), maker(mid)
    cost = r.get("list_cost_per_chapter")
    axes = "".join(
        f"<tr><th scope=\"row\" style=\"text-align:left\">{a.capitalize()}</th>"
        f"<td>{r['axis_means'][a]:.1f} / 5</td><td style=\"text-align:left\" class=\"hide-sm\">{AXIS_HELP[a]}</td></tr>"
        for a in AXES if r["axis_means"].get(a) is not None
    )
    better_cheaper = [
        o for o in rows
        if o is not r and o.get("list_cost_per_chapter") is not None and cost is not None
        and o["list_cost_per_chapter"] <= cost and o["composite"] > r["composite"]
    ]
    if cost is None:
        verdict = "Its cost per chapter was not recorded in this run."
    elif not better_cheaper:
        verdict = "No model in this test scored higher for the same price or less."
    else:
        alt = max(better_cheaper, key=lambda o: o["composite"])
        verdict = (f"{name(alt['model_id'])} scored higher ({alt['composite']:.1f}) for "
                   f"{money(alt['list_cost_per_chapter'])} a chapter.")
    jc = r.get("judge_composites") or {}
    judge_line = " and ".join(
        f"{JUDGES[j][0]} gave {jc[j]:.1f}" for j in JUDGES if jc.get(j) is not None
    )
    caveat = (f"<p>{esc(judge_line)}.</p>" if judge_line else "") + (
        "<p>This model is also one of the two judges, so one of those scores comes "
        "from a judge of its own family.</p>" if mid in SAME_FAMILY_AS_JUDGE else "")
    body = f"""
<p><a href="../../">All models</a></p>
<h1>{esc(n)} for web novel translation</h1>
<p class="lede">How {esc(n)} by {esc(mk)} translated eleven chapters of a Korean web novel
into English: rank {r['rank']} of {len(rows)}.</p>
<div class="stat">
<div><b>{r['composite']:.1f}</b><span>quality score of 100</span></div>
<div><b>{money(cost)}</b><span>per chapter at list price</span></div>
<div><b>{per_100(cost)}</b><span>per 100 chapters</span></div>
<div><b>{f"{r['rating']:.0f}" if r.get('rating') else "–"}</b><span>head-to-head rating</span></div>
</div>
<div class="prose">
<p>{esc(verdict)}</p>
{caveat}
<h2>Score by quality</h2>
<div class="table-wrap"><table><tbody>{axes}</tbody></table></div>
<p>{r['scored']} of {r['total_tasks']} chapters were scored; a chapter the model refused
or failed is counted separately. {f"It wrote about {r['tokens_out_per_chapter']:,.0f} output tokens per chapter, reasoning included." if r.get('tokens_out_per_chapter') else ""}</p>
</div>
"""
    title = f"{n} for Web Novel Translation: Score {r['composite']:.1f}, {money(cost)} per Chapter"
    description = (f"How well {n} ({mk}) translates Korean web novels into English: quality "
                   f"{r['composite']:.1f}/100, rank {r['rank']} of {len(rows)}, "
                   f"{money(cost)} per chapter at list price.")
    jsonld = [{
        "@context": "https://schema.org", "@type": "SoftwareApplication", "name": n,
        "applicationCategory": "AI language model", "creator": {"@type": "Organization", "name": mk},
        "review": {"@type": "Review", "author": {"@type": "Organization", "name": TITLE},
                   "reviewRating": {"@type": "Rating", "ratingValue": round(r["composite"], 1),
                                    "bestRating": 100, "worstRating": 0},
                   "reviewBody": f"Translation of Korean web novel chapters into English. {verdict}"},
    }]
    path = f"models/{slug(mid)}/"
    target = out / path
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.html").write_text(page(title, description, path, body, jsonld), encoding="utf-8")


def build(run: str, archive: Path, out: Path) -> None:
    data, rows, n_comparisons = load(run, archive)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    build_index(data, rows, n_comparisons, out)
    for r in rows:
        build_model_page(r, rows, out)
    urls = [SITE_URL] + [f"{SITE_URL}models/{slug(r['model_id'])}/" for r in rows]
    today = date.today().isoformat()
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"<url><loc>{u}</loc><lastmod>{today}</lastmod></url>\n" for u in urls)
        + "</urlset>\n",
        encoding="utf-8",
    )
    (out / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}sitemap.xml\n",
                                    encoding="utf-8")
    (out / "404.html").write_text(
        page(f"Page not found – {TITLE}", "This page does not exist.", "404.html",
             f'<h1>That page does not exist.</h1><p class="lede">The ranking is on '
             f'<a href="{SITE_URL}">the front page</a>.</p>', []),
        encoding="utf-8",
    )
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {out}: {len(rows)} model pages, {len(urls)} URLs in the sitemap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="2026-09")
    parser.add_argument("--archive", default="archive/2026-09-hwasan")
    parser.add_argument("--out", default="site")
    args = parser.parse_args()
    build(args.run, REPO / args.archive, REPO / args.out)


if __name__ == "__main__":
    main()
