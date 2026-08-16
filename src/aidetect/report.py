"""Self-contained HTML report generation.

Images are embedded as ``data:`` URIs and all CSS is inline, so a report is a
single portable file that renders correctly offline and in both light and dark
themes. No JavaScript, no external assets.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .io_utils import atomic_write_text, data_uri
from .logging_utils import get_logger
from .types import AnalysisResult

logger = get_logger(__name__)

_CSS = """
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #16181d; --muted: #5c6270;
  --line: #e2e5ea; --fake: #d3383d; --real: #16875a; --accent: #3355dd;
  --bar-track: #e8eaef;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1115; --panel: #171a21; --ink: #e8eaf0; --muted: #9aa1b1;
    --line: #262b35; --fake: #ff6b70; --real: #4fd39a; --accent: #7c9bff;
    --bar-track: #232833;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1115; --panel: #171a21; --ink: #e8eaf0; --muted: #9aa1b1;
  --line: #262b35; --fake: #ff6b70; --real: #4fd39a; --accent: #7c9bff;
  --bar-track: #232833;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 64px; background: var(--bg); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.02em; }
h2 { font-size: 18px; margin: 0 0 12px; }
.sub { color: var(--muted); margin: 0 0 28px; font-size: 14px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 28px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
.card .value { font-size: 24px; font-weight: 600; margin-top: 4px; }
.item { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px; margin-bottom: 20px; }
.head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; margin-bottom: 12px; }
.name { font-weight: 600; word-break: break-all; }
.badge { font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 999px; color: #fff; }
.badge.fake { background: var(--fake); } .badge.real { background: var(--real); }
.badge.error { background: var(--muted); }
.meta { color: var(--muted); font-size: 13px; }
.figs { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 6px; }
.fig { flex: 0 0 auto; text-align: center; }
.fig img { max-height: 210px; border-radius: 8px; border: 1px solid var(--line); display: block; }
.fig span { display: block; color: var(--muted); font-size: 12px; margin-top: 5px; }
.desc { margin: 14px 0; padding: 12px 14px; background: var(--bar-track); border-left: 3px solid var(--accent); border-radius: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
.bar { height: 7px; border-radius: 4px; background: var(--bar-track); overflow: hidden; min-width: 90px; }
.bar > i { display: block; height: 100%; background: var(--accent); }
.scroll { overflow-x: auto; }
footer { color: var(--muted); font-size: 12px; margin-top: 36px; text-align: center; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _figure(path: str, caption: str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    try:
        uri = data_uri(file_path)
    except OSError:  # pragma: no cover
        return ""
    return f'<div class="fig"><img src="{uri}" alt="{_escape(caption)}"><span>{_escape(caption)}</span></div>'


def _artifact_table(result: AnalysisResult) -> str:
    if not result.artifacts or not result.artifacts.matches:
        return ""
    rows = []
    for match in result.artifacts.matches:
        width = max(2, min(100, int(match.score * 100)))
        rows.append(
            "<tr>"
            f"<td>{match.rank}</td>"
            f"<td>{_escape(match.descriptor)}</td>"
            f"<td>{_escape(match.category)}</td>"
            f'<td><div class="bar"><i style="width:{width}%"></i></div></td>'
            f"<td>{match.score:.3f}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>#</th><th>Artifact</th>'
        "<th>Family</th><th>Score</th><th></th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _result_block(result: AnalysisResult) -> str:
    detection = result.detection
    if not result.ok:
        badge = '<span class="badge error">ERROR</span>'
        meta = _escape(result.error or "unknown error")
    else:
        css = "fake" if detection.is_fake else "real"
        badge = f'<span class="badge {css}">{_escape(detection.prediction)}</span>'
        meta = (
            f"{detection.confidence:.1f}% confidence · "
            f"FAKE p={detection.fake_prob:.3f} · "
            f"{_escape(detection.model_type)} checkpoint · "
            f"{detection.latency_ms:.0f} ms"
        )

    figures = "".join(
        _figure(path, label)
        for label, path in (
            ("annotated", result.saved_files.get("annotated", "")),
            ("saliency overlay", result.saved_files.get("overlay", "")),
            ("suspicious region", result.saved_files.get("crop", "")),
            ("raw saliency", result.saved_files.get("raw", "")),
        )
        if path
    )

    description = (
        f'<div class="desc">{_escape(result.description)}</div>' if result.description else ""
    )
    regions = ""
    if result.regions:
        parts = [
            f"#{r.rank} {r.position_phrase()} ({r.width}×{r.height}px, score {r.score:.2f})"
            for r in result.regions
        ]
        regions = f'<p class="meta">Regions: {_escape(" · ".join(parts))}</p>'

    return f"""
    <section class="item">
      <div class="head">
        <span class="name">{_escape(Path(result.image_path).name)}</span>
        {badge}
        <span class="meta">{meta}</span>
      </div>
      {f'<div class="figs">{figures}</div>' if figures else ''}
      {description}
      {regions}
      {_artifact_table(result)}
    </section>
    """


def _summary_cards(results: Sequence[AnalysisResult]) -> str:
    ok = [r for r in results if r.ok]
    fake = sum(1 for r in ok if r.detection.is_fake)
    mean_conf = sum(r.detection.confidence for r in ok) / len(ok) if ok else 0.0
    failed = len(results) - len(ok)

    cards = [
        ("Images", str(len(results))),
        ("Flagged FAKE", str(fake)),
        ("Classified REAL", str(len(ok) - fake)),
        ("Mean confidence", f"{mean_conf:.1f}%"),
    ]
    if failed:
        cards.append(("Failed", str(failed)))

    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="label">{_escape(label)}</div>'
        f'<div class="value">{_escape(value)}</div></div>'
        for label, value in cards
    ) + "</div>"


def build_html(
    results: Sequence[AnalysisResult],
    title: str = "AI Image Detection Report",
    subtitle: Optional[str] = None,
) -> str:
    """Render an entire run into one standalone HTML document."""
    blocks = "".join(_result_block(result) for result in results)
    subtitle = subtitle or f"{len(results)} image(s) analysed"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
  <div class="wrap">
    <h1>{_escape(title)}</h1>
    <p class="sub">{_escape(subtitle)}</p>
    {_summary_cards(results)}
    {blocks}
    <footer>Generated by <code>aidetect</code> — DINOv2 detection · SigLIP artifact classification · VLM explanation</footer>
  </div>
</body>
</html>
"""


def write_report(
    results: Sequence[AnalysisResult],
    output_path: str | Path,
    title: str = "AI Image Detection Report",
    subtitle: Optional[str] = None,
) -> Path:
    path = atomic_write_text(output_path, build_html(results, title, subtitle))
    logger.info("Wrote HTML report for %d image(s) -> %s", len(results), path)
    return path


def load_results(paths: Iterable[str | Path]) -> List[AnalysisResult]:
    """Rebuild results from ``result.json`` files or a ``results.jsonl``."""
    from .io_utils import load_json, read_jsonl

    results: List[AnalysisResult] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            results.extend(load_results(sorted(path.rglob("result.json"))))
            jsonl = path / "results.jsonl"
            if jsonl.is_file():
                results.extend(AnalysisResult.from_dict(rec) for rec in read_jsonl(jsonl))
        elif path.suffix == ".jsonl":
            results.extend(AnalysisResult.from_dict(rec) for rec in read_jsonl(path))
        elif path.suffix == ".json":
            payload = load_json(path)
            records = payload if isinstance(payload, list) else [payload]
            results.extend(AnalysisResult.from_dict(rec) for rec in records)
    return results
