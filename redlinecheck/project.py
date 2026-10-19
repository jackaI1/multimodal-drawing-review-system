"""Whole-project comparison: match a folder of redlined PDFs to a folder of revised
PDFs, compare every pair, and aggregate the findings into one result.

This is the real engineer workflow — drop the entire project (one PDF per block /
sheet) on each side. Junk files (non-PDFs, corrupt PDFs, drawings with no
counterpart) are skipped SILENTLY: the tool never errors out or stops just because
an unsupported file was in the folder.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import fitz

from .config import Config
from . import pipeline, prematch
from .report import build_report


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]


@dataclass
class ProjectInputs:
    valid_redlined: List[str] = field(default_factory=list)   # (label, path)
    valid_revised: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (name, why)


def is_valid_pdf(path: str) -> Tuple[bool, str]:
    """A file we can actually compare: a readable PDF with at least one page.
    Anything else (a .docx, a .dwg, a 0-byte file, a corrupt PDF) is 'unsupported'
    and should be skipped, not error the whole run."""
    if not os.path.isfile(path):
        return False, "not a file"
    if os.path.splitext(path)[1].lower() != ".pdf":
        return False, "not a PDF"
    try:
        doc = fitz.open(path)
        try:
            if doc.page_count < 1:
                return False, "empty PDF (0 pages)"
            if doc.needs_pass:
                return False, "password-protected PDF"
        finally:
            doc.close()
    except Exception as e:
        return False, f"unreadable PDF ({type(e).__name__})"
    return True, ""


def collect_inputs(redlined_paths: List[str], revised_paths: List[str]) -> ProjectInputs:
    """Filter both sides to valid, comparable PDFs; record what was skipped."""
    pi = ProjectInputs()
    for p in redlined_paths:
        ok, why = is_valid_pdf(p)
        (pi.valid_redlined if ok else pi.skipped).append(
            p if ok else (os.path.basename(p), why))
    for p in revised_paths:
        ok, why = is_valid_pdf(p)
        (pi.valid_revised if ok else pi.skipped).append(
            p if ok else (os.path.basename(p), why))
    return pi


@dataclass
class Pair:
    redlined: str
    revised: str
    score: float
    rsig: object = None
    fsig: object = None


def _name_key(path: str) -> str:
    """Normalise a filename to its drawing key for name-based matching:
    '0000-EXAMPLE-BLOCK A - DLE Markup#1.PDF' and 'EXAMPLE-BLOCK A.PDF' -> 'exampleblocka'."""
    b = os.path.splitext(os.path.basename(path))[0].lower()
    b = re.sub(r"(?<![a-z])redlined(?![a-z])", " ", b)
    # Strip decoration but KEEP the sheet number: '#1' is the markup revision marker
    # (drop it), whereas a bare trailing number like '-8'/'-10' is the sheet id and
    # MUST be kept (else EXAMPLE-8 and EXAMPLE-10 collide to the same key).
    b = re.sub(r"dle|markup|#\s*\d+|revised|final|tender|side\s*upg|\bsu\b", " ", b)
    # If your drawing sets prefix every filename with a job number, add those
    # prefixes to this pattern (e.g. r"12345|67890|rev\.?\s*\d+") so they are
    # stripped before matching.
    b = re.sub(r"rev\.?\s*\d+", " ", b)
    b = re.sub(r"[^a-z0-9]+", "", b)
    return b


def match_files(redlined: List[str], revised: List[str], cfg: Config,
                min_score: float = 0.55) -> Tuple[List[Pair], List[str], List[str]]:
    """Pair each redlined file with its best revised counterpart, using a structural
    content fingerprint backed up by filename similarity. Returns (pairs, unmatched
    redlined, unmatched revised). A redlined file with no good match is reported, not
    forced onto an unrelated drawing."""
    def sig(path):
        doc = fitz.open(path)
        try:
            return prematch.doc_signature(doc, cfg, annots=True)
        finally:
            doc.close()
    rsigs = [sig(p) for p in redlined]
    fsigs = [sig(p) for p in revised]
    rkeys = [_name_key(p) for p in redlined]
    fkeys = [_name_key(p) for p in revised]

    used = set()
    pairs: List[Pair] = []
    unmatched_r: List[str] = []
    # Greedy by best score; each revised file used at most once.
    order = []
    for i in range(len(redlined)):
        for j in range(len(revised)):
            s = float(np.dot(rsigs[i], fsigs[j]))
            # strong name agreement is a big boost (handles re-plots that look different)
            if rkeys[i] and rkeys[i] == fkeys[j]:
                s = max(s, 0.99)
            order.append((s, i, j))
    order.sort(reverse=True)
    matched_r = set()
    for s, i, j in order:
        if i in matched_r or j in used:
            continue
        if s < min_score:
            continue
        matched_r.add(i); used.add(j)
        pairs.append(Pair(redlined=redlined[i], revised=revised[j], score=s))
    unmatched_r = [redlined[i] for i in range(len(redlined)) if i not in matched_r]
    unmatched_f = [revised[j] for j in range(len(revised)) if j not in used]
    pairs.sort(key=lambda p: os.path.basename(p.redlined).lower())
    return pairs, unmatched_r, unmatched_f


def _write_project_index(out_dir, pairs, rows, skipped, un_r, un_f) -> None:
    """A one-page project overview: per-drawing counts linking to each drawing's full
    summary.html, plus matched pairs and ignored files. The single entry point an
    engineer opens for a whole project."""
    import html as _html
    from collections import Counter, defaultdict
    MISS = {"NO_CHANGE_DETECTED", "MISSED"}
    VERIFY = {"CHANGED_AT_TARGET", "COORDINATION", "COPY_REF", "COPY_CHANGE",
              "CHANGED_OTHER", "MARKED_REGION"}
    by_src = defaultdict(Counter)
    for r in rows:
        by_src[r.get("source", "")][r["status"]] += 1
    sub_of = {os.path.splitext(os.path.basename(p.redlined))[0]:
              f"{i:02d}_{_safe(os.path.splitext(os.path.basename(p.redlined))[0])}"
              for i, p in enumerate(pairs)}

    def cell(c, keys):
        return sum(c[k] for k in keys)

    trows = []
    for src in sorted(by_src):
        c = by_src[src]
        miss, un = cell(c, MISS), c.get("UNEXPECTED", 0)
        ver, ok = cell(c, VERIFY), c.get("CAPTURED", 0)
        link = sub_of.get(src, "")
        name = (f'<a href="{_html.escape(link)}/summary.html">{_html.escape(src)}</a>'
                if link else _html.escape(src))
        trows.append(
            f"<tr><td>{name}</td><td class='m'>{miss}</td><td class='u'>{un}</td>"
            f"<td class='v'>{ver}</td><td class='o'>{ok}</td></tr>")
    tot = Counter()
    for c in by_src.values():
        tot.update(c)
    total_row = (f"<tr class='tot'><td>All drawings</td>"
                 f"<td class='m'>{cell(tot, MISS)}</td><td class='u'>{tot.get('UNEXPECTED',0)}</td>"
                 f"<td class='v'>{cell(tot, VERIFY)}</td><td class='o'>{tot.get('CAPTURED',0)}</td></tr>")

    extra_html = ""
    if un_r or un_f or skipped:
        parts = ["<h2>Files needing attention</h2><ul>"]
        for p in un_r:
            parts.append(f"<li>🟠 <b>{_html.escape(os.path.basename(p))}</b> — redlined "
                         "drawing with no matching revised file.</li>")
        for p in un_f:
            parts.append(f"<li>🟣 <b>{_html.escape(os.path.basename(p))}</b> — revised "
                         "drawing with no matching redlined file.</li>")
        for name, why in skipped:
            parts.append(f"<li>⚪ <b>{_html.escape(name)}</b> — ignored ({_html.escape(why)}).</li>")
        parts.append("</ul>")
        extra_html = "\n".join(parts)

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Redline check — project</title>
<style>body{{font:14px/1.45 -apple-system,Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1b1f24}}
header{{background:#1b1f24;color:#fff;padding:18px 26px}} header h1{{margin:0;font-size:19px}}
main{{max-width:900px;margin:0 auto;padding:22px}}
table{{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #eef0f3}}
th{{background:#fafbfc;font-size:12px;color:#566}} td.m,td.u,td.v,td.o{{text-align:center;font-weight:600}}
td.m{{color:#d20000}} td.u{{color:#a500c8}} td.v{{color:#e68c00}} td.o{{color:#009600}}
tr.tot td{{border-top:2px solid #d0d3d8;font-weight:700;background:#fafbfc}}
a{{color:#0b6cc0}} h2{{font-size:15px;margin:24px 0 8px}} li{{margin:3px 0}}</style></head><body>
<header><h1>📐 Redline check — project overview</h1></header><main>
<p>{len(pairs)} drawing pair(s) compared. Click a drawing to open its full report.</p>
<table><tr><th>Drawing</th><th>Possible misses</th><th>Unexpected</th><th>To verify</th><th>Addressed</th></tr>
{''.join(trows)}{total_row}</table>
{extra_html}
</main></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(doc)


def compare_project(redlined_paths: List[str], revised_paths: List[str], cfg: Config,
                    out_dir: str, verifier=None, progress_cb=None) -> dict:
    """Compare a whole project. Aggregates every matched pair's findings into one
    result dict (rows carry a `source` = drawing name and a globally-unique `gpidx`).
    Skipped/unmatched files are surfaced under `extra`, never fatal."""
    os.makedirs(out_dir, exist_ok=True)
    pi = collect_inputs(redlined_paths, revised_paths)
    pairs, un_r, un_f = match_files(pi.valid_redlined, pi.valid_revised, cfg)

    agg_rows: List[dict] = []
    agg_overviews: dict = {}
    agg_extra: List[dict] = []
    counts: Counter = Counter()
    flattened: List[str] = []
    n = len(pairs)
    for i, pr in enumerate(pairs):
        label = os.path.splitext(os.path.basename(pr.redlined))[0]
        sub = f"{i:02d}_{_safe(label)}"
        pair_out = os.path.join(out_dir, sub)
        try:
            rep = pipeline.run(pr.redlined, pr.revised, cfg, verifier=verifier,
                               progress=False)
            res = build_report(rep, pair_out)
        except Exception as e:
            agg_extra.append({"kind": "error", "label": f"{label}",
                              "note": f"could not compare automatically: {e}", "img": ""})
            if progress_cb:
                progress_cb(i + 1, n)
            continue
        for row in res["rows"]:
            row["source"] = label
            row["gpidx"] = f"{i}:{row['pidx']}"
            for k in ("before", "after", "before_wide", "after_wide"):
                if row.get(k):
                    row[k] = f"{sub}/{row[k]}"
            agg_rows.append(row)
        for pidx, ov in res["overviews"].items():
            agg_overviews[f"{i}:{pidx}"] = f"{sub}/{ov}"
        for e in res["extra"]:
            if e.get("img"):
                e["img"] = f"{sub}/{e['img']}"
            e["label"] = f"{label}: {e['label']}"
            agg_extra.append(e)
        for k, v in res["counts"].items():
            counts[k] += v
        for fp in res.get("flattened_pages", []):
            flattened.append(f"{label} p{fp}")
        if progress_cb:
            progress_cb(i + 1, n)

    # Unmatched / skipped files -> visible but non-fatal notices.
    for p in un_r:
        agg_extra.append({"kind": "unmatched", "img": "",
                          "label": os.path.basename(p),
                          "note": "redlined drawing with no matching revised file."})
    for p in un_f:
        agg_extra.append({"kind": "inserted", "img": "",
                          "label": os.path.basename(p),
                          "note": "revised drawing with no matching redlined file "
                                  "(new / not marked up)."})
    counts["extra_pages"] = sum(1 for e in agg_extra
                                if e["kind"] in ("inserted", "unmatched", "lowconf", "error"))
    _write_project_index(out_dir, pairs, agg_rows, pi.skipped, un_r, un_f)
    return {
        "rows": agg_rows, "overviews": agg_overviews, "counts": dict(counts),
        "extra": agg_extra, "out_dir": out_dir, "flattened_pages": flattened,
        "project": True,
        "pairs": [{"redlined": os.path.basename(p.redlined),
                   "revised": os.path.basename(p.revised),
                   "score": round(p.score, 2)} for p in pairs],
        "skipped": [{"name": n, "why": w} for n, w in pi.skipped],
        "n_pairs": n,
    }
