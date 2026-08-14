"""Orchestrate the full comparison and collect results."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import fitz

from .config import Config
from .extract import extract_comments
from .align import prepare_page, PageData
from .diff import compute_change, DiffResult
from .classify import classify_page, Finding, Unexpected, Verifier
from . import prematch
from .raster import render_rgb


@dataclass
class PageResult:
    page_index: int            # redlined page index (the engineer's sheet)
    page: PageData
    diff: DiffResult
    findings: List[Finding]
    unexpected: List[Unexpected]
    final_index: int = -1      # matched revised page index


@dataclass
class ExtraPage:
    kind: str                  # 'inserted' | 'unmatched' | 'error'
    label: str
    thumb: np.ndarray          # small RGB preview
    note: str = ""


@dataclass
class Report:
    redlined_path: str
    final_path: str
    config: Config
    pages: List[PageResult] = field(default_factory=list)
    extra: List[ExtraPage] = field(default_factory=list)
    flattened_pages: List[int] = field(default_factory=list)  # 1-based sheets whose
    # markups were FLATTENED (no annotations -> caught via red-ink fallback)

    @property
    def all_findings(self) -> List[Finding]:
        return [f for p in self.pages for f in p.findings]

    @property
    def all_unexpected(self) -> List[Unexpected]:
        return [u for p in self.pages for u in p.unexpected]


def choose_dpi(doc: "fitz.Document", cfg: Config) -> int:
    """Pick a render DPI from the drawing's text size so body text lands at
    ~auto_dpi_target_px pixels tall (smaller CAD text -> higher DPI). Falls back
    to auto_dpi_default when the page has no extractable text."""
    sizes = []
    for pno in range(min(doc.page_count, 6)):
        try:
            blocks = doc[pno].get_text("dict")["blocks"]
        except Exception:
            continue
        for b in blocks:
            for ln in b.get("lines", []):
                for s in ln.get("spans", []):
                    if len(s.get("text", "").strip()) >= 2 and s.get("size", 0) > 0:
                        sizes.append(s["size"])
    if not sizes:
        return cfg.auto_dpi_default
    sizes.sort()
    median = sizes[len(sizes) // 2]
    dpi = int(round((cfg.auto_dpi_target_px / median * 72.0) / 25) * 25)
    return max(cfg.auto_dpi_min, min(cfg.auto_dpi_max, dpi))


def _region_fingerprint(final_rgb: np.ndarray, bbox, grid: int) -> "np.ndarray | None":
    """A small normalised grayscale fingerprint of the revised-image crop at a
    change region — used to recognise the SAME content (a stamp/signature/revision
    row) recurring across sheets, regardless of its exact position."""
    from PIL import Image
    h, w = final_rgb.shape[:2]
    x0 = max(0, int(bbox[0])); y0 = max(0, int(bbox[1]))
    x1 = min(w, int(bbox[2])); y1 = min(h, int(bbox[3]))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    crop = final_rgb[y0:y1, x0:x1]
    g = np.asarray(Image.fromarray(crop).convert("L").resize((grid, grid),
                                                             Image.BILINEAR),
                   dtype=np.float32).ravel()
    g -= g.mean()
    nrm = float(np.linalg.norm(g))
    return (g / nrm) if nrm > 1e-6 else None


def _suppress_boilerplate(report: "Report", cfg: Config) -> None:
    """Remove 'unexpected' changes that are boilerplate (the architect's licence
    stamp/signature appearing when a PRELIMINARY set is signed, a revision-table row,
    a titleblock edit) rather than design changes.

    Two complementary tests, unioned:
      1) CONTENT replication (primary) — fingerprint each region's revised crop and
         drop any cluster of look-alike regions spanning >= boilerplate_min_pages
         distinct sheets. Position-independent, so a stamp that re-plots at a slightly
         shifting centre — and only cleanly separates on some sheets — is still caught.
      2) LOCATION key (legacy, conservative) — same rounded centre on a large fraction
         of sheets. Catches fixed-position boilerplate whose content varies per sheet.
    """
    from collections import defaultdict
    pages = report.pages
    n = len(pages)
    if n < cfg.boilerplate_min_pages:
        return

    # Flatten all unexpected regions with a stable handle.
    items = []   # (pr, u, fingerprint, (bw, bh))
    for pr in pages:
        for u in pr.unexpected:
            fp = _region_fingerprint(pr.page.final_rgb, u.region.bbox_px,
                                     cfg.boilerplate_fp_grid)
            bw = u.region.bbox_px[2] - u.region.bbox_px[0]
            bh = u.region.bbox_px[3] - u.region.bbox_px[1]
            items.append([pr, u, fp, (bw, bh), False])   # last = drop flag

    # 1) Content-replication clustering (greedy, single-link by similarity + size).
    tol = cfg.boilerplate_size_tol
    for i, a in enumerate(items):
        if a[2] is None:
            continue
        cluster = [a]
        pgs = {a[0].page_index}
        for b in items[i + 1:]:
            if b[2] is None or b[4]:
                continue
            sim = float(np.dot(a[2], b[2]))
            if sim < cfg.boilerplate_fp_sim:
                continue
            (aw, ah), (bw, bh) = a[3], b[3]
            if not (abs(bw - aw) <= tol * max(aw, 1) and abs(bh - ah) <= tol * max(ah, 1)):
                continue
            cluster.append(b)
            pgs.add(b[0].page_index)
        if len(pgs) >= cfg.boilerplate_min_pages:
            for m in cluster:
                m[4] = True

    # 2) Legacy location-key suppression (high-fraction, conservative).
    def _key(pr, u):
        h, w = pr.page.base_ink.shape
        x0, y0, x1, y1 = u.region.bbox_px
        return (round((x0 + x1) / 2 / w, 2), round((y0 + y1) / 2 / h, 2))

    seen = defaultdict(set)
    for pr in pages:
        for u in pr.unexpected:
            seen[_key(pr, u)].add(pr.page_index)
    thresh = max(cfg.boilerplate_min_pages, int(round(cfg.boilerplate_min_frac * n)))
    boiler = {k for k, pgs in seen.items() if len(pgs) >= thresh}
    for it in items:
        if _key(it[0], it[1]) in boiler:
            it[4] = True

    drop = {(id(it[0]), id(it[1])) for it in items if it[4]}
    if not drop:
        return
    for pr in pages:
        pr.unexpected = [u for u in pr.unexpected
                         if (id(pr), id(u)) not in drop]


def _thumb(page, cfg: Config, annots: bool, max_w: int = 700) -> np.ndarray:
    rgb = render_rgb(page, cfg.extra_thumb_dpi, annots=annots)
    h, w = rgb.shape[:2]
    if w > max_w:
        from PIL import Image
        im = Image.fromarray(rgb).resize((max_w, int(h * max_w / w)))
        rgb = np.asarray(im)
    return rgb


def run(redlined_path: str, final_path: str, cfg: Config,
        verifier: Optional[Verifier] = None, progress=True, progress_cb=None,
        match: "prematch.MatchResult | None" = None) -> Report:
    rdoc = fitz.open(redlined_path)
    fdoc = fitz.open(final_path)
    report = Report(redlined_path=redlined_path, final_path=final_path, config=cfg)

    if cfg.auto_dpi:
        cfg.analysis_dpi = choose_dpi(rdoc, cfg)
        # OCR is calibrated at cfg.ocr_render_dpi (300). Don't let auto-DPI push the
        # render higher when OCR is on — Tesseract mis-reads short labels (L1/B1) at
        # 350+, inflating false misses. (At 300 the diff is unaffected: its grain is
        # in physical points, so it's DPI-invariant.)
        if cfg.use_ocr:
            cfg.analysis_dpi = min(cfg.analysis_dpi, cfg.ocr_render_dpi)
        if progress:
            print(f"  auto-DPI: rendering at {cfg.analysis_dpi} (from drawing text size)",
                  file=sys.stderr, flush=True)

    # Precheck: structurally match pages (architect may insert/split pages).
    if match is None:
        if progress:
            print("  precheck: matching pages structurally...", file=sys.stderr, flush=True)
        match = prematch.match_documents(rdoc, fdoc, cfg)

    pairs = match.pairs
    total = len(pairs)
    for n, (ridx, fidx) in enumerate(pairs, start=1):
        if progress:
            print(f"  page {ridx + 1}->{fidx + 1} ({n}/{total}) ...", file=sys.stderr, flush=True)
        try:
            comments = extract_comments(rdoc[ridx], ridx, cfg)
            page = prepare_page(rdoc[ridx], fdoc[fidx], ridx, cfg)
            diff = compute_change(page, cfg)
            # Fallback: markups flattened into the page (no annotations) -> detect
            # them from the red markup ink we already rendered.
            flattened = False
            if cfg.flatten_markups and not comments:
                from .extract import detect_flattened_markups
                comments = detect_flattened_markups(page.redlined_rgb, ridx, cfg,
                                                    final_rgb=page.final_rgb)
                flattened = bool(comments)   # only flag if red markups were actually found
            findings, unexpected = classify_page(page, comments, diff, cfg, verifier)
            if flattened:
                report.flattened_pages.append(ridx + 1)
            # Heavily re-plotted / unalignable sheet: the diff can't tell a real
            # change from a moved dimension, so DON'T emit per-region "unexpected"
            # noise — record one page-level "review manually" notice instead.
            if page.similarity < cfg.min_align_sim_unexpected and unexpected:
                report.extra.append(ExtraPage(
                    kind="lowconf", label=f"Sheet {ridx + 1}",
                    thumb=_thumb(fdoc[fidx], cfg, annots=True),
                    note=(f"heavily re-plotted — only {page.similarity:.0%} ink overlap "
                          "after alignment, so automatic change-detection is unreliable "
                          "here. Per-change flags were suppressed to avoid false "
                          "positives; review this whole sheet by eye. Your redline "
                          "targets (misses/captures) are still checked.")))
                unexpected = []
            if page.similarity < 0.15 and progress:
                print(f"    warning: low ink overlap ({page.similarity:.2f}) on "
                      f"redlined p{ridx + 1}", file=sys.stderr)
            report.pages.append(PageResult(ridx, page, diff, findings, unexpected,
                                           final_index=fidx))
        except Exception as e:  # one bad page must not sink the whole comparison
            if progress:
                print(f"    skipped redlined p{ridx + 1} / revised p{fidx + 1}: {e}",
                      file=sys.stderr)
            report.extra.append(ExtraPage(
                kind="error", label=f"Redlined p{ridx + 1} ↔ revised p{fidx + 1}",
                thumb=_thumb(fdoc[fidx], cfg, annots=True),
                note=f"could not compare automatically: {e}"))
        if progress_cb:
            progress_cb(n, total)

    _suppress_boilerplate(report, cfg)

    # Pages the architect inserted (in revision, no redline counterpart).
    for fidx in match.extra_revised:
        report.extra.append(ExtraPage(
            kind="inserted", label=f"Revised page {fidx + 1}",
            thumb=_thumb(fdoc[fidx], cfg, annots=True),
            note="in the revision but not in your redlined set — added by the architect"))
    # Redlined pages with no counterpart in the revision.
    for ridx in match.extra_redlined:
        report.extra.append(ExtraPage(
            kind="unmatched", label=f"Redlined page {ridx + 1}",
            thumb=_thumb(rdoc[ridx], cfg, annots=True),
            note="in your redlined set but no matching page found in the revision"))
    return report
