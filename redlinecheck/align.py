"""Render a matched page pair into a common analysis frame and register them.

The analysis frame is defined by the *redlined* page rendered without markups
(`annots=False`). The revised page is rendered as-displayed and shifted to line
up with it. Both sample PDFs share a coordinate system, so the shift is usually
near zero, but phase correlation handles small re-plot offsets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import fitz

from .config import Config
from .raster import render_rgb, ink_mask


@dataclass
class PageData:
    page_index: int
    zoom: float
    base_rgb: np.ndarray      # redlined page, markups removed (architect original)
    redlined_rgb: np.ndarray  # redlined page WITH your markups (for before-crops)
    final_rgb: np.ndarray     # revised page, registered onto base frame
    base_ink: np.ndarray      # bool
    final_ink: np.ndarray     # bool, registered
    shift: tuple              # (dy, dx) applied to final
    similarity: float         # ink overlap after registration (0..1 sanity check)
    redlined_page: "fitz.Page"
    final_page: "fitz.Page"


def _phase_shift(a: np.ndarray, b: np.ndarray, max_shift: int) -> tuple:
    """Estimate integer (dy,dx) to move b onto a; clamp to +/-max_shift."""
    af = a.astype(np.float32); bf = b.astype(np.float32)
    af -= af.mean(); bf -= bf.mean()
    F = np.fft.rfft2(af); G = np.fft.rfft2(bf)
    R = F * np.conj(G)
    R /= np.abs(R) + 1e-6
    corr = np.fft.irfft2(R, s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    if abs(dy) > max_shift or abs(dx) > max_shift:
        return 0, 0
    return int(dy), int(dx)


def _iou_at(a: np.ndarray, b: np.ndarray, shift, sub: int = 3) -> float:
    """Ink intersection-over-union of a vs b shifted by (dy,dx), on a subsampled
    grid for speed."""
    dy, dx = shift
    if dy or dx:
        b = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
    a2 = a[::sub, ::sub]; b2 = b[::sub, ::sub]
    union = int(np.logical_or(a2, b2).sum())
    if not union:
        return 1.0
    return int(np.logical_and(a2, b2).sum()) / union


def _search_shift(a: np.ndarray, b: np.ndarray, cy: int, cx: int, rad: int):
    """Brute-force the (dy,dx) within +/-rad of (cy,cx) that maximises ink overlap."""
    best, byx = -1, (cy, cx)
    for dy in range(cy - rad, cy + rad + 1):
        sb = np.roll(b, dy, axis=0)
        for dx in range(cx - rad, cx + rad + 1):
            inter = int(np.logical_and(a, np.roll(sb, dx, axis=1)).sum())
            if inter > best:
                best, byx = inter, (dy, dx)
    return byx


def _brute_shift(a: np.ndarray, b: np.ndarray, max_shift: int):
    """Robust coarse-to-fine overlap registration for sparse re-plotted drawings,
    where phase correlation can lock onto the wrong peak. Returns (dy,dx) in full px."""
    fc = 16
    cyc, cxc = _search_shift(a[::fc, ::fc], b[::fc, ::fc], 0, 0, max(1, max_shift // fc))
    ff = 4
    cy, cx = cyc * fc // ff, cxc * fc // ff
    fy, fx = _search_shift(a[::ff, ::ff], b[::ff, ::ff], cy, cx, fc // ff + 1)
    return fy * ff, fx * ff


def _iou_nowrap(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> float:
    """IoU of `a` and `b` shifted by (dy,dx) using SLICING (no wrap-around, which
    would invent spurious overlap on a small window and pick a bogus peak)."""
    H, W = a.shape
    ay0 = max(0, dy); ax0 = max(0, dx)
    ay1 = min(H, H + dy); ax1 = min(W, W + dx)
    ah = ay1 - ay0; aw = ax1 - ax0
    if ah <= 0 or aw <= 0:
        return 0.0
    by0 = max(0, -dy); bx0 = max(0, -dx)
    aa = a[ay0:ay1, ax0:ax1]; bb = b[by0:by0 + ah, bx0:bx0 + aw]
    u = int(np.logical_or(aa, bb).sum())
    if not u:
        return 0.0
    return int(np.logical_and(aa, bb).sum()) / u


def _search_iou(a, b, cy, cx, rad):
    best, byx = -1.0, (cy, cx)
    for dy in range(cy - rad, cy + rad + 1):
        for dx in range(cx - rad, cx + rad + 1):
            s = _iou_nowrap(a, b, dy, dx)
            if s > best:
                best, byx = s, (dy, dx)
    return byx, best


def _pool(m: np.ndarray, k: int) -> np.ndarray:
    """Downsample a bool mask by OR-pooling kxk blocks. Unlike subsampling
    (m[::k]), this KEEPS thin 1-2px CAD lines (a block is True if any pixel is),
    so the coarse correlation isn't run on a near-empty mask."""
    if k <= 1:
        return m
    H, W = m.shape
    H2, W2 = H // k, W // k
    if H2 == 0 or W2 == 0:
        return m[:max(1, H2), :max(1, W2)]
    return m[:H2 * k, :W2 * k].reshape(H2, k, W2, k).any(axis=(1, 3))


def local_shift(base: np.ndarray, final: np.ndarray, box, max_px: int):
    """Best (dy,dx) aligning `final` to `base` within a window around `box`, i.e.
    sampling final at (box - shift) lines it up with the base-frame box. Used to
    re-centre the after-crop when the architect re-plotted content at a shifted spot.

    Robust by design: no-wrap IoU (a small window + np.roll wrap used to invent
    overlap and return random shifts), coarse-to-fine, and a CONFIDENCE GUARD — it
    only returns a non-zero shift when the aligned overlap is clearly better than no
    shift; otherwise (0,0), so the crop is never moved to a wrong, random spot."""
    H, W = base.shape
    x0 = max(0, int(box[0]) - max_px); y0 = max(0, int(box[1]) - max_px)
    x1 = min(W, int(box[2]) + max_px); y1 = min(H, int(box[3]) + max_px)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return (0, 0)
    a = base[y0:y1, x0:x1]; b = final[y0:y1, x0:x1]
    if int(a.sum()) < 50 or int(b.sum()) < 50:
        return (0, 0)
    # OR-pool (not subsample) so thin lines survive the coarse search.
    fc = 8
    af, bf = _pool(a, fc), _pool(b, fc)
    (cy, cx), _ = _search_iou(af, bf, 0, 0, max(1, max_px // fc))
    ff = 2
    a2, b2 = _pool(a, ff), _pool(b, ff)
    (fy, fx), score = _search_iou(a2, b2, cy * fc // ff, cx * fc // ff, fc // ff + 1)
    dy, dx = fy * ff, fx * ff
    if dy == 0 and dx == 0:
        return (0, 0)
    base_score = _iou_nowrap(a2, b2, 0, 0)
    if score < 0.20 or score < base_score * 1.4:   # not clearly an improvement
        return (0, 0)
    return (dy, dx)


def prepare_page(redlined_page, final_page, page_index: int, cfg: Config) -> PageData:
    zoom = cfg.zoom()
    base_rgb = render_rgb(redlined_page, cfg.analysis_dpi, annots=False)
    # Render the marked-up page once here (annots on) so report crops don't have
    # to re-render the page per finding — a big speedup on heavily-marked sheets.
    redlined_rgb = render_rgb(redlined_page, cfg.analysis_dpi, annots=True)
    final_rgb = render_rgb(final_page, cfg.analysis_dpi, annots=True)

    h = min(base_rgb.shape[0], final_rgb.shape[0], redlined_rgb.shape[0])
    w = min(base_rgb.shape[1], final_rgb.shape[1], redlined_rgb.shape[1])
    base_rgb = base_rgb[:h, :w]
    redlined_rgb = redlined_rgb[:h, :w]
    final_rgb = final_rgb[:h, :w]

    base_ink = ink_mask(base_rgb, cfg)
    final_ink = ink_mask(final_rgb, cfg)

    # Register on a downsampled mask for speed, then scale the shift back.
    f = 4
    a_small = base_ink[::f, ::f]
    b_small = final_ink[::f, ::f]
    max_shift_small = max(1, int(cfg.max_register_shift_frac * max(a_small.shape)))
    dy_s, dx_s = _phase_shift(a_small, b_small, max_shift_small)
    dy, dx = dy_s * f, dx_s * f

    # Phase correlation can lock onto the wrong peak on sparse, re-plotted sheets
    # (returns ~0 while the sheet is actually offset). If the result aligns poorly,
    # fall back to a robust brute-force overlap search and keep the best-scoring of
    # {no shift, phase-corr, brute-force} — so we never make alignment worse.
    if _iou_at(base_ink, final_ink, (dy, dx)) < cfg.register_min_iou:
        max_shift = max(1, int(cfg.max_register_shift_frac * max(base_ink.shape)))
        cands = [(0, 0), (dy, dx), _brute_shift(base_ink, final_ink, max_shift)]
        dy, dx = max(cands, key=lambda s: _iou_at(base_ink, final_ink, s))

    if dy or dx:
        final_ink = np.roll(np.roll(final_ink, dy, axis=0), dx, axis=1)
        final_rgb = np.roll(np.roll(final_rgb, dy, axis=0), dx, axis=1)

    inter = np.logical_and(base_ink, final_ink).sum()
    union = np.logical_or(base_ink, final_ink).sum()
    similarity = float(inter / union) if union else 1.0

    return PageData(
        page_index=page_index, zoom=zoom,
        base_rgb=base_rgb, redlined_rgb=redlined_rgb, final_rgb=final_rgb,
        base_ink=base_ink, final_ink=final_ink,
        shift=(dy, dx), similarity=similarity,
        redlined_page=redlined_page, final_page=final_page,
    )
