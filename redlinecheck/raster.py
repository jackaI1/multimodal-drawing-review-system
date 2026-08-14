"""Rendering and pixel-mask helpers shared across the pipeline."""
from __future__ import annotations

import numpy as np
import fitz

from .config import Config


def render_rgb(page: "fitz.Page", dpi: int, annots: bool = True,
               clip: "fitz.Rect | None" = None) -> np.ndarray:
    """Render a page (as displayed, i.e. honouring /Rotate) to an HxWx3 uint8 array."""
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), annots=annots,
                          clip=clip, alpha=False, colorspace=fitz.csRGB)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.stride // pix.n, pix.n)[:, :pix.width, :]
    return np.ascontiguousarray(arr[:, :, :3])


def markup_mask(rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """True where a pixel is a SATURATED red/magenta redline stroke.

    A true redline ink — red (~220,30,30) or magenta (~230,30,230) — has a bright
    red channel, a *low* green channel, and high saturation. The architect's own
    drawings are full of desaturated pastel fills (peach area-coverage shading
    ~(255,201,174), pink ~(249,175,222)) whose red channel is also high; an
    R-minus-G-only test let those through and fabricated phantom redline findings.
    Requiring low green AND high saturation cleanly separates ink (sat ~0.86) from
    pastel fills (sat ~0.30)."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    mn = np.minimum(g, b)
    sat = (r - mn) / np.maximum(r, 1)          # HSV saturation (max channel is r here)
    return ((r >= cfg.markup_red_min)
            & ((r - g) >= cfg.markup_red_minus_green)
            & (g <= cfg.markup_green_max)
            & (sat >= cfg.markup_min_sat))


def ink_mask(rgb: np.ndarray, cfg: Config) -> np.ndarray:
    """True where a pixel is dark structural ink (excludes markup colours and
    the mid-grey 'PRELIMINARY' watermark)."""
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    lum = (299 * r + 587 * g + 114 * b) // 1000
    dark = lum < cfg.ink_lum_threshold
    return dark & ~markup_mask(rgb, cfg)
