"""Fast structural precheck: match corresponding pages between the redlined and
revised PDFs *before* the heavy comparison.

The architect routinely inserts extra pages (so page i no longer lines up with
page i) or splits one redlined PDF into several smaller revised PDFs. Comparing
by raw page index then pairs unrelated pages and corrupts the diff (and can crash
on mismatched coordinates).

This module renders a tiny low-DPI fingerprint per page and runs a monotonic
sequence alignment (Needleman–Wunsch) to:
  - match each redlined page to its revised counterpart,
  - flag revised pages with no counterpart  -> inserted by the architect,
  - flag redlined pages with no counterpart -> not carried into the revision.

Everything here is cheap and content-light: pages are rendered at ~36 DPI to a
32x32 fingerprint, so even hundreds of pages stay fast and low-memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import fitz
from PIL import Image

from .config import Config

NEG_INF = -1.0e9


@dataclass
class PageSig:
    index: int
    vec: np.ndarray       # normalised fingerprint (mean-removed, unit norm)
    landscape: bool       # displayed orientation (width >= height)


@dataclass
class MatchResult:
    pairs: List[Tuple[int, int]] = field(default_factory=list)   # (redlined_idx, revised_idx)
    extra_revised: List[int] = field(default_factory=list)       # inserted by architect
    extra_redlined: List[int] = field(default_factory=list)      # dropped / not carried over
    scores: dict = field(default_factory=dict)                   # (ri,fi) -> similarity


def _page_sig(page: "fitz.Page", index: int, cfg: Config, annots: bool) -> PageSig:
    zoom = cfg.prematch_dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), annots=annots,
                          alpha=False, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.stride)[:, :pix.width]
    g = cfg.prematch_grid
    im = Image.fromarray(arr).resize((g, g), Image.BILINEAR)
    v = 255.0 - np.asarray(im, dtype=np.float32).ravel()   # ink = high
    v -= v.mean()
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        v /= n
    return PageSig(index=index, vec=v, landscape=(pix.width >= pix.height))


def fingerprint_doc(doc: "fitz.Document", cfg: Config, annots: bool) -> List[PageSig]:
    return [_page_sig(doc[i], i, cfg, annots) for i in range(doc.page_count)]


def _similarity_matrix(rsigs: List[PageSig], fsigs: List[PageSig]) -> np.ndarray:
    if not rsigs or not fsigs:
        return np.zeros((len(rsigs), len(fsigs)), dtype=np.float32)
    R = np.stack([s.vec for s in rsigs])      # (n, d) unit vectors
    F = np.stack([s.vec for s in fsigs])      # (m, d)
    sim = R @ F.T                             # cosine similarity in [-1, 1]
    # Pages of different orientation can't be the same drawing.
    ro = np.array([s.landscape for s in rsigs])[:, None]
    fo = np.array([s.landscape for s in fsigs])[None, :]
    sim[ro != fo] = NEG_INF
    return sim.astype(np.float32)


def _align(sim: np.ndarray, gap: float):
    """Needleman–Wunsch global alignment; returns (pairs, extra_r, extra_f)."""
    n, m = sim.shape
    H = np.zeros((n + 1, m + 1), dtype=np.float64)
    H[:, 0] = np.arange(n + 1) * gap
    H[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        srow = sim[i - 1]
        for j in range(1, m + 1):
            H[i, j] = max(H[i - 1, j - 1] + srow[j - 1],
                          H[i - 1, j] + gap,
                          H[i, j - 1] + gap)
    pairs, extra_r, extra_f = [], [], []
    i, j = n, m
    while i > 0 and j > 0:
        diag = H[i - 1, j - 1] + sim[i - 1, j - 1]
        if H[i, j] == diag:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif H[i, j] == H[i - 1, j] + gap:
            extra_r.append(i - 1); i -= 1
        else:
            extra_f.append(j - 1); j -= 1
    while i > 0:
        extra_r.append(i - 1); i -= 1
    while j > 0:
        extra_f.append(j - 1); j -= 1
    pairs.reverse(); extra_r.reverse(); extra_f.reverse()
    return pairs, extra_r, extra_f


def match_documents(rdoc: "fitz.Document", fdoc: "fitz.Document",
                    cfg: Config) -> MatchResult:
    """Align redlined pages (markups removed) to revised pages by content."""
    rsigs = fingerprint_doc(rdoc, cfg, annots=False)
    fsigs = fingerprint_doc(fdoc, cfg, annots=True)
    return match_signatures(rsigs, fsigs, cfg)


def match_signatures(rsigs: List[PageSig], fsigs: List[PageSig],
                     cfg: Config) -> MatchResult:
    sim = _similarity_matrix(rsigs, fsigs)
    pairs, extra_r, extra_f = _align(sim, cfg.prematch_gap)
    # Drop weak "matches" the alignment was forced into; treat both sides as extra.
    good, scores = [], {}
    for ri, fi in pairs:
        s = float(sim[ri, fi])
        if s >= cfg.prematch_min_match:
            good.append((ri, fi)); scores[(ri, fi)] = s
        else:
            extra_r.append(ri); extra_f.append(fi)
    return MatchResult(pairs=good, extra_revised=sorted(extra_f),
                       extra_redlined=sorted(extra_r), scores=scores)


def doc_signature(doc: "fitz.Document", cfg: Config, annots: bool,
                  max_pages: int = 12) -> np.ndarray:
    """A single fingerprint for a whole file (mean of evenly-sampled page sigs),
    used to match files to files when several are uploaded per side."""
    n = doc.page_count
    if n == 0:
        return np.zeros(cfg.prematch_grid ** 2, dtype=np.float32)
    idx = np.unique(np.linspace(0, n - 1, min(max_pages, n)).round().astype(int))
    vecs = [_page_sig(doc[int(i)], int(i), cfg, annots).vec for i in idx]
    v = np.mean(vecs, axis=0)
    nrm = float(np.linalg.norm(v))
    return (v / nrm) if nrm > 1e-6 else v
