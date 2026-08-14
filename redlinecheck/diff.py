"""Density-based change detection between the registered base and final ink masks.

The architect's revised sheet is a full re-plot, so a raw pixel diff lights up
along every line (sub-pixel jitter). Instead we downsample each ink mask to an
ink-density grid and compare densities: jitter cancels, genuine added/removed
content stands out. In-place text swaps of equal ink (e.g. W200X52 -> W310X52)
are intentionally NOT visible here -- those require the OCR/AI tier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import Config
from .align import PageData


@dataclass
class ChangeRegion:
    bbox_px: tuple        # (x0, y0, x1, y1) in analysis pixels
    added_px: int         # approx pixels of ink added (final has, base didn't)
    removed_px: int       # approx pixels of ink removed
    magnitude: float      # summed |density delta| over the cluster cells
    symbol: bool = False  # detected via the symbol-callout interior comparison

    @property
    def real_px(self) -> int:
        return self.added_px + self.removed_px

    @property
    def directionality(self) -> float:
        t = self.added_px + self.removed_px
        return abs(self.added_px - self.removed_px) / t if t else 0.0

    @property
    def kind(self) -> str:
        return "added" if self.added_px >= self.removed_px else "removed"


@dataclass
class DiffResult:
    added: np.ndarray         # full-res bool, ink added
    removed: np.ndarray       # full-res bool, ink removed
    added_cells: np.ndarray   # cell-grid bool
    removed_cells: np.ndarray
    delta: np.ndarray         # cell-grid float density delta (final - base)
    cell: int
    regions: list
    fine_added_cells: np.ndarray = None    # finer grid, for comment-target activity
    fine_removed_cells: np.ndarray = None
    fine_cell: int = 0


def _density(mask: np.ndarray, cell: int) -> np.ndarray:
    h, w = mask.shape
    H, W = h // cell, w // cell
    m = mask[:H * cell, :W * cell].astype(np.float32)
    return m.reshape(H, cell, W, cell).mean(axis=(1, 3))


def compute_change(page: PageData, cfg: Config) -> DiffResult:
    cell = max(4, round(cfg.px(cfg.density_cell_pt)))
    base_d = _density(page.base_ink, cell)
    final_d = _density(page.final_ink, cell)
    if cfg.density_smooth_sigma:
        base_d = ndimage.gaussian_filter(base_d, cfg.density_smooth_sigma)
        final_d = ndimage.gaussian_filter(final_d, cfg.density_smooth_sigma)
    delta = final_d - base_d

    added_c = delta > cfg.density_add_thresh
    removed_c = delta < -cfg.density_remove_thresh
    change_c = added_c | removed_c
    if cfg.cluster_merge_cells:
        change_c = ndimage.binary_dilation(change_c, iterations=cfg.cluster_merge_cells)

    lbl, n = ndimage.label(change_c)
    page_area = float(page.base_ink.shape[0] * page.base_ink.shape[1])
    regions = []
    if n:
        objs = ndimage.find_objects(lbl)
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            comp = (lbl[sl] == i)
            ncells = int(comp.sum())
            sub_delta = delta[sl]
            mag = float(np.abs(sub_delta[comp]).sum())
            if ncells < cfg.min_cluster_cells or mag < cfg.min_change_magnitude:
                continue
            add_px = float(np.clip(sub_delta[comp], 0, None).sum()) * cell * cell
            rem_px = float(np.clip(-sub_delta[comp], 0, None).sum()) * cell * cell
            ys, xs = sl
            bbox = (xs.start * cell, ys.start * cell, xs.stop * cell, ys.stop * cell)
            r = ChangeRegion(bbox_px=bbox, added_px=int(add_px),
                             removed_px=int(rem_px), magnitude=mag)
            # Reject re-plot churn: balanced add/remove, or page-spanning blobs.
            area_frac = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / page_area
            if r.directionality < cfg.min_directionality:
                continue
            if area_frac > cfg.max_region_area_frac:
                continue
            regions.append(r)
    regions = _suppress_shift_pairs(regions, cfg)
    if cfg.suppress_moved_content:
        regions = _suppress_moved_content(page, regions, cfg)
        # Drop regions whose changed ink simply MOVED / re-plotted a hair away (checked
        # by local re-alignment). This catches the architect re-plotting a plan at a
        # shifted position on a poorly-aligned sheet, AND — when run on every page —
        # the dense-hatch re-plot churn on well-aligned elevation sheets.
        if page.similarity < cfg.register_min_iou:
            regions = _suppress_local_shifts(page, regions, cfg,
                                             frac=cfg.suppress_move_xor_frac)
        elif cfg.suppress_local_shifts_always:
            # well-aligned page: only drop near-identical re-plot churn, so real
            # added content among dense hatch survives.
            regions = _suppress_local_shifts(page, regions, cfg,
                                             frac=cfg.suppress_move_xor_frac_strict)
    # Add in-place edits inside closed callouts (hexnote "8c"->"8" etc.) that the
    # density diff can't see. Append after the density filters; dedup by overlap.
    if cfg.detect_symbols:
        for sr in detect_symbol_changes(page, cfg):
            if not any(boxes_overlap(sr.bbox_px, r.bbox_px) for r in regions):
                regions.append(sr)
    regions.sort(key=lambda r: r.magnitude, reverse=True)

    added = np.kron(added_c, np.ones((cell, cell), dtype=bool))
    removed = np.kron(removed_c, np.ones((cell, cell), dtype=bool))

    # Finer grid used only for per-comment target activity (not region detection).
    fcell = max(3, round(cfg.px(cfg.target_cell_pt)))
    fdelta = _density(page.final_ink, fcell) - _density(page.base_ink, fcell)
    fine_added = fdelta > cfg.density_add_thresh
    fine_removed = fdelta < -cfg.density_remove_thresh

    return DiffResult(added=added, removed=removed, added_cells=added_c,
                      removed_cells=removed_c, delta=delta, cell=cell, regions=regions,
                      fine_added_cells=fine_added, fine_removed_cells=fine_removed,
                      fine_cell=fcell)


def detect_symbol_changes(page: PageData, cfg: Config):
    """Find closed callout symbols (hexnotes, circles, boxes) whose *interior*
    changed between base and final — e.g. a hexnote '8c' -> '8'. The density diff
    is blind to such a tiny in-place edit and OCR can't read digits in a hexagon,
    but the symbol's enclosed interior is a clean place to compare directly.

    A symbol is an enclosed white region (a hole in the ink) that contains a digit
    in BOTH drawings; we flag it when the interior ink differs by a mid-range
    amount (a partial change = a digit swap, not text appearing/disappearing)."""
    base, final = page.base_ink, page.final_ink
    lblmask, n = ndimage.label(ndimage.binary_fill_holes(base) & ~base)
    if not n:
        return []
    lo = cfg.px(cfg.symbol_min_pt); hi = cfg.px(cfg.symbol_max_pt)
    minpx = max(1, int(cfg.px(cfg.symbol_min_change_pt) ** 2))
    R = int(cfg.symbol_align_px)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lblmask), start=1):
        if sl is None:
            continue
        bh = sl[0].stop - sl[0].start; bw = sl[1].stop - sl[1].start
        if not (0.4 <= bw / max(1, bh) <= 2.2 and lo <= bw <= hi and lo <= bh <= hi):
            continue
        # Shape gate: a real callout is a polygon/circle (fills ~0.75-0.78 of its
        # bbox); a dimension rectangle fills ~1.0 (excluded in standard).
        solidity = ndimage.binary_fill_holes(lblmask[sl] == i).sum() / float(bh * bw)
        if not (cfg.symbol_solidity_min <= solidity <= cfg.symbol_solidity_max):
            continue
        y0, x0, y1, x1 = sl[0].start, sl[1].start, sl[0].stop, sl[1].stop
        cb = base[y0:y1, x0:x1]; cf = final[y0:y1, x0:x1]
        bsum = int(cb.sum()); fsum = int(cf.sum())
        if min(bsum, fsum) < minpx:            # a digit must be present in BOTH
            continue
        # Micro-align the interiors so re-plot jitter cancels; a genuine content
        # change (the removed 'c') survives.
        ch = int(np.count_nonzero(cb ^ cf))
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                if dy or dx:
                    ch = min(ch, int(np.count_nonzero(
                        cb ^ np.roll(np.roll(cf, dy, 0), dx, 1))))
        frac = 2.0 * ch / (bsum + fsum)
        if ch < minpx or not (cfg.symbol_min_frac <= frac <= cfg.symbol_max_frac):
            continue
        out.append(ChangeRegion(
            bbox_px=(x0, y0, x1, y1),
            added_px=int((cf & ~cb).sum()), removed_px=int((cb & ~cf).sum()),
            magnitude=float(ch), symbol=True))
    return out


def _suppress_shift_pairs(regions, cfg: Config):
    """Drop added/removed region pairs that are close together and similar in
    size -- these are the same content re-plotted at a slightly shifted position
    (common for dimension strings / labels), not a real change."""
    gap = cfg.px(cfg.suppress_shift_gap_pt)
    added = [r for r in regions if r.kind == "added"]
    removed = [r for r in regions if r.kind == "removed"]
    drop = set()
    for a in added:
        for r in removed:
            if id(r) in drop:
                continue
            lo, hi = sorted((a.magnitude, r.magnitude))
            if hi <= 0 or lo / hi < cfg.suppress_shift_ratio:
                continue
            if boxes_overlap(a.bbox_px, r.bbox_px, gap):
                drop.add(id(a)); drop.add(id(r))
                break
    return [r for r in regions if id(r) not in drop]


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _crop_bool(mask, bbox):
    h, w = mask.shape
    x0 = max(0, int(bbox[0])); y0 = max(0, int(bbox[1]))
    x1 = min(w, int(bbox[2])); y1 = min(h, int(bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return mask[y0:y1, x0:x1]


def _content_matches(old_crop, new_crop, J: int, frac_thresh: float) -> bool:
    """True if the OLD ink (base @ removed box) and NEW ink (final @ added box) are
    the same content once aligned — i.e. the blob just moved. Pads both to a common
    size and takes the min XOR over a small jitter search; compares the residual to
    the ink present."""
    if old_crop is None or new_crop is None:
        return False
    bsum, fsum = int(old_crop.sum()), int(new_crop.sum())
    if bsum == 0 or fsum == 0:
        return False
    H = max(old_crop.shape[0], new_crop.shape[0]) + 2 * J
    W = max(old_crop.shape[1], new_crop.shape[1]) + 2 * J
    pb = np.zeros((H, W), dtype=bool); pf = np.zeros((H, W), dtype=bool)
    pb[J:J + old_crop.shape[0], J:J + old_crop.shape[1]] = old_crop
    pf[J:J + new_crop.shape[0], J:J + new_crop.shape[1]] = new_crop
    best = int(np.count_nonzero(pb ^ pf))
    for dy in range(-J, J + 1):
        for dx in range(-J, J + 1):
            if dy or dx:
                best = min(best, int(np.count_nonzero(
                    pb ^ np.roll(np.roll(pf, dy, 0), dx, 1))))
    return (2.0 * best / (bsum + fsum)) <= frac_thresh


def _suppress_moved_content(page: PageData, regions, cfg: Config):
    """Drop add/remove region pairs that are the SAME content relocated a short
    distance (a dimension/label nudged left or right with no value change). Unlike
    the overlap-based shift filter, this directly checks that the removed ink and the
    added ink match, so genuinely different add/remove edits are kept."""
    base, final = page.base_ink, page.final_ink
    added = [r for r in regions if r.kind == "added"]
    removed = [r for r in regions if r.kind == "removed"]
    if not added or not removed:
        return regions
    maxmove = cfg.px(cfg.suppress_move_max_pt)
    J = max(1, int(cfg.px(cfg.suppress_move_jitter_pt)))
    minink = max(1, int(cfg.px(cfg.suppress_move_min_pt) ** 2))
    drop = set()
    for a in added:
        if id(a) in drop:
            continue
        acx, acy = _center(a.bbox_px)
        aw = a.bbox_px[2] - a.bbox_px[0]; ah = a.bbox_px[3] - a.bbox_px[1]
        new_crop = _crop_bool(final, a.bbox_px)
        if new_crop is None or int(new_crop.sum()) < minink:
            continue
        for r in removed:
            if id(r) in drop:
                continue
            rcx, rcy = _center(r.bbox_px)
            dist = ((acx - rcx) ** 2 + (acy - rcy) ** 2) ** 0.5
            if dist < 1.0 or dist > maxmove:
                continue
            rw = r.bbox_px[2] - r.bbox_px[0]; rh = r.bbox_px[3] - r.bbox_px[1]
            # require similar footprint (same thing moved, not two different blobs)
            if not (0.5 <= (rw + 1) / (aw + 1) <= 2.0 and 0.5 <= (rh + 1) / (ah + 1) <= 2.0):
                continue
            old_crop = _crop_bool(base, r.bbox_px)
            if old_crop is None or int(old_crop.sum()) < minink:
                continue
            if _content_matches(old_crop, new_crop, J, cfg.suppress_move_xor_frac):
                drop.add(id(a)); drop.add(id(r))
                break
    return [r for r in regions if id(r) not in drop]


def _min_uncovered_frac(chg, other, x0, y0, R, step):
    """Fraction of the changed-ink template `chg` (located at base coords x0,y0) that
    is NOT found in `other` for the best local shift within +/-R. A moved-plan line
    shows up as add-only / remove-only ink whose template reappears in the other
    image a short distance away -> low uncovered fraction. A genuine add/remove has
    no match anywhere -> high. Coarse-to-fine over the SHIFT (full-res templates)."""
    H, W = other.shape
    th, tw = chg.shape
    total = int(chg.sum())
    if total == 0:
        return 1.0

    def uncovered(dy, dx):
        oy0 = y0 - dy; ox0 = x0 - dx
        sy0 = max(0, oy0); sx0 = max(0, ox0)
        sy1 = min(H, oy0 + th); sx1 = min(W, ox0 + tw)
        if sy1 <= sy0 or sx1 <= sx0:
            return total
        sub = other[sy0:sy1, sx0:sx1]
        csub = chg[sy0 - oy0:sy1 - oy0, sx0 - ox0:sx1 - ox0]
        return total - int(np.count_nonzero(csub & sub))   # chg ink covered by other

    best = total
    byx = (0, 0)
    for dy in range(-R, R + 1, step):
        for dx in range(-R, R + 1, step):
            u = uncovered(dy, dx)
            if u < best:
                best, byx = u, (dy, dx)
    by, bx = byx
    for dy in range(by - step, by + step + 1):
        for dx in range(bx - step, bx + step + 1):
            u = uncovered(dy, dx)
            if u < best:
                best = u
    return best / total


def _suppress_local_shifts(page: PageData, regions, cfg: Config, frac=None):
    """Drop change regions whose changed INK is the same content simply MOVED a
    little — the architect nudging a dimension, re-plotting the plan at a shifted
    spot inside a static frame (which global registration can't undo), or re-plotting
    dense elevation hatch with local churn. For an added region we look for that ink
    in the BASE nearby; for a removed region, in the FINAL nearby. A genuine
    add/remove/edit has no nearby match, so it survives. `frac` is the residual-
    mismatch bar (lower = stricter = only near-identical re-plots dropped)."""
    base, final = page.base_ink, page.final_ink
    R = max(2, int(cfg.px(cfg.suppress_move_max_pt)))
    step = max(3, R // 18)
    minink = max(1, int(cfg.px(cfg.suppress_move_min_pt) ** 2))
    if frac is None:
        frac = cfg.suppress_move_xor_frac
    # Dilate the "other" image so a re-plotted (1-2px jittery) line still counts as
    # matched when we find the moved content. Genuine changes have no match anyway.
    tol = max(1, int(cfg.px(cfg.suppress_move_tol_pt)))
    base_d = ndimage.binary_dilation(base, iterations=tol)
    final_d = ndimage.binary_dilation(final, iterations=tol)
    keep = []
    for r in regions:
        if r.symbol:
            keep.append(r); continue
        x0, y0, x1, y1 = (int(v) for v in r.bbox_px)
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(base.shape[1], x1); y1 = min(base.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            keep.append(r); continue
        b = base[y0:y1, x0:x1]; f = final[y0:y1, x0:x1]
        if r.kind == "added":
            chg = f & ~b           # ink final has here that base didn't
            other = base_d         # ... look for it in the (dilated) base nearby
        else:
            chg = b & ~f           # ink base had here that final dropped
            other = final_d        # ... look for it in the (dilated) final nearby
        if int(chg.sum()) < minink:
            keep.append(r); continue
        if _min_uncovered_frac(chg, other, x0, y0, R, step) <= frac:
            continue               # the changed ink reappears nearby -> it just moved
        keep.append(r)
    return keep


def boxes_overlap(a, b, margin: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + margin < bx0 or bx1 + margin < ax0 or
                ay1 + margin < by0 or by1 + margin < ay0)
