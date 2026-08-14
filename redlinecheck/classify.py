"""Turn comments + pixel changes into findings.

Pixel-only verdicts (always available):
  CHANGED_AT_TARGET   something was added/removed where the comment points
  NO_CHANGE_DETECTED  nothing changed there (likely missed, or an in-place text
                      swap the pixel diff can't see -> escalate to OCR/AI)
  COORDINATION        a "TO ARCH" note (semantic, manual review)
  MARKER              a small symbol with no text (informational)
  UNEXPECTED          a change with no redline near it

OCR/AI verifiers may upgrade a SPEC verdict to CAPTURED / MISSED / CHANGED_OTHER.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .config import Config
from .align import PageData
from .diff import DiffResult, ChangeRegion, boxes_overlap
from .extract import Comment, SPEC, TO_ARCH, MARKER, COPYREF

# verdicts
CHANGED_AT_TARGET = "CHANGED_AT_TARGET"
NO_CHANGE_DETECTED = "NO_CHANGE_DETECTED"
COORDINATION = "COORDINATION"
MARKER_INFO = "MARKER"
CAPTURED = "CAPTURED"
MISSED = "MISSED"
CHANGED_OTHER = "CHANGED_OTHER"
UNEXPECTED = "UNEXPECTED"
COPY_REF = "COPY_REF"        # "copy comments from sheet X" note (show whole sheet)
COPY_CHANGE = "COPY_CHANGE"  # an architect change on a copy-comments sheet (to verify)
MARKED_REGION = "MARKED_REGION"  # the engineer boxed/clouded an area (or a flattened
                                 # markup) with no specific instruction — "review this
                                 # highlighted area", never a possible miss.

# severity buckets for reporting/sorting
SEVERITY = {
    MISSED: 0, NO_CHANGE_DETECTED: 0, UNEXPECTED: 0,
    CHANGED_OTHER: 1, COORDINATION: 1, COPY_REF: 1, COPY_CHANGE: 1, MARKED_REGION: 1,
    CHANGED_AT_TARGET: 2, CAPTURED: 3, MARKER_INFO: 4,
}


@dataclass
class Finding:
    comment: Comment
    status: str
    target_box_px: tuple
    matched_regions: List[ChangeRegion] = field(default_factory=list)
    local_change_ratio: float = 0.0
    confidence: str = "pixel"          # pixel | ocr
    verified_value: Optional[str] = None
    note: str = ""


@dataclass
class Unexpected:
    region: ChangeRegion
    page_index: int
    to_verify: bool = False    # on a "copy comments" sheet -> verify, not unexpected


# verifier signature: (page, comment, target_box_px) -> (status, value, note) | None
Verifier = Callable[[PageData, Comment, tuple], Optional[tuple]]


def _expand(box, by):
    x0, y0, x1, y1 = box
    return (x0 - by, y0 - by, x1 + by, y1 + by)


def _contains(outer, inner) -> bool:
    """True if the centre of `inner` falls within `outer` (box around a note)."""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _ignore_boxes_px(page: PageData, cfg: Config):
    h, w = page.base_ink.shape
    out = []
    for (rx0, ry0, rx1, ry1) in cfg.ignore_zones:
        out.append((rx0 * w, ry0 * h, rx1 * w, ry1 * h))
    return out


def target_box_px(comment: Comment, page: PageData, cfg: Config) -> tuple:
    z = page.zoom
    radius = cfg.px(cfg.target_radius_pt)
    xs0, ys0, xs1, ys1 = comment.target_rect
    box = _expand((xs0 * z, ys0 * z, xs1 * z, ys1 * z), radius)
    # Enforce a minimum side so a small leader tip still yields a box big enough
    # to contain the referenced drawing label.
    minpx = cfg.px(cfg.target_min_pt)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw = max(minpx, box[2] - box[0]) / 2
    hh = max(minpx, box[3] - box[1]) / 2
    box = (cx - hw, cy - hh, cx + hw, cy + hh)
    h, w = page.base_ink.shape
    return (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))


def _changed_cell_frac(diff: DiffResult, box):
    """(fraction, count) of density cells inside the target box that changed. Uses the
    FINER grid (if available) so small in-place edits at a redline's target still
    count as activity — the coarse grid is for jitter-robust region detection. The
    absolute count lets a small added label register even when it's a tiny fraction of
    a large marked-region box."""
    if diff.fine_added_cells is not None:
        cell = diff.fine_cell
        added_cells, removed_cells = diff.fine_added_cells, diff.fine_removed_cells
    else:
        cell = diff.cell
        added_cells, removed_cells = diff.added_cells, diff.removed_cells
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    cx0, cy0 = x0 // cell, y0 // cell
    cx1, cy1 = (x1 + cell - 1) // cell, (y1 + cell - 1) // cell
    sub_a = added_cells[cy0:cy1, cx0:cx1]
    sub_r = removed_cells[cy0:cy1, cx0:cx1]
    if sub_a.size == 0:
        return 0.0, 0
    count = int(np.count_nonzero(sub_a | sub_r))
    return count / sub_a.size, count


def classify_page(page: PageData, comments: List[Comment], diff: DiffResult,
                  cfg: Config, verifier: Optional[Verifier] = None):
    findings: List[Finding] = []
    margin = cfg.px(cfg.target_change_margin_pt)
    page_comments = [c for c in comments if c.page == page.page_index]
    # A "copy comments from sheet X" note means the architect was told to reuse
    # markups here, so unmarked changes on this sheet are expected (-> to verify).
    page_has_copyref = any(c.kind == COPYREF for c in page_comments)
    copyref_boxes = [target_box_px(c, page, cfg)
                     for c in page_comments if c.kind == COPYREF]

    target_boxes = []
    for c in page_comments:
        tb = target_box_px(c, page, cfg)
        target_boxes.append(tb)
        matched = [r for r in diff.regions if boxes_overlap(tb, r.bbox_px, margin)]
        ratio, ccount = _changed_cell_frac(diff, tb)
        changed = (bool(matched) or ratio >= cfg.target_changed_cell_frac
                   or ccount >= cfg.target_changed_cell_min)

        if c.kind == MARKER:
            status = MARKER_INFO
        elif c.kind == COPYREF:
            status = COPY_REF
        elif c.kind == TO_ARCH:
            status = COORDINATION
        elif c.text == "(marked region)":
            # A red cloud/box the engineer drew purely to POINT the architect at where
            # the redline markups are — not itself an instruction. Ignore it (it still
            # claims its area below so in-cloud architect edits aren't flagged
            # unexpected), unless it boxes an "apply same markups" note.
            status = COPY_REF if any(_contains(tb, cb) for cb in copyref_boxes) else MARKER_INFO
        elif c.text == "(redline markup)":
            # A flattened markup (no live annotation, no readable instruction) — there's
            # no value to grade, so it's "review this highlighted area", never a miss.
            status = MARKED_REGION
        else:  # SPEC
            status = CHANGED_AT_TARGET if changed else NO_CHANGE_DETECTED

        f = Finding(comment=c, status=status, target_box_px=tb,
                    matched_regions=matched, local_change_ratio=ratio)

        # Optional stronger verification for spec comments (not area highlights).
        if (verifier is not None and c.kind == SPEC and c.text
                and c.text not in ("(marked region)", "(redline markup)")):
            res = verifier(page, c, tb, changed)
            if res:
                status2, value, note = res
                # If the verifier couldn't confirm the value (MISSED) but the pixel
                # diff clearly saw a change here, don't call it a confident miss —
                # downgrade to "change at target, review" so it isn't lost as green
                # nor over-claimed as red.
                if status2 == MISSED and changed:
                    status2 = CHANGED_AT_TARGET
                    note = (note + "; but pixel diff saw a change here — review").lstrip("; ")
                    value = ""
                f.status = status2
                f.verified_value = value
                f.note = note
                f.confidence = "ocr"
        findings.append(f)

    # Any of the engineer's markups claims the area around it: its text box, its
    # leader, and its target. A change the architect made there is something the
    # engineer FLAGGED, not an unmarked "unexpected" change — even when the leader tip
    # lands a little off the callout the architect then edits/removes. (A beam-size
    # redline pointing at a steel callout the architect deleted was otherwise
    # double-flagged as unexpected.) TO-ARCH clouds / "copy comments" boxes / marked
    # regions additionally claim their full drawn extent.
    z = page.zoom
    clear = cfg.px(cfg.unexpected_clear_margin_pt)
    claim_boxes = []
    for c in page_comments:
        ax0, ay0, ax1, ay1 = c.anchor                  # text/markup box (PDF pts)
        claim_boxes.append((ax0 * z, ay0 * z, ax1 * z, ay1 * z))
        for (tx, ty) in c.targets:                     # leader tips / referenced labels
            claim_boxes.append((tx * z, ty * z, tx * z, ty * z))

    # Unexpected = change regions not near any comment target/markup, and not in
    # ignore zones.
    ignore = _ignore_boxes_px(page, cfg)
    unexpected: List[Unexpected] = []
    for r in diff.regions:
        if any(boxes_overlap(tb, r.bbox_px, margin) for tb in target_boxes):
            continue
        if any(boxes_overlap(cb, r.bbox_px, clear) for cb in claim_boxes):
            continue
        if any(boxes_overlap(iz, r.bbox_px) for iz in ignore):
            continue
        unexpected.append(Unexpected(region=r, page_index=page.page_index,
                                     to_verify=page_has_copyref))

    return findings, unexpected
