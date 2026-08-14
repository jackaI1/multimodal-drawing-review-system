"""Extract the engineer's redline markups from the marked-up PDF and group them
into logical comments with a target location on the drawing."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz

from .config import Config

Point = Tuple[float, float]

SPEC = "spec"          # a change instruction (e.g. "W310X52") with a leader
TO_ARCH = "to_arch"    # coordination note ("TO ARCH: ...") — often semantic
MARKER = "marker"      # small symbol with no text (checkmarks, bearing dots, ...)
COPYREF = "copyref"    # "copy comments from sheet X" / "apply same markups as X"

import re as _re

# A meta-instruction telling the architect to reuse markups from another sheet,
# e.g. "PLEASE COPY COMMENTS FROM ... A1.01 UNLESS NOTED OTHERWISE" or
# "PLS APPLY THE SAME MARKUPS AS PAGE A1.01". Not a spec to verify per se.
_COPYREF_RE = _re.compile(
    r"COPY\s+COMMENT|SAME\s+MARKUP|APPLY\s+THE\s+SAME|COPY\s+THE\s+(SAME|MARKUP)"
    r"|SAME\s+COMMENT", _re.IGNORECASE)


def _is_copy_ref(text: str) -> bool:
    return bool(_COPYREF_RE.search(text or ""))


# A redline addressed TO the architect asking them to check/confirm/coordinate
# something is a COORDINATION note, not a change-instruction — so it must not be
# graded as a "possible miss" when no value changes at its target. Examples seen in
# the wild: "ARCH PLS CHECK L1 CANOPY BEAM ALIGNMENT", "PLS PROVIDE BOX-OUT FRAMING
# DETAIL FOR REVIEW", "CONFIRM ...?". Kept deliberately specific (request verbs /
# a question mark / an "ARCH ..." address) so genuine specs like "ADD HEXNOTE 15B"
# or "W310X52" are untouched.
_COORD_RE = _re.compile(
    r"\?"                                              # any question
    r"|\bARCH(?:'?S)?\b\.?\s"                          # addressed to the architect
    r"|\bPL(?:S|EASE|Z)?\b[^.]{0,40}\b(CHECK|CONFIRM|VERIFY|REVIEW|"
    r"COORDINATE|CO-?ORDINATE|ADVISE|CLARIFY)\b"       # "pls ... <verb>"
    r"|\b(CHECK|CONFIRM|VERIFY|CLARIFY|COORDINATE|CO-?ORDINATE|ADVISE)\b"
    r"|\bFOR\s+REVIEW\b"
    # Reference / pointer notes — "see/refer to X", "if/as required". These point the
    # architect elsewhere; they are NOT a value to grade as a possible miss. (On the
    # example11 set ~45% of "possible misses" were notes like these or area highlights.)
    r"|\b(REFER\s+TO|SEE\s+(COMMENT|NOTE|PAGE|SHEET|DWG|DRAWING|DETAIL|BLOCK|PLAN|ABOVE|BELOW))\b"
    r"|\bPL(?:S|EASE|Z)?\s+SEE\b"
    r"|\b(IF|AS)\s+REQ(?:'?D|UIRED)?\b",
    _re.IGNORECASE)


def _is_coordination(text: str) -> bool:
    return bool(_COORD_RE.search(text or ""))


@dataclass
class Comment:
    id: int
    page: int
    kind: str
    text: str
    anchor: Tuple[float, float, float, float]   # PDF points (rotation-0 page space)
    targets: List[Point] = field(default_factory=list)
    is_magenta: bool = False
    annot_types: List[str] = field(default_factory=list)

    @property
    def target_rect(self) -> Tuple[float, float, float, float]:
        """A bbox (PDF pts) covering all leader tips, or the anchor if none."""
        if not self.targets:
            return self.anchor
        xs = [p[0] for p in self.targets]
        ys = [p[1] for p in self.targets]
        return (min(xs), min(ys), max(xs), max(ys))


def _vertices(annot) -> List[Point]:
    v = annot.vertices
    if v:
        # PolyLine/Line return a flat list of points; Line is [p0, p1]
        return [(float(p[0]), float(p[1])) for p in v]
    r = annot.rect
    return [(r.x0, r.y0), (r.x1, r.y1)]


def _dist_point_rect(p: Point, rect) -> float:
    x, y = p
    dx = max(rect.x0 - x, 0, x - rect.x1)
    dy = max(rect.y0 - y, 0, y - rect.y1)
    return (dx * dx + dy * dy) ** 0.5


def _extend(near: Point, far: Point, dist_pt: float) -> Optional[Point]:
    """Continue from `near` through `far` an extra `dist_pt` points, so the target
    lands on the referenced drawing label rather than exactly on the arrowhead."""
    dx, dy = far[0] - near[0], far[1] - near[1]
    L = (dx * dx + dy * dy) ** 0.5
    if L < 1e-3:
        return None
    return (far[0] + dx / L * dist_pt, far[1] + dy / L * dist_pt)


def _rgb3(c) -> Optional[Tuple[float, float, float]]:
    if not c:
        return None
    c = tuple(c)
    if len(c) == 1:          # grayscale
        return (c[0], c[0], c[0])
    if len(c) >= 3:
        return (c[0], c[1], c[2])
    return None


def _is_markup_color(c) -> bool:
    rgb = _rgb3(c)
    if not rgb:
        return False
    r, g, b = rgb
    return r >= 0.45 and (r - g) >= 0.18


def _is_magenta(c) -> bool:
    rgb = _rgb3(c)
    if not rgb:
        return False
    r, g, b = rgb
    return r >= 0.45 and b >= 0.45 and g <= 0.4


def extract_comments(page: "fitz.Page", page_index: int, cfg: Config,
                     start_id: int = 0) -> List[Comment]:
    """Pull annotations from one redlined page and assemble Comment objects.

    Annotation geometry is stored in *unrotated* PDF space, but we render and
    compare pages *as displayed*. So when the page has a /Rotate (e.g. 270, common
    for this architect's sheets), transform every rect/vertex by the page's
    rotation matrix to match the rendered image. Identity when rotation == 0, so
    upright sheets are unaffected."""
    rot = page.rotation_matrix

    def _R(rect):
        r = fitz.Rect(rect) * rot
        return fitz.Rect(min(r.x0, r.x1), min(r.y0, r.y1),
                         max(r.x0, r.x1), max(r.y0, r.y1))

    def _P(pt):
        p = fitz.Point(pt[0], pt[1]) * rot
        return (p.x, p.y)

    freetexts = []      # (rect, content, magenta)
    leaders = []        # (near_pt_candidates list, color, magenta, type)
    markers = []        # (rect, type, color)

    for a in page.annots() or []:
        atype = a.type[1]
        stroke = a.colors.get("stroke")
        fill = a.colors.get("fill")
        magenta = _is_magenta(stroke) or _is_magenta(fill)
        if atype == "FreeText":
            content = (a.info.get("content") or "").replace("\r", " ").strip()
            if content:
                freetexts.append((_R(a.rect), content, magenta))
        elif atype in ("Line", "PolyLine"):
            leaders.append(([_P(v) for v in _vertices(a)], stroke, magenta, atype))
        elif atype in ("Square", "Circle", "Polygon", "Stamp", "Ink", "Caret"):
            # Only count a shape as an engineer markup if it's actually visible.
            # Some architects' PDFs carry hundreds of COLOURLESS boxes (stroke and
            # fill both unset — e.g. an OCR/text-region layer); those aren't redlines
            # and would otherwise flood the report as phantom "(marked region)"
            # misses. A real redline always has a colour (red stroke, or a
            # red/green marker). FreeText/leaders are handled above and unaffected.
            if not stroke and not fill:        # None or empty () -> no colour set
                continue
            markers.append((_R(a.rect), atype, stroke))

    comments: List[Comment] = []
    cid = start_id

    # Assign each leader to its NEAREST FreeText within tolerance (a leader points
    # at exactly one callout; greedy first-come used to mis-pair adjacent notes).
    leader_for_ft = {fi: [] for fi in range(len(freetexts))}
    for pts, color, l_magenta, ltype in leaders:
        best_fi, best_d = None, float("inf")
        for fi, (rect, _content, _m) in enumerate(freetexts):
            d = min(_dist_point_rect(p, rect) for p in pts)
            if d < best_d:
                best_d, best_fi = d, fi
        if best_fi is not None and best_d <= cfg.leader_attach_pt:
            leader_for_ft[best_fi].append((pts, color, l_magenta, ltype))

    for fi, (rect, content, ft_magenta) in enumerate(freetexts):
        targets: List[Point] = []
        is_mag = ft_magenta or content.upper().startswith("TO ARCH")
        atypes = ["FreeText"]
        for pts, color, l_magenta, ltype in leader_for_ft[fi]:
            near = min(pts, key=lambda p: _dist_point_rect(p, rect))
            far = max(pts, key=lambda p: _dist_point_rect(p, rect))
            targets.append(far)
            ext = _extend(near, far, cfg.leader_extend_pt)  # land on the label, not the arrow tip
            if ext:
                targets.append(ext)
            atypes.append(ltype)
            if l_magenta:
                is_mag = True
        # No leader: the label sits on the spot -> target is the whole text box
        # (target_rect falls back to the anchor when `targets` is empty).
        if _is_copy_ref(content):
            kind = COPYREF
        elif is_mag or _is_coordination(content):
            kind = TO_ARCH
        else:
            kind = SPEC
        comments.append(Comment(
            id=cid, page=page_index, kind=kind, text=content,
            anchor=(rect.x0, rect.y0, rect.x1, rect.y1),
            targets=targets, is_magenta=is_mag, annot_types=atypes,
        ))
        cid += 1

    # Markers: small symbols are info; large red squares box a region to inspect
    # (use the square itself as the target so the crop covers the whole region).
    for rect, atype, color in markers:
        big = max(rect.width, rect.height) > 30
        comments.append(Comment(
            id=cid, page=page_index, kind=SPEC if big else MARKER,
            text="(marked region)" if big else "",
            anchor=(rect.x0, rect.y0, rect.x1, rect.y1),
            targets=[] if big else [((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)],
            annot_types=[atype],
        ))
        cid += 1

    return comments


def detect_flattened_markups(redlined_rgb, page_index: int, cfg: Config,
                             start_id: int = 0, final_rgb=None) -> List[Comment]:
    """Fallback for redlines FLATTENED into the page (no Acrobat annotations).

    Finds clusters of SATURATED red/magenta markup ink in the already-rendered
    marked-up page and turns each into a SPEC comment whose target is the cluster
    region — so a baked-in redline still gets a before/after crop and a verdict.

    Conservative by design (real jobs almost always have live annotations, so this is
    a rare safety net and must not fabricate findings):
      - the colour test (markup_mask) now requires a saturated, low-green red, so the
        architect's pastel/orange area-coverage FILLS no longer qualify;
      - a PERSISTENCE filter drops any red that also appears in the revision at the
        same spot — that is part of the drawing, not your redline (your red never
        reaches the architect's black revision);
      - a stroke gate drops large solid colour blobs (a redline is hollow linework).

    Coordinates are DISPLAYED pixels (the array is rendered as-displayed) -> divide
    by zoom to get the displayed points classify.target_box_px expects. Only called
    when a page yielded no annotation comments."""
    from scipy import ndimage
    from .raster import markup_mask

    mk = markup_mask(redlined_rgb, cfg)
    z = cfg.zoom()
    minside = max(2, int(cfg.px(cfg.flatten_min_side_pt)))
    if int(mk.sum()) < minside * minside:
        return []
    # Red that persists into the revision = part of the drawing, not a redline.
    persist = None
    if final_rgb is not None:
        fmk = markup_mask(final_rgb, cfg)
        d = max(1, int(cfg.px(cfg.flatten_persist_dilate_pt)))
        persist = ndimage.binary_dilation(fmk, iterations=d)
    it = max(1, int(cfg.px(cfg.flatten_dilate_pt)))
    lbl, n = ndimage.label(ndimage.binary_dilation(mk, iterations=it))
    if not n:
        return []
    objs = ndimage.find_objects(lbl)
    min_area = minside * minside
    solid_min = cfg.px(cfg.flatten_solid_min_side_pt)
    cand = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        comp = (lbl[sl] == (i + 1)) & mk[sl]            # this cluster's red ink
        area = int(comp.sum())
        if area < min_area:
            continue
        if (x1 - x0) < minside and (y1 - y0) < minside:
            continue
        # Persistence: how much of this red also shows up in the revision here?
        if persist is not None:
            kept = int((comp & persist[sl]).sum())
            if area and kept / area > cfg.flatten_persist_max_frac:
                continue                                # it's in the drawing, not a markup
        # Stroke gate: a big, mostly-solid colour blob is a fill, not linework.
        bw, bh = x1 - x0, y1 - y0
        if min(bw, bh) >= solid_min:
            fill_ratio = area / float(bw * bh)
            if fill_ratio > cfg.flatten_max_fill_ratio:
                continue
        cand.append((area, x0, y0, x1, y1))
    cand.sort(reverse=True)                              # biggest markups first
    comments: List[Comment] = []
    cid = start_id
    for _area, x0, y0, x1, y1 in cand[:cfg.flatten_max_clusters]:
        comments.append(Comment(
            id=cid, page=page_index, kind=SPEC, text="(redline markup)",
            anchor=(x0 / z, y0 / z, x1 / z, y1 / z), targets=[],
            annot_types=["flattened"],
        ))
        cid += 1
    return comments


def extract_all(doc: "fitz.Document", cfg: Config) -> List[Comment]:
    out: List[Comment] = []
    for pno in range(doc.page_count):
        out.extend(extract_comments(doc[pno], pno, cfg, start_id=len(out)))
    return out
