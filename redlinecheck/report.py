"""Render the comparison results into an annotated PDF, an HTML review page with
before/after crops, and a CSV."""
from __future__ import annotations

import csv
import datetime as _dt
import html
import io
import os
import shutil
from collections import Counter
from typing import List

import numpy as np
import fitz
from PIL import Image, ImageDraw, ImageFont

from .config import Config
from .pipeline import Report
from .classify import (Finding, MARKER_INFO, COORDINATION,
                       NO_CHANGE_DETECTED, MISSED, UNEXPECTED, CHANGED_AT_TARGET,
                       CAPTURED, CHANGED_OTHER, COPY_REF, COPY_CHANGE, MARKED_REGION)

# status -> (rgb, short label, html css class, bucket)
STYLE = {
    MISSED:             ((210, 0, 0),   "MISSED",            "miss"),
    NO_CHANGE_DETECTED: ((210, 0, 0),   "No activity",       "miss"),
    UNEXPECTED:         ((165, 0, 200), "UNEXPECTED",        "unexp"),
    CHANGED_OTHER:      ((230, 140, 0), "Changed (other)",   "verify"),
    COORDINATION:       ((230, 140, 0), "Coordination",      "verify"),
    CHANGED_AT_TARGET:  ((230, 140, 0), "Change at target (review)", "verify"),
    COPY_REF:           ((230, 140, 0), "Copy-comments note", "verify"),
    COPY_CHANGE:        ((230, 140, 0), "Change on copy-comments sheet", "verify"),
    MARKED_REGION:      ((230, 140, 0), "Highlighted region (review)", "verify"),
    CAPTURED:           ((0, 150, 0),   "Addressed (verified)", "ok"),
    MARKER_INFO:        ((130, 130, 130), "Marker",          "marker"),
}


def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_box(draw, box, color, label, font, width=4):
    x0, y0, x1, y1 = (int(v) for v in box)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    if label:
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 4
        ty = max(0, y0 - th - 2 * pad)
        draw.rectangle([x0, ty, x0 + tw + 2 * pad, ty + th + 2 * pad], fill=color)
        draw.text((x0 + pad, ty + pad), label, fill=(255, 255, 255), font=font)


def _bounds(box, pad, w, h):
    x0 = max(0, int(box[0] - pad)); y0 = max(0, int(box[1] - pad))
    x1 = min(w, int(box[2] + pad)); y1 = min(h, int(box[3] + pad))
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, min(w, 10), min(h, 10)
    return x0, y0, x1, y1


def _wide_window(box, pad, factor):
    """A crop window ~`factor`x the normal (box + pad) window, centred on the box —
    the 'zoom out' view so you can see where your markup sits on the sheet."""
    cx = (box[0] + box[2]) / 2.0; cy = (box[1] + box[3]) / 2.0
    hw = (box[2] - box[0]) / 2.0 + pad
    hh = (box[3] - box[1]) / 2.0 + pad
    return (cx - hw * factor, cy - hh * factor, cx + hw * factor, cy + hh * factor)


def _draw_target(img: Image.Image, origin, mark_box, color):
    """Outline the analysed region (absolute px) inside a crop whose top-left is `origin`."""
    if mark_box is None:
        return
    ox, oy = origin
    x0, y0 = mark_box[0] - ox, mark_box[1] - oy
    x1, y1 = mark_box[2] - ox, mark_box[3] - oy
    if x1 < x0:
        x0, x1 = x1, x0           # normalise: a mismatched page can invert the box
    if y1 < y0:
        y0, y1 = y1, y0
    width = max(2, img.width // 170)
    ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], outline=color, width=width)


def _finish(img: Image.Image, max_w: int) -> Image.Image:
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    return img


def _crop_after(rgb: np.ndarray, box, pad: float, mark_box, color,
                shift=(0, 0), max_w: int = 520):
    """Crop from the (registered) revision array; outline the target region.

    `shift` (dy,dx) locally re-aligns the revision content to the base frame so the
    target box lands on the right spot even when the architect re-plotted the plan at
    a moved position (which global registration can't fix). The box stays at its base
    coords; only the sampled content is offset."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = _bounds(box, pad, w, h)        # base-frame window
    dy, dx = shift
    if dy or dx:
        Hc, Wc = y1 - y0, x1 - x0
        out = np.full((Hc, Wc, 3), 255, np.uint8)
        sy0, sx0 = y0 - dy, x0 - dx                 # sample final at box - shift
        vy0, vx0 = max(0, sy0), max(0, sx0)
        vy1, vx1 = min(h, sy0 + Hc), min(w, sx0 + Wc)
        if vy1 > vy0 and vx1 > vx0:
            out[vy0 - sy0:vy0 - sy0 + (vy1 - vy0),
                vx0 - sx0:vx0 - sx0 + (vx1 - vx0)] = rgb[vy0:vy1, vx0:vx1]
        img = Image.fromarray(out).convert("RGB")
    else:
        img = Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGB")
    _draw_target(img, (x0, y0), mark_box, color)
    return _finish(img, max_w)


def _crop_before(page, box, pad: float, mark_box, color, max_w: int = 520):
    """Crop from the pre-rendered marked-up page (so your red leader/box shows
    where you pointed), then outline the target region. No per-finding render."""
    rgb = page.redlined_rgb
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = _bounds(box, pad, w, h)
    img = Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGB")
    _draw_target(img, (x0, y0), mark_box, color)
    return _finish(img, max_w)


def _save_wide(page, wide_box, mark_box, color, shift, crops_dir, stem,
               max_w: int = 820):
    """Render and save the zoomed-OUT before/after crops (wider field of view) so the
    app's one-click 'zoom out' can swap them in instantly. Returns (before_rel,
    after_rel) paths."""
    before = _crop_before(page, wide_box, 0, mark_box, color, max_w=max_w)
    after = _crop_after(page.final_rgb, wide_box, 0, mark_box, color, shift=shift,
                        max_w=max_w)
    bname = f"{stem}_before_wide.png"
    aname = f"{stem}_after_wide.png"
    before.save(os.path.join(crops_dir, bname))
    after.save(os.path.join(crops_dir, aname))
    return f"crops/{bname}", f"crops/{aname}"


def _display_no(f: Finding) -> str:
    return f"{f.comment.page + 1}.{f.comment.id}"


def build_report(report: Report, out_dir: str) -> dict:
    cfg = report.config
    crops_dir = os.path.join(out_dir, "crops")
    shutil.rmtree(crops_dir, ignore_errors=True)   # avoid stale crops from prior runs
    os.makedirs(crops_dir, exist_ok=True)
    pad = cfg.px(cfg.crop_pad_pt)

    rows = []          # dicts for html/csv
    ucount = 0

    from .align import local_shift
    move_R = int(cfg.px(cfg.suppress_move_max_pt))
    for pr in report.pages:
        page = pr.page
        h, w = page.base_ink.shape
        full_box = (0, 0, w, h)
        # On poorly-aligned sheets the architect moved content within a static frame,
        # so re-align the revision crop locally to each box (else the box looks offset).
        poorly = page.similarity < cfg.register_min_iou

        def _ashift(box):
            if not poorly:
                return (0, 0)
            return local_shift(page.base_ink, page.final_ink, box, move_R)

        for f in pr.findings:
            if f.status == MARKER_INFO:
                continue
            color = STYLE[f.status][0]
            ashift = _ashift(f.target_box_px)
            if f.status == COPY_REF:
                # show the whole revised sheet (this note says "reuse markups here")
                before = _crop_before(page, full_box, 0, f.target_box_px, color)
                after = _crop_after(page.final_rgb, full_box, 0, f.target_box_px, color)
                wide_box = full_box                      # already the whole sheet
            else:
                before = _crop_before(page, f.target_box_px, pad, f.target_box_px, color)
                after = _crop_after(page.final_rgb, f.target_box_px, pad, f.target_box_px,
                                    color, shift=ashift)
                wide_box = _wide_window(f.target_box_px, pad, cfg.wide_crop_factor)
            bname = f"p{page.page_index}_c{f.comment.id}_before.png"
            aname = f"p{page.page_index}_c{f.comment.id}_after.png"
            before.save(os.path.join(crops_dir, bname))
            after.save(os.path.join(crops_dir, aname))
            bwide, awide = _save_wide(page, wide_box, f.target_box_px, color, ashift,
                                      crops_dir, f"p{page.page_index}_c{f.comment.id}")
            rows.append({
                "no": _display_no(f), "page": page.page_index + 1,
                "kind": f.comment.kind, "status": f.status,
                "text": f.comment.text, "verified_value": f.verified_value or "",
                "confidence": f.confidence, "note": f.note,
                "activity": f"{f.local_change_ratio:.3f}",
                "before": f"crops/{bname}", "after": f"crops/{aname}",
                "before_wide": bwide, "after_wide": awide,
                "pidx": page.page_index, "mag": f.local_change_ratio,
            })

        for u in pr.unexpected:
            ucount += 1
            tag = f"U{ucount}"
            ustatus = COPY_CHANGE if u.to_verify else UNEXPECTED
            ucolor = STYLE[ustatus][0]
            before = _crop_before(page, u.region.bbox_px, pad, u.region.bbox_px, ucolor)
            after = _crop_after(page.final_rgb, u.region.bbox_px, pad, u.region.bbox_px,
                                ucolor, shift=_ashift(u.region.bbox_px))
            bname = f"p{page.page_index}_{tag}_before.png"
            aname = f"p{page.page_index}_{tag}_after.png"
            before.save(os.path.join(crops_dir, bname))
            after.save(os.path.join(crops_dir, aname))
            uwide_box = _wide_window(u.region.bbox_px, pad, cfg.wide_crop_factor)
            bwide, awide = _save_wide(page, uwide_box, u.region.bbox_px, ucolor,
                                      _ashift(u.region.bbox_px), crops_dir,
                                      f"p{page.page_index}_{tag}")
            if u.to_verify:
                # a 'copy comments' sheet takes priority: the whole sheet must be
                # checked against the referenced one (covers symbol edits too).
                text = ("(change on a 'copy comments' sheet — verify it matches the "
                        "referenced sheet)")
            elif getattr(u.region, "symbol", False):
                text = "(callout/symbol content changed — e.g. a hexnote edit; verify)"
            else:
                text = "(architect changed this; not marked up)"
            rows.append({
                "no": tag, "page": page.page_index + 1, "kind": "unexpected",
                "status": ustatus, "text": text,
                "verified_value": "", "confidence": "pixel", "note": "",
                "activity": f"+{u.region.added_px}/-{u.region.removed_px}px",
                "before": f"crops/{bname}", "after": f"crops/{aname}",
                "before_wide": bwide, "after_wide": awide,
                "pidx": page.page_index, "mag": u.region.magnitude,
                "kindword": u.region.kind,
            })

    # Extra pages (inserted by the architect / unmatched / failed-to-compare).
    extra = []
    for i, ep in enumerate(report.extra):
        name = f"extra_{i}_{ep.kind}.png"
        Image.fromarray(ep.thumb).convert("RGB").save(os.path.join(crops_dir, name))
        extra.append({"kind": ep.kind, "label": ep.label, "note": ep.note,
                      "img": f"crops/{name}"})

    overviews = _write_overviews(report, crops_dir)
    _write_annotated_pdf(report, out_dir, cfg)
    _write_csv(rows, out_dir)
    counts = _write_html(report, rows, overviews, extra, out_dir)
    return {"counts": counts, "rows": rows, "overviews": overviews,
            "extra": extra, "out_dir": out_dir,
            "flattened_pages": list(report.flattened_pages)}


def _stamp_page(pr) -> Image.Image:
    """Return the revised page image with all finding/unexpected boxes drawn."""
    label_font = _font(30)
    stamp = Image.fromarray(pr.page.final_rgb.copy())
    draw = ImageDraw.Draw(stamp)
    for f in pr.findings:
        if f.status == MARKER_INFO or f.status == COPY_REF:
            continue  # copy-ref note refers to the whole sheet, not a spot
        _draw_box(draw, f.target_box_px, STYLE[f.status][0], _display_no(f), label_font)
    for i, u in enumerate(pr.unexpected, start=1):
        color = STYLE[COPY_CHANGE if u.to_verify else UNEXPECTED][0]
        _draw_box(draw, u.region.bbox_px, color, f"U{i}", label_font)
    return stamp


def _write_overviews(report: Report, crops_dir: str, width: int = 1100) -> dict:
    """Save a per-page annotated overview thumbnail; return {pidx: relpath}."""
    out = {}
    for pr in report.pages:
        if not (any(f.comment.kind != "marker" for f in pr.findings) or pr.unexpected):
            continue
        stamp = _stamp_page(pr)
        if stamp.width > width:
            stamp = stamp.resize((width, int(stamp.height * width / stamp.width)))
        name = f"overview_p{pr.page_index}.png"
        stamp.convert("RGB").save(os.path.join(crops_dir, name))
        out[pr.page_index] = f"crops/{name}"
    return out


def _write_annotated_pdf(report: Report, out_dir: str, cfg: Config):
    out = fitz.open()
    scale = cfg.report_dpi / cfg.analysis_dpi
    for pr in report.pages:
        if not (any(f.comment.kind != "marker" for f in pr.findings) or pr.unexpected):
            continue
        stamp = _stamp_page(pr)
        if scale != 1.0:
            stamp = stamp.resize((max(1, int(stamp.width * scale)),
                                  max(1, int(stamp.height * scale))))
        buf = io.BytesIO()
        stamp.convert("RGB").save(buf, format="JPEG", quality=80)
        rect = pr.page.final_page.rect
        newpage = out.new_page(width=rect.width, height=rect.height)
        newpage.insert_image(newpage.rect, stream=buf.getvalue())
    path = os.path.join(out_dir, "annotated.pdf")
    if out.page_count:
        out.save(path, deflate=True)
    out.close()


def _write_csv(rows: List[dict], out_dir: str):
    path = os.path.join(out_dir, "findings.csv")
    cols = ["no", "page", "kind", "status", "text", "verified_value",
            "confidence", "note", "activity"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_html(report: Report, rows: List[dict], overviews: dict,
                extra: List[dict], out_dir: str) -> dict:
    counts = Counter(r["status"] for r in rows)
    n_markers = sum(1 for p in report.pages for f in p.findings
                    if f.status == MARKER_INFO)

    def row_html(r):
        _, lbl, cls = STYLE[r["status"]]
        vv = f' &rarr; <b>{html.escape(r["verified_value"])}</b>' if r["verified_value"] else ""
        note = f'<div class="note">{html.escape(r["note"])}</div>' if r["note"] else ""
        return f'''
        <div class="row {cls}">
          <div class="meta">
            <span class="no">{html.escape(r["no"])}</span>
            <span class="pill {cls}">{html.escape(lbl)}</span>
            <span class="pg">p.{r["page"]}</span>
            <span class="conf">{html.escape(r["confidence"])}</span>
          </div>
          <div class="text">{html.escape(r["text"])}{vv}</div>
          {note}
          <div class="imgs">
            <figure><img src="{r["before"]}"><figcaption>original (your markup)</figcaption></figure>
            <figure><img src="{r["after"]}"><figcaption>architect&#39;s revision</figcaption></figure>
          </div>
        </div>'''

    def section(title, statuses, desc, grouped=False):
        sel = [r for r in rows if r["status"] in statuses]
        if not sel:
            return ""
        out = [f'<h2>{html.escape(title)} <span class="cnt">{len(sel)}</span></h2>',
               f'<p class="desc">{html.escape(desc)}</p>']
        if not grouped:
            out.extend(row_html(r) for r in sel)
            return "\n".join(out)
        by_page = {}
        for r in sel:
            by_page.setdefault(r["pidx"], []).append(r)
        for pidx in sorted(by_page, key=lambda p: -len(by_page[p])):
            prows = sorted(by_page[pidx], key=lambda r: -r["mag"])
            ov = overviews.get(pidx, "")
            open_attr = "open" if len(prows) <= 20 else ""
            ov_html = (f'<a href="{ov}" target="_blank"><img class="ov" src="{ov}"></a>'
                       if ov else "")
            out.append(f'''<details {open_attr} class="pg-group">
              <summary>Sheet {pidx + 1} &mdash; {len(prows)} item(s)</summary>
              {ov_html}
              {"".join(row_html(r) for r in prows)}
            </details>''')
        return "\n".join(out)

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    head = f'''<!doctype html><html><head><meta charset="utf-8">
<title>Redline check</title><style>
body{{font:14px/1.45 -apple-system,Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1b1f24}}
header{{background:#1b1f24;color:#fff;padding:18px 26px}}
header h1{{margin:0 0 4px;font-size:19px}} header .sub{{opacity:.8;font-size:12px}}
main{{max-width:1100px;margin:0 auto;padding:22px}}
.summary{{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 18px}}
.card{{background:#fff;border-radius:8px;padding:12px 16px;min-width:110px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.card b{{font-size:24px;display:block}}
.card.miss b{{color:#d20000}} .card.unexp b{{color:#a500c8}} .card.ok b{{color:#009600}} .card.verify b{{color:#e68c00}}
h2{{font-size:16px;margin:26px 0 6px;border-bottom:2px solid #e2e4e8;padding-bottom:4px}}
h2 .cnt{{background:#e2e4e8;border-radius:10px;padding:1px 9px;font-size:12px;vertical-align:middle}}
.desc{{color:#566;margin:0 0 12px;font-size:13px}}
.row{{background:#fff;border-radius:8px;padding:12px 14px;margin:10px 0;box-shadow:0 1px 3px rgba(0,0,0,.07);border-left:5px solid #ccc}}
.row.miss{{border-left-color:#d20000}} .row.unexp{{border-left-color:#a500c8}}
.row.ok{{border-left-color:#009600}} .row.verify{{border-left-color:#e68c00}} .row.marker{{border-left-color:#bbb}}
.meta{{display:flex;gap:10px;align-items:center;margin-bottom:6px}}
.no{{font-weight:700}} .pg,.conf{{color:#789;font-size:12px}}
.pill{{color:#fff;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600}}
.pill.miss{{background:#d20000}} .pill.unexp{{background:#a500c8}} .pill.ok{{background:#009600}} .pill.verify{{background:#e68c00}} .pill.marker{{background:#bbb}}
.text{{font-size:15px;margin-bottom:8px}}
.note{{color:#667;font-size:12px;margin-bottom:8px;font-style:italic}}
.imgs{{display:flex;gap:14px;flex-wrap:wrap}}
figure{{margin:0}} figure img{{border:1px solid #d0d3d8;border-radius:4px;max-width:520px;background:#fff}}
figcaption{{font-size:11px;color:#789;text-align:center;margin-top:2px}}
.pg-group{{background:#fbfbfc;border:1px solid #e2e4e8;border-radius:8px;padding:6px 12px;margin:10px 0}}
.pg-group>summary{{cursor:pointer;font-weight:600;padding:6px 0}}
img.ov{{max-width:100%;border:1px solid #d0d3d8;border-radius:4px;margin:6px 0;background:#fff}}
</style></head><body>
<header><h1>Redline check</h1>
<div class="sub">redlined: {html.escape(os.path.basename(report.redlined_path))} &nbsp;vs&nbsp; revised: {html.escape(os.path.basename(report.final_path))} &middot; generated {now}</div>
</header><main>'''

    summary = f'''<div class="summary">
<div class="card miss"><b>{counts.get(MISSED,0)+counts.get(NO_CHANGE_DETECTED,0)}</b>possible misses</div>
<div class="card unexp"><b>{counts.get(UNEXPECTED,0)}</b>unexpected</div>
<div class="card verify"><b>{counts.get(COORDINATION,0)+counts.get(CHANGED_OTHER,0)+counts.get(CHANGED_AT_TARGET,0)+counts.get(COPY_REF,0)+counts.get(COPY_CHANGE,0)+counts.get(MARKED_REGION,0)}</b>to verify</div>
<div class="card ok"><b>{counts.get(CAPTURED,0)}</b>addressed (verified)</div>
<div class="card"><b>{n_markers}</b>markers</div>
<div class="card unexp"><b>{len(extra)}</b>extra pages</div>
</div>'''

    flat = list(getattr(report, "flattened_pages", []) or [])
    if flat:
        shown = ", ".join(str(p) for p in flat[:15]) + ("…" if len(flat) > 15 else "")
        summary += (
            '<div style="background:#fff4d6;border-left:5px solid #e6a700;'
            'border-radius:8px;padding:12px 16px;margin:0 0 18px;font-size:13px">'
            f'<b>⚠️ Flattened redlines detected</b> on {len(flat)} sheet(s) (sheet '
            f'{html.escape(shown)}). The markups are baked into the page, not editable '
            'annotations, so they were caught with a red-ink fallback (no markup text '
            'to read &mdash; those appear under <i>To verify</i> / <i>Possible misses</i>). '
            'For the most accurate results, use the original un-flattened PDF with live '
            'annotations.</div>')

    body = [
        section("Possible misses — review", {MISSED, NO_CHANGE_DETECTED},
                "Redlines with no detected change at the target. Could be ignored by the "
                "architect, or an in-place text swap the pixel diff can't see — verify each. "
                "Run with --ocr for a confident captured/missed verdict.",
                grouped=True),
        section("Unexpected changes", {UNEXPECTED},
                "The architect changed these areas, but you didn't mark them up. Grouped by "
                "sheet; sheets that were heavily revised show the biggest changes first.",
                grouped=True),
        section("To verify — change seen, value not confirmed",
                {COORDINATION, CHANGED_OTHER, CHANGED_AT_TARGET, COPY_REF, COPY_CHANGE,
                 MARKED_REGION},
                "Something changed at the target; a TO-ARCH or reference note; an area you "
                "highlighted; or a 'copy comments from sheet X' note and the changes on its "
                "sheet. The value wasn't auto-confirmed — eyeball each. Run --ocr to confirm "
                "spec values.", grouped=True),
        section("Addressed — verified", {CAPTURED},
                "The instructed value was read at the target in the revision (OCR/AI). "
                "These are confirmed handled.", grouped=True),
    ]
    if extra:
        ex = ['<h2>Extra pages <span class="cnt">%d</span></h2>' % len(extra),
              '<p class="desc">Pages the architect inserted/split out, or that could not '
              'be matched and compared. Review these manually.</p>']
        for e in extra:
            ex.append(f'''<div class="row unexp">
              <div class="meta"><span class="no">{html.escape(e["label"])}</span>
              <span class="pill unexp">{html.escape(e["kind"])}</span></div>
              <div class="note">{html.escape(e["note"])}</div>
              <div class="imgs"><figure><img src="{e["img"]}"></figure></div>
            </div>''')
        body.append("\n".join(ex))
    tail = "</main></body></html>"
    path = os.path.join(out_dir, "summary.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(head + summary + "\n".join(body) + tail)
    return {
        "possible_misses": counts.get(MISSED, 0) + counts.get(NO_CHANGE_DETECTED, 0),
        "unexpected": counts.get(UNEXPECTED, 0),
        "to_verify": (counts.get(COORDINATION, 0) + counts.get(CHANGED_OTHER, 0)
                      + counts.get(CHANGED_AT_TARGET, 0) + counts.get(COPY_REF, 0)
                      + counts.get(COPY_CHANGE, 0) + counts.get(MARKED_REGION, 0)),
        "likely_addressed": counts.get(CAPTURED, 0),
        "markers": n_markers,
        "extra_pages": len(extra),
    }
