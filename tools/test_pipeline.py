import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from PIL import Image
import fitz

from redlinecheck.config import Config
from redlinecheck.extract import extract_comments
from redlinecheck.align import prepare_page
from redlinecheck.diff import compute_change
from redlinecheck import classify as C

PRELIM = "sample-prelim-redlined.pdf"
FINAL = "sample-final-revised.pdf"

COLORS = {
    C.CHANGED_AT_TARGET: (0, 160, 0), C.CAPTURED: (0, 160, 0),
    C.NO_CHANGE_DETECTED: (220, 0, 0), C.MISSED: (220, 0, 0),
    C.COORDINATION: (255, 140, 0), C.CHANGED_OTHER: (255, 140, 0),
    C.MARKER_INFO: (130, 130, 130),
}


def draw_box(arr, box, color, t=3):
    x0, y0, x1, y1 = (int(v) for v in box)
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(arr.shape[1] - 1, x1); y1 = min(arr.shape[0] - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return
    arr[y0:y1, x0:x0 + t] = color
    arr[y0:y1, x1 - t:x1] = color
    arr[y0:y0 + t, x0:x1] = color
    arr[y1 - t:y1, x0:x1] = color


def main(pno=1):
    rdoc = fitz.open(PRELIM); fdoc = fitz.open(FINAL)
    cfg = Config()
    comments = extract_comments(rdoc[pno], pno, cfg)
    page = prepare_page(rdoc[pno], fdoc[pno], pno, cfg)
    print(f"shift={page.shift} similarity={page.similarity:.3f}  "
          f"frame={page.base_ink.shape}")
    diff = compute_change(page, cfg)
    print(f"change regions: {len(diff.regions)}")
    findings, unexpected = C.classify_page(page, comments, diff, cfg)

    print("STATUS COUNTS:", dict(Counter(f.status for f in findings)))
    print(f"UNEXPECTED: {len(unexpected)}")
    for f in findings:
        if f.comment.kind == "marker":
            continue
        print(f"  #{f.comment.id:<3} {f.status:<18} ratio={f.local_change_ratio:.3f} "
              f"reg={len(f.matched_regions)} {f.comment.text[:40]!r}")
    for u in unexpected:
        b = u.region.bbox_px
        print(f"  UNEXPECTED bbox={tuple(int(v) for v in b)} add={u.region.added_px} rem={u.region.removed_px}")

    viz = page.final_rgb.copy()
    for f in findings:
        draw_box(viz, f.target_box_px, COLORS.get(f.status, (0, 0, 255)))
    for u in unexpected:
        draw_box(viz, u.region.bbox_px, (160, 0, 200), t=4)
    h, w = viz.shape[:2]
    Image.fromarray(viz).resize((w // 3, h // 3)).save(f"tools/_debug/pipeline_p{pno}.png")
    Image.fromarray(viz).save(f"tools/_debug/pipeline_p{pno}_full.png")
    print(f"saved tools/_debug/pipeline_p{pno}.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
