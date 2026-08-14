"""Dev tool: inspect the internal structure of the two sample PDFs.

Run: python tools/inspect_pdf.py "<pdf path>"
"""
import sys
import fitz  # PyMuPDF


def describe(path):
    doc = fitz.open(path)
    print("=" * 80)
    print(f"FILE: {path}")
    print(f"pages: {doc.page_count}  is_pdf: {doc.is_pdf}  metadata: {doc.metadata.get('producer')!r}")
    for pno in range(doc.page_count):
        page = doc[pno]
        text = page.get_text("text")
        words = page.get_text("words")
        drawings = page.get_drawings()
        images = page.get_images(full=True)
        annots = list(page.annots()) if page.first_annot else []
        print("-" * 80)
        print(f"PAGE {pno}: size={tuple(round(v,1) for v in page.rect[2:])} rot={page.rotation}")
        print(f"  text_chars={len(text)}  words={len(words)}  drawings={len(drawings)}  "
              f"images={len(images)}  annots={len(annots)}")
        # Annotation breakdown
        if annots:
            print("  ANNOTATIONS:")
            for i, a in enumerate(annots):
                info = a.info
                content = (info.get("content") or "").replace("\n", " ").strip()
                stroke = a.colors.get("stroke")
                fill = a.colors.get("fill")
                r = a.rect
                print(f"    [{i:>2}] type={a.type[1]:<12} "
                      f"rect=({r.x0:.0f},{r.y0:.0f},{r.x1:.0f},{r.y1:.0f}) "
                      f"stroke={_fmt(stroke)} fill={_fmt(fill)} "
                      f"content={content[:60]!r}")
        # Sample of red-ish text spans in the page content itself (flattened markups?)
        red_spans = []
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    rgb = _int_to_rgb(span.get("color", 0))
                    if _is_red(rgb):
                        red_spans.append((span["text"], rgb, span["bbox"]))
        if red_spans:
            print(f"  RED TEXT SPANS in page content: {len(red_spans)} (flattened markups?)")
            for t, rgb, bb in red_spans[:8]:
                print(f"      {t!r:40} rgb={tuple(round(c,2) for c in rgb)}")
    doc.close()


def _fmt(c):
    if not c:
        return "None"
    return "(" + ",".join(f"{x:.2f}" for x in c) + ")"


def _int_to_rgb(c):
    return ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)


def _is_red(rgb):
    r, g, b = rgb
    return r > 0.5 and g < 0.4 and b < 0.4


if __name__ == "__main__":
    for p in sys.argv[1:]:
        describe(p)
