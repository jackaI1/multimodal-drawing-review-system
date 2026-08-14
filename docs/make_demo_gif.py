"""Build the animated demo used at the top of the README.

Everything shown is generated: the drawings come from
``examples/make_sample_drawings.py`` and the findings come from running the real
comparator over them. No real drawing is involved at any point.

    python docs/make_demo_gif.py

Writes ``docs/demo.gif``.
"""

import os
import subprocess
import sys
import tempfile

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
EXAMPLES = os.path.join(ROOT, "examples")

CANVAS_W, CANVAS_H = 1000, 700
BAR_H = 62
BG = (255, 255, 255)
BAR_BG = (24, 28, 34)
BAR_FG = (238, 240, 244)
ACCENT = (255, 138, 76)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # very old Pillow
        return ImageFont.load_default()


def render_pdf_page(pdf_path: str, page_no: int, target_w: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[page_no]
    scale = target_w / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def frame(img: Image.Image, step: str, caption: str) -> Image.Image:
    """Fit *img* onto the canvas under a caption bar."""
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)

    area_h = CANVAS_H - BAR_H
    scale = min((CANVAS_W - 32) / img.width, (area_h - 32) / img.height)
    w, h = int(img.width * scale), int(img.height * scale)
    canvas.paste(img.resize((w, h), Image.LANCZOS),
                 ((CANVAS_W - w) // 2, BAR_H + (area_h - h) // 2))

    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, CANVAS_W, BAR_H], fill=BAR_BG)
    d.text((22, 15), step, font=font(15), fill=ACCENT)
    d.text((22, 36), caption, font=font(15), fill=BAR_FG)
    d.rectangle([0, BAR_H - 2, CANVAS_W, BAR_H], fill=ACCENT)
    return canvas


def main() -> None:
    os.makedirs(DOCS, exist_ok=True)

    print("generating sample drawings...")
    subprocess.run([sys.executable, os.path.join(EXAMPLES, "make_sample_drawings.py"),
                    "--scenario", "basic"], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)

    report_dir = os.path.join(tempfile.mkdtemp(prefix="rc_demo_"), "report")
    print("running the comparator...")
    subprocess.run([sys.executable, os.path.join(ROOT, "comparator.py"),
                    os.path.join(EXAMPLES, "basic_redlined.pdf"),
                    os.path.join(EXAMPLES, "basic_revised.pdf"),
                    "-o", report_dir], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)

    redlined = render_pdf_page(os.path.join(EXAMPLES, "basic_redlined.pdf"), 0, 1400)
    revised = render_pdf_page(os.path.join(EXAMPLES, "basic_revised.pdf"), 0, 1400)
    annotated_pdf = os.path.join(report_dir, "annotated.pdf")
    annotated = render_pdf_page(annotated_pdf, 0, 1400)

    # Crop tightly around the area the markups point at, for the close-ups.
    def crop(img: Image.Image, box=(0.14, 0.28, 0.62, 0.68)) -> Image.Image:
        x0, y0, x1, y1 = box
        return img.crop((int(img.width * x0), int(img.height * y0),
                         int(img.width * x1), int(img.height * y1)))

    frames = [
        (frame(redlined, "STEP 1", "The marked-up drawing: two instructions in red"), 2600),
        (frame(revised, "STEP 2", "The revision that came back"), 2600),
        (frame(annotated, "STEP 3", "Every finding boxed and numbered"), 3000),
        (frame(crop(annotated), "MISSED",
               "An instruction with no change at the spot it pointed to"), 2600),
        (frame(crop(annotated, (0.30, 0.10, 0.78, 0.50)), "UNEXPECTED",
               "A change nobody asked for"), 2600),
        (frame(annotated, "RESULT",
               "2 possible misses  ·  1 unexpected change  ·  fully local"), 3200),
    ]

    out = os.path.join(DOCS, "demo.gif")
    imgs = [f for f, _ in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=[d for _, d in frames], loop=0, optimize=True)
    print(f"wrote {out} ({os.path.getsize(out):,} bytes, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
