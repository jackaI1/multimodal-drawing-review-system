"""Generate public, synthetic sheets for local vector-search examples."""
from pathlib import Path
import fitz


def make_drawings(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sheets = [
        ("S1", "STEEL FRAMING PLAN", "Steel beams and columns support the floor slab."),
        ("M2", "PLUMBING LAYOUT", "Water supply pipes, drain lines and sanitary sewer connections."),
        ("E3", "LIGHTING CIRCUITS", "Electrical panel, switches and ceiling light fixtures."),
    ]
    paths = []
    for revision, order in (("r1", [0, 1, 2]), ("r2", [2, 0, 1])):
        path = folder / f"search_{revision}.pdf"
        with fitz.open() as doc:
            for i in order:
                code, title, text = sheets[i]
                page = doc.new_page(width=600, height=400)
                page.insert_text((30, 35), f"{code} {title}", fontsize=18)
                page.insert_text((30, 60), text, fontsize=11)
                if i == 0:
                    for x in range(60, 550, 80):
                        page.draw_line((x, 100), (x, 320), width=5)
                    for y in (110, 210, 310):
                        page.draw_line((60, y), (540, y), width=3)
                elif i == 1:
                    page.draw_polyline([(60, 130), (460, 130), (460, 300), (150, 300)], width=4)
                    for x in (100, 200, 300):
                        page.draw_circle((x, 200), 23, width=2)
                        page.draw_line((x, 130), (x, 177), width=2)
                else:
                    for x in (100, 250, 400):
                        for y in (150, 270):
                            page.draw_circle((x, y), 12, width=2)
                            page.draw_line((x-8, y-8), (x+8, y+8), width=2)
                    page.draw_rect((480, 150, 530, 270), width=2)
                page.insert_text((420, 380), f"REVISION {revision.upper()}", fontsize=10)
            doc.save(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for path in make_drawings(Path(__file__).parent):
        print(path)
