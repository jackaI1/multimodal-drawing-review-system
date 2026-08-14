"""Generate synthetic drawing pairs so the comparison can be run end to end.

No real drawing is needed and nothing here is derived from one -- every sheet is
drawn from scratch and all text is invented. Each scenario targets a different
part of the pipeline, so you can see the behaviour that part was built for:

  basic         One instruction applied as an in-place text swap, one ignored,
                and one change nobody asked for. The applied swap is invisible
                to a pixel diff, so it stays in "possible misses" until --ocr
                reads the label.

  inserted      The revision has an extra sheet inserted in the middle.
                Comparing by page index would misalign everything after it;
                the Needleman-Wunsch precheck should pair correctly and report
                the inserted sheet as extra.

  dropped       The mirror case: a sheet present in the markup set is missing
                from the revision.

  flattened     Markups baked into the page as red linework with no Acrobat
                annotations at all, exercising the red-ink fallback.

  rotated       The sheet is stored portrait and displayed landscape via
                /Rotate. Markup coordinates must be transformed to match.

  boilerplate   A four-sheet set where the title block changes on every sheet
                (a PRELIMINARY marking is replaced by a signed block). That is
                boilerplate, not a design change, and should not flood the
                unexpected list.

  copyref       A markup that says to copy the comments from another sheet,
                which must be routed to "to verify" rather than counted as a
                miss.

  replot        The revision is re-plotted at a large offset with content
                dropped. Registration compensates for part of it, and what is
                left surfaces as unexpected changes. This is the honest noise
                case: a heavily re-plotted sheet costs false positives, which
                is why the tool ships crops for every finding rather than
                asking anyone to trust the count.

Usage:

    python examples/make_sample_drawings.py                    # basic
    python examples/make_sample_drawings.py --scenario all     # every scenario
    python examples/make_sample_drawings.py --scenario inserted

Then compare a pair, for example:

    python comparator.py examples/basic_redlined.pdf examples/basic_revised.pdf -o report
"""

import argparse
import os

import fitz

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

RED = (1.0, 0.0, 0.0)
BLACK = (0.0, 0.0, 0.0)
GREY = (0.4, 0.4, 0.4)

PAGE_W, PAGE_H = 1224, 792          # a modest landscape sheet, in points


# ---------------------------------------------------------------------------
# Base sheet
# ---------------------------------------------------------------------------

def draw_sheet(page, sheet_id: str, beam_label: str = "BEAM W200X52",
               joist_label: str = "JOIST J1", extra_note: str | None = None,
               titleblock_note: str = "PRELIMINARY", offset: float = 0.0,
               dense: bool = True) -> None:
    """Draw one synthetic plan sheet.

    ``offset`` shifts the whole plan, which is how the re-plot scenario makes
    two sheets fail to register.
    """
    ox = offset
    page.draw_rect(fitz.Rect(24, 24, PAGE_W - 24, PAGE_H - 24), color=BLACK, width=1.5)

    # Title block, bottom-right.
    tb = fitz.Rect(PAGE_W - 300, PAGE_H - 150, PAGE_W - 24, PAGE_H - 24)
    page.draw_rect(tb, color=BLACK, width=1.0)
    page.draw_line(fitz.Point(tb.x0, tb.y0 + 42), fitz.Point(tb.x1, tb.y0 + 42),
                   color=BLACK, width=0.8)
    page.draw_line(fitz.Point(tb.x0, tb.y0 + 84), fitz.Point(tb.x1, tb.y0 + 84),
                   color=BLACK, width=0.8)
    page.insert_text((tb.x0 + 10, tb.y0 + 26), "SYNTHETIC PROJECT", fontsize=9, color=BLACK)
    page.insert_text((tb.x0 + 10, tb.y0 + 68), titleblock_note, fontsize=10, color=BLACK)
    page.insert_text((tb.x1 - 70, tb.y0 + 112), sheet_id, fontsize=14, color=BLACK)

    page.insert_text((44, 52), f"SYNTHETIC PLAN {sheet_id} - NOT A REAL DRAWING",
                     fontsize=10, color=BLACK)

    # Floor outline and partitions.
    plan = fitz.Rect(110 + ox, 110, 700 + ox, 520)
    page.draw_rect(plan, color=BLACK, width=1.4)
    for x in (250, 390, 530):
        page.draw_line(fitz.Point(x + ox, plan.y0), fitz.Point(x + ox, plan.y1),
                       color=BLACK, width=0.9)
    page.draw_line(fitz.Point(plan.x0, 330), fitz.Point(plan.x1, 330),
                   color=BLACK, width=0.9)

    # Grid bubbles along the top.
    for i, x in enumerate((250, 390, 530)):
        page.draw_circle(fitz.Point(x + ox, 92), 11, color=BLACK, width=0.9)
        page.insert_text((x + ox - 4, 96), chr(65 + i), fontsize=8, color=BLACK)

    # Member labels -- these are what the markups point at.
    page.insert_text((300 + ox, 300), beam_label, fontsize=11, color=BLACK)
    page.insert_text((300 + ox, 470), joist_label, fontsize=11, color=BLACK)

    if dense:
        # A schedule table, so the sheet has ordinary content to ignore.
        tbl = fitz.Rect(760, 110, 1180, 430)
        page.draw_rect(tbl, color=BLACK, width=1.0)
        page.insert_text((tbl.x0 + 8, tbl.y0 + 16), "MEMBER SCHEDULE",
                         fontsize=8, color=BLACK)
        for i in range(1, 10):
            y = tbl.y0 + 24 + i * 32
            page.draw_line(fitz.Point(tbl.x0, y), fitz.Point(tbl.x1, y),
                           color=BLACK, width=0.7)
        for i in range(9):
            y = tbl.y0 + 24 + i * 32 + 20
            page.insert_text((tbl.x0 + 8, y), f"MEMBER M{i + 1:02d}", fontsize=6, color=BLACK)
            page.insert_text((tbl.x0 + 200, y), f"{180 + i * 20} mm", fontsize=6, color=BLACK)

        # A block of general notes.
        for i in range(8):
            page.insert_text((760, 470 + i * 13),
                             f"{i + 1}. GENERIC GENERAL NOTE LINE FOR THIS SHEET",
                             fontsize=6, color=GREY)

    if extra_note:
        page.insert_text((470 + ox, 200), extra_note, fontsize=11, color=BLACK)


def add_markup(page, target: fitz.Point, text: str, box_at: fitz.Point) -> None:
    """A red callout plus a leader line pointing at *target*."""
    rect = fitz.Rect(box_at.x, box_at.y, box_at.x + 200, box_at.y + 44)
    annot = page.add_freetext_annot(rect, text, fontsize=9, text_color=RED)
    annot.update()
    leader = page.add_line_annot(fitz.Point(rect.x0, rect.y1), target)
    leader.set_colors(stroke=RED)
    leader.set_border(width=1.2)
    leader.update()


def add_flattened_markup(page, target: fitz.Point, text: str, box_at: fitz.Point) -> None:
    """The same markup baked into the page as red linework, with no annotation."""
    rect = fitz.Rect(box_at.x, box_at.y, box_at.x + 200, box_at.y + 44)
    page.draw_rect(rect, color=RED, width=1.4)
    page.insert_text((rect.x0 + 6, rect.y0 + 18), text, fontsize=9, color=RED)
    page.draw_line(fitz.Point(rect.x0, rect.y1), target, color=RED, width=1.4)
    page.draw_circle(target, 16, color=RED, width=1.4)


def save(doc, name: str) -> str:
    path = os.path.join(OUT_DIR, name)
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Scenarios -- each returns a list of (path, description)
# ---------------------------------------------------------------------------

def scenario_basic() -> list:
    r = fitz.open()
    p = r.new_page(width=PAGE_W, height=PAGE_H)
    draw_sheet(p, "S1.01")
    add_markup(p, fitz.Point(310, 296), "CHANGE TO W310X52", fitz.Point(770, 560))
    add_markup(p, fitz.Point(310, 466), "REVISE TO JOIST J2", fitz.Point(770, 660))

    f = fitz.open()
    p2 = f.new_page(width=PAGE_W, height=PAGE_H)
    draw_sheet(p2, "S1.01", beam_label="BEAM W310X52", extra_note="ADDED POST P3",
               titleblock_note="ISSUED FOR CONSTRUCTION")

    return [(save(r, "basic_redlined.pdf"), "1 applied (in-place swap), 1 missed, 1 unexpected"),
            (save(f, "basic_revised.pdf"), "")]


def scenario_inserted() -> list:
    r = fitz.open()
    for i, sid in enumerate(("S1.01", "S1.02", "S1.03")):
        p = r.new_page(width=PAGE_W, height=PAGE_H)
        draw_sheet(p, sid, beam_label=f"BEAM W200X{40 + i * 4}")
        add_markup(p, fitz.Point(310, 296), f"CHANGE TO W310X{40 + i * 4}",
                   fitz.Point(770, 560))

    f = fitz.open()
    # Same three sheets, with a brand-new sheet inserted between 1 and 2.
    order = [("S1.01", 0), ("S1.01A", None), ("S1.02", 1), ("S1.03", 2)]
    for sid, src in order:
        p = f.new_page(width=PAGE_W, height=PAGE_H)
        if src is None:
            draw_sheet(p, sid, beam_label="BEAM W250X45", joist_label="JOIST J9",
                       titleblock_note="ISSUED FOR CONSTRUCTION")
        else:
            draw_sheet(p, sid, beam_label=f"BEAM W310X{40 + src * 4}",
                       titleblock_note="ISSUED FOR CONSTRUCTION")

    return [(save(r, "inserted_redlined.pdf"), "3 sheets marked up"),
            (save(f, "inserted_revised.pdf"), "4 sheets: one inserted at position 2")]


def scenario_dropped() -> list:
    r = fitz.open()
    for sid in ("S2.01", "S2.02", "S2.03"):
        p = r.new_page(width=PAGE_W, height=PAGE_H)
        draw_sheet(p, sid)
        add_markup(p, fitz.Point(310, 296), "CHANGE TO W310X52", fitz.Point(770, 560))

    f = fitz.open()
    for sid in ("S2.01", "S2.03"):        # S2.02 never came back
        p = f.new_page(width=PAGE_W, height=PAGE_H)
        draw_sheet(p, sid, beam_label="BEAM W310X52",
                   titleblock_note="ISSUED FOR CONSTRUCTION")

    return [(save(r, "dropped_redlined.pdf"), "3 sheets marked up"),
            (save(f, "dropped_revised.pdf"), "2 sheets: the middle one is missing")]


def scenario_flattened() -> list:
    r = fitz.open()
    p = r.new_page(width=PAGE_W, height=PAGE_H)
    draw_sheet(p, "S3.01")
    # No annotations at all -- the markup is part of the page content.
    add_flattened_markup(p, fitz.Point(310, 296), "CHANGE TO W310X52", fitz.Point(770, 560))

    f = fitz.open()
    p2 = f.new_page(width=PAGE_W, height=PAGE_H)
    draw_sheet(p2, "S3.01", beam_label="BEAM W310X52",
               titleblock_note="ISSUED FOR CONSTRUCTION")

    return [(save(r, "flattened_redlined.pdf"), "markup baked in as red linework, no annotations"),
            (save(f, "flattened_revised.pdf"), "")]


def scenario_rotated() -> list:
    r = fitz.open()
    p = r.new_page(width=PAGE_H, height=PAGE_W)     # stored portrait
    p.set_rotation(270)                             # displayed landscape
    draw_sheet(p, "S4.01")
    add_markup(p, fitz.Point(310, 296), "CHANGE TO W310X52", fitz.Point(500, 620))

    f = fitz.open()
    p2 = f.new_page(width=PAGE_H, height=PAGE_W)
    p2.set_rotation(270)
    draw_sheet(p2, "S4.01", beam_label="BEAM W310X52",
               titleblock_note="ISSUED FOR CONSTRUCTION")

    return [(save(r, "rotated_redlined.pdf"), "page stored portrait, displayed landscape"),
            (save(f, "rotated_revised.pdf"), "")]


def scenario_boilerplate() -> list:
    r = fitz.open()
    for sid in ("S5.01", "S5.02", "S5.03", "S5.04"):
        p = r.new_page(width=PAGE_W, height=PAGE_H)
        draw_sheet(p, sid, titleblock_note="PRELIMINARY")
        add_markup(p, fitz.Point(310, 296), "CHANGE TO W310X52", fitz.Point(770, 560))

    f = fitz.open()
    for sid in ("S5.01", "S5.02", "S5.03", "S5.04"):
        p = f.new_page(width=PAGE_W, height=PAGE_H)
        # The same title-block change lands on every single sheet.
        draw_sheet(p, sid, beam_label="BEAM W310X52", titleblock_note="SIGNED / SEALED")

    return [(save(r, "boilerplate_redlined.pdf"), "4 sheets, all marked PRELIMINARY"),
            (save(f, "boilerplate_revised.pdf"), "same change on all 4 title blocks")]


def scenario_copyref() -> list:
    r = fitz.open()
    p = r.new_page(width=PAGE_W, height=PAGE_H)
    draw_sheet(p, "S6.01")
    add_markup(p, fitz.Point(310, 296), "PLEASE COPY COMMENTS FROM SHEET S6.02",
               fitz.Point(770, 560))

    f = fitz.open()
    p2 = f.new_page(width=PAGE_W, height=PAGE_H)
    draw_sheet(p2, "S6.01", beam_label="BEAM W310X52", extra_note="REVISED HEADER H2",
               titleblock_note="ISSUED FOR CONSTRUCTION")

    return [(save(r, "copyref_redlined.pdf"), "a copy-comments instruction"),
            (save(f, "copyref_revised.pdf"), "changes on the same sheet")]


def scenario_replot() -> list:
    r = fitz.open()
    p = r.new_page(width=PAGE_W, height=PAGE_H)
    draw_sheet(p, "S7.01")
    add_markup(p, fitz.Point(310, 296), "CHANGE TO W310X52", fitz.Point(770, 560))

    f = fitz.open()
    p2 = f.new_page(width=PAGE_W, height=PAGE_H)
    # Whole plan shifted a long way AND the schedule dropped: the two sheets
    # share a title block but almost nothing else, so registration fails and
    # per-region flags should give way to a single page-level notice.
    draw_sheet(p2, "S7.01", beam_label="BEAM W310X52", offset=330.0,
               titleblock_note="ISSUED FOR CONSTRUCTION", dense=False)

    return [(save(r, "replot_redlined.pdf"), "baseline"),
            (save(f, "replot_revised.pdf"),
             "re-plotted at a large offset; the residual shows up as unexpected")]


SCENARIOS = {
    "basic": scenario_basic,
    "inserted": scenario_inserted,
    "dropped": scenario_dropped,
    "flattened": scenario_flattened,
    "rotated": scenario_rotated,
    "boilerplate": scenario_boilerplate,
    "copyref": scenario_copyref,
    "replot": scenario_replot,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="basic")
    args = ap.parse_args()

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        print(f"--- {name} " + "-" * (58 - len(name)))
        for path, note in SCENARIOS[name]():
            tail = f"   {note}" if note else ""
            print(f"  {os.path.basename(path)}{tail}")
        print(f"  compare: python comparator.py examples/{name}_redlined.pdf "
              f"examples/{name}_revised.pdf -o report")
        print()


if __name__ == "__main__":
    main()
