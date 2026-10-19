"""Run advertised workflows against drawings generated entirely in this repo."""
from collections import Counter
from pathlib import Path
import subprocess
import sys

import fitz
import numpy as np
import pytest

from examples import make_sample_drawings as samples
from redlinecheck import pipeline, report, ocr
from redlinecheck.config import Config
from redlinecheck.classify import NO_CHANGE_DETECTED, MARKED_REGION, COPY_REF, CHANGED_AT_TARGET


@pytest.mark.parametrize("scenario,pairs,statuses,unexpected,extras", [
    ("basic", [(0, 0)], {NO_CHANGE_DETECTED: 2}, 1, []),
    ("inserted", [(0, 0), (1, 2), (2, 3)], {NO_CHANGE_DETECTED: 3}, 0, ["inserted"]),
    ("dropped", [(0, 0), (2, 1)], {NO_CHANGE_DETECTED: 2}, 0, ["unmatched"]),
    ("flattened", [(0, 0)], {MARKED_REGION: 1}, 0, []),
    ("rotated", [(0, 0)], {NO_CHANGE_DETECTED: 1}, 0, []),
    ("boilerplate", [(0, 0), (1, 1), (2, 2), (3, 3)], {NO_CHANGE_DETECTED: 4}, 0, []),
    ("copyref", [(0, 0)], {COPY_REF: 1}, 1, []),
    ("replot", [(0, 0)], {CHANGED_AT_TARGET: 1}, 2, []),
])
def test_documented_scenarios(tmp_path, monkeypatch, scenario, pairs, statuses, unexpected, extras):
    monkeypatch.setattr(samples, "OUT_DIR", str(tmp_path))
    paths = samples.SCENARIOS[scenario]()
    rep = pipeline.run(paths[0][0], paths[1][0],
                       Config(analysis_dpi=150, auto_dpi=False), progress=False)
    assert [(p.page_index, p.final_index) for p in rep.pages] == pairs
    assert dict(Counter(f.status for f in rep.all_findings)) == statuses
    assert len(rep.all_unexpected) == unexpected
    assert [e.kind for e in rep.extra] == extras
    assert rep.flattened_pages == ([1] if scenario == "flattened" else [])
    if scenario == "copyref":
        assert all(u.to_verify for u in rep.all_unexpected)
    result = report.build_report(rep, str(tmp_path / "report"))
    for row in result["rows"]:
        for key in ("before", "after", "before_wide", "after_wide"):
            assert (tmp_path / "report" / row[key]).is_file()
    assert (tmp_path / "report/summary.html").is_file()
    assert (tmp_path / "report/findings.csv").is_file()


def test_unchanged_document_still_exports_annotated_pdf(tmp_path):
    path = tmp_path / "plain.pdf"
    with fitz.open() as doc:
        p = doc.new_page(width=300, height=200)
        p.insert_text((30, 70), "SYNTHETIC UNCHANGED SHEET")
        doc.save(path)
    rep = pipeline.run(str(path), str(path), Config(analysis_dpi=100, auto_dpi=False), progress=False)
    result = report.build_report(rep, str(tmp_path / "report"))
    assert result["rows"] == []
    with fitz.open(tmp_path / "report/annotated.pdf") as annotated:
        assert annotated.page_count == 1


def test_all_unmatched_pages_still_export_annotated_pdf(tmp_path):
    paths = []
    for name, width, height in [("portrait", 200, 300), ("landscape", 300, 200)]:
        path = tmp_path / f"{name}.pdf"
        with fitz.open() as doc:
            doc.new_page(width=width, height=height).insert_text((30, 70), "SYNTHETIC")
            doc.save(path)
        paths.append(str(path))
    rep = pipeline.run(*paths, Config(analysis_dpi=100, auto_dpi=False), progress=False)
    result = report.build_report(rep, str(tmp_path / "report"))
    assert result["counts"]["extra_pages"] == 2
    with fitz.open(tmp_path / "report/annotated.pdf") as annotated:
        assert annotated.page_count == 2


def test_auto_dpi_uses_text_size_and_bounds():
    with fitz.open() as doc:
        doc.new_page().insert_text((30, 70), "SYNTHETIC", fontsize=10)
        assert pipeline.choose_dpi(doc, Config()) == 225
    with fitz.open() as doc:
        doc.new_page().insert_text((30, 70), "SYNTHETIC", fontsize=2)
        assert pipeline.choose_dpi(doc, Config()) == 500
    with fitz.open() as doc:
        doc.new_page().insert_text((30, 70), "SYNTHETIC", fontsize=50)
        assert pipeline.choose_dpi(doc, Config()) == 200


def test_content_boilerplate_is_suppressed_even_when_it_moves():
    from types import SimpleNamespace
    from redlinecheck.classify import Unexpected
    from redlinecheck.diff import ChangeRegion
    pages = []
    for i in range(3):
        rgb = np.full((160, 240, 3), 255, np.uint8)
        x = 10 + i * 60
        rgb[20:28, x:x + 20] = 0
        rgb[28:40, x:x + 7] = 0
        region = ChangeRegion((x, 20, x + 20, 40), 200, 0, 10)
        pages.append(SimpleNamespace(page_index=i, page=SimpleNamespace(final_rgb=rgb,
                     base_ink=np.zeros((160, 240), bool)), unexpected=[Unexpected(region, i)]))
    pipeline._suppress_boilerplate(SimpleNamespace(pages=pages), Config())
    assert all(not p.unexpected for p in pages)


def test_cli_quick_start_writes_all_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(samples, "OUT_DIR", str(tmp_path))
    paths = samples.scenario_basic()
    out = tmp_path / "report"
    run = subprocess.run([sys.executable, "comparator.py", paths[0][0], paths[1][0],
                          "-o", str(out)], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert "Possible misses (review):   2" in run.stdout
    assert "Unexpected changes:         1" in run.stdout
    assert all((out / f).is_file() for f in ("summary.html", "annotated.pdf", "findings.csv"))


def test_four_film_verdicts_end_to_end_with_real_ocr(tmp_path):
    if not ocr.check_availability()[0]:
        pytest.skip("local Tesseract is required for actual OCR integration")
    paths = []
    for revised in (False, True):
        path = tmp_path / ("revised.pdf" if revised else "redlined.pdf")
        with fitz.open() as doc:
            p = doc.new_page(width=800, height=600)
            p.draw_rect(fitz.Rect(20, 20, 780, 580))
            for x, y, label, requested in [
                (90, 160, "W310X52" if revised else "W200X52", "CHANGE TO W310X52"),
                (480, 160, "J1", "REVISE TO J2"),
                (90, 420, "L4" if revised else "L3", "REVISE TO L5"),
            ]:
                p.insert_text((x, y), label, fontsize=24)
                if not revised:
                    samples.add_markup(p, fitz.Point(x + (60 if label.startswith("W") else 15), y - 5), requested,
                                       fitz.Point(x + 10, y - 95))
            if revised:
                p.draw_rect(fitz.Rect(600, 400, 625, 465), fill=(0, 0, 0))
            doc.save(path)
        paths.append(str(path))
    cfg = Config(analysis_dpi=200, auto_dpi=False, use_ocr=True, ignore_zones=[],
                 ocr_rotations=())
    rep = pipeline.run(*paths, cfg, verifier=ocr.make_verifier(cfg), progress=False)
    verdicts = {f.comment.text: f.status for f in rep.all_findings}
    assert verdicts == {"CHANGE TO W310X52": "CAPTURED", "REVISE TO J2": "MISSED",
                        "REVISE TO L5": "CHANGED_OTHER"}
    assert len(rep.all_unexpected) == 1
    result = report.build_report(rep, str(tmp_path / "report"))
    assert result["counts"]["likely_addressed"] == 1
    assert result["counts"]["possible_misses"] == 1
    assert result["counts"]["to_verify"] == 1
    assert result["counts"]["unexpected"] == 1


def test_density_diff_preserves_physical_change_across_dpi(tmp_path):
    paths = []
    for revised in (False, True):
        path = tmp_path / f"scan-{revised}.pdf"
        with fitz.open() as vector, fitz.open() as scan:
            p = vector.new_page(width=300, height=200)
            p.draw_rect(fitz.Rect(20, 20, 280, 180))
            p.insert_text((30, 60), "SYNTHETIC SCAN")
            for y in range(75, 175, 20):
                p.draw_line(fitz.Point(30, y), fitz.Point(260, y))
            if revised:
                p.draw_rect(fitz.Rect(80, 90, 100, 105), fill=(0, 0, 0))
            # Embed raster-only content: no extractable text or vector drawing.
            pix = p.get_pixmap(matrix=fitz.Matrix(2, 2))
            scan.new_page(width=300, height=200).insert_image(p.rect, stream=pix.tobytes("png"))
            scan.save(path)
        paths.append(str(path))
    for dpi in (200, 400):
        rep = pipeline.run(*paths, Config(analysis_dpi=dpi, auto_dpi=False, ignore_zones=[]),
                           progress=False)
        assert len(rep.all_unexpected) == 1
        assert rep.pages[0].page.final_page.get_text() == ""


def test_project_run_matches_files_and_skips_synthetic_junk(tmp_path, monkeypatch):
    import shutil
    from redlinecheck.project import compare_project
    monkeypatch.setattr(samples, "OUT_DIR", str(tmp_path))
    paths = samples.scenario_basic()
    rpaths, fpaths = [], []
    for label in ("A", "B"):
        rpath, fpath = tmp_path / f"{label}-redlined.pdf", tmp_path / f"{label}.pdf"
        shutil.copyfile(paths[0][0], rpath)
        shutil.copyfile(paths[1][0], fpath)
        rpaths.append(str(rpath))
        fpaths.append(str(fpath))
    junk = tmp_path / "synthetic.txt"
    junk.write_text("synthetic non-PDF fixture")
    result = compare_project(rpaths + [str(junk)], list(reversed(fpaths)),
                             Config(analysis_dpi=150, auto_dpi=False),
                             str(tmp_path / "project"))
    assert result["n_pairs"] == 2
    assert {(p["redlined"], p["revised"]) for p in result["pairs"]} == {
        ("A-redlined.pdf", "A.pdf"), ("B-redlined.pdf", "B.pdf")}
    assert result["counts"]["possible_misses"] == 4
    assert result["counts"]["unexpected"] == 2
    assert result["skipped"] == [{"name": "synthetic.txt", "why": "not a PDF"}]
    assert (tmp_path / "project/index.html").is_file()
    assert {row["source"] for row in result["rows"]} == {"A-redlined", "B-redlined"}
