"""Synthetic read-back regressions; OCR tests use the real local Tesseract."""
from types import SimpleNamespace
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from redlinecheck import classify as C, ocr, vlm
from redlinecheck.config import Config
from redlinecheck.extract import Comment, SPEC


def text_image(text):
    image = Image.new("RGB", (800, 180), "white")
    # PyMuPDF's bundled font keeps fixtures portable without OS fonts.
    import fitz
    font = ImageFont.truetype(io.BytesIO(fitz.Font("helv").buffer), 40)
    ImageDraw.Draw(image).text((50, 55), text, fill="black", font=font)
    return np.asarray(image)


def ocr_result(before, after, instruction, changed=False):
    if not ocr.check_availability()[0]:
        pytest.skip("local Tesseract is required for actual OCR integration")
    cfg = Config(analysis_dpi=300, auto_dpi=False, ocr_rotations=())
    page = SimpleNamespace(base_rgb=text_image(before), final_rgb=text_image(after))
    comment = Comment(1, 0, SPEC, instruction, (0, 0, 1, 1))
    return ocr.make_verifier(cfg)(page, comment, (0, 0, 800, 180), changed)


def test_ocr_wrong_in_place_label_is_review_change():
    result = ocr_result("L3", "L4", "REVISE TO L5")
    assert result[0] == C.CHANGED_OTHER
    assert result[1] == "L4"


def test_ocr_placeholder_removal_does_not_verify_requested_value():
    result = ocr_result("BM. BY ENG.", "BEAM", "CHANGE TO W310X52", True)
    assert result[0] != C.CAPTURED


def test_ocr_partial_multi_value_instruction_is_not_verified():
    result = ocr_result("W200X52", "W310X52", "ADD W310X52 AND W410X60", True)
    assert result[0] != C.CAPTURED


def test_ocr_reports_actual_read_back_value():
    result = ocr_result("W200X52", "W310X52", "CHANGE TO W310X52")
    assert result[:2] == (C.CAPTURED, "W310X52")


def test_ocr_unchanged_instruction_is_missed():
    assert ocr_result("J1", "J1", "REVISE TO JOIST J2")[0] == C.MISSED


def test_page_wide_hexnote_read_is_review_not_target_verification():
    if not ocr.check_availability()[0]:
        pytest.skip("local Tesseract is required for actual OCR integration")
    after = text_image("BEAM").copy()
    image = Image.fromarray(after)
    import fitz
    font = ImageFont.truetype(io.BytesIO(fitz.Font("helv").buffer), 40)
    ImageDraw.Draw(image).text((650, 55), "17", fill="black", font=font)
    page = SimpleNamespace(base_rgb=text_image("BEAM"), final_rgb=np.asarray(image))
    comment = Comment(1, 0, SPEC, "ADD HEXNOTE 17", (0, 0, 1, 1))
    cfg = Config(analysis_dpi=300, auto_dpi=False, ocr_rotations=())
    result = ocr.make_verifier(cfg)(page, comment, (0, 0, 180, 180), False)
    assert result[0] != C.CAPTURED


def test_replacement_instruction_does_not_count_the_old_value_as_proof():
    assert ocr._instruction_tokens("CHANGE FROM L3 TO L5") == ["L5"]
    assert ocr._instruction_tokens("REPLACE W200X52 WITH W310X52") == ["W310X52"]


@pytest.mark.parametrize("content", ['[]', '"yes"', 'null', '123'])
def test_vlm_non_object_json_is_uncertain(content):
    assert vlm._parse(content) is None


@pytest.mark.parametrize("tail", [[], [{"error": "synthetic interrupted inference"}]])
def test_interrupted_model_stream_cannot_supply_a_verified_verdict(monkeypatch, tail):
    class Response(io.BytesIO):
        pass

    answer = '{"applied":"yes","value":"L5","reason":"read label"}'
    events = [{"message": {"content": answer}, "done": False}] + tail
    response = Response(b"".join(json.dumps(e).encode() + b"\n" for e in events))
    monkeypatch.setattr(vlm.urllib.request, "urlopen", lambda *a, **kw: response)
    result = vlm._chat(Config(), "REVISE TO L5", "before", "after", None)
    assert result is None
    assert response.closed


def test_completed_model_stream_parses_read_back_and_closes_response(monkeypatch):
    response = io.BytesIO((json.dumps({"message": {"content":
        '{"applied":"yes","value":"L5","reason":"read label"}'}, "done": True}) + "\n").encode())
    monkeypatch.setattr(vlm.urllib.request, "urlopen", lambda *a, **kw: response)
    assert vlm._chat(Config(), "REVISE TO L5", "before", "after", None)["value"] == "L5"
    assert response.closed


@pytest.mark.parametrize("answer,changed,want", [
    ({"applied": "yes", "value": "L4", "reason": "read label"}, False, C.CHANGED_OTHER),
    ({"applied": "no", "value": "L4", "reason": "wrong label"}, True, C.CHANGED_OTHER),
    ({"applied": "yes", "value": "", "reason": ""}, True, None),
    ({"applied": "uncertain", "value": "", "reason": ""}, False, None),
    ({"applied": "yes", "value": "L5", "reason": "read label"}, True, C.CAPTURED),
])
def test_vlm_read_back_policy(monkeypatch, answer, changed, want):
    # Model inference is external; keep crop encoding and verdict policy real.
    monkeypatch.setattr(vlm, "check_availability", lambda cfg: (True, "fixture"))
    monkeypatch.setattr(vlm, "_chat", lambda *args: answer)
    rgb = text_image("L3")
    page = SimpleNamespace(redlined_rgb=rgb, final_rgb=rgb)
    comment = Comment(1, 0, SPEC, "REVISE TO L5", (0, 0, 1, 1))
    result = vlm.make_verifier(Config())(page, comment, (0, 0, 800, 180), changed)
    assert (result[0] if result else None) == want


def test_sample_generator_imports_and_runs_on_supported_python(tmp_path, monkeypatch):
    from examples import make_sample_drawings as samples
    monkeypatch.setattr(samples, "OUT_DIR", str(tmp_path))
    samples.scenario_basic()
    assert (tmp_path / "basic_redlined.pdf").is_file()
    assert (tmp_path / "basic_revised.pdf").is_file()
