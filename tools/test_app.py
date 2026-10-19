"""Headless UI smoke tests using generated drawings, never private PDFs."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from examples import make_sample_drawings as samples
from redlinecheck.config import Config
from redlinecheck import pipeline, report as report_mod

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_empty_state():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Redline Check" in title.value for title in at.title)
    assert at.button[0].disabled


def test_results_and_view_switches(tmp_path, monkeypatch):
    monkeypatch.setattr(samples, "OUT_DIR", str(tmp_path))
    paths = samples.scenario_basic()
    rep = pipeline.run(paths[0][0], paths[1][0],
                       Config(analysis_dpi=150, auto_dpi=False), progress=False)
    out = tmp_path / "report"
    result = report_mod.build_report(rep, str(out))
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["result"] = result
    at.session_state["out_dir"] = str(out)
    at.session_state["names"] = ("synthetic-redlined.pdf", "synthetic-revised.pdf")
    at.run()
    assert not at.exception
    assert len(at.metric) == 6
    assert len(at.get("download_button")) == 4
    view = next(r for r in at.radio if r.label == "View")
    for value in ["miss", "unexp", "verify", "ov", "extra"]:
        view.set_value(value).run()
        assert not at.exception
