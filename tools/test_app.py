"""Headless smoke test for the Streamlit app using AppTest."""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest
from redlinecheck.config import Config
from redlinecheck import pipeline, report as report_mod

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
PRELIM = "sample-prelim-redlined.pdf"
FINAL = "sample-final-revised.pdf"

# 1) empty state — script must execute with no exception
at = AppTest.from_file(APP, default_timeout=60).run()
print("empty-state exception:", at.exception)
assert not at.exception, at.exception
assert any("Redline Check" in (getattr(t, "value", "") or "") for t in at.title)

# 2) results state — inject a real result and confirm the results UI renders
cfg = Config(analysis_dpi=150)            # lower dpi for a faster test
cfg.auto_dpi = False                      # keep the test fast/deterministic
rep = pipeline.run(PRELIM, FINAL, cfg, progress=False)
out = tempfile.mkdtemp(prefix="apptest_")
result = report_mod.build_report(rep, out)

at2 = AppTest.from_file(APP, default_timeout=180)
at2.session_state["result"] = result
at2.session_state["out_dir"] = out
at2.session_state["names"] = ("a.pdf", "b.pdf")
at2.run()
print("results-state exception:", at2.exception)
assert not at2.exception, at2.exception
print("metrics rendered:", len(at2.metric))
print("tabs rendered:", len(at2.tabs))
print("download buttons:", len(at2.button) + len([d for d in at2.get("download_button")]))
print("APP SMOKE TEST OK")
