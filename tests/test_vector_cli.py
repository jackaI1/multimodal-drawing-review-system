import json

import fitz
import pytest

from examples.make_search_drawings import make_drawings
from redlinecheck import cli, pipeline, report
from redlinecheck.config import Config
from redlinecheck.prematch import MatchResult
from redlinecheck import vector_cli
from test_vector_search import index


def test_index_search_and_matching_commands(tmp_path, monkeypatch, capsys):
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient
    from redlinecheck.vector_search import DrawingIndex
    from test_vector_search import FixtureEncoder
    def connect(**kw):
        return DrawingIndex(QdrantClient(path=str(tmp_path / "database")), FixtureEncoder())
    monkeypatch.setattr(vector_cli.DrawingIndex, "connect", connect)
    old, new = make_drawings(tmp_path)
    assert vector_cli.main(["index", str(new), "--project", "demo", "--revision", "r2"]) == 0
    assert json.loads(capsys.readouterr().out)["indexed_pages"] == 3
    # Fixture vectors understand beam/pipe; sheet filter makes this exact.
    assert vector_cli.main(["search", "--text", "pipe", "--project", "demo", "--revision", "r2", "--sheet", "M2"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["page"] == 3
    assert vector_cli.main(["match", str(old), "--page", "2", "--project", "demo", "--revision", "r2", "--sheet", "M2"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["page"] == 3


def test_report_exposes_scoped_candidates_and_previews(index, tmp_path):
    old, new = make_drawings(tmp_path)
    cfg = Config(analysis_dpi=72, auto_dpi=False)
    # Force one uncertain page to exercise the fallback consumer.
    match = MatchResult(extra_redlined=[1], extra_revised=[0, 1, 2])
    rep = pipeline.run(str(old), str(new), cfg, match=match, progress=False,
                       vector_index=index, vector_project="demo", vector_revision="r2")
    assert rep.vector_candidates[0]["source_page"] == 2
    assert rep.vector_candidates[0]["matches"]
    assert rep.pages == []  # Suggestions do not approve a drawing pair.
    output = tmp_path / "report"
    report.build_report(rep, str(output))
    data = json.loads((output / "page_candidates.json").read_text())
    assert data[0]["source_page"] == 2
    text = (output / "summary.html").read_text()
    assert "Page match candidates" in text and "Review before comparing" in text
    assert (output / "candidates" / "source-2.png").exists()
    assert list((output / "candidates").glob("target-*.png"))


def test_confirmed_pair_replaces_conflicting_pair_only():
    from redlinecheck.vector_match import confirm_pairs
    match = MatchResult(pairs=[(0, 0), (1, 1), (2, 2)], scores={(0, 0): 1., (1, 1): .5, (2, 2): 1.})
    confirmed = confirm_pairs(match, [(1, 2)], source_count=3, target_count=3)
    assert confirmed.pairs == [(0, 0), (1, 2)]
    assert confirmed.extra_redlined == [2] and confirmed.extra_revised == [1]
    assert match.pairs == [(0, 0), (1, 1), (2, 2)]
    with pytest.raises(ValueError, match="range"):
        confirm_pairs(match, [(9, 0)], source_count=3, target_count=3)
    with pytest.raises(ValueError, match="unique"):
        confirm_pairs(match, [(0, 1), (2, 1)], source_count=3, target_count=3)


def test_cli_rejects_partial_vector_configuration(tmp_path, capsys):
    old, new = make_drawings(tmp_path)
    assert cli.main([str(old), str(new), "--vector-project", "demo"]) == 2
    assert "revision" in capsys.readouterr().err


def test_reused_report_removes_stale_candidate_images(tmp_path):
    from redlinecheck.vector_report import candidate_section
    from types import SimpleNamespace
    folder = tmp_path / "candidates"
    folder.mkdir()
    (folder / "old.png").write_bytes(b"old")
    assert candidate_section(SimpleNamespace(vector_candidates=[]), str(tmp_path)) == ""
    assert not folder.exists()


def test_confirmed_pair_is_excluded_before_fallback_extraction(index, tmp_path, monkeypatch):
    from redlinecheck import vector_match
    old, new = make_drawings(tmp_path)
    def no_extract(*args, **kwargs):
        raise AssertionError("confirmed pages must not require fallback extraction")
    original = vector_match.fallback_candidates
    monkeypatch.setattr(vector_match, "fallback_candidates",
                        lambda *a, **kw: original(*a, **kw, extract=no_extract))
    cfg = Config(analysis_dpi=72, auto_dpi=False)
    match = MatchResult(extra_redlined=[0], extra_revised=[0, 1, 2])
    rep = pipeline.run(str(old), str(new), cfg, match=match, progress=False,
                       vector_index=index, vector_project="demo", vector_revision="r2",
                       page_pairs=[(0, 1), (1, 2), (2, 0)])
    assert rep.vector_candidates == []
    assert [(p.page_index, p.final_index) for p in rep.pages] == [(0, 1), (1, 2), (2, 0)]


def test_changed_target_is_rejected_before_candidate_previews(index, tmp_path):
    old, new = make_drawings(tmp_path)
    rep = pipeline.run(str(old), str(new), Config(analysis_dpi=72, auto_dpi=False),
                       match=MatchResult(extra_redlined=[1], extra_revised=[0, 1, 2]),
                       progress=False, vector_index=index, vector_project="demo", vector_revision="r2")
    new.write_bytes(old.read_bytes())
    with pytest.raises(ValueError, match="changed"):
        report.build_report(rep, str(tmp_path / "report"))
