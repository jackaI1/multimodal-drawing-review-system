import fitz
import numpy as np

from redlinecheck.config import Config
from redlinecheck.prematch import MatchResult
from redlinecheck.vector_match import uncertain_pages, fallback_candidates


def test_only_unmatched_weak_or_ambiguous_pairs_need_candidates():
    match = MatchResult(pairs=[(0, 0), (1, 1), (2, 2)], extra_redlined=[3],
                        scores={(0, 0): .98, (1, 1): .5, (2, 2): .9})
    sim = np.array([[.98, .2, .3], [.1, .5, .2], [.1, .89, .9], [.1, .2, .1]])
    assert uncertain_pages(match, sim) == [1, 2, 3]


def test_fallback_is_a_scoped_suggestion_and_keeps_alignment(tmp_path):
    class Index:
        encoder = object()
        def search_page(self, record, **filters):
            assert filters["project"] == "demo" and filters["revision"] == "r2"
            assert filters["document"] == "target" and filters["document_hash"] == "hash"
            return [{"page": 2, "sheet": "S2", "score": .03,
                     "text_score": .95, "image_score": .8}]

    path = tmp_path / "source.pdf"
    with fitz.open() as doc:
        p = doc.new_page(width=300, height=200)
        p.insert_text((30, 70), "S2 BEAM PLAN")
        doc.save(path)
    match = MatchResult(extra_redlined=[0], extra_revised=[0, 1])
    with fitz.open(path) as doc:
        hits = fallback_candidates(doc, match, np.zeros((1, 2)), Index(),
                                   project="demo", revision="r2", document="target",
                                   document_hash="hash", extract=lambda *a, **k: object())
    assert hits[0]["source_page"] == 1
    assert hits[0]["matches"][0]["page"] == 2
    assert match.pairs == [] and match.extra_revised == [0, 1]


def test_no_candidates_for_confident_alignment():
    class NoCalls:
        def search_page(self, *a, **kw):
            raise AssertionError("strong match should not query database")
    match = MatchResult(pairs=[(0, 0)], scores={(0, 0): 1.})
    assert fallback_candidates([], match, np.ones((1, 1)), NoCalls(),
                               project="p", revision="r", document="d", document_hash="h") == []
