"""Tests for the page-alignment precheck (Needleman-Wunsch).

This is what stops an inserted page shifting every subsequent pairing and
corrupting the diff, so the interesting cases are insertions and deletions rather
than the happy path. Fixtures are hand-built similarity matrices and synthetic
``PageSig`` vectors -- no PDF is rendered.
"""

import numpy as np
import pytest

from redlinecheck import prematch
from redlinecheck.config import Config


def sig(index: int, vec, landscape: bool = True) -> prematch.PageSig:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        v = v / n
    return prematch.PageSig(index=index, vec=v, landscape=landscape)


# --------------------------------------------------------------------------
# _align
# --------------------------------------------------------------------------

def test_identical_sequences_pair_straight_through():
    sim = np.eye(3, dtype=np.float64)
    pairs, extra_r, extra_f = prematch._align(sim, -0.5)

    assert pairs == [(0, 0), (1, 1), (2, 2)]
    assert extra_r == [] and extra_f == []


def test_a_page_inserted_by_the_architect_is_reported_not_mispaired():
    # Redlined page 1 corresponds to revised page 2; revised page 1 is new.
    sim = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])
    pairs, extra_r, extra_f = prematch._align(sim, -0.5)

    assert pairs == [(0, 0), (1, 2)]
    assert extra_f == [1]
    assert extra_r == []


def test_a_page_dropped_from_the_revision_is_reported():
    sim = np.array([[1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 1.0]])
    pairs, extra_r, extra_f = prematch._align(sim, -0.5)

    assert pairs == [(0, 0), (2, 1)]
    assert extra_r == [1]
    assert extra_f == []


def test_alignment_is_monotonic():
    """Pairings may never cross: page order is preserved on both sides."""
    rng = np.random.default_rng(3)
    sim = rng.random((6, 6))
    pairs, _, _ = prematch._align(sim, -0.3)

    assert [r for r, _ in pairs] == sorted(r for r, _ in pairs)
    assert [f for _, f in pairs] == sorted(f for _, f in pairs)


def test_every_index_is_accounted_for_exactly_once():
    rng = np.random.default_rng(11)
    n, m = 5, 7
    sim = rng.random((n, m))
    pairs, extra_r, extra_f = prematch._align(sim, -0.4)

    assert sorted([r for r, _ in pairs] + extra_r) == list(range(n))
    assert sorted([f for _, f in pairs] + extra_f) == list(range(m))


def test_empty_redlined_side_makes_every_revised_page_extra():
    sim = np.zeros((0, 3))
    pairs, extra_r, extra_f = prematch._align(sim, -0.5)

    assert pairs == []
    assert extra_r == []
    assert extra_f == [0, 1, 2]


def test_a_large_gap_penalty_forces_pairing_over_skipping():
    sim = np.array([[0.1, 0.0], [0.0, 0.1]])
    pairs, _, _ = prematch._align(sim, -10.0)
    assert pairs == [(0, 0), (1, 1)]


# --------------------------------------------------------------------------
# _similarity_matrix
# --------------------------------------------------------------------------

def test_identical_fingerprints_score_one():
    a = sig(0, [1.0, 0.0, 0.0])
    b = sig(0, [1.0, 0.0, 0.0])
    sim = prematch._similarity_matrix([a], [b])
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_fingerprints_score_zero():
    a = sig(0, [1.0, 0.0])
    b = sig(0, [0.0, 1.0])
    sim = prematch._similarity_matrix([a], [b])
    assert sim[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_pages_of_different_orientation_can_never_match():
    portrait = sig(0, [1.0, 0.0], landscape=False)
    landscape = sig(0, [1.0, 0.0], landscape=True)
    sim = prematch._similarity_matrix([portrait], [landscape])
    assert sim[0, 0] == pytest.approx(prematch.NEG_INF, rel=1e-3)


def test_empty_input_yields_an_empty_matrix():
    assert prematch._similarity_matrix([], []).shape == (0, 0)
    assert prematch._similarity_matrix([sig(0, [1.0])], []).shape == (1, 0)


# --------------------------------------------------------------------------
# match_signatures
# --------------------------------------------------------------------------

def test_strong_matches_are_kept_and_scored():
    cfg = Config()
    a = [sig(0, [1.0, 0.0]), sig(1, [0.0, 1.0])]
    b = [sig(0, [1.0, 0.0]), sig(1, [0.0, 1.0])]

    res = prematch.match_signatures(a, b, cfg)

    assert res.pairs == [(0, 0), (1, 1)]
    assert res.extra_revised == [] and res.extra_redlined == []
    assert all(s == pytest.approx(1.0, abs=1e-5) for s in res.scores.values())


def test_weak_forced_matches_are_demoted_to_extras_on_both_sides():
    """The aligner always produces a full alignment; anything below the
    confidence floor must be reported as unmatched rather than compared."""
    cfg = Config()
    a = [sig(0, [1.0, 0.0])]
    b = [sig(0, [0.0, 1.0])]        # orthogonal -> similarity 0

    res = prematch.match_signatures(a, b, cfg)

    assert res.pairs == []
    assert res.extra_redlined == [0]
    assert res.extra_revised == [0]


def test_weak_matches_cannot_steal_a_better_valid_alignment():
    a = [sig(0, [1, 0, 0]), sig(1, [0, 1, 0])]
    b = [sig(0, [0.4, 0.34, np.sqrt(1 - 0.4**2 - 0.34**2)]),
         sig(1, [0.9, 0.34, np.sqrt(1 - 0.9**2 - 0.34**2)])]
    result = prematch.match_signatures(a, b, Config())
    assert result.pairs == [(0, 1)]
    assert result.extra_redlined == [1]
    assert result.extra_revised == [0]
