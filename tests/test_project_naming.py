"""Tests for filename normalisation used to pair drawing files.

``_name_key`` decides which redlined file belongs with which revised file when a
whole folder is dropped in. Getting it wrong pairs unrelated drawings, so the
sheet-number-preservation rule matters more than it looks.
"""

from redlinecheck.project import _name_key


def test_markup_decoration_is_stripped_so_a_pair_shares_one_key():
    redlined = _name_key("EXAMPLE-BLOCK A - DLE Markup#1.PDF")
    revised = _name_key("EXAMPLE-BLOCK A.PDF")
    assert redlined == revised == "exampleblocka"


def test_sheet_numbers_are_preserved():
    """'-8' and '-10' are sheet ids, not revision markers: they must not collide."""
    assert _name_key("PLAN-8.pdf") != _name_key("PLAN-10.pdf")


def test_revision_markers_are_dropped():
    assert _name_key("PLAN rev.2.pdf") == _name_key("PLAN.pdf")


def test_decoration_words_are_dropped():
    base = _name_key("PLAN.pdf")
    for decorated in ("PLAN revised.pdf", "PLAN final.pdf", "PLAN tender.pdf"):
        assert _name_key(decorated) == base


def test_key_is_case_insensitive():
    assert _name_key("Plan-A.pdf") == _name_key("PLAN-A.PDF")


def test_separators_do_not_affect_the_key():
    assert _name_key("PLAN_A.pdf") == _name_key("PLAN-A.pdf") == _name_key("PLAN A.pdf")


def test_directory_and_extension_are_ignored():
    assert _name_key(r"some/deep/folder/PLAN-A.pdf") == _name_key("PLAN-A.PDF")


def test_unrelated_drawings_get_different_keys():
    assert _name_key("FOUNDATION.pdf") != _name_key("ROOF FRAMING.pdf")


def test_markup_revision_number_does_not_change_the_key():
    assert _name_key("PLAN - Markup#1.pdf") == _name_key("PLAN - Markup#2.pdf")
