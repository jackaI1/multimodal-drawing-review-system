"""Tests for the OCR text-matching helpers.

These are the functions that decide whether an instructed value was actually read
back at the target, so they are the difference between a trustworthy green
"Addressed" verdict and a false one. All are pure string in / string out.
"""

from redlinecheck import ocr


# --------------------------------------------------------------------------
# _base / _norm -- normalisation
# --------------------------------------------------------------------------

def test_base_uppercases_and_strips_punctuation():
    assert ocr._base("wl-1") == "WL1"
    assert ocr._base("W310x52") == "W310X52"
    assert ocr._base('36"x48"') == "36X48"


def test_base_preserves_the_letter_o():
    """_base must not fold O->0: it is also used to classify prose tokens."""
    assert ocr._base("NOTE") == "NOTE"


def test_norm_folds_letter_o_to_zero_for_fuzzy_comparison():
    assert ocr._norm("WL-O1") == "WL01"
    assert ocr._norm("O") == "0"


def test_norm_and_base_agree_when_no_o_is_present():
    for s in ("WL1", "W310X52", "PF9"):
        assert ocr._norm(s) == ocr._base(s)


# --------------------------------------------------------------------------
# _fold -- confusable characters
# --------------------------------------------------------------------------

def test_fold_maps_each_documented_confusable_pair():
    assert ocr._fold("B") == "8"
    assert ocr._fold("S") == "5"
    assert ocr._fold("I") == "1"
    assert ocr._fold("L") == "1"
    assert ocr._fold("Z") == "2"
    assert ocr._fold("G") == "6"
    assert ocr._fold("D") == "0"
    assert ocr._fold("Q") == "0"


def test_fold_is_idempotent():
    once = ocr._fold("15B")
    assert ocr._fold(once) == once


def test_fold_leaves_unambiguous_characters_alone():
    assert ocr._fold("W310X52") == "W310X52"


# --------------------------------------------------------------------------
# _fuzzy_contains
# --------------------------------------------------------------------------

def test_exact_substring_matches():
    assert ocr._fuzzy_contains("W310X52", "BEAM W310X52 TYP", 0)


def test_confusable_misread_still_matches():
    """Tesseract reading '15B' as '158' must not cost us the match."""
    assert ocr._fuzzy_contains("15B", "TAG 158 HERE", 0)


def test_confusable_matching_is_symmetric():
    assert ocr._fuzzy_contains("158", "TAG 15B HERE", 0)


def test_gap_tolerance_bridges_a_stray_character():
    assert ocr._fuzzy_contains("ABC", "AXBXC", 1)


def test_gap_tolerance_is_bounded():
    assert not ocr._fuzzy_contains("ABC", "AXXBXXC", 1)


def test_zero_gap_rejects_interrupted_tokens():
    assert not ocr._fuzzy_contains("ABC", "AXBXC", 0)


def test_empty_token_never_matches():
    """A blank instruction must not silently match everything."""
    assert not ocr._fuzzy_contains("", "ANYTHING AT ALL", 5)


def test_absent_token_is_not_found():
    assert not ocr._fuzzy_contains("W310X52", "COMPLETELY UNRELATED", 2)


# --------------------------------------------------------------------------
# _instruction_tokens
# --------------------------------------------------------------------------

def test_keeps_alphanumeric_part_labels():
    assert ocr._instruction_tokens("CHANGE TO W310X52") == ["W310X52"]


def test_drops_prose_words():
    """Prose must never become a matchable token -- a wrong 'addressed' is worse
    than a missed one."""
    assert ocr._instruction_tokens("PLEASE REVIEW AND CONFIRM") == []


def test_keeps_known_all_letter_material_codes():
    assert "LVL" in ocr._instruction_tokens("USE LVL HEADER")


def test_drops_short_bare_numbers():
    assert ocr._instruction_tokens("SEE NOTE 5") == []


def test_keeps_long_bare_numeric_codes():
    assert "1234" in ocr._instruction_tokens("REFER 1234")


def test_splits_on_slashes_rather_than_fusing_labels():
    got = ocr._instruction_tokens("PF9/150 PF8")
    assert "PF9" in got and "PF8" in got


def test_tokens_are_sorted_and_deduplicated():
    got = ocr._instruction_tokens("WL1 AND WL1 AND F5")
    assert got == sorted(set(got))


def test_hyphenated_and_bare_labels_normalise_together():
    """'WL-1' and 'WL1' must produce the same token so either spelling matches."""
    assert ocr._instruction_tokens("ADD WL-1") == ocr._instruction_tokens("ADD WL1")
