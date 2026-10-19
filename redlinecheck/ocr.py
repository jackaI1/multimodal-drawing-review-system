"""Optional local OCR verification tier (Tesseract). OFF by default.

Fully local — nothing leaves your machine. Reads the original and revised crops
at each spec comment's target and compares the recognised text against the
instruction, turning a pixel "no activity" hint into a CAPTURED / MISSED verdict
(including the in-place beam-size swaps the pixel diff can't see).

Requires `pip install pytesseract` and the Tesseract binary. The binary is
auto-detected on PATH and in the standard Windows install location, so a winget
/ UB-Mannheim install just works without editing PATH.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from .config import Config
from .align import PageData
from .extract import Comment
from . import classify as C

_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def _locate_binary() -> Optional[str]:
    onpath = shutil.which("tesseract")
    if onpath:
        return onpath
    for p in _WINDOWS_PATHS:
        if os.path.isfile(p):
            return p
    return None


def check_availability() -> Tuple[bool, str]:
    """(ok, message) — for the CLI/app to report OCR readiness."""
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract not installed (pip install pytesseract)"
    binary = _locate_binary()
    if not binary:
        return (False, "Tesseract binary not found. Install it "
                       "(Windows: winget install UB-Mannheim.TesseractOCR).")
    pytesseract.pytesseract.tesseract_cmd = binary
    try:
        ver = pytesseract.get_tesseract_version()
    except Exception as e:
        return False, f"Tesseract found at {binary} but failed to run: {e}"
    return True, f"Tesseract {ver} ({binary})"


def _base(s: str) -> str:
    """Case-folded, punctuation-stripped — but letters preserved (no O->0).
    Used to decide whether a token *looks* like a part label."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def _norm(s: str) -> str:
    """Like _base, but also folds O->0 for FUZZY MATCHING — CAD fonts and OCR
    routinely confuse the letter O with zero. (Only used to compare strings,
    never to classify prose: O->0 would turn 'NOTE' into a digit-bearing token.)"""
    return _base(s).replace("O", "0")


# Pure-letter structural/material labels worth matching on (they carry no digit).
# Anything else without a digit is treated as prose and ignored.
_ALPHA_LABELS = {
    "BBFM", "LVL", "LSL", "PSL", "GLB", "GLULAM", "PWF", "WWF", "DBL",
    "HSS", "SOG", "GLT",
}

# The architect pre-prints "BM. BY ENG." (beam by engineer) where the engineer
# must specify the member. If that is present in the original at the target and
# gone in the revision, the spec was actioned — even when the thin replacement
# label won't OCR. Use the short core "BYEN" because OCR truncates the vertical
# text ("BY ENG" -> "BY EN"). NOTE: do NOT add "BY F.T.M"/"BM BY" — those are
# permanent joist/truss notes that persist in both drawings (false positives).
_PLACEHOLDERS = ("BYEN",)


def _instruction_tokens(text: str):
    """Distinctive part/label tokens to look for in the revision.

    Keeps structural labels case- and punctuation-insensitively:
      - anything with a digit (F5, WL1, WL-1, W310X52, 200X6, 36"x48"x16")
      - known all-letter material codes (BBFM, LVL, ...)
    Drops prose words and bare short numbers so we never claim a match on
    something generic — a wrong "addressed" is worse than a missed one.
    """
    up = text.upper()
    # The old label in an explicit replacement is context, never evidence that
    # the requested new label was applied.
    replacement = re.search(r"\b(?:CHANGE|REVISE)\s+FROM\b.+?\bTO\s+(.+)", up)
    replacement = replacement or re.search(r"\bREPLACE\b.+?\bWITH\s+(.+)", up)
    if replacement:
        up = replacement.group(1)
    toks = set()
    # Split on slash/space/comma so "PF9/150 PF8" -> PF9, 150, PF8 (don't fuse).
    for m in re.findall(r'[A-Z0-9][A-Z0-9\-.+X"\']*', up):
        b = _base(m)
        if len(b) < 2:
            continue
        has_letter = any(c.isalpha() for c in b)
        has_digit = any(c.isdigit() for c in b)
        keep = ((has_letter and has_digit)                      # WL1, W310X52, 6X6
                or (has_digit and not has_letter and len(b) >= 4)  # long numeric codes
                or (not has_digit and b in _ALPHA_LABELS))       # BBFM, LVL
        if keep:
            toks.add(_norm(m))   # store O->0 folded for fuzzy matching
    return sorted(toks)


# Characters Tesseract routinely confuses in thin CAD lettering. Folded to one
# representative on BOTH the instruction and the OCR text, so a clear "15B" still
# matches an OCR read of "158" / "1SB", etc. (per-channel, symmetric).
_CONFUSE = str.maketrans({
    "B": "8",                       # B <-> 8
    "S": "5",                       # S <-> 5
    "I": "1", "L": "1",             # I, l, L <-> 1
    "Z": "2",                       # Z <-> 2
    "G": "6",                       # G <-> 6
    "D": "0", "Q": "0",             # D, Q, O <-> 0 (O->0 already in _norm)
})


def _fold(s: str) -> str:
    return s.translate(_CONFUSE)


def _fuzzy_contains(token: str, text: str, gap: int) -> bool:
    """True if `token` appears in `text` allowing (a) confusable-character
    near-misses (B/8, S/5, I/L/1, ...) and (b) up to `gap` stray chars between
    consecutive token chars. Tesseract mangles thin CAD fonts badly ('L3+L7' ->
    'L3eL7', '15B' -> '158'), which exact substring matching misses."""
    if not token:
        return False
    token, text = _fold(token), _fold(text)
    if token in text:                       # fast path
        return True
    if gap <= 0:
        return False
    pat = ("." + "{0,%d}" % gap).join(re.escape(c) for c in token)
    return re.search(pat, text) is not None


def _words(raw: str):
    """Split raw OCR text into normalised word tokens (preserves word boundaries
    so short labels like 'B3' don't match a substring of 'SB3')."""
    return [w for w in (_norm(x) for x in re.split(r"[^A-Za-z0-9]+", raw)) if w]


def _match(tok: str, words, blob: str, gap: int) -> bool:
    """Does `tok` appear in an OCR read?
    - long, distinctive tokens (>=4 chars, e.g. W310X52): substring anywhere
      (tolerates merges/dimensions), with confusable + 1-char fuzz.
    - short labels (B3, L3, SB4, 15B): must equal a WHOLE OCR word (confusable
      fold only) — otherwise 'B3' would wrongly match 'SB3'/'SB1' everywhere."""
    ft = _fold(tok)
    if len(ft) >= 4:
        return _fuzzy_contains(tok, blob, gap)
    return ft in {_fold(w) for w in words}


def _hexnote_numbers(text: str):
    """The label after 'HEXNOTE' / 'ADD HEX NOTE' (e.g. 17, 42, 15B). The word
    'HEXNOTE' never appears in the drawing — the architect draws the label inside
    a hexagon — so we only look for that number/letter appearing in the revision
    (and confirm it wasn't there before). Captures a trailing letter (15B)."""
    return [m.group(1) for m in
            re.finditer(r'HEX\s*-?\s*NOTE\s*#?\s*(\d{1,3}[A-Z]?)', text.upper())]


def _different_value(requested, before_words, after_words):
    """Read a newly present alternative in the same label family.

    Only a single, unambiguous replacement earns this more specific review
    verdict. Unrelated new prose or labels are not evidence of a wrong edit.
    """
    if len(requested) != 1:
        return None
    wanted = requested[0]
    family = re.sub(r"\d+", "#", wanted)
    if not re.match(r"^[A-Z]+\d", wanted):
        return None
    candidates = {word for word in after_words
                  if re.sub(r"\d+", "#", word) == family
                  and _fold(word) != _fold(wanted)
                  and _fold(word) not in {_fold(w) for w in before_words}}
    return next(iter(candidates)) if len(candidates) == 1 else None


def _upscale(cfg: Config) -> float:
    """Adaptive enlargement: target ~ocr_target_dpi effective resolution so a
    high-DPI render isn't over-upscaled (which blurs thin strokes and hurts OCR)."""
    up = cfg.ocr_target_dpi / max(1, cfg.analysis_dpi)
    return max(1.0, min(cfg.ocr_upscale_max, up))


def _prep(crop_rgb: np.ndarray, cfg: Config, binarize: bool = True) -> Image.Image:
    img = Image.fromarray(crop_rgb).convert("L")
    img = ImageOps.autocontrast(img)
    up = _upscale(cfg)
    if up > 1.01:
        img = img.resize((max(1, int(img.width * up)), max(1, int(img.height * up))),
                         Image.LANCZOS)
    if binarize:
        thr = cfg.ocr_binarize_threshold
        return img.point(lambda p: 255 if p >= thr else 0)
    return img


def make_verifier(cfg: Config):
    ok, msg = check_availability()
    if not ok:
        raise RuntimeError(msg)
    import pytesseract

    pad = cfg.px(cfg.crop_pad_pt) * cfg.ocr_search_mult
    rotations = tuple(cfg.ocr_rotations)
    gap = cfg.ocr_fuzzy_gap
    page_cache: dict = {}    # id(page) -> ((fwords,fblob),(bwords,bblob)) whole-sheet OCR

    def _read_full(rgb, max_w: int = 3200) -> str:
        """OCR a whole sheet — used to find a HEXNOTE hexagon placed anywhere on
        the page, not at the leader tip."""
        img = Image.fromarray(rgb).convert("L")
        if img.width > max_w:
            img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
        img = ImageOps.autocontrast(img).point(
            lambda p: 255 if p >= cfg.ocr_binarize_threshold else 0)
        return pytesseract.image_to_string(img, config="--psm 11")

    def _page_text(page):
        key = id(page)
        if key not in page_cache:
            fr, br = _read_full(page.final_rgb), _read_full(page.base_rgb)
            page_cache[key] = ((_words(fr), _norm(fr)), (_words(br), _norm(br)))
        return page_cache[key]

    def _read(rgb, box, angles=(0,), full=True, gray=False) -> str:
        """OCR the padded crop at each angle, returning the raw recognised text.
        `full` adds the psm-6 (block) pass; `gray` adds a grayscale (non-binarised)
        variant — a cheap 'second model'. Both are used only for the primary
        horizontal read to keep the rotated/before passes fast."""
        h, w = rgb.shape[:2]
        x0 = max(0, int(box[0] - pad)); y0 = max(0, int(box[1] - pad))
        x1 = min(w, int(box[2] + pad)); y1 = min(h, int(box[3] + pad))
        if x1 <= x0 or y1 <= y0:
            return ""
        crop = rgb[y0:y1, x0:x1]
        variants = [_prep(crop, cfg, binarize=True)]
        if gray:
            variants.append(_prep(crop, cfg, binarize=False))
        out = []
        for a in angles:
            for vi, base_img in enumerate(variants):
                im = base_img if a == 0 else base_img.rotate(a, expand=True)
                out.append(pytesseract.image_to_string(im, config="--psm 11"))
                if full and vi == 0:
                    out.append(pytesseract.image_to_string(im, config="--psm 6"))
        return " ".join(out)

    def verify(page: PageData, comment: Comment, box, changed: bool = False) -> Optional[tuple]:
        # `changed` = the pixel diff saw activity at this target. It guards the green
        # verdict: OCR alone coincidentally matches the dense, REPEATED short beam tags
        # (L1/L3/L7/B1…) on these drawings, so requiring a real pixel change too keeps
        # 'Addressed (verified)' trustworthy (the dangerous failure is a false green).
        labels = _instruction_tokens(comment.text)
        hexnums = _hexnote_numbers(comment.text)
        search = labels + [n for n in hexnums if n not in labels]
        if not search:
            return None  # nothing distinctive to look for (e.g. "CANT.")

        # 1) primary horizontal read of the revision (binarised + grayscale).
        araw = _read(page.final_rgb, box, (0,), full=True, gray=True)
        awords, ablob = _words(araw), _norm(araw)
        hit = next((t for t in search if _match(t, awords, ablob, gap)), None)
        # 2) only if that misses, retry rotated — the architect plots many beam
        #    labels vertically, which a horizontal pass can't read.
        if not hit and rotations:
            rraw = _read(page.final_rgb, box, rotations, full=False)
            awords += _words(rraw); ablob += _norm(rraw)
            hit = next((t for t in search if _match(t, awords, ablob, gap)), None)

        # Read the original at the target (all angles) — to tell a newly-appeared
        # value from one already there, and to detect a resolved placeholder.
        braw = _read(page.base_rgb, box, (0,) + rotations, full=False)
        bwords, bblob = _words(braw), _norm(braw)

        if hit:
            missing = [t for t in search if not _match(t, awords, ablob, gap)]
            if missing:
                return (C.CHANGED_AT_TARGET, "",
                        "only part of the instruction was read back; missing "
                        + ", ".join(missing) + " — review")
            disp = f"HEXNOTE {hit}" if hit in hexnums else hit
            new_values = [t for t in search if not _match(t, bwords, bblob, gap)]
            appeared = bool(new_values)
            long_hit = all(len(_fold(t)) >= 4 for t in new_values)
            # A SHORT token (L9, B1) only earns green when the instruction is a BARE
            # label (e.g. 'L12'), not a phrase. A phrase like 'CONT. AND CANT. L9'
            # matched on its lone short token is unreliable — OCR coincidentally finds an
            # 'L9' amid a real-but-different change, a false green.
            bare_label = len(_base(comment.text)) <= 8
            if appeared and (long_hit or (changed and bare_label)):
                # The instructed value is newly present AND either it's a long/distinctive
                # code (hard to match by chance) or a real pixel change was seen at a
                # bare-label target. These guards keep 'Addressed (verified)' trustworthy.
                why = "distinctive code" if long_hit else "a change was seen at the target"
                return (C.CAPTURED, ", ".join(search),
                        f"value '{disp}' now present in the revision (was not there "
                        f"before) and {why} — OCR")
            if appeared:
                # Value newly READ but with no pixel change and a short label: this is an
                # in-place swap (e.g. SB3->SB5, which has no density change) OR an OCR
                # mis-read of a similar tag. Don't claim green — flag it to verify.
                return (C.CHANGED_AT_TARGET, "",
                        f"OCR read '{disp}' at the target but saw no pixel change — "
                        "verify it's the instructed edit")
            # The value is present in the BEFORE crop too -> it coincidentally matches a
            # label already there (a nearby beam tag), not a confirmed change. Defer to
            # the pixel verdict rather than over-claim a capture.
            return None

        other = _different_value(search, bwords, awords)
        if other:
            return (C.CHANGED_OTHER, other,
                    f"read a new value '{other}', not the instructed value — review")

        # The architect's drawing pre-prints a placeholder ("BM. BY ENG.") where
        # the engineer must specify the member. The thin final label often won't
        # OCR, but that big placeholder reads reliably — so if it was there before
        # and is gone after, the spec was actioned.
        if any(p in bblob for p in _PLACEHOLDERS) and \
           not any(p in ablob for p in _PLACEHOLDERS):
            return (C.CHANGED_AT_TARGET, "",
                    "the placeholder disappeared, but the instructed value was not "
                    "read back — review")

        # HEXNOTE fallback: the architect draws the numbered hexagon anywhere on
        # the sheet, not necessarily at the leader tip. Search the whole revised
        # page for the number; require it to be new vs the original.
        if hexnums:
            (fw, fb), (bw_, bb) = _page_text(page)
            hn = next((n for n in hexnums
                       if _match(n, fw, fb, gap) and not _match(n, bw_, bb, gap)), None)
            if hn:
                return (C.CHANGED_AT_TARGET, f"HEXNOTE {hn}",
                        f"HEXNOTE {hn} appeared elsewhere on this sheet; not read at "
                        "the target — review")

        if bblob or ablob:  # we read text here, but not the instructed value
            return (C.MISSED, "",
                    "instructed value not found in revision at the target — OCR")
        return None  # OCR read nothing — fall back to the pixel verdict

    return verify
