"""Tunable configuration for the comparison pipeline.

All pixel thresholds are expressed at the analysis DPI so they scale together
if you change `analysis_dpi`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# A relative rectangle (x0, y0, x1, y1) in [0,1] page fractions.
RelRect = Tuple[float, float, float, float]


@dataclass
class Config:
    # ---- rendering -------------------------------------------------------
    analysis_dpi: int = 300            # DPI for the visual comparison (used when auto_dpi is off)
    report_dpi: int = 150              # DPI for the rendered report pages
    # Auto-DPI: pick the render DPI from the drawing's text size so labels land at
    # a readable pixel height (smaller CAD text -> higher DPI). On by default.
    auto_dpi: bool = True
    auto_dpi_target_px: float = 30.0   # aim to render body text ~this many px tall
    auto_dpi_min: int = 200
    auto_dpi_max: int = 500
    auto_dpi_default: int = 300         # fallback when the drawing has no extractable text

    # ---- ink / markup detection -----------------------------------------
    # A pixel is "structural ink" if its luminance is below this AND it is not
    # a markup colour. Kept low enough to exclude the light "PRELIMINARY"
    # watermark (which is mid-grey).
    ink_lum_threshold: int = 140
    # Markup colour test (applied to rendered pixels): a real redline is a
    # SATURATED red/magenta stroke. The architect's drawings, however, are full
    # of *desaturated pastel fills* (peach/salmon area-coverage shading ~
    # (255,201,174), pink ~ (249,175,222)) that an R-G-only test wrongly flags as
    # "redline ink" — which then fabricated phantom "(redline markup)" findings on
    # un-annotated sheets. So we additionally require LOW green and HIGH saturation,
    # which a true red (220,30,30)/magenta (230,30,230) easily clears (sat ~0.86)
    # but a pastel fill (sat ~0.30) cannot.
    markup_red_min: int = 110          # R channel at least this
    markup_red_minus_green: int = 45   # R - G at least this (red or magenta)
    markup_green_max: int = 110        # G at most this. Excludes the architect's
                                       # ORANGE linework/fills (G ~ 135-210); a true
                                       # red/magenta redline has G ~ 20-80.
    markup_min_sat: float = 0.45       # HSV-style saturation (r - min(g,b)) / r

    # ---- flattened (baked-in) redline fallback --------------------------
    # Some markups are FLATTENED into the page content (vector red linework), not
    # live Acrobat annotations, so page.annots() returns nothing. When a page has
    # NO annotation-derived comments, fall back to detecting clusters of red markup
    # INK and treating each as a markup region to inspect. On by default; it only
    # fires when there are no annotations, so annotated PDFs are unaffected.
    flatten_markups: bool = True
    flatten_dilate_pt: float = 6.0     # merge red strokes within this gap into one callout
    flatten_min_side_pt: float = 2.5   # ignore red clusters smaller than this (specks)
    flatten_max_clusters: int = 200    # safety cap per page
    # STAY-SILENT guard: real jobs almost always have live annotations, so the
    # flattened fallback should only fire on genuine red LINEWORK, never a solid
    # colour fill that slipped past the markup test. Reject a cluster whose red ink
    # fills more than this fraction of its bbox AND is larger than a small callout —
    # that's a solid fill, not a stroke. (A drawn red box/cloud is hollow, so its
    # fill ratio is low and it survives.)
    flatten_max_fill_ratio: float = 0.55
    flatten_solid_min_side_pt: float = 14.0  # only apply the fill-ratio gate above this size
    # The engineer's red mark exists ONLY in the redlined PDF; the architect's
    # revision is black linework and never carries your red. So any "red" cluster
    # whose ink also appears red at the SAME spot in the revision is part of the
    # drawing (a red detail/logo, or an anti-aliased fill boundary), NOT a redline —
    # drop it. This cleanly removes residual colour-fill artefacts the colour test
    # alone can't, while keeping true baked-in redlines (which vanish in the revision).
    flatten_persist_max_frac: float = 0.35   # drop if > this fraction persists in revision
    flatten_persist_dilate_pt: float = 2.0   # slack for re-plot jitter when matching

    # ---- alignment -------------------------------------------------------
    max_register_shift_frac: float = 0.04   # ignore registration shifts larger than this
    # Phase correlation sometimes locks onto the wrong peak on sparse, re-plotted
    # line drawings (it returns ~0 while the sheet is actually offset 30-120px),
    # which then floods the report with false "unexpected" and offsets the crop
    # boxes. When the phase-corr alignment scores below this ink IoU, fall back to a
    # robust coarse-to-fine brute-force overlap search and keep whichever shift
    # (including no shift) aligns best.
    register_min_iou: float = 0.6

    # ---- page precheck / matching (architect may insert/split pages) -----
    # Cheap low-DPI fingerprint per page, then a monotonic sequence alignment
    # matches corresponding pages and flags inserted/unmatched ones.
    prematch_dpi: int = 36             # tiny render just for fingerprinting (fast)
    prematch_grid: int = 32            # fingerprint resized to grid x grid
    prematch_gap: float = -0.15        # alignment gap penalty (insert/delete a page)
    prematch_min_match: float = 0.35   # below this similarity, a pair isn't a real match
    extra_thumb_dpi: int = 80          # DPI for "Extra Pages" thumbnails

    # ---- diff (density-based, robust to full re-plot jitter) -------------
    # Compare ink *density* over small cells. A line shifting 1-2px barely
    # changes a cell's density; genuinely added/removed content changes it a lot.
    # Cell size in POINTS (physical), converted to px at the analysis DPI, so the
    # diff grain is DPI-invariant: a higher DPI no longer catches finer re-plot /
    # line-weight jitter and explode false positives. 4.32pt == 12px at 200 DPI
    # (the original calibration) == 18px at 300 DPI.
    density_cell_pt: float = 4.32
    density_smooth_sigma: float = 0.0  # gaussian blur (cells); 0 = off (smoothing hurt here)
    suppress_shift_gap_pt: float = 22.0  # add+remove pairs closer than this & similar size = re-plot shift
    suppress_shift_ratio: float = 0.4    # min(magA,magR)/max(...) above this => treat as shift pair
    # MOVED-CONTENT suppression: the architect often nudges a dimension/label a bit
    # left/right WITHOUT changing its value -> a "removed" blob at the old spot and an
    # "added" blob at the new spot. If the OLD ink (base @ removed-box) matches the NEW
    # ink (final @ added-box) once aligned, it's the same content relocated, not a real
    # change -> drop both. Strict (content must actually match) so real edits survive.
    suppress_moved_content: bool = True
    # Run the per-region "did this ink just move/re-plot nearby?" suppression on EVERY
    # page (not only badly-aligned ones)? Tempting for dense-hatch elevations, but it
    # is DANGEROUS on drawings with REPEATED content: a real edit (e.g. relabel one of
    # seven identical 'DEN' rooms to 'GUEST BDRM', or add a 3'-6" dimension in a railing
    # bay) matches an unchanged copy a short distance away and gets wrongly suppressed.
    # On the example11 set this erased ~32 genuine architect changes to remove ~21 hatch
    # false positives — a bad ~1:1 trade against recall. So it's OFF: local-shift runs
    # only where global registration genuinely failed (the architect re-plotted the
    # whole plan at a shifted spot), where there's no nearby duplicate to false-match.
    suppress_local_shifts_always: bool = False
    # Residual-mismatch threshold for calling a region "just moved/re-plotted":
    #  - lenient on genuinely mis-registered sheets (a lot legitimately shifted),
    #  - STRICT on well-aligned sheets (the always-on pass), so only near-identical
    #    re-plot churn is dropped and a region carrying real NEW ink (a dimension/label
    #    the architect added among the hatch) keeps enough unmatched ink to survive.
    # Without the strict bar the always-on pass also erased real changes on dense
    # elevation sheets (on a dense sample set, unexpected findings fell to 0).
    suppress_move_xor_frac_strict: float = 0.16
    suppress_move_max_pt: float = 54.0   # max distance the content can have moved
    suppress_move_xor_frac: float = 0.34 # residual mismatch after best align <= this => same content
    suppress_move_jitter_pt: float = 2.0 # extra alignment slack for re-plot jitter
    suppress_move_min_pt: float = 3.0    # ignore blobs with less ink than this (side, points)
    suppress_move_tol_pt: float = 0.9    # dilate the other image by this so re-plotted
                                         # (1-2px jittery) lines still count as "found"
    max_unexpected_per_page: int = 10    # above this, summarize the page instead of listing all
    # When the architect RE-PLOTS a sheet so heavily that it won't register (the two
    # drawings barely overlap even after the brute-force alignment search), the pixel
    # diff is dominated by alignment noise: dimension strings and labels that merely
    # moved get flagged as "unexpected" changes (verified on the SD/example10 set,
    # where such sheets register at 14-20% ink overlap and ~half their "unexpected"
    # flags were re-plot artifacts). Below this post-registration ink overlap we DON'T
    # trust per-region unexpected detection — we suppress those flags and instead
    # surface ONE honest page-level "heavily re-plotted; review manually" notice. The
    # redline TARGET findings (misses/captures) are kept: they're anchored to the spot
    # you marked and, when OCR/VLM-confirmed, robust to misalignment.
    min_align_sim_unexpected: float = 0.30
    # Drop "unexpected" changes that recur on most sheets — titleblock / stamp /
    # revision-table boilerplate (e.g. the architect's licence stamp appearing when
    # a PRELIMINARY set is signed), not design changes. Matching by CONTENT rather
    # than by position is deliberate: boilerplate re-plots at a slightly shifting
    # centre, so a location-keyed test lets it through on the sheets where it moves
    # and floods "Unexpected". Instead each unexpected region's revised-image crop is
    # fingerprinted, and any cluster of look-alike regions spanning
    # >= boilerplate_min_pages distinct sheets is dropped, independent of position.
    # boilerplate_min_frac feeds the location path only and is no longer the gate.
    boilerplate_min_pages: int = 3
    boilerplate_min_frac: float = 0.6    # (legacy location-key path only)
    boilerplate_fp_grid: int = 16        # region crop fingerprint resized to NxN
    boilerplate_fp_sim: float = 0.82     # cosine similarity to count as the same content
    boilerplate_size_tol: float = 0.40   # bbox sides must match within +/- this fraction
    density_add_thresh: float = 0.16   # cell counts as "ink added" if density rises this much
    density_remove_thresh: float = 0.16
    # Comment-target activity (CHANGED_AT_TARGET vs NO_CHANGE) uses a FINER grid
    # than unexpected-region detection, so a small in-place edit at a redline's
    # target still registers as "activity" (decoupled from the coarse, jitter-
    # robust region grid so neither over- nor under-fires).
    target_cell_pt: float = 2.9
    min_cluster_cells: int = 6         # ignore change clusters smaller than this many cells
    min_change_magnitude: float = 3.0  # ... and below this summed |density delta|
    cluster_merge_cells: int = 1       # dilate (in cells) to merge adjacent change cells
    # A redline TARGET counts as having "activity" (-> CHANGED_AT_TARGET / "to verify",
    # not a possible miss) if this fraction of its cells changed, OR at least
    # target_changed_cell_min cells did. The absolute-count floor matters because a
    # small label the architect ADDED (e.g. 'SBFA') is only a few cells inside a large
    # marked-region box, so its fraction is tiny — but it's a real change and should be
    # "to verify", per the reported errors. (Pure in-place swaps like SB3->SB5 still
    # have ~0 density change and need OCR/VLM — no pixel threshold can see them.)
    target_changed_cell_frac: float = 0.05
    target_changed_cell_min: int = 8        # ... or this many changed cells, absolute
    # A full re-plot makes texture (brick/shingle hatching) churn with BALANCED
    # add/remove. Genuine edits are directional (mostly added OR removed). Drop
    # low-directionality regions and page-spanning blobs as re-plot noise.
    min_directionality: float = 0.55   # |add-rem| / (add+rem)
    max_region_area_frac: float = 0.35 # bbox area as fraction of page

    # ---- symbol-callout change detection --------------------------------
    # Catches an in-place edit *inside* a closed callout (hexnote/circle), e.g. a
    # hexnote "8c" -> "8" — too small for the density diff and unreadable by OCR
    # (digits inside a hexagon). Detects each enclosed symbol, micro-aligns its
    # interior (kills re-plot jitter), and flags a mid-range change. OFF by default
    # (opt-in checkbox / --hexnotes): it adds noise on heavily re-plotted sheets and
    # isn't worth it unless you're specifically chasing callout edits. When on, it's
    # restricted to polygon/circle callouts via a solidity window (a hexagon/circle
    # fills ~0.75-0.78 of its bbox; a dimension RECTANGLE ~0.95+, excluded).
    detect_symbols: bool = False
    symbol_min_pt: float = 3.0         # enclosed-interior side, min (points)
    symbol_max_pt: float = 24.0        # ... and max
    symbol_min_change_pt: float = 2.0  # min changed ink (side, points) to count
    symbol_min_frac: float = 0.34      # changed / total interior ink, min
    symbol_max_frac: float = 1.15      # ... and max (above this = content appeared/
                                       #   disappeared, i.e. text add/remove, not a swap)
    symbol_solidity_min: float = 0.62  # filled-interior area / bbox: polygon/circle band
    symbol_solidity_max: float = 0.88  # (rectangles ~0.95+ excluded in standard)
    symbol_align_px: int = 2           # micro-alignment radius to cancel re-plot jitter

    # ---- comment geometry (in PDF points, scaled to analysis px) ---------
    target_radius_pt: float = 30.0     # search radius around a leader tip / comment
    leader_attach_pt: float = 18.0     # max gap to attach a leader line to a FreeText
    leader_extend_pt: float = 16.0     # push the target past the arrowhead onto the referenced label
    target_min_pt: float = 48.0        # minimum target-box side so a label is fully contained
    target_change_margin_pt: float = 10.0  # slack when matching clusters to a target
    # A change region this close to ANY of the engineer's markups (a comment's text
    # box, its leader, or its target) is treated as something they DID flag — so it is
    # not reported as an "unexpected" (un-marked) change. Bigger than the target-match
    # margin because a leader tip often lands a little off the callout the architect
    # then edits/removes (seen on the example11 set: a 'W200X27' redline pointing at a
    # steel callout the architect deleted was double-flagged as unexpected).
    unexpected_clear_margin_pt: float = 26.0

    # ---- ignore zones (revision tables, titleblock, etc.) ---------------
    # Default: the bottom-right titleblock block of these sheets.
    ignore_zones: List[RelRect] = field(
        default_factory=lambda: [(0.60, 0.60, 1.00, 1.00)]
    )
    ignore_watermark: bool = True

    # ---- comment classification -----------------------------------------
    # Annotation stroke/fill colours that mark a "TO ARCH" coordination note.
    # Detected structurally too (text starting with "TO ARCH").
    # ---- local VLM verification tier (Ollama) ---------------------------
    # A local on-device vision model reads the BEFORE crop (your instruction) and
    # the AFTER crop (the architect's revision) for items the pixel layer flagged
    # (possible misses + change-at-target) and judges whether the instruction was
    # applied. Only an explicit "applied" with a read-back value turns a finding
    # green; "uncertain" stays in manual review (never over-claims). Fully local —
    # talks to an Ollama server on localhost. Off by default; opt-in like OCR.
    use_vlm: bool = False
    vlm_endpoint: str = "http://localhost:11434"   # Ollama default
    # Default to the SMALL/fast model: on CPU-only machines (no usable GPU) the 7B is
    # ~140-240 s/call and pins every core — not worth it for bulk. moondream is ~3x
    # faster (~45 s/call) and good enough for a yes/no "was it applied" second opinion.
    # Switch to "qwen2.5vl:7b" for a more careful value read when you have the time/GPU.
    vlm_model: str = "moondream"
    vlm_timeout: float = 300.0          # seconds per request (CPU inference is slow)
    # Cap the model's output length. The answer is a short JSON object, so a few
    # hundred tokens is plenty — capping it bounds the GENERATION time. On a fast GPU
    # this changes nothing.
    vlm_num_predict: int = 200
    # VLM call gating. The VLM is chained AFTER OCR, so it only ever runs on spec
    # redlines OCR could NOT already confirm — that alone bounds calls to the genuinely
    # uncertain items. We deliberately do NOT additionally require a pixel change:
    # the most valuable VLM case is the in-place text swap (e.g. SB3->SB5), which has
    # NO density change at all, so a pixel-change gate would skip exactly those. Set
    # this True to bound calls further (skips swaps) on a slow machine.
    vlm_require_pixel_change: bool = False
    # CPU-thread cap for the VLM (Ollama num_thread). Default 0 = use all cores. We
    # learned the hard way that capping it is counterproductive: on a 4-core CPU,
    # leaving 2 free made a single moondream call go from ~45s to >180s (it then hit
    # the timeout and the half-done generation lingered and lagged the machine — the
    # exact symptom we were trying to avoid). Full cores finishes fast and frees the
    # machine sooner. A NEWER computer offloads to the GPU, so the CPU stays free
    # regardless. Raise this only if you specifically want to throttle a CPU run.
    vlm_leave_free_cores: int = 0
    # GPU offload control. None = let Ollama decide. On machines with a tiny/old GPU
    # (e.g. a 2 GB card below CUDA compute 5.0), Ollama's partial offload CRASHES the
    # model runner (HTTP 500: "connection forcibly closed"); set this to 0 to force
    # pure-CPU and it runs fine (just slower). The client also AUTO-falls back to CPU
    # on the first 500, so this is mostly a way to skip that wasted first attempt.
    vlm_num_gpu: "int | None" = None
    vlm_crop_pad_pt: float = 28.0       # context around the target for the VLM crop
    vlm_max_side_px: int = 700          # downscale crops sent to the model. Kept
                                        # modest because CPU inference cost scales with
                                        # pixels; 700 reads CAD labels fine. Raise it
                                        # if you have a capable GPU.
    # Only run the VLM on pixel-FLAGGED items (per your choice): the misses and the
    # change-at-target "to verify" bucket. It never runs on clearly-fine items.

    # ---- verification tiers ---------------------------------------------
    use_ocr: bool = False              # local Tesseract OCR (must be installed)
    # OCR is calibrated at this DPI. When OCR is on, auto-DPI won't render above it:
    # Tesseract mis-reads short labels (L1/B1) at 350+, which inflated false misses.
    # The pixel diff is DPI-invariant (its grain is in physical points) so capping
    # at 300 for OCR runs doesn't change unexpected/miss screening.
    ocr_render_dpi: int = 300
    # Enlarge crops so the text hits Tesseract's sweet spot. Upscale is ADAPTIVE:
    # we target ~ocr_target_dpi effective resolution, so a 500-DPI render isn't
    # over-upscaled (which actually hurt OCR). upscale = clamp(target/dpi, 1, max).
    ocr_target_dpi: int = 600          # aim for this effective text resolution
    ocr_upscale_max: float = 3.0       # never enlarge more than this
    ocr_binarize_threshold: int = 165  # grayscale cutoff to black/white for OCR
    ocr_search_mult: float = 1.25      # OCR crop padding vs report pad; tight = cleaner read
    ocr_rotations: tuple = (90, 270)   # extra angles to retry when horizontal OCR misses
                                       # (architect plots beam labels vertically)
    ocr_fuzzy_gap: int = 1             # allow up to N stray chars between value chars when
                                       # matching (OCR reads "L3+L7" as "L3eL7", "4" as "A")

    # ---- output ----------------------------------------------------------
    crop_pad_pt: float = 40.0          # padding around finding crops in the report
    # One-click "zoom out" in the app shows a wider crop (more of the sheet around
    # the finding) so you can see where your markup actually is. Pre-rendered at
    # report time so the toggle just swaps the image (no re-render, no full reload).
    wide_crop_factor: float = 3.0      # zoomed-out crop covers ~this much more area

    def zoom(self, dpi: int | None = None) -> float:
        return (dpi or self.analysis_dpi) / 72.0

    def px(self, points: float) -> float:
        """Convert PDF points to analysis-resolution pixels."""
        return points * self.zoom()
