# Drawing comparison reference

How sheets are matched, changes are checked and reports are produced.

## How it works

1. **Match sheets.** [`redlinecheck/prematch.py`](../redlinecheck/prematch.py)
   renders each page at 36 DPI and reduces it to a mean-centred, unit-normalised
   32 × 32 ink fingerprint. Cosine similarity scores candidate pairs. Different
   displayed orientations and scores below `0.35` are excluded before alignment.
2. **Align the sequences.** `_align` in the same module implements
   Needleman–Wunsch. Each dynamic-programming cell maximises diagonal similarity,
   a deletion with a `-0.15` gap score, or an insertion with that gap score.
   Boundary rows accumulate gaps; traceback returns ordered pairs and unmatched
   sheets. Global alignment preserves order across the whole set while allowing
   added and dropped sheets. For `n` and `m` pages, the DP uses O(nm) time and
   memory; fingerprint dimension is fixed. See
   [`tests/test_prematch_align.py`](../tests/test_prematch_align.py).
3. **Extract targets and register images.**
   [`redlinecheck/extract.py`](../redlinecheck/extract.py) reads annotations and
   leaders, with a red-ink fallback for flattened markups.
   [`redlinecheck/align.py`](../redlinecheck/align.py) renders the original without
   annotations and registers the revision using phase correlation and a
   coarse-to-fine overlap fallback. Both images use a common analysis frame.
4. **Find content changes.** [`redlinecheck/diff.py`](../redlinecheck/diff.py)
   compares ink-density cells whose size is expressed in PDF points. This reduces
   sensitivity to small plotting shifts. A finer grid checks activity at targets;
   clustered additions and removals identify changes elsewhere. Equal-ink text
   replacements can remain invisible to this stage.
5. **Assign review states.**
   [`redlinecheck/classify.py`](../redlinecheck/classify.py) associates changes with
   target regions. Pixel activity produces a change-to-review state, never a
   verified requested value. Coordination notes and marked areas have separate
   states rather than being treated as failed text instructions.
6. **Optionally read the label.** [`redlinecheck/ocr.py`](../redlinecheck/ocr.py)
   uses Tesseract, normalises common character confusions and limits fuzzy gaps.
   Short labels require whole-word matches. [`redlinecheck/vlm.py`](../redlinecheck/vlm.py)
   can ask an opt-in Ollama vision model about unresolved targets. Verification
   requires a supported requested-value read-back; uncertain or interrupted
   responses remain review work. Policy regressions are in
   [`tests/test_verification.py`](../tests/test_verification.py).
7. **Export evidence.** [`redlinecheck/pipeline.py`](../redlinecheck/pipeline.py)
   suppresses repeated boilerplate by crop content and a position fallback.
   [`redlinecheck/report.py`](../redlinecheck/report.py) writes an HTML review with
   before/after crops, CSV findings and an annotated PDF.
   [`redlinecheck/project.py`](../redlinecheck/project.py) handles multiple input
   files, matching files one-to-one before comparing sheets.
## Design decisions and trade-offs

- Conservative verification keeps an unreadable correction in the queue. It
  costs reviewer time, but avoids treating pixel activity as semantic proof.
- Density comparison reduces plotting noise and misses some in-place text edits.
  OCR addresses that separate problem; it also adds a native dependency and
  recognition errors.
- Global matching permits insertions and deletions but assumes sheet order is
  preserved. It does not solve arbitrary page permutation.
- Automatic DPI targets roughly 30-pixel text, clamped to 200–500 DPI. OCR caps
  analysis at its configured 300 DPI. Larger rasters cost memory and runtime.
- Unconditional local-shift suppression is off by default: repeated nearby
  content can otherwise be mistaken for the target and hide a real change.
- Boilerplate suppression reduces repeated report entries but can hide a genuine
  repeated edit. Low-overlap sheets receive a whole-sheet review notice.

Configuration and defaults are in
[`redlinecheck/config.py`](../redlinecheck/config.py).
## Results and evaluation

Reported real-world use: revision checking went from 2 hours to 30 minutes during
on-the-job use. This is a workflow measurement from real use, not a
benchmark reproducible from this repository's synthetic drawings.

Reproducible here, using the basic fixture and commands below:

| Mode | Possible misses | Unexpected changes | Verified addressed |
| --- | ---: | ---: | ---: |
| Default pixel comparison | 2 | 1 | 0 |
| With `--ocr` and Tesseract 5.5.3 | 1 | 1 | 1 |

The fixture applies one requested label change, leaves another unchanged and
adds an unrequested item. The pixel-only result intentionally demonstrates its
text limitation. [`tests/test_pipeline_parity.py`](../tests/test_pipeline_parity.py)
checks the CLI exports and eight generated scenarios. Its
`test_four_film_verdicts_end_to_end_with_real_ocr` exercises real OCR on a
separate synthetic sheet. No detection-accuracy percentage is established.
## C++17 pixel comparison

`pip install .` builds the optional pybind11 extension with CMake and
scikit-build-core. A C++17 compiler is required (Apple Command Line Tools on
macOS, or GCC/Clang on Linux); build dependencies are installed automatically.
Use `pip install -e '.[test]'` when developing from this checkout. Installation
includes the existing Streamlit UI dependencies.

Use the editable install when running `comparator.py`, the benchmark or tests
from this checkout: the source package otherwise shadows the installed wheel.
A regular `pip install .` compiles and installs the native wheel for imports
outside the checkout, including `python -m redlinecheck` from another directory.

The extension accelerates the repeated shifted-XOR search used to compare
callout interiors and suppress moved drawing content. It counts differing ink
pixels for every small circular shift, matching the previous NumPy implementation
exactly. It releases the Python GIL, uses a single native thread and avoids
allocating rolled images. Density thresholds and review states are unchanged.

Source-only use with `pip install -r requirements.txt` automatically uses the
Python/NumPy fallback. To install without a compiler explicitly:

```sh
python -m pip install . -Ccmake.define.REDLINECHECK_BUILD_CPP=OFF
```

Use `REDLINECHECK_PIXEL_BACKEND=python` to force fallback during a comparison;
`cpp` requires the extension and fails clearly if it is missing. The default is
`auto`. Check availability with
`python -c 'from redlinecheck._pixels import NATIVE_AVAILABLE; print(NATIVE_AVAILABLE)'`.

### Reproducible benchmark

[`benchmarks/benchmark_pixels.py`](../benchmarks/benchmark_pixels.py) generates a
seeded synthetic drawing pair, applies a shift and a real edit, asserts identical
results, warms both implementations and alternates timing order. Timings include
dispatch and input normalization, excluding fixture generation and compilation.
The Python baseline is the existing vectorized NumPy roll/XOR loop.

```sh
python -m pip install -e .
python benchmarks/benchmark_pixels.py --size 128 --radius 3 --repeat 9
python benchmarks/benchmark_pixels.py --size 512 --radius 3 --repeat 9
```

Measured on Apple M5 Pro, macOS arm64 (Darwin 27.0.0), CPython 3.12.14,
NumPy 2.5.4, Apple Clang Release build, on 2026-11-03:

| Synthetic mask pair | Python/NumPy median | C++17 median | Kernel speedup |
| --- | ---: | ---: | ---: |
| 128 × 128, radius 3 | 0.289 ms | 0.070 ms | 4.12× |
| 512 × 512, radius 3 | 1.500 ms | 1.358 ms | 1.10× |

Both backends return exactly **24 differing pixels** after the best shift.
Small crops benefit most because the Python path allocates temporary arrays for
each shift. These are comparison-kernel measurements, not whole-PDF speedups;
results vary by CPU, crop size and radius. Only generated synthetic masks are used.
## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
```

The suite includes the original 84 regressions plus 45 pixel-comparison and
benchmark tests. Native-only checks are skipped when the extension is absent;
CI requires a successful native import before running the full native suite.
It groups:

- Sequence scoring, weak-match rejection and inserted/dropped pages in
  [`tests/test_prematch_align.py`](../tests/test_prematch_align.py).
- OCR character normalisation and short-label boundaries in
  [`tests/test_ocr_matching.py`](../tests/test_ocr_matching.py), plus read-back and
  incomplete model-stream policy in [`tests/test_verification.py`](../tests/test_verification.py).
- Generated PDF pipelines, DPI scaling, boilerplate, real OCR and report files in
  [`tests/test_pipeline_parity.py`](../tests/test_pipeline_parity.py).
- File-name matching in [`tests/test_project_naming.py`](../tests/test_project_naming.py).
- Exact C++/Python counts, circular boundaries, empty/tiny and strided masks,
  read-only/noncanonical boolean storage, fallback and both comparison consumers
  in [`tests/test_pixels.py`](../tests/test_pixels.py).

The model-response tests use controlled responses; they do not measure live
vision-model quality. The real OCR test is skipped if Tesseract is unavailable.
PyMuPDF may emit upstream SWIG deprecation warnings.
## Limitations and known issues

- Synthetic regressions demonstrate specific behaviours, not general accuracy on
  drawing standards or independent client datasets.
- Scans lose annotation/text structure. Flattened red ink identifies review areas,
  but does not reconstruct the instruction's semantics.
- Re-plots with major geometry changes and cropped page dimensions can defeat
  registration. Review low-confidence sheets in full.
- OCR verification targets readable FreeText specification comments, not every
  markup type. Other annotation or colour conventions require extraction tuning.
- OCR may confuse similar labels; a vision model may return an incorrect reading.
  Review findings and drawing context before accepting a revision.
- Multi-file matching is one-to-one. Combine a revision split across files before
  comparing it against a single original file.
- Optional Ollama models, their weights and Tesseract are not bundled. Live model
  quality and the Windows installer are outside the fresh-copy checks reported here.
