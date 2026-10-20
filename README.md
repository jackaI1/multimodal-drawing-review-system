# Multimodal Drawing Review System

A local PDF comparison tool that checks marked-up drawings against revised sheets and produces a review queue of possible misses, verified labels and unrequested changes.

https://github.com/user-attachments/assets/e7ba8c98-791b-4ca5-bac0-df6a75b7bbf6

## Overview

A revised drawing can contain most requested corrections while leaving one label
unchanged. A page-by-page visual check also becomes unreliable when sheets are
inserted, removed or re-plotted with small offsets.

The system first matches sheets, then compares registered drawing content around
redline targets. Pixel changes are evidence of activity; optional OCR and a vision
model read labels to determine whether a requested value appears at the target.
The output is a queue for a reviewer, rather than an approval to issue drawings.

Built during an engineering co-op, this public version includes generated
synthetic drawings in place of project records, and so does the demo video.
The default comparison needs neither credentials nor a model service.

## How it works

1. **Match sheets.** [`redlinecheck/prematch.py`](redlinecheck/prematch.py)
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
   [`tests/test_prematch_align.py`](tests/test_prematch_align.py).
3. **Extract targets and register images.**
   [`redlinecheck/extract.py`](redlinecheck/extract.py) reads annotations and
   leaders, with a red-ink fallback for flattened markups.
   [`redlinecheck/align.py`](redlinecheck/align.py) renders the original without
   annotations and registers the revision using phase correlation and a
   coarse-to-fine overlap fallback. Both images use a common analysis frame.
4. **Find content changes.** [`redlinecheck/diff.py`](redlinecheck/diff.py)
   compares ink-density cells whose size is expressed in PDF points. This reduces
   sensitivity to small plotting shifts. A finer grid checks activity at targets;
   clustered additions and removals identify changes elsewhere. Equal-ink text
   replacements can remain invisible to this stage.
5. **Assign review states.**
   [`redlinecheck/classify.py`](redlinecheck/classify.py) associates changes with
   target regions. Pixel activity produces a change-to-review state, never a
   verified requested value. Coordination notes and marked areas have separate
   states rather than being treated as failed text instructions.
6. **Optionally read the label.** [`redlinecheck/ocr.py`](redlinecheck/ocr.py)
   uses Tesseract, normalises common character confusions and limits fuzzy gaps.
   Short labels require whole-word matches. [`redlinecheck/vlm.py`](redlinecheck/vlm.py)
   can ask an opt-in Ollama vision model about unresolved targets. Verification
   requires a supported requested-value read-back; uncertain or interrupted
   responses remain review work. Policy regressions are in
   [`tests/test_verification.py`](tests/test_verification.py).
7. **Export evidence.** [`redlinecheck/pipeline.py`](redlinecheck/pipeline.py)
   suppresses repeated boilerplate by crop content and a position fallback.
   [`redlinecheck/report.py`](redlinecheck/report.py) writes an HTML review with
   before/after crops, CSV findings and an annotated PDF.
   [`redlinecheck/project.py`](redlinecheck/project.py) handles multiple input
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
[`redlinecheck/config.py`](redlinecheck/config.py).

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
text limitation. [`tests/test_pipeline_parity.py`](tests/test_pipeline_parity.py)
checks the CLI exports and eight generated scenarios. Its
`test_four_film_verdicts_end_to_end_with_real_ocr` exercises real OCR on a
separate synthetic sheet. No detection-accuracy percentage is established.

## Getting started

Use CPython 3.12; fresh-install verification used 3.12.14 on macOS arm64.
Runtime packages and their audited transitive versions are pinned in
[`requirements.txt`](requirements.txt) and
[`requirements-lock.txt`](requirements-lock.txt). Run from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/make_sample_drawings.py
python comparator.py examples/basic_redlined.pdf examples/basic_revised.pdf -o report
```

The generator writes synthetic PDF pairs into `examples/`. The comparison prints
2 possible misses, 1 unexpected change and 0 verified addressed findings. Open
`report/summary.html`; the same folder contains `findings.csv`,
`annotated.pdf` and supporting crops.

For OCR, install the Tesseract executable separately and ensure it is on `PATH`.
The tested version is 5.5.3; `pytesseract` is installed by the requirements file.
Then run:

```sh
python comparator.py examples/basic_redlined.pdf examples/basic_revised.pdf -o report-ocr --ocr
```

This produces the second row of the evaluation table. OCR output can vary with
Tesseract versions. For the optional browser UI:

```sh
python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
```

Open `http://127.0.0.1:8501` and upload both PDFs. Stop with Ctrl-C. The CLI
leaves OCR off unless requested; the UI enables it initially when Tesseract is
available. Ollama is opt-in and requires a separately installed model. Its
configured endpoint receives target crops; use a loopback endpoint for local-only
processing. [`INSTALL.md`](INSTALL.md) describes the Windows installer path;
that installer was not reverified by this documentation pass.

## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The current suite has 84 passing tests with Tesseract installed. It groups:

- Sequence scoring, weak-match rejection and inserted/dropped pages in
  [`tests/test_prematch_align.py`](tests/test_prematch_align.py).
- OCR character normalisation and short-label boundaries in
  [`tests/test_ocr_matching.py`](tests/test_ocr_matching.py), plus read-back and
  incomplete model-stream policy in [`tests/test_verification.py`](tests/test_verification.py).
- Generated PDF pipelines, DPI scaling, boilerplate, real OCR and report files in
  [`tests/test_pipeline_parity.py`](tests/test_pipeline_parity.py).
- File-name matching in [`tests/test_project_naming.py`](tests/test_project_naming.py).

The model-response tests use controlled responses; they do not measure live
vision-model quality. The real OCR test is skipped if Tesseract is unavailable.
PyMuPDF may emit upstream SWIG deprecation warnings.

## Project structure

```text
redlinecheck/                 matching, extraction, registration, diff and reports
comparator.py                 single-pair CLI entry point
app.py                        Streamlit review interface
examples/make_sample_drawings.py  synthetic PDF scenarios
tests/                       unit and integration regressions
tools/run_project.py         multi-file comparison entry point
installer/                    Windows setup and launcher scripts
```

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

## License

Original project source is MIT; see [`LICENSE`](LICENSE). PyMuPDF 1.26.7 is an
AGPL-3.0 dependency: distributing a combined application requires applicable
AGPL corresponding-source and licence compliance, and network deployments may
require source access. MIT does not replace those terms. Other native-library,
MPL and font obligations are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
