# Multimodal Drawing Review System

I built a PDF drawing checker because it was easy to miss a requested correction when comparing revisions by hand.

## Overview

Checking a revised drawing set means finding each requested correction again.
It's easy to miss an unchanged label, especially when pages have moved or new
sheets have been added. This tool pairs the sheets, compares the marked areas
and makes a review queue with before-and-after crops. It also flags changes
outside the marked areas.

I built this during an engineering co-op. This public project and its demo use
generated drawings instead of client records. To try it, generate the sample
pair and open the report with the commands below. No account is needed.

## Try it

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/make_sample_drawings.py
python comparator.py examples/basic_redlined.pdf examples/basic_revised.pdf -o report
```

Open `report/summary.html`. The folder also contains a findings CSV, an annotated
PDF and supporting crops. The sample leaves one correction undone and adds an
unrequested item. With the default image comparison it reports two possible
misses and one unexpected change: it can see changed pixels, but cannot confirm
that the requested text is there.

For a browser interface:

```sh
python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
```

Open <http://127.0.0.1:8501> and upload the two generated PDFs. Stop with Ctrl-C.
The [Windows installation guide](INSTALL.md) describes the installer option.

## Reading changed labels

To check text as well as pixels, install Tesseract separately and put it on
`PATH`, then run:

```sh
python comparator.py examples/basic_redlined.pdf examples/basic_revised.pdf -o report-ocr --ocr
```

On the sample, this confirms one requested label and leaves one possible miss.
The browser interface initially enables text recognition when Tesseract is
available. An optional local Ollama vision model can look at unresolved targets;
it needs separate setup and receives crops at the configured endpoint.
Uncertain readings stay in the queue for a person to check.

## Faster comparisons

Repeatedly checking small offsets between images can take time. An optional
compiled comparison speeds up that part while keeping the same pixel counts and
review decisions. It is used automatically when installed:

```sh
python -m pip install -e .
```

This needs a C++17 compiler, such as Apple's Command Line Tools on macOS or
GCC/Clang on Linux. The source-only setup above works without a compiler.
The [comparison reference](docs/comparison.md#c17-pixel-comparison) covers
installation options, backend selection and the synthetic timing measurements.
Those timings cover the pixel comparison, not a whole drawing review.

## Limits and tests

The output is a queue to review, not approval to issue drawings. Scans, unreadable
labels, major layout changes and unfamiliar markup conventions can leave work
for the reviewer. Check low-confidence sheets in full. The
[comparison reference](docs/comparison.md) explains matching, text checks,
configuration and test fixtures. The reported reduction from two hours to thirty
minutes came from on-the-job use; the public synthetic examples cannot reproduce
that measurement.

```sh
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
```

The suite checks generated PDFs, reports, sheet matching, text recognition and
agreement between the two pixel implementations. Native-only tests need the
compiled extension; the real text-recognition test needs Tesseract.

## Demo

https://github.com/user-attachments/assets/de7ea1df-c432-4912-ae5f-b7f1004bc5f5

## License

Developed at D.L. Engineering Inc. and published with the firm's written
permission. Client drawings, firm documents, internal identifiers and proprietary
data are excluded.

Original source is MIT; see [LICENSE](LICENSE). PyMuPDF is AGPL-3.0, with source
and licence obligations for combined distributions and some network deployments.
Other dependency terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
