# Multimodal Drawing Review System

I built a drawing comparison tool because I needed to check whether requested corrections had made it into a revised PDF.

## Overview

I used to find each marked correction again in the revised drawing set. A changed
page order or an extra sheet made that harder. I can give this tool the marked-up set
and the revised set, then work through a list of possible misses with
before-and-after image crops. It pairs corresponding sheets and also flags changes
outside the marked areas. I can also search for a sheet by its contents or find the
same sheet after a revision changes the page order. Uncertain page matches get
previews to check before I confirm a pair.

The original tool came from my engineering co-op. I maintain this public version as a
personal project with generated drawings, so I can share it without client records. To try it,
generate the sample pair and open the report with the commands below. No account is
needed.

## Try it

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/make_sample_drawings.py
python comparator.py examples/basic_redlined.pdf examples/basic_revised.pdf -o report
```

Open `report/summary.html`. The folder also contains a findings CSV, a table I can
open in a spreadsheet, plus an annotated PDF and supporting image crops. The sample
leaves one correction undone and adds an unrequested item. With the default image
comparison it reports two possible misses and one unexpected change: it can see
changed pixels, but cannot confirm that the requested text is there.

For a browser interface:

```sh
python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false
```

Open <http://127.0.0.1:8501> and upload the two generated PDFs. Stop with Ctrl-C. The
[Windows installation guide](INSTALL.md) describes the installer option.

## Reading changed labels

To check text as well as pixels, I use Tesseract for optical character recognition
(OCR), which reads letters from images. Install it separately and put it on `PATH`,
then run:

```sh
python comparator.py examples/basic_redlined.pdf examples/basic_revised.pdf -o report-ocr --ocr
```

On the sample, this confirms one requested label and leaves one possible miss. The
browser interface initially enables text recognition when Tesseract is available. An
optional vision model, an image-reading model run locally with Ollama, can look at
unresolved targets; it needs separate setup and receives crops at the configured
endpoint. Uncertain readings stay in the queue for a person to check.

## Finding a sheet across revisions

When page order or layout changes, I search the drawing contents instead of hunting
through every sheet. The optional local search database stores text and image
representations of each page, along with its revision and sheet label. I can search
in plain language or give it a page and ask for similar pages in another revision.

With Docker running and Tesseract installed (`brew install tesseract` on macOS):

```sh
python -m pip install -e '.[vector,test]'
docker compose -p drawing-search up -d
python examples/make_search_drawings.py
python -m redlinecheck.vector_cli index examples/search_r2.pdf --project sample --revision r2
python -m redlinecheck.vector_cli search --text 'drinking water and wastewater drainage' --project sample --revision r2
python -m redlinecheck.vector_cli match examples/search_r1.pdf --page 2 --project sample --revision r2
```

The sample's plumbing sheet moves from page 2 to page 3. Both searches rank it first.
The first run downloads the public text and image models; drawing content stays on
my computer with this localhost setup. To add candidate previews to a comparison:

```sh
python comparator.py examples/search_r1.pdf examples/search_r2.pdf -o report-search --vector-project sample --vector-revision r2 --dpi 72
```

Open `report-search/summary.html` and read **Page match candidates**. I inspect the
source and revised page previews, then use the displayed `--page-pair` command to
confirm a choice. Suggestions do not silently change the comparison. This search
workflow uses the command line and HTML report. Stop the database with
`docker compose -p drawing-search down`; the index is preserved for the next run.

I use Qdrant, a self-hosted vector database that finds nearby content representations,
with separate text and image indexes. Text comes from the PDF or OCR, and image
representations come from page crops. The [search design and setup guide](docs/vector-search.md)
explains filtering, matching, model downloads, storage, deletion and limitations.

## Faster comparisons

I compare nearby image positions so a small drawing shift is not mistaken for an
edit. An optional C++ component speeds up this repeated search while keeping the same
pixel counts and review decisions. Python calls the compiled component through
pybind11, a bridge between the two languages. It is used automatically when
installed:

```sh
python -m pip install -e .
```

This needs a C++17 compiler, such as Apple's Command Line Tools on macOS or GCC/Clang
on Linux. The source-only setup above works without a compiler. The [comparison
reference](docs/comparison.md#c17-pixel-comparison) covers installation options,
backend selection and the synthetic timing measurements. Those timings cover the
pixel comparison, not a whole drawing review.

## Limits and tests

The output is a queue to review, not approval to issue drawings. Scans, unreadable
labels, major layout changes and unfamiliar markup conventions can leave work for the
reviewer. Check low-confidence sheets in full. The [comparison
reference](docs/comparison.md) explains matching, text checks, configuration and test
fixtures. The reported reduction from two hours to thirty minutes came from
on-the-job use; the public synthetic examples cannot reproduce that measurement.

```sh
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
```

The suite checks generated PDFs, reports, sheet matching, text recognition and
agreement between the two pixel implementations. Native-only tests need the compiled
extension; the real text-recognition test needs Tesseract. The search CI job runs
against a Qdrant container with actual text/image models and scanned fixture drawings:

```sh
QDRANT_URL=http://127.0.0.1:6333 RUN_VECTOR_MODELS=1 python -m pytest -q
```

That full setup passes 153 tests, including all 129 existing checks. On the generated
three-sheet revision set, all 3 page queries rank the matching reordered page first;
the scanned plumbing sheet and a plain-language water/drainage query also rank its
match first. These are fixture results, not an accuracy claim for client drawings.

## Demo

https://github.com/user-attachments/assets/de7ea1df-c432-4912-ae5f-b7f1004bc5f5

## License

Developed at D.L. Engineering Inc. and published with the firm's written permission.
Client drawings, firm documents, internal identifiers and proprietary data are
excluded.

Original source is MIT; see [LICENSE](LICENSE). PyMuPDF is AGPL-3.0, with source and
licence obligations for combined distributions and some network deployments. Other
dependency terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
