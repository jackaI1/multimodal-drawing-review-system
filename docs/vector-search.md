# Finding the same sheet in another drawing set

I use page search when a revised set changes order or the page layout changes enough
that the normal comparison cannot confidently pair the sheets. Search returns a
short list to inspect. It does not decide that a requested correction was made.

## Run it locally

Use Python 3.12, Docker with Compose, and Tesseract on `PATH`. On macOS I install
Tesseract with `brew install tesseract`; on Ubuntu, `sudo apt-get install tesseract-ocr`.
From the repository root, with the virtual environment active:

```sh
python -m pip install -e '.[vector,test]'
docker compose -p drawing-search up -d
curl --fail http://127.0.0.1:6333/readyz
python examples/make_search_drawings.py
python -m redlinecheck.vector_cli index examples/search_r2.pdf --project sample --revision r2
python -m redlinecheck.vector_cli search --text 'drinking water and wastewater drainage' --project sample --revision r2
python -m redlinecheck.vector_cli match examples/search_r1.pdf --page 2 --project sample --revision r2
python comparator.py examples/search_r1.pdf examples/search_r2.pdf -o report-search --vector-project sample --vector-revision r2 --dpi 72
```

The first search should rank sheet M2, revised page 3, first. The page query finds the
same plumbing drawing even though it moved from page 2 to page 3. The JSON output
includes one-based page numbers, sheet labels, document names, revision labels and
scores. `--document search_r2` or `--sheet M2` narrows a query. To index several PDFs
in one revision, repeat `index` with the same project and revision. Give different
files unique `--document` labels if their filenames are the same.

Open `report-search/summary.html`. In **Page match candidates**, inspect the source
and target previews, then confirm a pair by running the comparison again. For this
sample, source page 3 moved to revised page 1:

```sh
python comparator.py examples/search_r1.pdf examples/search_r2.pdf -o report-confirmed --page-pair 3:1 --dpi 72
```

A confirmed pair can cross page order. It displaces conflicting layout pairs to
manual review, keeps other pairs, and never assigns the same target twice. Repeat
`--page-pair` for more choices. The search and fallback workflow is currently in the
command line and generated HTML report; the Streamlit interface keeps its existing
layout matching workflow.

Stop the database when finished:

```sh
docker compose -p drawing-search down
```

This preserves the local index volume. `docker compose -p drawing-search down -v`
removes this project's stored index as well. Do not run it on an index you want to
keep. The fixture PDFs and report folders are generated locally and ignored by Git.

## What I store

An embedding is a list of numbers representing content, so related content can be
retrieved without requiring identical words or page positions. I run both models on
the CPU through FastEmbed 0.9.1:

- BGE small English v1.5 produces a 384-number text embedding. I read native PDF text
  when present and use Tesseract OCR, which reads letters from images, for pages
  without native text. Long text is split into 160-word chunks and averaged.
- CLIP ViT-B/32 vision produces a 512-number image embedding. I embed the full page
  and its four quarter-page crops, then average them. Each render is bounded to a
  1,600-pixel long edge so large paper sizes do not require huge images in memory.

I chose self-hosted Qdrant rather than adding PostgreSQL just for page search. One
point, Qdrant's name for a stored record, holds two named vectors, `text` and `image`.
Both use cosine similarity, a comparison of vector direction. The payload, the
record's metadata, holds project, revision, document, SHA-256 document hash, sheet,
page, title, OCR/native text, orientation and the embedding-version signature. Pages
with no readable text can still be searched through the image vector.

Qdrant performs the filtered vector queries. Project and revision are required;
document and sheet are optional. A page query also checks orientation. Fallback
queries require the current target's document name and hash, so another upload with
the same filename cannot supply stale candidates. Search labels organize data; they
are not user authentication or access control.

Text queries search BGE vectors. Page queries retrieve up to four times the requested
count from each vector index, then combine rankings with reciprocal rank fusion:
each position contributes `1 / (60 + rank)`. This avoids presenting unrelated text
and image cosine scores as a shared probability. Combined scores are ranking aids,
not confidence percentages. Ties use document and page order.

## Index lifecycle and comparison fallback

`redlinecheck/vector_search.py` owns extraction, embeddings, indexing and retrieval.
Each point has a deterministic ID from project, revision, document, hash and page.
Re-indexing is idempotent. I hash the PDF before and after extraction and abort if
the file changes; the comparison and preview stages also verify these hashes. I finish extraction and validate records before writing,
upsert the replacement, and then delete stale pages for that document. This is not a
multi-request database transaction: an interrupted write/delete or concurrent writes
to the same document can leave mixed versions. Re-index once with a single writer to
repair it. Exact-hash filtering protects comparison candidates, while ordinary text
search may see both versions until that repair.

I reject different model signatures or vector dimensions rather than silently mix
incompatible embeddings. Changing models or crop policy needs a fresh collection
(`--collection` for the search commands) and a reindex. Comparisons use the default
`drawing_pages_v1` collection.

The normal matcher uses Needleman-Wunsch, a sequence algorithm that preserves page
order. `vector_match.py` asks for candidates for unmatched pages, matched layout
scores below 0.65, or a margin below 0.03 between the chosen page and an alternative.
These are review heuristics, not calibrated uncertainty estimates. When enabled,
`pipeline.py` indexes the exact revised PDF and attaches candidates without altering
the alignment. `vector_report.py` writes previews and `page_candidates.json` beside
the HTML report. A reviewer can use `--page-pair` to compare a confirmed candidate.

## Privacy and limits

The Compose service binds only to localhost and does not restart automatically. It
stores text, vectors and metadata in a local Docker volume. The first embedding run
downloads public model weights from their hosting services; drawing content is
processed locally. Keep the database private. An overridden `--url`/`--qdrant-url`
will send extracted text, metadata and vectors to that address.

Sheet labels are inferred from the first sheet-like token in text, so a body
reference can be mistaken for the sheet label. Check that label before using a
`--sheet` filter; page and document metadata still identify the returned drawing.
English OCR quality, small labels, sparse drawings and repeated title blocks can
mislead the models. The image crops lose precise geometry and fine dimensions when
resized. This generated three-sheet fixture checks the plumbing, framing and lighting
examples; it is not a retrieval-accuracy study on a real engineering drawing archive.
For a scan, missing Tesseract is an explicit indexing error and does not erase the
previous index. A failed database connection is reported as an error; ordinary
comparisons still work without installing the optional search dependencies.

## Tests and references

```sh
python -m pytest -q
QDRANT_URL=http://127.0.0.1:6333 RUN_VECTOR_MODELS=1 python -m pytest -q
```

The first command runs database contract tests with Qdrant's embedded local engine;
the model test is opt-in. The second uses the container, real BGE/CLIP weights and
actual Tesseract on a generated raster-only PDF. CI runs that second mode on Linux
and keeps the existing C++ and Python suites on Linux and macOS. Fixture collections
have unique names and are deleted in `finally` blocks; CI removes its service
container when the job ends.

I used the official references for [named vectors](https://qdrant.tech/documentation/concepts/vectors/#named-vectors),
[metadata filtering](https://qdrant.tech/documentation/search/filtering/) and
[local embeddings](https://qdrant.tech/documentation/embeddings/fastembed/).
