"""Database contract tests; actual learned-embedding smoke tests live separately."""
import pytest
import fitz

from redlinecheck.vector_search import DrawingIndex, PageRecord, page_record


class FixtureEncoder:
    """Small deterministic vectors isolate database behavior from model downloads."""
    dimension_text = 3
    dimension_image = 3
    signature = "fixture-v1"

    def text(self, text):
        return [float("beam" in text.lower()), float("pipe" in text.lower()), 0.1]

    def images(self, images):
        return [1.0, 0.0, 0.1]


@pytest.fixture
def index():
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient
    import os
    from uuid import uuid4
    client = QdrantClient(url=os.environ["QDRANT_URL"]) if os.getenv("QDRANT_URL") else QdrantClient(":memory:")
    collection = "test_" + uuid4().hex
    try:
        with DrawingIndex(client, FixtureEncoder(), collection) as idx:
            yield idx
    finally:
        # The context closes the HTTP client, so use a fresh handle for teardown.
        cleanup = QdrantClient(url=os.environ["QDRANT_URL"]) if os.getenv("QDRANT_URL") else None
        if cleanup:
            try:
                cleanup.delete_collection(collection)
            finally:
                cleanup.close()


def record(page, text="beam plan", sheet="S1"):
    return PageRecord(page, text, sheet, text, False, [1., 0., .1], [1., 0., .1])


def test_index_is_idempotent_and_removes_stale_pages(index):
    index.replace_document("demo", "r1", "structural", "hash1", [record(1), record(2)])
    index.replace_document("demo", "r1", "structural", "hash2", [record(1)])
    hits = index.search_text("beam", project="demo", revision="r1")
    assert len(hits) == 1
    assert hits[0]["page"] == 1
    assert hits[0]["document_hash"] == "hash2"
    assert hits[0]["sheet"] == "S1"


def test_query_filters_and_ranking(index):
    index.replace_document("demo", "r2", "structural", "a", [record(1)])
    p = record(2, "pipe layout", "M2")
    p.text_vector = [0., 1., .1]
    index.replace_document("demo", "r2", "mechanical", "b", [p])
    index.replace_document("other", "r2", "structural", "c", [record(1)])
    index.replace_document("demo", "r1", "structural", "d", [record(1)])
    hits = index.search_text("pipe", project="demo", revision="r2")
    assert [h["sheet"] for h in hits] == ["M2", "S1"]
    assert hits[0]["score"] > hits[1]["score"]
    assert len(index.search_text("pipe", project="demo", revision="r2", sheet="M2")) == 1
    assert len(index.search_text("pipe", project="demo", revision="r2", document="structural")) == 1
    assert index.search_text("pipe", project="demo", revision="missing") == []


def test_scanned_page_uses_ocr_and_crops(monkeypatch):
    from redlinecheck import vector_search
    calls = []
    monkeypatch.setattr(vector_search, "read_ocr", lambda image: calls.append(image.size) or "S2 BEAM PLAN")
    with fitz.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.draw_rect(fitz.Rect(30, 30, 270, 170))
        r = page_record(page, 1, FixtureEncoder())
    assert calls
    assert r.text == "S2 BEAM PLAN"
    assert r.sheet == "S2"
    assert r.text_vector != [] and r.image_vector != []


def test_page_match_is_scoped_to_exact_content(index):
    index.replace_document("demo", "new", "drawing", "current", [record(1)])
    index.replace_document("demo", "new", "another", "old", [record(1)])
    hits = index.search_page(record(1), project="demo", revision="new", document="drawing", document_hash="current")
    assert len(hits) == 1 and hits[0]["document_hash"] == "current"
    assert index.search_page(record(1), project="demo", revision="new", document="drawing", document_hash="stale") == []


def test_embedding_version_mismatch_is_rejected(index):
    index.replace_document("demo", "r1", "drawing", "a", [record(1)])
    index.encoder.signature = "different-model"
    with pytest.raises(ValueError, match="embedding"):
        index.search_text("beam", project="demo", revision="r1")


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_page_search_rejects_invalid_limits(index, limit):
    with pytest.raises(ValueError, match="limit"):
        index.search_page(record(1), project="demo", revision="r2", limit=limit)


def test_invalid_records_do_not_partially_replace_document(index):
    index.replace_document("demo", "r2", "d", "old", [record(1)])
    invalid = record(0)
    with pytest.raises(ValueError, match="page"):
        index.replace_document("demo", "r2", "d", "new", [record(1), invalid])
    hits = index.search_text("beam", project="demo", revision="r2")
    assert len(hits) == 1 and hits[0]["document_hash"] == "old"


def test_blank_query_does_not_return_unrelated_pages(index):
    index.replace_document("demo", "r2", "d", "h", [record(1)])
    assert index.search_text("  ", project="demo", revision="r2") == []


def test_image_search_without_readable_text_uses_crop_vectors(index):
    a, b = record(1), record(2, "pipe layout", "M2")
    b.image_vector = [0., 1., .1]
    index.replace_document("demo", "r2", "drawing", "h", [a, b])
    source = record(1)
    source.text_vector = []
    source.image_vector = [0., 1., .1]
    hits = index.search_page(source, project="demo", revision="r2")
    assert [h["page"] for h in hits] == [2, 1]
    assert hits[0]["image_score"] > hits[1]["image_score"]
    assert "text_score" not in hits[0]
    source.landscape = True
    assert index.search_page(source, project="demo", revision="r2") == []


def test_incompatible_empty_collection_is_rejected(index):
    encoder = FixtureEncoder()
    encoder.dimension_text = 4
    with pytest.raises(ValueError, match="dimensions"):
        DrawingIndex(index.client, encoder, index.collection)


def test_pdf_changed_during_extraction_does_not_write(index, tmp_path, monkeypatch):
    from redlinecheck import vector_search
    source = tmp_path / "drawing.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_text((30, 60), "S1 BEAM PLAN")
        doc.save(source)
    extract = vector_search.page_record
    def replace_after_read(page, number, encoder):
        result = extract(page, number, encoder)
        with fitz.open() as changed:
            changed.new_page().insert_text((30, 60), "M2 PIPE PLAN")
            changed.save(tmp_path / "changed.pdf")
        (tmp_path / "changed.pdf").replace(source)
        return result
    monkeypatch.setattr(vector_search, "page_record", replace_after_read)
    with pytest.raises(ValueError, match="changed"):
        index.index_pdf(source, project="demo", revision="r2")
    assert index.search_text("beam", project="demo", revision="r2") == []
