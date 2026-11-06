"""Opt-in real BGE/CLIP + Tesseract + Qdrant test, also required in Linux CI."""
import os
from uuid import uuid4

import fitz
import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_VECTOR_MODELS") != "1", reason="set RUN_VECTOR_MODELS=1 for learned models")


def test_learned_text_crops_scans_and_reordered_revision(tmp_path):
    from qdrant_client import QdrantClient
    from examples.make_search_drawings import make_drawings
    from redlinecheck.vector_search import DrawingIndex, LocalEncoder, page_record
    original, revised = make_drawings(tmp_path)
    encoder = LocalEncoder(os.getenv("FASTEMBED_CACHE_PATH"))
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    collection = "learned_" + uuid4().hex
    try:
        with DrawingIndex(client, encoder, collection) as index:
            assert index.index_pdf(revised, project="sample", revision="r2") == 3
            hits = index.search_text("drinking water and wastewater drainage", project="sample", revision="r2")
            assert hits[0]["sheet"] == "M2"
            assert hits[0]["page"] == 3
            with fitz.open(original) as doc:
                for source_page, target in ((0, 2), (1, 3), (2, 1)):
                    record = page_record(doc[source_page], source_page + 1, encoder)
                    found = index.search_page(record, project="sample", revision="r2")
                    assert found[0]["page"] == target
                    assert "image_score" in found[0] and "text_score" in found[0]
                # Raster-only PDF has no native text: exercise actual Tesseract.
                pix = doc[1].get_pixmap(matrix=fitz.Matrix(2, 2))
            with fitz.open() as scan:
                p = scan.new_page(width=600, height=400)
                p.insert_image(p.rect, stream=pix.tobytes("png"))
                scanned = page_record(p, 1, encoder)
            from redlinecheck import prematch
            from redlinecheck.config import Config
            from redlinecheck.vector_match import fallback_candidates
            with fitz.open(original) as src, fitz.open(revised) as dst:
                cfg = Config()
                layout = prematch.match_documents(src, dst, cfg)
                sim = prematch._similarity_matrix(prematch.fingerprint_doc(src, cfg, False),
                                                 prematch.fingerprint_doc(dst, cfg, True))
                candidates = fallback_candidates(src, layout, sim, index, project="sample",
                    revision="r2", document=revised.stem,
                    document_hash=__import__("hashlib").sha256(revised.read_bytes()).hexdigest())
                assert any(c["source_page"] == 3 and c["matches"][0]["page"] == 1 for c in candidates)
            assert "WATER" in scanned.text.upper()
            assert len(scanned.image_vector) == 512 and len(scanned.text_vector) == 384
            assert index.search_page(scanned, project="sample", revision="r2")[0]["page"] == 3
    finally:
        cleanup = QdrantClient(url=os.environ["QDRANT_URL"])
        try:
            cleanup.delete_collection(collection)
        finally:
            cleanup.close()
