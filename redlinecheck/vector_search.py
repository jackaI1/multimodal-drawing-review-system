"""Optional local semantic page retrieval backed by a real Qdrant collection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from uuid import NAMESPACE_URL, uuid5

import fitz
import numpy as np
from PIL import Image


@dataclass
class PageRecord:
    page: int                 # one-based, as displayed to a reviewer
    text: str
    sheet: str
    title: str
    landscape: bool
    text_vector: list[float]
    image_vector: list[float]


class LocalEncoder:
    """Learned embeddings on CPU; imports and model downloads are opt-in only."""
    dimension_text = 384
    dimension_image = 512
    signature = "bge-small-en-v1.5+clip-ViT-B-32-vision/crops-v1"

    def __init__(self, cache_dir=None):
        from fastembed import ImageEmbedding, TextEmbedding
        self._text = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=cache_dir, threads=2)
        self._image = ImageEmbedding("Qdrant/clip-ViT-B-32-vision", cache_dir=cache_dir, threads=2)

    def text(self, text, *, query=False):
        if not text.strip():
            return []
        # Keep long OCR pages inside the model context rather than truncating to
        # the title block. Average bounded chunks and let Qdrant normalise cosine.
        words = text.split()
        chunks = [" ".join(words[i:i + 160]) for i in range(0, len(words), 160)]
        vectors = self._text.query_embed(chunks) if query else self._text.embed(chunks)
        return np.mean(list(vectors), axis=0).tolist()

    def images(self, images):
        return np.mean(list(self._image.embed(images)), axis=0).tolist()


def read_ocr(image):
    import pytesseract
    return pytesseract.image_to_string(image, config="--psm 11", timeout=30).strip()


def page_record(page, number, encoder, *, annots=False):
    """Read native PDF text or OCR a scan, and embed five bounded page crops."""
    # Respect displayed rotation. Clip to a 1600-pixel long edge, including scans
    # and large-format sheets, so indexing memory does not scale with paper size.
    zoom = min(2., 1600. / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB,
                          alpha=False, annots=annots)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    text = page.get_text().strip()
    if not text:
        text = read_ocr(image)
    w, h = image.size
    crops = [image] + [image.crop(box) for box in (
        (0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h), (w // 2, h // 2, w, h))]
    sheet_match = re.search(r"\b[A-Z]{1,3}[-.]?\d{1,4}(?:\.\d+)?\b", text)
    return PageRecord(number, text, sheet_match.group() if sheet_match else "",
                      " ".join(text.splitlines()[:2])[:200], w >= h,
                      encoder.text(text), encoder.images(crops))


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DrawingIndex:
    """Separate text/image cosine indexes with mandatory project isolation."""
    def __init__(self, client, encoder, collection="drawing_pages_v1"):
        from qdrant_client import models
        self.client, self.encoder, self.collection = client, encoder, collection
        self.models = models
        if not client.collection_exists(collection):
            client.create_collection(collection, vectors_config={
                "text": models.VectorParams(size=encoder.dimension_text, distance=models.Distance.COSINE),
                "image": models.VectorParams(size=encoder.dimension_image, distance=models.Distance.COSINE),
            })
            for key in ("project", "revision", "document", "document_hash", "sheet", "encoder"):
                # Qdrant uses these indexes to avoid scanning other drawing sets.
                if client.__class__.__name__ != "QdrantClient" or client._client.__class__.__name__ != "QdrantLocal":
                    client.create_payload_index(collection, key, models.PayloadSchemaType.KEYWORD, wait=True)
        vectors = client.get_collection(collection).config.params.vectors
        expected = {"text": encoder.dimension_text, "image": encoder.dimension_image}
        if not isinstance(vectors, dict) or any(
                name not in vectors or vectors[name].size != size or
                vectors[name].distance != models.Distance.COSINE for name, size in expected.items()):
            raise ValueError("embedding dimensions or distance differ; use a new collection and reindex")
        self._check_encoder()

    @classmethod
    def connect(cls, url="http://127.0.0.1:6333", *, cache_dir=None, collection="drawing_pages_v1"):
        from qdrant_client import QdrantClient
        encoder = LocalEncoder(cache_dir)
        client = QdrantClient(url=url, timeout=30)
        try:
            return cls(client, encoder, collection)
        except Exception:
            client.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def _check_encoder(self):
        records, _ = self.client.scroll(self.collection, limit=1, with_vectors=False)
        if records and records[0].payload.get("encoder") != self.encoder.signature:
            raise ValueError("embedding model/version differs; use a new collection and reindex")

    def _filter(self, **values):
        return self.models.Filter(must=[
            self.models.FieldCondition(key=k, match=self.models.MatchValue(value=v))
            for k, v in values.items() if v is not None])

    def replace_document(self, project, revision, document, document_hash, records):
        """Upsert before deleting stale pages; failed embedding leaves old index intact."""
        self._check_encoder()
        if not all((project, revision, document, document_hash)):
            raise ValueError("project, revision, document and hash must be nonempty")
        records = list(records)
        # Validate the entire replacement before the first write.
        seen = set()
        for r in records:
            if r.page < 1 or r.page in seen:
                raise ValueError("page numbers must be unique and start at one")
            seen.add(r.page)
            for name, vector, size in (("image", r.image_vector, self.encoder.dimension_image),
                                       ("text", r.text_vector, self.encoder.dimension_text)):
                if name == "text" and not vector:
                    continue
                if len(vector) != size or not np.isfinite(vector).all():
                    raise ValueError(f"invalid {name} embedding")
        ids, points = [], []
        for r in records:
            if r.page < 1:
                raise ValueError("page numbers start at one")
            identity = "\0".join((project, revision, document, document_hash, str(r.page)))
            point_id = str(uuid5(NAMESPACE_URL, identity))
            vectors = {"image": r.image_vector}
            if r.text_vector:
                vectors["text"] = r.text_vector
            payload = dict(project=project, revision=revision, document=document,
                           document_hash=document_hash, page=r.page, sheet=r.sheet,
                           title=r.title, text=r.text, landscape=r.landscape,
                           encoder=self.encoder.signature)
            points.append(self.models.PointStruct(id=point_id, vector=vectors, payload=payload))
            ids.append(point_id)
        if points:
            self.client.upsert(self.collection, points, wait=True)
        stale = self._filter(project=project, revision=revision, document=document)
        if ids:
            stale.must_not = [self.models.HasIdCondition(has_id=ids)]
        self.client.delete(self.collection, points_selector=stale, wait=True)
        return len(ids)

    def index_pdf(self, path, *, project, revision, document=None):
        content_hash = file_hash(path)
        with fitz.open(path) as doc:
            # Finish extraction before any DB writes; e.g. missing Tesseract must
            # not erase the previously indexed revision of this document.
            records = [page_record(p, i + 1, self.encoder) for i, p in enumerate(doc)]
        if file_hash(path) != content_hash:
            raise ValueError("PDF changed while indexing; retry with a stable file")
        return self.replace_document(project, revision, document or Path(path).stem,
                                     content_hash, records)

    def _query(self, vector, using, filters, limit):
        self._check_encoder()
        if not filters.get("project") or not filters.get("revision"):
            raise ValueError("project and revision are required")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not vector:
            return []
        points = self.client.query_points(self.collection, query=vector, using=using,
                                         query_filter=self._filter(**filters), limit=limit).points
        return [dict(p.payload, score=float(p.score)) for p in points]

    def search_text(self, text, *, project, revision, document=None, sheet=None, limit=5):
        if not text.strip():
            return self._query([], "text", dict(project=project, revision=revision), limit)
        if isinstance(self.encoder, LocalEncoder):
            vector = self.encoder.text(text, query=True)
        else:
            vector = self.encoder.text(text)
        return self._query(vector, "text", dict(project=project, revision=revision,
                           document=document, sheet=sheet), limit)

    def search_page(self, record, *, project, revision, document=None, document_hash=None, sheet=None, limit=5):
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        filters = dict(project=project, revision=revision, document=document,
                       document_hash=document_hash, sheet=sheet, landscape=record.landscape)
        # Both modalities are searched by Qdrant, then combined by reciprocal
        # rank fusion. Rank fusion avoids treating CLIP/BGE scores as calibrated
        # probabilities. Retrieve more than the displayed count before fusing.
        image = self._query(record.image_vector, "image", filters, min(100, limit * 4))
        text = self._query(record.text_vector, "text", filters, min(100, limit * 4))
        fused = {}
        for modality, hits in (("image", image), ("text", text)):
            for rank, hit in enumerate(hits, 1):
                key = (hit["document"], hit["page"])
                row = fused.setdefault(key, dict(hit, score=0.0))
                row["score"] += 1.0 / (60 + rank)
                row[modality + "_score"] = hit["score"]
        return sorted(fused.values(), key=lambda h: (-h["score"], h["document"], h["page"]))[:limit]
