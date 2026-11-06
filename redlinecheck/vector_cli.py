"""Opt-in drawing search commands: python -m redlinecheck.vector_cli --help."""
from __future__ import annotations
import argparse
import json
import os
import sys

import fitz

from .vector_search import DrawingIndex, page_record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Index and search drawing pages in a local Qdrant database.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "search", "match"):
        p = commands.add_parser(name)
        p.add_argument("--project", required=True)
        p.add_argument("--revision", required=True, help="revision to index or search")
        p.add_argument("--document", help="document label within this revision")
        p.add_argument("--url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
        p.add_argument("--cache-dir", default=os.getenv("FASTEMBED_CACHE_PATH"))
        p.add_argument("--collection", default="drawing_pages_v1")
        if name in ("index", "match"):
            p.add_argument("pdf")
        if name == "search":
            p.add_argument("--text", required=True)
        if name != "index":
            p.add_argument("--sheet")
            p.add_argument("--limit", type=int, default=5)
        if name == "match":
            p.add_argument("--page", required=True, type=int, help="one-based source page number")
    args = parser.parse_args(argv)
    try:
        with DrawingIndex.connect(url=args.url, cache_dir=args.cache_dir, collection=args.collection) as index:
            if args.command == "index":
                data = {"indexed_pages": index.index_pdf(args.pdf, project=args.project,
                        revision=args.revision, document=args.document)}
            elif args.command == "search":
                data = index.search_text(args.text, project=args.project, revision=args.revision,
                                        document=args.document, sheet=args.sheet, limit=args.limit)
            else:
                with fitz.open(args.pdf) as doc:
                    if not 1 <= args.page <= len(doc):
                        raise ValueError("page is outside the source PDF")
                    record = page_record(doc[args.page - 1], args.page, index.encoder)
                data = index.search_page(record, project=args.project, revision=args.revision,
                                         document=args.document, sheet=args.sheet, limit=args.limit)
        if isinstance(data, list):
            data = [dict(row, text=row.get("text", "")[:300]) for row in data]
        print(json.dumps(data, indent=2))
        return 0
    except Exception as error:
        print(f"error: vector search: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
