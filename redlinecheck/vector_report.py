"""Human-readable candidate evidence in the existing static review report."""
import html
import json
from pathlib import Path

import fitz


def candidate_section(report, out_dir):
    rows = report.vector_candidates
    if rows:
        from .vector_search import file_hash
        if (file_hash(report.redlined_path) != report.vector_hashes["source"] or
                file_hash(report.final_path) != report.vector_hashes["target"]):
            raise ValueError("PDF changed before candidate previews; rerun comparison")
    # Always overwrite this file so a reused output folder cannot show old matches.
    root = Path(out_dir)
    (root / "page_candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    import shutil
    folder = root / "candidates"
    shutil.rmtree(folder, ignore_errors=True)
    if not rows:
        return ""
    folder.mkdir()
    result = ['<section><h2>Page match candidates</h2><p class="desc">'
              'Review before comparing. These are suggestions for uncertain layout matches. '
              'A similarity score is not proof that two drawings match. '
              'Confirm a choice with <code>--page-pair SOURCE:TARGET</code> and run again.</p>']
    with fitz.open(report.redlined_path) as source, fitz.open(report.final_path) as target:
        def preview(doc, page, prefix):
            name = f"{prefix}-{page}.png"
            p = doc[page - 1]
            zoom = min(1., 850. / max(p.rect.width, p.rect.height))
            p.get_pixmap(matrix=fitz.Matrix(zoom, zoom), annots=False).save(folder / name)
            return f"candidates/{name}"
        for row in rows:
            page = row["source_page"]
            src = preview(source, page, "source")
            paired = row.get("aligned_page")
            result.append(f'<div class="row"><h3>Source page {page}</h3><p>'
                          f'{html.escape(row["reason"])}. Current layout pair: {paired or "none"}.</p>'
                          f'<figure><img src="{src}" alt="Source drawing page {page}"></figure>')
            if not row["matches"]:
                result.append('<p>No candidate in the selected revision. Review the full set manually.</p>')
            for rank, hit in enumerate(row["matches"], 1):
                target_page = int(hit["page"])
                dst = preview(target, target_page, "target")
                label = html.escape(f'{hit.get("sheet", "")} {hit.get("title", "")}')
                result.append(f'<h4>Candidate {rank}: revised page {target_page}, {label}</h4>'
                              f'<p>Combined rank score: {hit["score"]:.5f}. '
                              f'Confirm with <code>--page-pair {page}:{target_page}</code>.</p>'
                              f'<figure><img src="{dst}" alt="Candidate revised page {target_page}"></figure>')
            result.append('</div>')
    return "".join(result) + '</section>'
