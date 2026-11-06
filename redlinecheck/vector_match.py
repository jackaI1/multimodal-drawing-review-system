"""Retrieve review candidates without changing structural alignment decisions."""
from __future__ import annotations

from .vector_search import page_record


def uncertain_pages(match, similarities, *, minimum=.65, margin=.03):
    uncertain = set(match.extra_redlined)
    for source, target in match.pairs:
        score = match.scores.get((source, target), 0.)
        alternatives = [float(s) for j, s in enumerate(similarities[source]) if j != target]
        if score < minimum or (alternatives and score - max(alternatives) < margin):
            uncertain.add(source)
    return sorted(uncertain)


def fallback_candidates(source_doc, match, similarities, index, *, project,
                        revision, document, document_hash, extract=page_record, exclude_sources=()):
    paired = dict(match.pairs)
    rows = []
    for source in uncertain_pages(match, similarities):
        if source in exclude_sources:
            continue
        record = extract(source_doc[source], source + 1, index.encoder, annots=False)
        hits = index.search_page(record, project=project, revision=revision,
                                 document=document, document_hash=document_hash, limit=3)
        rows.append(dict(source_page=source + 1,
                         aligned_page=paired[source] + 1 if source in paired else None,
                         reason="unmatched" if source not in paired else "weak or ambiguous layout match",
                         matches=hits))
    return rows


def confirm_pairs(match, overrides, *, source_count, target_count):
    """Apply explicit reviewer choices; displace collisions to manual review."""
    from .prematch import MatchResult
    sources, targets = set(), set()
    for source, target in overrides:
        if not 0 <= source < source_count or not 0 <= target < target_count:
            raise ValueError("confirmed page pair is outside the PDF page range")
        if source in sources or target in targets:
            raise ValueError("confirmed page pairs must use unique source and target pages")
        sources.add(source)
        targets.add(target)
    pairs = sorted([(a, b) for a, b in match.pairs if a not in sources and b not in targets] + list(overrides))
    return MatchResult(pairs=pairs,
                       extra_redlined=sorted(set(range(source_count)) - {a for a, _ in pairs}),
                       extra_revised=sorted(set(range(target_count)) - {b for _, b in pairs}),
                       scores={pair: match.scores[pair] for pair in pairs if pair in match.scores})
