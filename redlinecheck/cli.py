"""Command-line entry point.

    python -m redlinecheck REDLINED.pdf REVISED.pdf -o report
    python comparator.py    REDLINED.pdf REVISED.pdf -o report
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import Config
from . import pipeline, report as report_mod


def _parse_zone(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ignore-zone must be x0,y0,x1,y1 in 0..1")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="redlinecheck",
        description="Compare a redlined PDF against the architect's revised PDF and "
                    "flag missed redlines, unexpected changes, and likely captures.")
    p.add_argument("redlined", help="your marked-up PDF (drawing + red redlines)")
    p.add_argument("revised", help="the architect's revised PDF that came back")
    p.add_argument("-o", "--out", default="report", help="output folder (default: report)")
    p.add_argument("--dpi", default="auto",
                   help="analysis DPI, or 'auto' (default) to pick it from the drawing's "
                        "text size; pass a number (e.g. 300, 500) to force it")
    p.add_argument("--ignore-zone", action="append", type=_parse_zone, default=None,
                   metavar="x0,y0,x1,y1",
                   help="page-fraction rect to ignore (e.g. titleblock). Repeatable.")
    p.add_argument("--no-default-ignore", action="store_true",
                   help="don't ignore the default bottom-right titleblock zone")
    p.add_argument("--ocr", action="store_true",
                   help="use local Tesseract OCR to verify spec changes (must be installed)")
    p.add_argument("--vlm", action="store_true",
                   help="use a local on-device VLM (Ollama) to read each flagged "
                        "target and confirm captures the pixel diff/OCR can't. Reads "
                        "before+after crops; fully local. `ollama pull moondream`")
    p.add_argument("--vlm-model", default=None,
                   help=f"override the Ollama vision model (default: {Config().vlm_model})")
    p.add_argument("--hexnotes", action="store_true",
                   help="also flag in-place edits inside closed callouts (e.g. a hexnote "
                        "8c->8); off by default, noisier on heavily re-plotted sheets")
    return p


def make_config(args) -> Config:
    if str(args.dpi).lower() == "auto":
        cfg = Config(use_ocr=args.ocr)          # auto_dpi defaults to True
    else:
        cfg = Config(analysis_dpi=int(args.dpi), use_ocr=args.ocr)
        cfg.auto_dpi = False
    cfg.use_vlm = args.vlm
    if args.vlm_model:
        cfg.vlm_model = args.vlm_model
    if args.hexnotes:
        cfg.detect_symbols = True
    zones = []
    if not args.no_default_ignore:
        zones.extend(Config().ignore_zones)
    if args.ignore_zone:
        zones.extend(args.ignore_zone)
    cfg.ignore_zones = zones
    return cfg


def _make_verifier(cfg: Config, args):
    """Build the verification callable. When both OCR and the VLM are on they are
    CHAINED: cheap OCR settles the easy captures, the VLM only works the rest."""
    verifiers = []
    if cfg.use_ocr:
        from . import ocr
        verifiers.append(ocr.make_verifier(cfg))
    if cfg.use_vlm:
        from . import vlm
        verifiers.append(vlm.make_verifier(cfg))
    if not verifiers:
        return None
    if len(verifiers) == 1:
        return verifiers[0]
    from .vlm import chained
    return chained(*verifiers)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.redlined, args.revised):
        if not os.path.isfile(path):
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
    cfg = make_config(args)

    try:
        verifier = _make_verifier(cfg, args)
    except Exception as e:  # missing optional dep / key
        print(f"error: could not enable verifier: {e}", file=sys.stderr)
        return 2

    print("Comparing (this renders every page, ~a few seconds per page)...", file=sys.stderr)
    rep = pipeline.run(args.redlined, args.revised, cfg, verifier=verifier)
    os.makedirs(args.out, exist_ok=True)
    counts = report_mod.build_report(rep, args.out)["counts"]

    print("\n==================  SUMMARY  ==================")
    print(f"  Possible misses (review):   {counts['possible_misses']}")
    print(f"  Unexpected changes:         {counts['unexpected']}")
    print(f"  To verify (coordination):   {counts['to_verify']}")
    print(f"  Addressed (verified):       {counts['likely_addressed']}")
    print(f"  Markers (info):             {counts['markers']}")
    print(f"  Extra / inserted pages:     {counts.get('extra_pages', 0)}")
    print("==============================================")
    print(f"\nReport written to: {os.path.abspath(args.out)}")
    print("  - summary.html   (open this)")
    print("  - annotated.pdf")
    print("  - findings.csv")
    if not cfg.use_ocr and not cfg.use_vlm:
        print("\nNote: run with --ocr (Tesseract) or --vlm (local Ollama vision "
              "model) to turn 'possible misses' into confident captured/missed "
                        "verdicts for beam-size and other in-place text edits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
