"""redlinecheck — compare a redlined PDF against an architect's revised PDF.

Flags redline comments the architect MISSED, changes they made that were
UNEXPECTED (not marked up), and confirms comments that were CAPTURED.

Core engine is fully local (PyMuPDF + numpy + scipy + pillow). Optional OCR
(Tesseract) and local vision-model (Ollama) tiers improve verification of
in-place text edits.
"""
from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
