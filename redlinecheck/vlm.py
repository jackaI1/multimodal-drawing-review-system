"""Optional local VLM verification tier (Ollama). OFF by default.

Fully local — talks only to an Ollama server on localhost; nothing leaves your
machine. Chained after OCR, so it only runs on spec redlines OCR could not confirm.
For each, it hands the model two small crops — the BEFORE (your red markup) and the
AFTER (the architect's revision of the same spot) — and asks whether the instruction
was applied. Reading the actual text the way an engineer would, it confirms captures
the pixel diff and even Tesseract can't (a typed-in note, an in-place beam swap).

Conservative by policy: only an explicit "applied: yes" WITH a read-back value turns
a finding green (CAPTURED); anything ambiguous returns None so the OCR/pixel verdict
stands — the VLM never over-claims.

Requests are STREAMED so that a timed-out call cancels the generation instead of
leaving it running and pinning the CPU. Per-call cost is bounded by `vlm_num_predict`.
Practicality note: a 7B model on a CPU is minutes-per-call; the tier is meant for
machines with a capable GPU (where Ollama offloads and the CPU stays free).

Setup (one time):
    winget install Ollama.Ollama        # or https://ollama.com/download
    ollama pull moondream              # fast default
    ollama pull qwen2.5vl:7b           # slower, more accurate (needs a GPU)
The server (http://localhost:11434) is auto-detected.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .config import Config
from .align import PageData
from .extract import Comment
from . import classify as C

# Instruction placeholders that carry no real value to verify.
_SKIP_TEXT = {"(marked region)", "(redline markup)", "(unlabeled markup)", ""}

_PROMPT = (
    "You are checking whether an architect applied a structural engineer's redline.\n"
    "The engineer's instruction (their red markup) reads:\n  \"{instr}\"\n"
    "Image 1 = BEFORE: the engineer's marked-up drawing (the red note/leader/cloud "
    "shows what they asked for and where).\n"
    "Image 2 = AFTER: the architect's revised drawing of the SAME area.\n\n"
    "Did the architect APPLY this instruction in the AFTER image? Look for the "
    "requested value/label/note/change actually present at that location in AFTER.\n\n"
    "Reply with ONLY a JSON object:\n"
    '{{"applied": "yes" | "no" | "uncertain", '
    '"value": "<exact text/value you can read at the target in AFTER, else empty>", '
    '"reason": "<short>"}}\n'
    "Rules: answer \"yes\" ONLY if you can actually SEE the instructed value/change "
    "in AFTER. If you cannot read it clearly, or it is ambiguous, answer "
    "\"uncertain\". Answer \"no\" only if the area is clearly unchanged and the "
    "instruction was not carried out. Do not guess."
)


def check_availability(cfg: Optional[Config] = None) -> Tuple[bool, str]:
    """(ok, message) — is an Ollama server reachable and is the model present?"""
    cfg = cfg or Config()
    base = cfg.vlm_endpoint.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/api/tags", timeout=4) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return (False, f"Ollama not reachable at {base} ({e.reason}). Install it "
                       "(winget install Ollama.Ollama) and `ollama serve`.")
    except Exception as e:
        return False, f"Ollama at {base} did not respond as expected: {e}"
    models = [m.get("name", "") for m in data.get("models", [])]
    want = cfg.vlm_model
    # accept an exact match or the same model with/without a :tag
    base_want = want.split(":")[0]
    have = any(m == want or m.split(":")[0] == base_want for m in models)
    if not have:
        avail = ", ".join(models) if models else "none"
        return (False, f"Ollama is running but model '{want}' isn't pulled "
                       f"(have: {avail}). Run: ollama pull {want}")
    return True, f"Ollama VLM ready: {want} (local, {base})"


def _num_thread(cfg: Config):
    """How many CPU threads Ollama may use — leaves `vlm_leave_free_cores` free so the
    machine stays usable during a (CPU-bound) VLM call. None = let Ollama use all."""
    free = getattr(cfg, "vlm_leave_free_cores", 0) or 0
    if free <= 0:
        return None
    try:
        cores = os.cpu_count() or 0
    except Exception:
        return None
    if not cores:
        return None
    return max(1, cores - free)


def _options(cfg: Config, num_gpu):
    opts = {"temperature": 0}
    np_cap = getattr(cfg, "vlm_num_predict", 0) or 0
    if np_cap > 0:
        opts["num_predict"] = np_cap        # bound output length -> bound call time
    if num_gpu is not None:
        opts["num_gpu"] = num_gpu
    nt = _num_thread(cfg)
    if nt is not None:
        opts["num_thread"] = nt
    return opts


def _b64_png(crop: np.ndarray, max_side: int) -> str:
    img = Image.fromarray(crop).convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _crop(rgb: np.ndarray, box, pad: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0 = max(0, int(box[0] - pad)); y0 = max(0, int(box[1] - pad))
    x1 = min(w, int(box[2] + pad)); y1 = min(h, int(box[3] + pad))
    if x1 <= x0 or y1 <= y0:
        return rgb[:1, :1]
    return rgb[y0:y1, x0:x1]


def _chat(cfg: Config, instr: str, before_b64: str, after_b64: str, num_gpu) -> Optional[dict]:
    """One STREAMING /api/chat call, returning the parsed JSON answer (or None).

    Streaming is deliberate and important: with a non-streaming request, if the client
    times out and disconnects, Ollama keeps generating to completion — the orphaned
    generation pins the CPU, and on a slow machine these pile up until it grinds to a
    halt (the reported 'timed out but kept running and lagged'). Streaming lets Ollama
    notice the disconnect on its next token write and CANCEL, so an aborted call frees
    the machine. Raises urllib.error.HTTPError on a connect-time server error (e.g. the
    too-small/old-GPU 500) so the caller's CPU fallback can kick in."""
    payload = {
        "model": cfg.vlm_model,
        "messages": [{"role": "user", "content": _PROMPT.format(instr=instr),
                      "images": [before_b64, after_b64]}],
        "stream": True,
        "format": "json",
        "options": _options(cfg, num_gpu),
    }
    req = urllib.request.Request(
        cfg.vlm_endpoint.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    deadline = time.monotonic() + cfg.vlm_timeout
    parts = []
    completed = False
    resp = urllib.request.urlopen(req, timeout=cfg.vlm_timeout)
    try:
        for raw in resp:
            if time.monotonic() > deadline:
                break                       # close() below -> Ollama sees the
                                            # disconnect and cancels the generation
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("error"):
                break
            parts.append((obj.get("message") or {}).get("content", ""))
            if obj.get("done"):
                completed = True
                break
    finally:
        resp.close()
    return _parse("".join(parts)) if completed else None


def _parse(content: str) -> Optional[dict]:
    """Pull the JSON object out of the model's reply (tolerant of stray prose)."""
    content = (content or "").strip()
    if not content:
        return None
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    i, j = content.find("{"), content.rfind("}")
    if 0 <= i < j:
        try:
            parsed = json.loads(content[i:j + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def make_verifier(cfg: Config):
    ok, msg = check_availability(cfg)
    if not ok:
        raise RuntimeError(msg)
    pad = cfg.px(cfg.vlm_crop_pad_pt)
    # Remembered GPU setting: starts at the configured value; if a request crashes the
    # runner with a 500 (tiny/old GPU offload), we drop to CPU (0) for the rest of the
    # run so we don't keep paying for the doomed first attempt.
    state = {"num_gpu": cfg.vlm_num_gpu}

    def _ask(instr: str, before_b64: str, after_b64: str) -> Optional[dict]:
        try:
            return _chat(cfg, instr, before_b64, after_b64, state["num_gpu"])
        except urllib.error.HTTPError as e:
            # 500 == the model runner crashed (commonly a too-small/old GPU). Retry
            # once on pure CPU and stick with CPU for the rest of the run.
            if e.code == 500 and state["num_gpu"] != 0:
                state["num_gpu"] = 0
                try:
                    return _chat(cfg, instr, before_b64, after_b64, 0)
                except Exception:
                    return None
            return None
        except Exception:
            return None

    def verify(page: PageData, comment: Comment, box, changed: bool = False) -> Optional[tuple]:
        instr = (comment.text or "").strip()
        if instr in _SKIP_TEXT:
            return None
        # Gate (config): only spend a VLM call where the pixel diff actually saw a
        # change at this change-expecting target. Keeps the bulk run bounded.
        if getattr(cfg, "vlm_require_pixel_change", False) and not changed:
            return None
        before = _b64_png(_crop(page.redlined_rgb, box, pad), cfg.vlm_max_side_px)
        after = _b64_png(_crop(page.final_rgb, box, pad), cfg.vlm_max_side_px)
        ans = _ask(instr, before, after)
        if not ans:
            return None
        applied = str(ans.get("applied", "")).strip().lower()
        value = str(ans.get("value", "") or "").strip()
        reason = str(ans.get("reason", "") or "").strip()
        if applied == "yes":
            # Only green when the model actually read a value back; a bare "yes"
            # with nothing read is treated as uncertain (never over-claim).
            if value:
                from .ocr import _instruction_tokens, _words, _norm, _match
                requested = _instruction_tokens(instr)
                if requested and not all(_match(t, _words(value), _norm(value), 0)
                                         for t in requested):
                    return (C.CHANGED_OTHER, value,
                            "VLM read-back differs from the instructed value — review")
                return (C.CAPTURED, value,
                        f"VLM confirmed the instruction was applied: '{value}'"
                        + (f" — {reason}" if reason else ""))
            return None
        if applied == "no":
            return (C.CHANGED_OTHER if changed else C.MISSED, value,
                    "VLM: the instruction does not appear applied at the target"
                    + (f" — {reason}" if reason else ""))
        return None  # uncertain -> keep the pixel/OCR verdict, leave for manual

    return verify


def chained(*verifiers):
    """Combine verifiers (e.g. OCR then VLM): take the first DEFINITE verdict.
    A cheap verifier (OCR) can settle the easy captures; the VLM only runs on
    what's left. A verifier returning None (no opinion) defers to the next."""
    verifiers = [v for v in verifiers if v is not None]
    if not verifiers:
        return None
    if len(verifiers) == 1:
        return verifiers[0]

    def verify(page, comment, box, changed: bool = False):
        last = None
        for v in verifiers:
            res = v(page, comment, box, changed)
            if res is None:
                continue
            status = res[0]
            if status == C.CAPTURED:      # a confident capture wins immediately
                return res
            last = last or res            # remember a MISSED, but let a later
        return last                        # verifier upgrade it to CAPTURED first

    return verify
