"""Streamlit GUI for redlinecheck.

    streamlit run app.py

Upload your redlined PDF and the architect's revised PDF, compare them, view the
findings (with before/after crops) in the browser, and download the report.
"""
from __future__ import annotations

# Cap math-library threads BEFORE numpy/scipy/fitz load, so a big comparison leaves
# a CPU core free and the machine stays usable while it runs.
from redlinecheck._compute import limit_threads
limit_threads(leave_free=1)

import io
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from html import escape as html_escape

import streamlit as st

from redlinecheck.config import Config
from redlinecheck import pipeline, ocr as ocr_mod, vlm as vlm_mod
from redlinecheck import report as report_mod
from redlinecheck.report import STYLE
from redlinecheck.classify import (
    MISSED, NO_CHANGE_DETECTED, UNEXPECTED, COORDINATION, CHANGED_OTHER,
    CHANGED_AT_TARGET, CAPTURED, COPY_REF, COPY_CHANGE, MARKED_REGION,
)

st.set_page_config(page_title="Redline Check", page_icon="📐", layout="wide")

# Colour the review buttons (green = mark addressed, yellow = report error) and
# show the "marked addressed" toast green in the top-right. Targets the per-widget
# `st-key-<key>` classes Streamlit emits, so only these buttons are styled.
st.markdown("""
<style>
div[class*="st-key-addr_"] button{background-color:#16a34a;border-color:#16a34a;color:#fff;}
div[class*="st-key-addr_"] button:hover{background-color:#15803d;border-color:#15803d;color:#fff;}
div[class*="st-key-addr_"] button:active,
div[class*="st-key-addr_"] button:focus{background-color:#15803d;border-color:#15803d;color:#fff;}
div[class*="st-key-err_"] button{background-color:#eab308;border-color:#eab308;color:#1b1f24;}
div[class*="st-key-err_"] button:hover{background-color:#ca8a04;border-color:#ca8a04;color:#1b1f24;}
div[class*="st-key-err_"] button:disabled{background-color:#9ca3af;border-color:#9ca3af;color:#fff;opacity:1;}
div[class*="st-key-zoom_"] button{background-color:#334155;border-color:#334155;color:#fff;}
div[class*="st-key-zoom_"] button:hover{background-color:#1e293b;border-color:#1e293b;color:#fff;}
div[data-testid="stToastContainer"]{top:3.5rem;bottom:auto;}
div[data-testid="stToast"]{background-color:#16a34a;color:#fff;}
div[data-testid="stToast"] *{color:#fff;}
</style>
""", unsafe_allow_html=True)

MISS = {MISSED, NO_CHANGE_DETECTED}
ADDRESSED = {CAPTURED}                                  # green = verified only
OTHER = {COORDINATION, CHANGED_OTHER, CHANGED_AT_TARGET, COPY_REF, COPY_CHANGE,
         MARKED_REGION}

# Local folder for "report classification error" exports (user-created).
ERROR_DIR = os.path.join(os.getcwd(), "classification errors")


@st.cache_data(show_spinner=False, ttl=30)
def _ocr_status():
    return ocr_mod.check_availability()


@st.cache_data(show_spinner=False, ttl=30)
def _vlm_status(model: str):
    # Cached so the localhost probe doesn't run (and block up to a few seconds when
    # Ollama isn't running) on every sidebar interaction. Re-checked every 30s.
    cfg = Config()
    cfg.vlm_model = model
    return vlm_mod.check_availability(cfg)


def _rid(r):
    """A stable, GLOBALLY-UNIQUE row id. The finding number (`no`) repeats across
    drawings in a project (every sheet has a 'U1', a '2.0'…), so keying session state
    or widget keys on `no` alone collides — marking one row addressed would silently
    affect another drawing's same-numbered row. The drawing-scoped `gpidx` makes it
    unique."""
    return f"{r.get('gpidx', '')}|{r['no']}"


def _eff_status(r):
    """A row's status after any one-click reclassification this session."""
    return st.session_state.get("overrides", {}).get(_rid(r), r["status"])


def _reclassify(rid):
    # on_click callback (fragment-scoped) so only this row reruns (no full reload).
    # The toast is emitted later in the fragment body — calling st.toast *inside* a
    # callback during a fragment rerun is unsupported and can shove elements to the
    # top of the page, so we just stash the intent here.
    st.session_state.setdefault("overrides", {})[rid] = CAPTURED
    st.session_state["_toast"] = ("addr", rid, "")


def _report_error(rid, r, out_dir):
    folder = _save_classification_error(r, out_dir)
    st.session_state.setdefault("reported", set()).add(rid)
    st.session_state["_toast"] = ("err", rid, os.path.basename(folder))


def _toggle_zoom(rid):
    # on_click callback (fragment-scoped): flips this row's zoom flag, so only the
    # results fragment reruns and swaps in the pre-rendered wide crops — no full-page
    # reload, no dim, no 10-second wait.
    z = st.session_state.setdefault("zoom", {})
    z[rid] = not z.get(rid, False)


def _save_classification_error(r, out_dir):
    """Save the before/after crops, text and all settings to ./classification errors
    so the model can be refined from real mistakes."""
    os.makedirs(ERROR_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(ERROR_DIR, f"{stamp}_{str(r['no']).replace('.', '-')}")
    os.makedirs(folder, exist_ok=True)
    for side in ("before", "after"):
        src = os.path.join(out_dir, r.get(side, ""))
        if r.get(side) and os.path.exists(src):
            shutil.copy(src, os.path.join(folder, f"{side}.png"))
    meta = {
        "finding": {k: r[k] for k in r if k not in ("before", "after")},
        "reported_as_wrong_status": r["status"],
        "should_be": "addressed (verified)",
        "settings": st.session_state.get("run_settings", {}),
        "reported_at": stamp,
    }
    with open(os.path.join(folder, "info.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return folder


# ───────────────────────── helpers ─────────────────────────
def _pill(status: str) -> str:
    rgb, label, _ = STYLE[status]
    color = "#%02x%02x%02x" % rgb
    return (f'<span style="background:{color};color:#fff;border-radius:4px;'
            f'padding:1px 8px;font-size:11px;font-weight:600">{label}</span>')


@st.cache_data(show_spinner=False)
def _zip_dir(out_dir: str) -> bytes:
    # Cached on out_dir (unique per run) so it's built ONCE, not on every rerun —
    # zipping all the crops each time was the main cause of the slow reload.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, out_dir))
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _write_combined_csv(rows, out_dir):
    """One findings.csv across the whole run (all drawings), with a `source` column."""
    import csv
    path = os.path.join(out_dir, "findings_all.csv")
    if os.path.exists(path):
        return
    cols = ["source", "no", "page", "kind", "status", "text", "verified_value",
            "confidence", "note", "activity"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _show_row(r: dict, out_dir: str):
    # Rendered inside the results fragment, so a button click reruns just the
    # results (counts/sheets update, the row leaves its section) — not the whole app.
    rid = _rid(r)
    reported = st.session_state.setdefault("reported", set())
    eff = _eff_status(r)
    src = r.get("source")
    where = (f"{html_escape(src)} · " if src else "") + f"sheet {r['page']} · {r['confidence']}"
    head = f'<b>{html_escape(str(r["no"]))}</b> &nbsp; {_pill(eff)} &nbsp; ' \
           f'<span style="color:#789;font-size:12px">{where}</span>'
    st.markdown(head, unsafe_allow_html=True)
    txt = r["text"] or "*(unlabeled markup)*"
    if r["verified_value"]:
        txt += f" &nbsp;→&nbsp; **{r['verified_value']}**"
    st.markdown(txt)
    if r["note"]:
        st.caption(r["note"])

    zoomed = st.session_state.get("zoom", {}).get(rid, False)

    def _img(side):
        # zoomed -> the pre-rendered wide crop; fall back to the normal crop if a
        # wide one wasn't produced (e.g. an older run).
        rel = r.get(f"{side}_wide") if zoomed else r.get(side)
        if not rel or not os.path.exists(os.path.join(out_dir, rel)):
            rel = r.get(side)
        return os.path.join(out_dir, rel)

    zcap = " · zoomed out" if zoomed else ""
    c1, c2 = st.columns(2)
    c1.image(_img("before"), caption="original (your markup)" + zcap, width="stretch")
    c2.image(_img("after"), caption="architect's revision" + zcap, width="stretch")

    b1, b2, b3 = st.columns(3)
    if eff != CAPTURED:
        b1.button("✓ Mark addressed", key=f"addr_{rid}",
                  help="Reclassify as Addressed (verified) and remove it from this list.",
                  on_click=_reclassify, args=(rid,), width="stretch")
    b3.button(("🔎 Zoom in" if zoomed else "🔍 Zoom out"), key=f"zoom_{rid}",
              help="Show ~3× more of the sheet around this finding so you can see "
                   "where your markup is. Instant — only this results area updates.",
              on_click=_toggle_zoom, args=(rid,), width="stretch")
    if rid in reported:
        b2.button("✓ Reported", key=f"err_{rid}", disabled=True, width="stretch",
                  help="Already exported to the 'classification errors' folder.")
    else:
        b2.button("⚠ Report error", key=f"err_{rid}",
                  help="Save the crops, text and settings to the local 'classification "
                       "errors' folder to refine the tool.",
                  on_click=_report_error, args=(rid, r, out_dir), width="stretch")
    st.divider()


def _ov_key(r):
    """Overview dict key for a row: a global 'pairIdx:pageIdx' in project mode, else
    the plain page index."""
    return r.get("gpidx", r.get("pidx"))


def _render_section(rows, statuses, overviews, out_dir, empty_msg, expand_small=True):
    sel = [r for r in rows if _eff_status(r) in statuses]
    if not sel:
        st.success(empty_msg)
        return
    # Group by drawing (project mode) then sheet, so a whole project reads cleanly.
    groups = {}
    for r in sel:
        groups.setdefault((r.get("source", ""), r["page"], _ov_key(r)), []).append(r)
    for (src, pg, ovk) in sorted(groups, key=lambda k: -len(groups[k])):
        prows = sorted(groups[(src, pg, ovk)], key=lambda r: -r["mag"])
        title = (f"{src} — sheet {pg}" if src else f"Sheet {pg}") + f" — {len(prows)} item(s)"
        with st.expander(title, expanded=(expand_small and len(prows) <= 20)):
            ov = overviews.get(ovk)
            if ov:
                st.image(os.path.join(out_dir, ov), width="stretch",
                         caption=f"{title} — all findings boxed")
            for r in prows:                       # show everything, no cap
                _show_row(r, out_dir)


def _save_uploads(files, dirpath):
    """Write uploaded files to disk, keeping only readable PDFs. Anything else (a
    stray .docx/.dwg dragged in with a folder, a corrupt or empty PDF) is silently
    dropped — never an error, never a halt. Returns (valid_paths, skipped[(name,why)])."""
    from redlinecheck import project as proj
    os.makedirs(dirpath, exist_ok=True)
    valid, skipped = [], []
    for f in files:
        name = os.path.basename(f.name)
        dest = os.path.join(dirpath, name)
        stem, ext = os.path.splitext(dest)
        k = 1
        while os.path.exists(dest):           # avoid same-name collisions
            dest = f"{stem}__{k}{ext}"; k += 1
        try:
            with open(dest, "wb") as fh:
                fh.write(f.getvalue())
        except Exception as e:
            skipped.append((name, f"could not save ({e})")); continue
        ok, why = proj.is_valid_pdf(dest)
        if ok:
            valid.append(dest)
        else:
            skipped.append((name, why))
            try:
                os.remove(dest)               # delete the unsupported file
            except OSError:
                pass
    return valid, skipped


def _run(redlined_files, revised_files, cfg, use_ocr, use_vlm=False):
    """Save uploads (dropping junk), then compare. One drawing per side -> a single
    comparison; several -> a whole-PROJECT comparison (auto-matched pairwise).
    Returns (result, out_dir)."""
    from redlinecheck import project as proj
    work = tempfile.mkdtemp(prefix="redlinecheck_")
    out = os.path.join(work, "report")
    rvalid, rskip = _save_uploads(redlined_files, os.path.join(work, "redlined"))
    vvalid, vskip = _save_uploads(revised_files, os.path.join(work, "revised"))
    skipped = [{"name": n, "why": w} for n, w in (rskip + vskip)]
    if not rvalid or not vvalid:
        raise RuntimeError("no readable PDF drawings on one side after filtering "
                           f"(redlined: {len(rvalid)}, revised: {len(vvalid)}).")

    # Build the verification tier: OCR (fast) and/or the local VLM, chained so OCR
    # settles the easy reads and the VLM only works what's left, including
    # in-place edits with no density change.
    verifiers = []
    if use_ocr:
        cfg.use_ocr = True
        verifiers.append(ocr_mod.make_verifier(cfg))
    if use_vlm:
        cfg.use_vlm = True
        verifiers.append(vlm_mod.make_verifier(cfg))   # raises if Ollama unavailable
    verifier = (verifiers[0] if len(verifiers) == 1
                else vlm_mod.chained(*verifiers) if verifiers else None)

    bar = st.progress(0.0, text="Rendering and comparing…")
    if len(rvalid) == 1 and len(vvalid) == 1:
        def cb(done, total):
            bar.progress(done / max(total, 1), text=f"Comparing page {done} of {total}…")
        rep = pipeline.run(rvalid[0], vvalid[0], cfg, verifier=verifier,
                           progress=False, progress_cb=cb)
        os.makedirs(out, exist_ok=True)
        result = report_mod.build_report(rep, out)
        result.setdefault("skipped", [])
    else:
        def cb(done, total):
            bar.progress(done / max(total, 1),
                         text=f"Comparing drawing {done} of {total}…")
        result = proj.compare_project(rvalid, vvalid, cfg, out, verifier=verifier,
                                      progress_cb=cb)
    result["skipped"] = (result.get("skipped") or []) + skipped
    bar.empty()
    return result, out


# ───────────────────────── sidebar ─────────────────────────
with st.sidebar:
    st.header("Options")
    dpi_mode = st.radio("Resolution (DPI)", ["Auto (recommended)", "Manual"], index=0,
                        help="Auto picks the DPI from the drawing's text size, so small-text "
                             "sheets render sharper and large sheets aren't over-rendered.")
    manual_dpi = None
    if dpi_mode == "Manual":
        manual_dpi = st.select_slider("Manual DPI", [200, 250, 300, 400, 500], value=300,
                                      help="Higher = sharper crops/text, but slower.")
    ignore_tb = st.checkbox("Ignore the titleblock corner", value=True,
                            help="Skip the bottom-right title/revision block so its routine "
                                 "edits don't show up as unexpected changes.")
    detect_hexnotes = st.checkbox("Detect hexnote / callout edits", value=False,
                                  help="Also flag in-place edits inside closed callouts "
                                       "(e.g. a hexnote 8c→8) that the normal diff and OCR "
                                       "can't see. Off by default — it adds noise on heavily "
                                       "re-plotted drawings. Turn on only when chasing these.")
    st.divider()
    st.subheader("Read the text at each target?")
    st.caption("The visual diff can't see an in-place swap like W200X52→W310X52, and "
               "it can't confirm the architect typed in the note you asked for. Turn on "
               "a reader to confirm captured/missed. Everything stays on your machine.")
    ocr_ok, ocr_msg = _ocr_status()
    use_ocr = st.checkbox("Local OCR (Tesseract)", value=ocr_ok,
                          help="Reads the label text at each flagged target with "
                               "Tesseract to confirm captured/missed. Fast, fully local. "
                               "This is the recommended workhorse.")
    if use_ocr:
        st.caption(("✅ " if ocr_ok else "⚠️ ") + ocr_msg)
    use_vlm = st.checkbox("Local VLM (Ollama vision model)", value=False,
                          help="A local on-device vision model reads the before/after "
                               "crop and confirms whether your instruction was applied — "
                               "for unresolved spec redlines, including in-place swaps. "
                               "Catches captures OCR can't read. Best with a "
                               "capable GPU; on a CPU it adds a model call per unresolved "
                               "target and is slow. Off by default.")
    if use_vlm:
        vlm_ok, vlm_msg = _vlm_status(Config().vlm_model)
        st.caption(("✅ " if vlm_ok else "⚠️ ") + vlm_msg)
        if vlm_ok:
            st.caption("Runs on unresolved spec targets, even without a pixel change. "
                       "On a slow machine each target can add minutes.")


# ──────────────── file matching for pick-a-pair ────────────────
@st.cache_data(show_spinner=False)
def _sig(data: bytes):
    """Whole-file content fingerprint (cached on bytes) for matching files."""
    import fitz
    from redlinecheck import prematch
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        return prematch.doc_signature(doc, Config(), annots=True)
    finally:
        doc.close()


def _best_matches(redlined_files, revised_files):
    """For each redlined file, (best_revised_index, score) using content + filename —
    same logic as whole-project matching. Junk/unreadable files score 0."""
    import numpy as np
    from redlinecheck.project import _name_key
    rs, fs = [], []
    for f in redlined_files:
        try:
            rs.append(_sig(f.getvalue()))
        except Exception:
            rs.append(None)
    for f in revised_files:
        try:
            fs.append(_sig(f.getvalue()))
        except Exception:
            fs.append(None)
    rk = [_name_key(f.name) for f in redlined_files]
    fk = [_name_key(f.name) for f in revised_files]
    best = {}
    for i in range(len(redlined_files)):
        sims = []
        for j in range(len(revised_files)):
            s = 0.0 if (rs[i] is None or fs[j] is None) else float(np.dot(rs[i], fs[j]))
            if rk[i] and rk[i] == fk[j]:
                s = max(s, 0.99)
            sims.append(s)
        j = int(np.argmax(sims)) if sims else 0
        best[i] = (j, sims[j] if sims else 0.0)
    return best


# ───────────────────────── main ─────────────────────────
st.title("📐 Redline Check")
st.markdown("Compare your **redlined PDF** against the architect's **revised PDF** — "
            "flag missed comments, unexpected changes, and confirmed captures.")

with st.expander("How to read the results"):
    st.markdown(
        "- **Possible misses** — your redline shows no change at the spot you pointed to "
        "(the architect ignored it, *or* it's an in-place text swap the visual diff can't "
        "see — turn on OCR for a definite verdict).\n"
        "- **Unexpected changes** — the architect changed something you never marked up.\n"
        "- **To verify** — a change was seen at the target but the value wasn't read/confirmed.\n"
        "- **Addressed (verified)** — OCR actually read your instructed value at the target.\n"
        "- **Extra pages** — pages the architect inserted/split out (no counterpart in your "
        "set). A structural precheck matches pages first, so inserted pages don't throw the "
        "comparison off.\n\n"
        "- **Highlighted region** — an area you clouded/boxed to point us at the markups; "
        "shown under *To verify* for a quick look, not graded as a miss.\n\n"
        "Drop a **whole project** (many drawings per side) and they're matched into pairs "
        "and compared together — unmatched or unsupported files are listed under *Extra "
        "pages*, never block the run. Each item shows your *original* next to the "
        "architect's *revision*, with a **🔍 Zoom out** for context. Everything runs "
        "locally — nothing is uploaded."
    )

c1, c2 = st.columns(2)
redlined_files = c1.file_uploader("① Redlined drawings — your markup",
                                  accept_multiple_files=True)
revised_files = c2.file_uploader("② Revised drawings — architect's",
                                 accept_multiple_files=True)
c1.caption("Drop one drawing, or the whole redlined set / folder.")
c2.caption("Drop the matching revised set. Junk/non-PDF files are ignored automatically.")

# Decide what to compare. With several files on BOTH sides, match them and let the
# user either pick one pair or run the whole project.
run_redlined = run_revised = None
sel_names = ("", "")
if redlined_files and revised_files:
    rn, vn = len(redlined_files), len(revised_files)
    if rn == 1 and vn == 1:
        run_redlined, run_revised = redlined_files, revised_files
        sel_names = (redlined_files[0].name, revised_files[0].name)
    elif rn > 1 and vn > 1:
        with st.spinner("Matching files…"):
            best = _best_matches(redlined_files, revised_files)
        scope = st.radio(
            "What to compare?",
            ["Pick a single pair", "Whole project (all matched pairs)"],
            index=0, horizontal=True,
            help="Pick one redlined↔revised drawing to compare just that pair, or "
                 "compare every matched pair in the project at once.")
        if scope.startswith("Pick"):
            ri = st.selectbox("① Redlined drawing", range(rn),
                              format_func=lambda i: redlined_files[i].name)
            gj, score = best.get(ri, (0, 0.0))
            st.caption(f"Best structural match: **{revised_files[gj].name}** "
                       f"(similarity {score:.2f}) — change it below if needed.")
            vi = st.selectbox("② Revised drawing", range(vn), index=gj,
                              format_func=lambda i: revised_files[i].name)
            run_redlined, run_revised = [redlined_files[ri]], [revised_files[vi]]
            sel_names = (redlined_files[ri].name, revised_files[vi].name)
        else:
            run_redlined, run_revised = redlined_files, revised_files
            sel_names = (f"{rn} redlined drawings", f"{vn} revised drawings")
            st.caption(f"Comparing all {rn}×{vn} as auto-matched pairs; unmatched/"
                       "unsupported files are listed under *Extra pages*.")
    else:
        # one side single, the other several -> project handles the matching
        run_redlined, run_revised = redlined_files, revised_files
        sel_names = (f"{rn} redlined", f"{vn} revised")
        st.info("Several files on one side — matched into pairs automatically.")

ready = bool(run_redlined and run_revised)
go = st.button("Compare ▶", type="primary", disabled=not ready, width="stretch")

if go:
    # clean up the previous run's temp dir
    old = st.session_state.pop("out_dir", None)
    if old and os.path.isdir(os.path.dirname(old)):
        shutil.rmtree(os.path.dirname(old), ignore_errors=True)
    cfg = Config()
    if manual_dpi is not None:
        cfg.analysis_dpi = manual_dpi
        cfg.auto_dpi = False
    if detect_hexnotes:
        cfg.detect_symbols = True
    if not ignore_tb:
        cfg.ignore_zones = []
    try:
        with st.spinner("Working… (precheck + comparison; a big project takes a few "
                        "minutes per drawing)"):
            result, out_dir = _run(run_redlined, run_revised, cfg, use_ocr, use_vlm)
    except Exception as e:
        st.error(f"Comparison failed: {e}")
        st.stop()
    import dataclasses
    st.session_state["result"] = result
    st.session_state["out_dir"] = out_dir
    st.session_state["names"] = sel_names
    st.session_state["dpi_used"] = cfg.analysis_dpi   # pipeline set this when auto
    st.session_state["dpi_auto"] = manual_dpi is None
    st.session_state["overrides"] = {}                # reset one-click reclassifications
    st.session_state["reported"] = set()              # reset reported-error buttons
    st.session_state["zoom"] = {}                     # reset per-row zoom-out toggles
    st.session_state.pop("active_bucket", None)       # start a fresh run on the first view
    st.session_state["run_settings"] = {
        "redlined": sel_names[0], "revised": sel_names[1],
        "ocr_enabled": bool(use_ocr), "vlm_enabled": bool(use_vlm),
        "dpi_used": cfg.analysis_dpi, "dpi_auto": manual_dpi is None,
        "detect_hexnotes": bool(detect_hexnotes), "ignore_titleblock": bool(ignore_tb),
        "config": dataclasses.asdict(cfg),
    }

# ───────────────────────── results ─────────────────────────
@st.fragment
def _render_results():
    # The whole results area is one fragment: clicking Mark addressed / Report error
    # reruns ONLY this (no full-screen 10s reload), so tab-heading counts update,
    # addressed rows leave their section, and sheets with nothing left disappear.
    result = st.session_state["result"]
    out_dir = st.session_state["out_dir"]
    rows = result["rows"]
    overviews = result["overviews"]
    counts = result["counts"]

    extra = result.get("extra", [])

    # Emit a queued toast (set by a button callback; shown here in the fragment body
    # rather than inside the callback, which Streamlit warns against).
    _t = st.session_state.pop("_toast", None)
    if _t:
        if _t[0] == "addr":
            st.toast("Marked addressed", icon="✅")
        else:
            st.toast(f"Reported — saved to {_t[2]}", icon="📝")

    # Call out FLATTENED redlines: the markups were baked into the page (no live
    # annotations), so we caught them via the red-ink fallback. Detection still ran,
    # but for the cleanest results the reviewer should use the original un-flattened
    # PDF with editable annotations.
    flat = result.get("flattened_pages") or []
    if flat:
        shown = ", ".join(str(p) for p in flat[:15]) + ("…" if len(flat) > 15 else "")
        st.warning(
            f"⚠️ **Flattened redlines detected** on {len(flat)} sheet(s) (sheet "
            f"{shown}). This PDF's markups are baked into the page, not editable "
            "annotations, so they were caught with a red-ink fallback (no markup text "
            "to read — those land in *To verify* / *Possible misses*). For the most "
            "accurate results, re-run with the original **un-flattened** PDF that "
            "still has live annotations.",
            icon="⚠️")

    # Project: show the matched drawing pairs and any files that were ignored
    # (unsupported / unreadable) — surfaced gently, never as a blocking error.
    proj_pairs = result.get("pairs")
    skipped = result.get("skipped") or []
    if proj_pairs or skipped:
        lab = f"Project files — {len(proj_pairs or [])} drawing pair(s) compared"
        if skipped:
            lab += f", {len(skipped)} file(s) ignored"
        with st.expander(lab, expanded=False):
            if proj_pairs:
                st.caption("Matched automatically (redlined → revised):")
                for p in proj_pairs:
                    st.markdown(f"- **{p['redlined']}** → {p['revised']} "
                                f"&nbsp;·&nbsp; match {p['score']}")
            if skipped:
                st.caption("Ignored — unsupported or unreadable, dropped so they don't "
                           "block the run:")
                for s in skipped:
                    st.markdown(f"- {html_escape(s['name'])} — _{html_escape(s['why'])}_")

    # Live counts that reflect one-click reclassifications this session.
    from collections import Counter as _Counter
    ec = _Counter(_eff_status(r) for r in rows)
    n_miss = sum(ec[s] for s in MISS)
    n_unexp = ec[UNEXPECTED]
    n_verify = sum(ec[s] for s in OTHER)
    n_addr = sum(ec[s] for s in ADDRESSED)
    n_over = len(st.session_state.get("overrides", {}))

    st.divider()
    st.subheader("Results")
    cap = []
    if st.session_state.get("dpi_used"):
        cap.append(f"Rendered at {st.session_state['dpi_used']} DPI "
                   f"({'auto' if st.session_state.get('dpi_auto') else 'manual'}).")
    if n_over:
        cap.append(f"{n_over} reclassified as addressed this session.")
    if cap:
        st.caption("  ".join(cap))
    m = st.columns(6)
    m[0].metric("⚠ Possible misses", n_miss)
    m[1].metric("⚠ Unexpected", n_unexp)
    m[2].metric("To verify", n_verify)
    m[3].metric("Addressed (verified)", n_addr)
    m[4].metric("Markers", counts["markers"])
    m[5].metric("Extra pages", counts.get("extra_pages", 0))

    # Per-drawing reports (summary.html/annotated.pdf/findings.csv) live at the run
    # root for a single comparison, or inside per-drawing subfolders for a project.
    # Always offer the combined CSV + the full zip; offer the single-file HTML/PDF only
    # when this was a one-drawing run.
    is_project = bool(result.get("project"))
    d = st.columns(4)
    _write_combined_csv(rows, out_dir)
    if not is_project and os.path.exists(os.path.join(out_dir, "summary.html")):
        d[0].download_button("⬇ summary.html",
                             data=_read_bytes(os.path.join(out_dir, "summary.html")),
                             file_name="summary.html", mime="text/html", width="stretch")
        d[1].download_button("⬇ annotated.pdf",
                             data=_read_bytes(os.path.join(out_dir, "annotated.pdf")),
                             file_name="annotated.pdf", mime="application/pdf", width="stretch")
    else:
        d[0].caption("Per-drawing summary.html & annotated.pdf are inside the .zip →")
    d[2].download_button("⬇ findings.csv",
                         data=_read_bytes(os.path.join(out_dir, "findings_all.csv")),
                         file_name="findings.csv", mime="text/csv", width="stretch")
    d[3].download_button("⬇ everything (.zip)", data=_zip_dir(out_dir),
                         file_name="redline_report.zip", mime="application/zip",
                         width="stretch")

    # Stateful view selector. We deliberately do NOT use st.tabs: a tab widget resets
    # to the first tab on every fragment rerun, so clicking "Mark addressed" inside the
    # Unexpected/To-verify view would snap the page back to "Possible misses". A radio
    # keyed on STABLE ids (not the count-bearing labels) preserves the selected view
    # across reruns — the count labels can change without losing your place.
    labels = {
        "miss":   f"⚠ Possible misses ({n_miss})",
        "unexp":  f"⚠ Unexpected ({n_unexp})",
        "verify": f"To verify / addressed ({n_verify}/{n_addr})",
        "ov":     "Sheet overviews",
        "extra":  f"📄 Extra pages ({counts.get('extra_pages', 0)})",
    }
    view = st.radio("View", list(labels), format_func=lambda k: labels[k],
                    key="active_bucket", horizontal=True, label_visibility="collapsed")
    if view == "miss":
        st.caption("Your redline showed no confirmed change at the target. With OCR/VLM "
                   "on, the rest are likely real misses — eyeball each.")
        _render_section(rows, MISS, overviews, out_dir,
                        empty_msg="No possible misses 🎉")
    elif view == "unexp":
        st.caption("The architect changed these areas; you didn't mark them up. "
                   "Heavily-revised sheets are collapsed — open to inspect.")
        _render_section(rows, {UNEXPECTED}, overviews, out_dir,
                        empty_msg="No unexpected changes 🎉")
    elif view == "verify":
        st.caption("A change was seen at a target, a TO-ARCH/reference note, an area you "
                   "highlighted, or a 'copy comments from sheet X' note — eyeball each.")
        _render_section(rows, OTHER, overviews, out_dir,
                        empty_msg="Nothing awaiting verification.")
        st.markdown("##### Addressed — verified")
        st.caption("The instructed value was actually read at the target (OCR/VLM). "
                   "These are confirmed handled.")
        _render_section(rows, ADDRESSED, overviews, out_dir, expand_small=False,
                        empty_msg="— (turn on OCR/VLM to confirm captures)")
    elif view == "ov":
        lbl = {}
        for r in rows:
            lbl.setdefault(_ov_key(r), (r.get("source", ""), r["page"]))
        if not overviews:
            st.caption("No per-sheet overviews to show.")
        for k in sorted(overviews, key=lambda x: str(x)):
            src, pg = lbl.get(k, ("", None))
            cap = (f"{src} — sheet {pg}" if src else
                   (f"Sheet {pg}" if pg else "Overview"))
            st.image(os.path.join(out_dir, overviews[k]), caption=cap, width="stretch")
    elif view == "extra":
        if not extra:
            st.success("No extra/inserted pages — the two sets line up page-for-page.")
        else:
            _KIND = {"inserted": ("🟣 Inserted by architect",
                                  "In the revision but not in your redlined set."),
                     "unmatched": ("🟠 No match in revision",
                                   "In your redlined set but not found in the revision."),
                     "lowconf": ("🔵 Heavily re-plotted — review manually",
                                 "Re-plotted so heavily it won't align, so automatic "
                                 "change-detection was suppressed to avoid false "
                                 "positives. Check these whole sheets by eye."),
                     "error": ("🔴 Could not compare",
                               "Pages couldn't be matched/compared automatically.")}
            st.caption("Pages the architect inserted or split out, re-plotted beyond "
                       "reliable alignment, or that couldn't be matched. Review by hand.")
            for kind in ("inserted", "unmatched", "lowconf", "error"):
                items = [e for e in extra if e["kind"] == kind]
                if not items:
                    continue
                head, desc = _KIND[kind]
                st.markdown(f"##### {head} ({len(items)})")
                st.caption(desc)
                for e in items:
                    with st.expander(e["label"]):
                        if e["note"]:
                            st.caption(e["note"])
                        if e.get("img"):
                            st.image(os.path.join(out_dir, e["img"]), width="stretch")


if "result" in st.session_state:
    _render_results()
else:
    st.info("Drop your redlined drawing(s) and the architect's revised drawing(s) — a "
            "single pair or a whole project — choose options in the sidebar, then click "
            "**Compare**. Unsupported files are ignored automatically.")
