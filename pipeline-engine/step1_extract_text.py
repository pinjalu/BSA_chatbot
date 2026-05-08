from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


# camelot uses `tempfile.TemporaryDirectory` to host the per-page PDFs
# it renders via ghostscript.  On Windows, ghostscript subprocesses can
# still hold a handle to those `page-N.pdf` files when Python tries to
# clean up at exit, raising `PermissionError: [WinError 32]`.  The
# error is cosmetic — the output JSON has already been written — but
# it floods stderr with scary-looking atexit tracebacks, especially in
# chunked mode where many temp dirs queue up.
#
# Python 3.10+ added `ignore_cleanup_errors=True` for exactly this
# case.  Patch the default so every TemporaryDirectory in the process
# (including camelot's internal ones) silently swallows cleanup errors.
_orig_tmpdir_init = tempfile.TemporaryDirectory.__init__

def _tmpdir_init_quiet(self, *args, **kwargs):
    try:
        kwargs.setdefault("ignore_cleanup_errors", True)
        return _orig_tmpdir_init(self, *args, **kwargs)
    except TypeError:
        # Pre-3.10 Python doesn't support the kwarg; fall back cleanly.
        kwargs.pop("ignore_cleanup_errors", None)
        return _orig_tmpdir_init(self, *args, **kwargs)

tempfile.TemporaryDirectory.__init__ = _tmpdir_init_quiet


# camelot ALSO registers `shutil.rmtree` directly as an atexit callback
# for some of its tempdirs (bypassing TemporaryDirectory entirely). The
# patch above doesn't cover those, so install an unraisable hook that
# silently drops the specific PermissionError pattern raised when
# ghostscript still owns `page-N.pdf` at exit. Anything else still
# surfaces normally.
_orig_unraisable = sys.unraisablehook

def _quiet_camelot_tempdir_unraisable(unraisable):  # noqa: D401
    exc = unraisable.exc_value
    msg = str(exc) if exc else ""
    if (isinstance(exc, PermissionError)
            and "WinError 32" in msg
            and "page-" in msg
            and "Temp" in msg):
        return  # cosmetic-only — output already written, swallow.
    return _orig_unraisable(unraisable)

sys.unraisablehook = _quiet_camelot_tempdir_unraisable


import fitz  # PyMuPDF
import pdfplumber


__all__ = [
    "extract_pdf_to_json",
    "extract_single_page_job",
    "extract_layout_text",
    "extract_page_text",
    "extract_tables",
    "extract_all_tables_camelot",
    "extract_tables_camelot_for_page",
    "ocr_page",
    "detect_page_rotation",
]


# `_logger` is the proper stdlib Logger.  `log(msg, quiet=...)` is a
# back-compat shim that forwards INFO-level messages to `_logger` while
# honouring the `quiet=True` kwarg used at every existing call site.
#
# Server callers should drive verbosity via `_setup_logging("INFO" / "DEBUG"
# / ...)` or by configuring the `pdf_to_json` logger directly.

_logger = logging.getLogger("pdf_to_json")


def _setup_logging(level: str | int) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="[PDF] %(asctime)s %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
    else:
        logging.getLogger().setLevel(level)


def log(msg: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    _logger.info("%s", msg)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    _atomic_write_bytes(path, text.encode("utf-8"))


_SMART_REPLACEMENTS = {
    "'": "'", "'": "'",
    """: '"', """: '"',
    "–": "-", "—": "-",
    "ﬁ": "fi", "ﬂ": "fl",
    "�": "'",
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    for k, v in _SMART_REPLACEMENTS.items():
        text = text.replace(k, v)
    lines = [ln.rstrip() for ln in text.splitlines()]
    out, blank = [], 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln)
    return "\n".join(out).strip()


def extract_layout_text(fitz_page,
                        fw_ratio: float = 0.70,
                        narrow_ratio: float = 0.55,
                        min_narrow_blocks: int = 4) -> tuple[str, str]:
    """Extract page text, respecting 2-column layouts when present.

    Returns (text, layout_mode) where layout_mode is "single" or "two-col".

    Logic:
      - Collect every text block on the page.
      - If the page has at least `min_narrow_blocks` blocks narrower than
        `narrow_ratio * page_width` AND those blocks are split between the
        left and right halves (>=2 on each side), treat the page as
        two-column. Full-width blocks (> fw_ratio * page_width) are treated
        as headings/banners that break the column flow.
      - Otherwise read row-wise (single column).
    """
    page_w = fitz_page.rect.width
    mid_x = page_w / 2

    raw = fitz_page.get_text("blocks")
    blocks = []
    for b in raw:
        # Block tuple: (x0, y0, x1, y1, text, block_no, type). type 0 = text.
        if len(b) < 7 or b[6] != 0:
            continue
        text = (b[4] or "").strip()
        if not text:
            continue
        blocks.append(b)

    if not blocks:
        return "", "single"

    narrow = [b for b in blocks if (b[2] - b[0]) <= page_w * narrow_ratio]
    left_narrow = [b for b in narrow if (b[0] + b[2]) / 2 < mid_x]
    right_narrow = [b for b in narrow if (b[0] + b[2]) / 2 >= mid_x]
    is_two_col = (
        len(narrow) >= min_narrow_blocks
        and len(left_narrow) >= 2
        and len(right_narrow) >= 2
    )

    if not is_two_col:
        blocks.sort(key=lambda b: (round(b[1], 1), b[0]))
        return "\n".join(b[4].strip() for b in blocks), "single"

    fw = [b for b in blocks if (b[2] - b[0]) > page_w * fw_ratio]
    fw_ids = {id(b) for b in fw}
    left = [b for b in blocks
            if id(b) not in fw_ids and (b[0] + b[2]) / 2 < mid_x]
    right = [b for b in blocks
             if id(b) not in fw_ids and (b[0] + b[2]) / 2 >= mid_x]

    fw.sort(key=lambda b: b[1])
    left.sort(key=lambda b: b[1])
    right.sort(key=lambda b: b[1])

    ordered = []
    li = ri = 0
    for f in fw:
        fy = f[1]
        while li < len(left) and left[li][1] < fy:
            ordered.append(left[li]); li += 1
        while ri < len(right) and right[ri][1] < fy:
            ordered.append(right[ri]); ri += 1
        ordered.append(f)
    ordered.extend(left[li:])
    ordered.extend(right[ri:])

    return "\n".join(b[4].strip() for b in ordered), "two-col"


_ROTATION_RE = re.compile(r"Rotate:\s*(\d+)")


def detect_page_rotation(fitz_page) -> int:
    """Detect the actual content rotation of a page, independent of declared
    `page.rotation` metadata.

    BSA workshop manuals often claim `rotation=0` but have content laid out
    at 90°/180°/270°. We render the page at low-res, ask tesseract's OSD
    what rotation would make the text upright, and return that value so
    callers can `fitz_page.set_rotation(rot)` before extracting text,
    tables, or running OCR.

    Returns 0, 90, 180, or 270.
    """
    import pytesseract
    from PIL import Image

    # Low-res is enough for orientation detection and keeps it fast.
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        osd = pytesseract.image_to_osd(img)
        m = _ROTATION_RE.search(osd)
        if m:
            rot = int(m.group(1)) % 360
            if rot in (90, 180, 270):
                return rot
    except Exception:
        pass
    return 0


def ocr_page(fitz_page, dpi: int = 300) -> str:
    import pytesseract
    from PIL import Image

    zoom = dpi / 72
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img) or ""


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Common English + workshop-manual words. A real OCR'd line in a manual
# almost always contains at least one of these; scrambled rotated-text
# fragments (`pasindas Pula`, `AJana Ul Aydde`) almost never do. Keep
# this list short and skewed toward function words + domain vocabulary
# rather than exhaustive — bigger lists wash out the discriminator.
_COMMON_TEXT_WORDS = frozenset({
    # Function words
    "the", "a", "an", "is", "are", "be", "been", "being", "of", "to", "in",
    "and", "or", "for", "on", "at", "with", "this", "that", "it", "as",
    "by", "from", "but", "not", "if", "then", "do", "does", "has", "have",
    "was", "were", "any", "no", "yes", "you", "we", "they", "all", "use",
    "see", "than", "such", "must", "should", "can", "will", "may", "shall",
    "into", "before", "after", "during", "while", "until", "since",
    "through", "between", "below", "above", "over", "under", "each", "per",
    # Workshop / parts vocabulary that recurs constantly in BSA manuals
    "engine", "battery", "fuel", "oil", "air", "water", "system", "valve",
    "switch", "remove", "install", "replace", "tighten", "check", "inspect",
    "fluid", "filter", "spark", "plug", "fuse", "wire", "connector",
    "torque", "bolt", "nut", "screw", "washer", "level", "fill", "drain",
    "flush", "clean", "service", "lubricate", "adjust", "hex", "pin", "ring",
    "shaft", "gear", "bearing", "seal", "gasket", "cover", "frame", "wheel",
    "brake", "clutch", "chain", "sprocket", "head", "cylinder", "piston",
    "spec", "specification", "specifications", "rpm", "max", "min", "type",
    "ratio", "pressure", "temperature", "speed", "voltage", "current",
    "front", "rear", "left", "right", "side", "top", "bottom", "inside",
    "outside", "page", "section", "chapter", "figure", "fig", "note",
    "caution", "warning", "important", "model", "part", "code", "color",
    "yellow", "blue", "green", "red", "black", "white", "kit", "assembly",
    "lubrication", "lubricant", "consumables", "maintenance", "procedure",
    "instructions", "operation", "function", "component", "components",
    "parts", "tools", "tool", "year", "date", "time", "step", "steps",
    "mode", "modes", "fault", "diagnostic", "ignition", "starter", "sensor",
    "module", "controller", "harness", "coolant", "radiator", "exhaust",
    "intake", "throttle", "injector", "alternator", "regulator",
})


def _is_quality_ocr_line(line: str) -> bool:
    """Heuristic: True when an OCR line looks like real English text from
    the manual rather than scrambled fragments OCR produces from rotated
    diagrams or upside-down labels.

    Filters used (any failure rejects the line):
      1. Mostly alphanumeric chars — rejects symbol soup.
      2. No 5+ consecutive consonants — `pasndspl` style impossible.
      3. Most multi-letter words contain a vowel.
      4. At least one word from the common-English / workshop-vocab list.

    The vocab check is the strongest signal: rotated-text OCR almost
    never produces a recognisable function word (`the`, `is`, `of`) or
    a manual-domain word (`engine`, `bolt`, `fuel`). Real OCR almost
    always does.
    """
    s = line.strip()
    if len(s) < 5:
        return False

    # 1. Junk-character ratio. Common manual punctuation is allowed
    # (.,:;-/&()'[] and digits + letters); anything else is OCR garbage.
    junk = sum(1 for c in s
               if not (c.isalnum() or c.isspace()
                       or c in ".,:;-/&()'[]+%°×*#"))
    if junk / len(s) > 0.20:
        return False

    alpha = sum(1 for c in s if c.isalpha())
    if alpha / len(s) < 0.45:
        return False

    if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", s, flags=re.IGNORECASE):
        return False

    long_words = re.findall(r"[A-Za-z]{3,}", s)
    if long_words:
        vowel_words = sum(1 for w in long_words
                          if re.search(r"[aeiouyAEIOUY]", w))
        if vowel_words / len(long_words) < 0.65:
            return False

    words = re.findall(r"[a-z]+", s.lower())
    if not any(w in _COMMON_TEXT_WORDS for w in words):
        return False

    return True


def unique_ocr_lines(ocr_text: str, existing_text: str,
                     min_line_chars: int = 5,
                     word_overlap_threshold: float = 0.75) -> list[str]:
    """Return OCR lines that don't already appear in existing_text and
    look like real text rather than rotated-diagram garbage.

    Filters (in order):
      1. Skip lines shorter than `min_line_chars`.
      2. Skip lines whose normalised form already appears in `existing_text`
         (catches exact repeats with minor formatting differences).
      3. Skip lines whose word set overlaps the existing-text word bag
         above the threshold — restatements of the same sentence.
      4. Skip lines that fail `_is_quality_ocr_line` — scrambled char soup
         from rotated-text OCR. This is what makes large-PDF augments
         clean: workshop manuals have many diagram pages whose OCR
         output is junk, and we now drop those lines at the source
         instead of polluting the page text with them.
    """
    existing_norm = _normalize(existing_text)
    existing_words = set(existing_norm.split())

    extras: list[str] = []
    seen_norm: set[str] = set()

    for raw in ocr_text.splitlines():
        s = raw.strip()
        if len(s) < min_line_chars:
            continue
        s_norm = _normalize(s)
        if not s_norm or s_norm in seen_norm:
            continue
        if s_norm in existing_norm:
            continue

        words = s_norm.split()
        if len(words) >= 3:
            overlap = sum(1 for w in words if w in existing_words) / len(words)
            if overlap >= word_overlap_threshold:
                continue

        if not _is_quality_ocr_line(s):
            continue

        seen_norm.add(s_norm)
        extras.append(s)
    return extras


def is_real_table(data: list[list]) -> bool:
    if not data or len(data) < 2:
        return False
    ncols = max(len(r) for r in data)
    if ncols < 2:
        return False
    non_empty = sum(
        1 for row in data for cell in row if cell and str(cell).strip()
    )
    return non_empty >= 3


# Function-words used to detect when a "table's" headers actually form
# an English sentence (= camelot stream parser fitting prose into a
# fake column grid).  Real column headers ("Sr. No.", "Part No.",
# "FRT", "QTY") never include these.
_PROSE_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "and", "or", "for", "on", "at", "with", "this",
    "that", "it", "as", "by", "from", "but", "not", "if", "so", "than",
    "into", "about", "before", "after", "during", "while", "until",
    "since", "through", "between", "below", "above", "over", "under",
    "such", "must", "should", "can", "will", "may", "shall", "have",
    "has", "had", "do", "does", "did",
})


def _headers_form_sentence(headers: list[str]) -> bool:
    """True when the headers concatenate into a coherent English sentence
    rather than being independent column labels.

    Example reject (this is body text mistakenly column-split):
      ["Here", "are some general troubleshooting steps for",
       "a starter motor", ":", ""]
        -> "Here are some general troubleshooting steps for a starter motor :"
        -> contains 'are', 'for', 'a' (function words) → likely prose.

    Example accept (real workshop-manual table headers):
      ["Sr. No.", "Part No.", "Part Description", "Frt.", "Qty.",
       "Remarks"] -> 0 function words → not prose.
    """
    joined = " ".join(str(h).strip() for h in headers if h).lower()
    words = re.findall(r"[a-z']+", joined)
    if len(words) < 4:
        return False
    fw = sum(1 for w in words if w in _PROSE_FUNCTION_WORDS)
    return fw >= 2


def _table_has_word_splits(rows: list[dict]) -> bool:
    """True when the rows show 'word split across columns' — the
    signature of camelot stream parser fitting prose into a fake grid.

    Two signals counted as a split:
      (a) cell ends with a lowercase fragment (1–4 chars, no terminal
          punctuation) AND next non-empty cell on the same row starts
          with a lowercase letter — `'and in good co' + 'ndition.\\n...'`
          where 'co' + 'ndition' = 'condition'.
      (b) cell contains an internal newline AND lines after the newline
          look like continuations of a previous word — typical when
          camelot treats a paragraph as one cell.
    """
    splits = 0
    rows_seen = 0
    for row in rows:
        if isinstance(row, dict):
            cells = list(row.values())
        elif isinstance(row, list):
            cells = list(row)
        else:
            continue
        rows_seen += 1
        prev_tail: str | None = None
        for cell in cells:
            s = str(cell or "").strip()
            if not s:
                continue
            if prev_tail is not None:
                tail_word = re.split(r"[\s\n]", prev_tail)[-1]
                tail_word = tail_word.strip(".,;:!?'\"()[]")
                if (1 <= len(tail_word) <= 4
                        and tail_word.isalpha()
                        and tail_word.islower()
                        and s[0].isalpha()
                        and s[0].islower()):
                    splits += 1
                    break
            prev_tail = s
    return rows_seen >= 3 and splits >= 2


def _looks_like_real_table(table: dict) -> bool:
    """Reject camelot output that's body text masquerading as a table.

    A table is dropped if ANY of these are true:
      * Headers concatenate into an English sentence (prose mistakenly
        column-split) — see `_headers_form_sentence`.
      * 2+ rows show mid-word splits across columns — see
        `_table_has_word_splits`.
      * Fewer than 2 plausible column headers AND average non-empty
        cells per row < 2.0 (no real grid structure).
    """
    headers = table.get("headers") or []
    rows = table.get("rows") or []
    if not rows:
        return False

    if _headers_form_sentence(headers):
        return False
    if _table_has_word_splits(rows):
        return False

    real_headers = sum(
        1 for h in headers
        if h and len(str(h).strip()) <= 50
        and not str(h).strip()[:1].isdigit()
    )
    cell_count = 0
    for row in rows:
        if isinstance(row, dict):
            cell_count += sum(1 for v in row.values()
                              if v and str(v).strip())
        elif isinstance(row, list):
            cell_count += sum(1 for v in row if v and str(v).strip())
    avg_cells = cell_count / max(len(rows), 1)
    return real_headers >= 2 or avg_cells >= 2.0


def _convert_camelot_table(t, flavor: str) -> dict | None:
    df = t.df
    if df.empty or df.shape[1] < 2 or df.shape[0] < 2:
        return None

    header = [str(c).strip().replace("\n", " ") for c in df.iloc[0]]
    rows: list[dict] = []
    for ridx in range(1, len(df)):
        per_col = [
            str(v).split("\n") if isinstance(v, str) else [""]
            for v in df.iloc[ridx]
        ]
        max_lines = max((len(c) for c in per_col), default=1)

        def _push(rec: dict):
            if any(v.strip() for v in rec.values() if isinstance(v, str)):
                rows.append(rec)

        if max_lines > 1:
            for li in range(max_lines):
                rec: dict = {}
                for i, lines in enumerate(per_col):
                    key = header[i] if i < len(header) and header[i] else f"col{i+1}"
                    rec[key] = lines[li].strip() if li < len(lines) else ""
                _push(rec)
        else:
            rec = {}
            for i, lines in enumerate(per_col):
                key = header[i] if i < len(header) and header[i] else f"col{i+1}"
                rec[key] = (lines[0] if lines else "").strip()
            _push(rec)

    if not rows:
        return None

    accuracy = round(t.parsing_report.get("accuracy", 0), 1)
    return {
        "headers": header,
        "row_count": len(rows),
        "rows": rows,
        "accuracy": accuracy,
        "flavor": flavor,
        "page": t.page,
    }


def extract_all_tables_camelot(pdf_path: Path, page_count: int,
                               min_accuracy: float = 50.0,
                               quiet: bool = False) -> dict[int, list[dict]]:
    """Pre-compute tables for every page using camelot.

    Strategy:
      1. First pass with flavor='lattice' (bordered tables, rotation-aware).
      2. For pages where lattice found nothing, try flavor='stream' (uses
         whitespace — good for borderless tables).
      3. Filter tables below `min_accuracy` to drop misfires.

    Returns {page_num: [table, ...]}.
    """
    try:
        import camelot
    except ImportError:
        log("[PDF] camelot not installed — tables disabled", quiet=quiet)
        return {i: [] for i in range(1, page_count + 1)}

    page_tables: dict[int, list[dict]] = {i: [] for i in range(1, page_count + 1)}

    try:
        result = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
    except Exception as e:
        log(f"[PDF] camelot lattice failed: {e}", quiet=quiet)
        result = []

    for t in result:
        if t.parsing_report.get("accuracy", 0) < min_accuracy:
            continue
        conv = _convert_camelot_table(t, "lattice")
        if conv and _looks_like_real_table(conv):
            page_tables[t.page].append(conv)

    missing = [p for p, ts in page_tables.items() if not ts]
    if missing:
        page_spec = ",".join(str(p) for p in missing)
        try:
            result = camelot.read_pdf(str(pdf_path), pages=page_spec, flavor="stream")
        except Exception:
            result = []
        for t in result:
            if t.parsing_report.get("accuracy", 0) < min_accuracy:
                continue
            conv = _convert_camelot_table(t, "stream")
            if conv and _looks_like_real_table(conv):
                page_tables[t.page].append(conv)

    return page_tables


def extract_tables_camelot_for_page(pdf_path: Path, page_num: int,
                                    min_accuracy: float = 50.0,
                                    quiet: bool = False) -> list[dict]:
    try:
        import camelot
    except ImportError:
        log("[PDF] camelot not installed — tables disabled", quiet=quiet)
        return []

    out: list[dict] = []

    try:
        result = camelot.read_pdf(str(pdf_path), pages=str(page_num), flavor="lattice")
    except Exception as e:
        log(f"[PDF] camelot lattice failed p{page_num}: {e}", quiet=quiet)
        result = []

    for t in result:
        if t.parsing_report.get("accuracy", 0) < min_accuracy:
            continue
        conv = _convert_camelot_table(t, "lattice")
        if conv and _looks_like_real_table(conv):
            out.append(conv)

    if not out:
        try:
            result = camelot.read_pdf(str(pdf_path), pages=str(page_num), flavor="stream")
        except Exception:
            result = []
        for t in result:
            if t.parsing_report.get("accuracy", 0) < min_accuracy:
                continue
            conv = _convert_camelot_table(t, "stream")
            if conv and _looks_like_real_table(conv):
                out.append(conv)

    return out


def extract_tables(fitz_page, camelot_tables: list[dict] | None = None) -> list[dict]:
    """Return a page's tables.

    Primary source is camelot (passed in via `camelot_tables`). Falls back
    to fitz's built-in `find_tables` if camelot returned nothing — gives
    at least SOMETHING for every page even when camelot misses.
    """
    if camelot_tables:
        return camelot_tables

    tables: list[dict] = []
    try:
        finder = fitz_page.find_tables()
    except Exception:
        return tables

    for t in finder:
        try:
            data = t.extract()
        except Exception:
            continue
        if not is_real_table(data):
            continue

        header = [(c or "").strip().replace("\n", " ") for c in data[0]]
        rows: list[dict] = []
        for row in data[1:]:
            per_col = [
                (c if isinstance(c, str) else ("" if c is None else str(c))).split("\n")
                for c in row
            ]
            max_lines = max((len(c) for c in per_col), default=1)

            def _push(record: dict):
                if any(v for v in record.values()):
                    rows.append(record)

            if max_lines > 1:
                for li in range(max_lines):
                    rec: dict = {}
                    for i, lines in enumerate(per_col):
                        key = header[i] if i < len(header) and header[i] else f"col{i+1}"
                        rec[key] = lines[li].strip() if li < len(lines) else ""
                    _push(rec)
            else:
                rec = {}
                for i, lines in enumerate(per_col):
                    key = header[i] if i < len(header) and header[i] else f"col{i+1}"
                    rec[key] = (lines[0] if lines else "").strip()
                _push(rec)

        bbox = getattr(t, "bbox", None)
        tables.append({
            "bbox": [round(v, 2) for v in bbox] if bbox else None,
            "headers": header,
            "row_count": len(rows),
            "rows": rows,
            "flavor": "fitz",
        })
    return tables


def format_table_as_text(table: dict) -> str:
    header = table["headers"]
    rows = table["rows"]
    if not header and not rows:
        return ""
    col_widths = [max(len(str(h)), 1) for h in header]
    for r in rows:
        for i, h in enumerate(header):
            v = str(r.get(h, ""))
            if i < len(col_widths):
                col_widths[i] = min(max(col_widths[i], len(v)), 40)

    def _fmt_row(cells):
        parts = [str(c)[:col_widths[i]].ljust(col_widths[i])
                 for i, c in enumerate(cells) if i < len(col_widths)]
        return " | ".join(parts).rstrip()

    lines = [_fmt_row(header)]
    lines.append("-+-".join("-" * w for w in col_widths))
    for r in rows:
        lines.append(_fmt_row([r.get(h, "") for h in header]))
    return "\n".join(lines)


def extract_page_text(fitz_page, pdfplumber_page,
                      min_text_chars: int,
                      sparse_threshold: int,
                      force_ocr: bool,
                      quiet: bool,
                      page_num: int,
                      total_pages: int,
                      ocr_dpi: int = 300) -> tuple[str, str, list[str]]:
    """Return (clean_text, method, action_tags).

    method is one of:
        "text"        — normal text extraction was complete
        "ocr_full"    — force-ocr or near-empty page; OCR used as sole source
        "ocr_swap"    — sparse text; OCR had notably more content; OCR used
        "ocr_augment" — sparse text; OCR appended missing lines
        "blank"       — nothing extractable at all
    """
    actions: list[str] = []

    rotation = fitz_page.rotation or 0
    if rotation:
        actions.append(f"rotation={rotation}")

    quick_text, layout_mode = extract_layout_text(fitz_page)
    actions.append(f"layout={layout_mode}")

    if not quick_text.strip():
        try:
            quick_text = pdfplumber_page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 — pdfplumber raises diverse errors
            _logger.debug("page %d: pdfplumber.extract_text failed: %s",
                          page_num, exc)
            quick_text = ""

    quick_clean = clean_text(quick_text)
    quick_len = len(quick_clean)

    if force_ocr or quick_len < min_text_chars:
        actions.append("OCR(full)" if force_ocr else f"OCR(fallback,{quick_len}ch)")
        ocr = ocr_page(fitz_page, dpi=ocr_dpi)
        ocr_clean = clean_text(ocr)
        if not ocr_clean:
            actions.append("blank")
            return "", "blank", actions
        return ocr_clean, "ocr_full", actions

    actions.append(f"text({quick_len}ch)")

    if quick_len < sparse_threshold:
        ocr = ocr_page(fitz_page, dpi=ocr_dpi)
        ocr_clean = clean_text(ocr)
        ocr_len = len(ocr_clean)

        if ocr_len > quick_len * 1.2 and ocr_len > quick_len + 100:
            actions.append(f"swap({quick_len}->{ocr_len}ch)")
            return ocr_clean, "ocr_swap", actions

        extras = unique_ocr_lines(ocr_clean, quick_clean)
        if extras:
            actions.append(f"augment+{len(extras)}")
            merged = quick_clean + "\n\n[OCR augment]\n" + "\n".join(extras)
            return merged, "ocr_augment", actions

        actions.append("OCR-no-gain")

    return quick_clean, "text", actions


def _build_extract_summary(completed: dict[int, dict], total: int,
                           pdf_path: Path, t_start: float,
                           rotated_count: int) -> dict:
    pages = sorted(completed.values(), key=lambda p: p.get("page", 0))
    method_counts: dict[str, int] = {}
    total_tables = 0
    total_rows = 0
    total_chars = 0
    for p in pages:
        method_counts[p["method"]] = method_counts.get(p["method"], 0) + 1
        total_chars += p.get("char_count", 0)
        for tbl in p.get("tables", []) or []:
            total_tables += 1
            total_rows += int(tbl.get("row_count", 0) or 0)
    return {
        "pdf_path":            str(pdf_path),
        "pdf_name":            pdf_path.name,
        "total_pages":         total,
        "total_chars":         total_chars,
        "total_tables":        total_tables,
        "total_table_rows":    total_rows,
        "total_rotated_pages": rotated_count,
        "duration_seconds":    round(time.time() - t_start, 2),
        "method_counts":       method_counts,
        "pages":               pages,
    }


def extract_pdf_to_json(pdf_path: Path, output_path: Path,
                        *,
                        min_text_chars: int = 30,
                        sparse_threshold: int = 1500,
                        force_ocr: bool = False,
                        quiet: bool = False,
                        page_workers: int = 1,
                        ocr_dpi: int = 300,
                        table_min_accuracy: float = 50.0,
                        chunk_size: int | None = None,
                        resume: bool = True) -> dict:
    """Extract per-page text + tables from `pdf_path` and write `output_path`.

    All thresholds are explicit keyword arguments so the function is safe
    to call as a library on a server with per-tenant config.

    Large-PDF support (parallel mode only):
      `chunk_size`  — process this many pages per batch and write a
                      `<output_stem>.partial.json` after each batch. A
                      crash mid-run loses at most `chunk_size` pages of
                      work; the next call resumes automatically.
      `resume`      — when True (default), look for a `.partial.json`
                      next to `output_path` and skip pages already
                      extracted. Set False to start fresh.
    """
    pdf_path = pdf_path.resolve() if isinstance(pdf_path, Path) else Path(pdf_path).resolve()
    output_path = (output_path.resolve()
                   if isinstance(output_path, Path) else Path(output_path).resolve())
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    t_start = time.time()
    log(f"[PDF] Opening {pdf_path.name}", quiet=quiet)

    # Page-parallel mode (single PDF): workers run as threads so the
    # extractor module — whose filename has spaces — does not need to be
    # re-imported in spawned processes (which fails on Windows). camelot's
    # ghostscript subprocess and pytesseract's native calls release the
    # GIL, so thread-level parallelism is real for this workload.
    page_workers = max(int(page_workers or 1), 1)
    if page_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with fitz.open(str(pdf_path)) as doc:
            total = doc.page_count

        log(f"[PDF] {total} pages | force_ocr={force_ocr} "
            f"sparse_threshold={sparse_threshold}ch  page_workers={page_workers}"
            + (f"  chunk_size={chunk_size}" if chunk_size else ""), quiet=quiet)

        # `completed` is keyed by page_num so reads from a partial file
        # merge naturally with new chunks.  On crash, the next call
        # finds the partial and skips pages already done.
        partial_path = output_path.parent / (output_path.stem + ".partial.json")
        completed: dict[int, dict] = {}
        rotated_count = 0

        if resume and partial_path.is_file():
            try:
                saved = json.loads(partial_path.read_text(encoding="utf-8"))
                for p in saved.get("pages", []):
                    pn = p.get("page")
                    if isinstance(pn, int) and 1 <= pn <= total:
                        completed[pn] = p
                rotated_count = int(saved.get("total_rotated_pages", 0) or 0)
                log(f"[PDF] Resuming: {len(completed)} pages already extracted",
                    quiet=quiet)
            except (OSError, json.JSONDecodeError) as exc:
                log(f"[PDF] Could not read partial output ({exc}); starting fresh",
                    quiet=quiet)
                completed = {}
                rotated_count = 0

        pending = [pn for pn in range(1, total + 1) if pn not in completed]

        # When chunk_size is unset, do a single batch over all pending pages
        # (preserves previous behaviour).
        if chunk_size and chunk_size > 0:
            batches = [pending[i:i + chunk_size]
                       for i in range(0, len(pending), chunk_size)]
        else:
            batches = [pending] if pending else []

        t0_all = time.time()

        for b_idx, batch in enumerate(batches, start=1):
            if not batch:
                continue
            log(f"[PDF] Chunk {b_idx}/{len(batches)}: pages {batch[0]}-{batch[-1]} "
                f"({len(batch)} pages, {len(completed)}/{total} done overall)",
                quiet=quiet)

            with ThreadPoolExecutor(max_workers=page_workers) as ex:
                futs = {
                    ex.submit(
                        extract_single_page_job,
                        str(pdf_path),
                        page_num,
                        min_text_chars=min_text_chars,
                        sparse_threshold=sparse_threshold,
                        force_ocr=force_ocr,
                        ocr_dpi=ocr_dpi,
                        table_min_accuracy=table_min_accuracy,
                    ): page_num
                    for page_num in batch
                }

                for fut in as_completed(futs):
                    page_num = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as exc:  # noqa: BLE001 — keep going
                        # One bad page should not nuke the whole 475-page
                        # run.  Record a stub and continue.
                        log(f"[PDF] page {page_num} extraction failed: {exc}",
                            quiet=quiet)
                        completed[page_num] = {
                            "page":       page_num,
                            "method":     "error",
                            "char_count": 0,
                            "text":       "",
                            "tables":     [],
                        }
                        continue

                    completed[r["page"]] = {
                        "page":       r["page"],
                        "method":     r["method"],
                        "char_count": r["char_count"],
                        "text":       r["text"],
                        "tables":     r["tables"],
                    }
                    if r.get("rotated"):
                        rotated_count += 1

                    if not quiet:
                        elapsed = max(time.time() - t0_all, 1e-6)
                        rate = len(completed) / elapsed if elapsed > 0 else 0
                        eta = ((total - len(completed)) / rate
                               if rate > 0 else 0.0)
                        log(
                            f"  [done {len(completed):>3}/{total}] "
                            f"p{r['page']:>3}  {r['char_count']:>5}ch  "
                            f"{r['dt_seconds']:>4.1f}s  {r['method']:<11}  "
                            f"ETA {eta:>5.1f}s",
                            quiet=quiet,
                        )

            # Persist progress between chunks so a crash loses at most
            # `chunk_size` pages of work.
            if chunk_size and chunk_size > 0:
                _atomic_write_json(
                    partial_path,
                    _build_extract_summary(completed, total, pdf_path,
                                           t_start, rotated_count),
                )
                log(f"[PDF] Saved partial: {len(completed)}/{total} pages "
                    f"-> {partial_path.name}", quiet=quiet)

        summary = _build_extract_summary(completed, total, pdf_path,
                                         t_start, rotated_count)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_path, summary)

        if partial_path.exists():
            try:
                partial_path.unlink()
            except OSError:
                pass

        log("", quiet=quiet)
        log(f"[PDF] DONE in {summary['duration_seconds']:.1f}s  "
            + "  ".join(f"{k}={v}" for k, v in summary['method_counts'].items())
            + f"  total_chars={summary['total_chars']}"
            + f"  tables={summary['total_tables']}  rows={summary['total_table_rows']}"
            + f"  rotated_pages={summary['total_rotated_pages']}", quiet=quiet)
        log(f"[PDF] Wrote {output_path}", quiet=quiet)
        return summary

    with fitz.open(str(pdf_path)) as doc, pdfplumber.open(str(pdf_path)) as pdf:
        total = doc.page_count
        log(f"[PDF] {total} pages | force_ocr={force_ocr} "
            f"sparse_threshold={sparse_threshold}ch  ocr_dpi={ocr_dpi}", quiet=quiet)

        pages: list[dict] = []
        method_counts: dict[str, int] = {}
        total_tables = 0
        total_rows = 0
        total_rotated = 0

        for i in range(total):
            page_num = i + 1
            t0 = time.time()
            fitz_page = doc[i]
            pdfp_page = pdf.pages[i]

            # Rotation detection runs Tesseract OSD on a rendered pixmap and
            # costs ~1.5s per page. On large workshop manuals where most
            # pages are upright, doing it for every page adds 10+ minutes
            # before the first page is even extracted. Cheap pre-check: if
            # the page already yields a healthy amount of text, the content
            # is upright and we can skip OSD entirely.
            applied_rotation = 0
            if not force_ocr:
                try:
                    quick_probe = fitz_page.get_text("text") or ""
                except Exception:
                    quick_probe = ""
                if len(quick_probe.strip()) < min_text_chars:
                    detected = detect_page_rotation(fitz_page)
                    if detected:
                        fitz_page.set_rotation(detected)
                        applied_rotation = detected
                        total_rotated += 1

            text, method, actions = extract_page_text(
                fitz_page, pdfp_page,
                min_text_chars=min_text_chars,
                sparse_threshold=sparse_threshold,
                force_ocr=force_ocr,
                quiet=quiet,
                page_num=page_num,
                total_pages=total,
                ocr_dpi=ocr_dpi,
            )

            if applied_rotation:
                actions.insert(0, f"rotate-to-upright={applied_rotation}")

            # Per-page camelot (lattice -> stream fallback). Doing this
            # per-page keeps memory bounded — `pages="all"` on a 470-page
            # PDF holds every rendered table in RAM at once and uses
            # several GB before returning anything.
            page_tables = extract_tables_camelot_for_page(
                pdf_path, page_num,
                min_accuracy=table_min_accuracy, quiet=quiet,
            )
            tables = extract_tables(
                fitz_page,
                camelot_tables=page_tables,
            )
            if tables:
                total_tables += len(tables)
                total_rows += sum(t["row_count"] for t in tables)
                actions.append(f"tables={len(tables)}({sum(t['row_count'] for t in tables)}rows)")
                text_parts = [text] if text else []
                for t_idx, tab in enumerate(tables, 1):
                    text_parts.append(f"\n[Table {t_idx}]")
                    text_parts.append(format_table_as_text(tab))
                text = "\n".join(text_parts).strip()

            method_counts[method] = method_counts.get(method, 0) + 1
            dt = time.time() - t0

            pages.append({
                "page": page_num,
                "method": method,
                "char_count": len(text),
                "text": text,
                "tables": tables,
            })

            log(
                f"  [{page_num:>3}/{total}] {len(text):>5}ch  {dt:>4.1f}s  "
                f"{method:<11}  " + "  ".join(actions),
                quiet=quiet,
            )

    total_dt = time.time() - t_start
    summary = {
        "pdf_path": str(pdf_path),
        "pdf_name": pdf_path.name,
        "total_pages": total,
        "total_chars": sum(p["char_count"] for p in pages),
        "total_tables": total_tables,
        "total_table_rows": total_rows,
        "total_rotated_pages": total_rotated,
        "duration_seconds": round(total_dt, 2),
        "method_counts": method_counts,
        "pages": pages,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, summary)

    log("", quiet=quiet)
    log(f"[PDF] DONE in {total_dt:.1f}s  "
        + "  ".join(f"{k}={v}" for k, v in method_counts.items())
        + f"  total_chars={summary['total_chars']}"
        + f"  tables={total_tables}  rows={total_rows}"
        + f"  rotated_pages={total_rotated}", quiet=quiet)
    log(f"[PDF] Wrote {output_path}", quiet=quiet)

    return summary


def extract_single_page_job(pdf_path_str: str, page_num: int,
                            *,
                            min_text_chars: int,
                            sparse_threshold: int,
                            force_ocr: bool,
                            ocr_dpi: int = 300,
                            table_min_accuracy: float = 50.0) -> dict:
    pdf_path = Path(pdf_path_str)
    t0 = time.time()

    with fitz.open(str(pdf_path)) as doc, pdfplumber.open(str(pdf_path)) as pdf:
        total = doc.page_count
        fitz_page = doc[page_num - 1]
        pdfp_page = pdf.pages[page_num - 1]

        rotated = False
        applied_rotation = 0
        if not force_ocr:
            try:
                quick_probe = fitz_page.get_text("text") or ""
            except Exception:
                quick_probe = ""
            if len(quick_probe.strip()) < min_text_chars:
                detected = detect_page_rotation(fitz_page)
                if detected:
                    fitz_page.set_rotation(detected)
                    applied_rotation = detected
                    rotated = True

        text, method, actions = extract_page_text(
            fitz_page, pdfp_page,
            min_text_chars=min_text_chars,
            sparse_threshold=sparse_threshold,
            force_ocr=force_ocr,
            quiet=True,
            page_num=page_num,
            total_pages=total,
            ocr_dpi=ocr_dpi,
        )
        if applied_rotation:
            actions.insert(0, f"rotate-to-upright={applied_rotation}")

        camelot_tables = extract_tables_camelot_for_page(
            pdf_path, page_num,
            min_accuracy=table_min_accuracy, quiet=True,
        )
        tables = extract_tables(
            fitz_page,
            camelot_tables=camelot_tables,
        )

        table_rows = sum(t.get("row_count", 0) for t in tables) if tables else 0
        if tables:
            text_parts = [text] if text else []
            for t_idx, tab in enumerate(tables, 1):
                text_parts.append(f"\n[Table {t_idx}]")
                text_parts.append(format_table_as_text(tab))
            text = "\n".join(text_parts).strip()

        return {
            "page": page_num,
            "method": method,
            "char_count": len(text),
            "text": text,
            "tables": tables,
            "tables_count": len(tables),
            "table_rows": table_rows,
            "rotated": rotated,
            "actions": actions,
            "dt_seconds": time.time() - t0,
        }


def _ensure_utf8_console() -> None:
    """Force UTF-8 on stdout/stderr (Windows defaults to cp1252).

    Without this, `--help` and log lines containing box-drawing / arrow
    characters crash on Windows servers.  Python 3.7+ supports `reconfigure`.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_utf8_console()

    default_sparse = _env_int("PDF2JSON_SPARSE_THRESHOLD", 1500)
    default_min = _env_int("PDF2JSON_MIN_TEXT_CHARS", 30)
    default_dpi = _env_int("PDF2JSON_OCR_DPI", 300)
    default_min_acc = _env_float("PDF2JSON_TABLE_MIN_ACCURACY", 50.0)
    default_workers = _env_int("PDF2JSON_PAGE_WORKERS", 1)
    default_chunk = _env_int("PDF2JSON_CHUNK_SIZE", 0)
    default_log_level = os.environ.get("PDF2JSON_LOG_LEVEL", "INFO")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path "
                             "(default: <pdf_stem>.text.json next to the PDF)")

    parser.add_argument("--force-ocr", action="store_true",
                        help="OCR every page, ignore selectable text")

    parser.add_argument("--sparse-threshold", type=int, default=default_sparse,
                        help=f"Below this char count, OCR augments the text "
                             f"(default: {default_sparse})")
    parser.add_argument("--min-text-chars", type=int, default=default_min,
                        help=f"Below this, page goes straight to full OCR "
                             f"(default: {default_min})")
    parser.add_argument("--ocr-dpi", type=int, default=default_dpi,
                        help=f"OCR rendering DPI (default: {default_dpi})")
    parser.add_argument("--table-min-accuracy", type=float,
                        default=default_min_acc,
                        help=f"Minimum camelot table accuracy "
                             f"(default: {default_min_acc})")

    parser.add_argument("--page-workers", type=int, default=default_workers,
                        help=f"Parallel worker processes per single PDF "
                             f"(default: {default_workers})")
    parser.add_argument("--chunk-size", type=int, default=default_chunk,
                        help=f"Process pages in batches of this size, "
                             f"saving a `.partial.json` checkpoint after "
                             f"each batch.  0 disables chunking.  Crash "
                             f"recovery resumes from the last checkpoint. "
                             f"(default: {default_chunk})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing `.partial.json` and "
                             "extract every page from scratch.")

    parser.add_argument("--log-level", default=default_log_level,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help=f"Log level (default: {default_log_level})")
    parser.add_argument("--quiet", action="store_true",
                        help="Shorthand for --log-level WARNING")
    args = parser.parse_args(argv)

    _setup_logging("WARNING" if args.quiet else args.log_level)

    if not args.pdf.exists():
        _logger.error("PDF not found: %s", args.pdf)
        return 1
    if not args.pdf.is_file():
        _logger.error("Not a file: %s", args.pdf)
        return 1

    output = args.output or args.pdf.with_suffix(".text.json")

    try:
        extract_pdf_to_json(
            args.pdf, output,
            min_text_chars=args.min_text_chars,
            sparse_threshold=args.sparse_threshold,
            force_ocr=args.force_ocr,
            quiet=args.quiet,
            page_workers=args.page_workers,
            ocr_dpi=args.ocr_dpi,
            table_min_accuracy=args.table_min_accuracy,
            chunk_size=args.chunk_size if args.chunk_size > 0 else None,
            resume=not args.no_resume,
        )
    except KeyboardInterrupt:
        _logger.warning("Cancelled.")
        return 130
    except FileNotFoundError as exc:
        _logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface anything else with a clean exit
        _logger.exception("Fatal: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
