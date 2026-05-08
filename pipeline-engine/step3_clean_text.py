from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


_PUA_RE = re.compile(r"[-]")

# Only collapse when the next line starts with a lowercase letter —
# uppercase usually means a new sentence/section, not a continuation.
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n([a-z])")

# Accept Title Case as well as ALL CAPS — different manuals use both.
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){0,3})\s+([A-Z][A-Za-z &/\-+',]{3,80})\s*[:\-]?\s*$"
)

# Plain `1. Start the engine` is NOT included here — it collides with
# numbered procedure steps that fill workshop manuals and would produce
# hundreds of false sub-topics.
_SUBTOPIC_RE = re.compile(
    r"^\s*(?:"
    r"[ivx]{1,4}\.\s+"           # i., ii., iii., iv.
    r"|[IVX]{1,4}\.\s+"          # I., II., III.
    r"|\([ivxIVX]{1,4}\)\s+"     # (i), (ii)
    r"|\([a-zA-Z]\)\s+"          # (a), (b)
    r")"
    r"(.{4,120})$"
)

# If a candidate sub-topic body opens with one of these, it's almost
# certainly a step, not a heading.
_IMPERATIVE_PREFIXES = frozenset({
    "start", "stop", "connect", "disconnect", "check", "refer", "replace",
    "remove", "install", "adjust", "inspect", "read", "repeat", "press",
    "tighten", "loosen", "apply", "use", "note", "ensure", "set", "open",
    "close", "release", "verify", "measure", "wait", "switch", "turn",
    "do", "if", "when", "then", "before", "after",
})

# Treat as body, not as sub-topics.
_CALLOUT_LABELS = frozenset({
    "note", "notes", "caution", "warning", "warnings", "important",
    "tip", "tips", "danger", "attention", "remarks", "remark", "hint",
})

# Unit / measurement tokens — never count as a heading body even when
# they sit after a number ("2.5 Liter", "12V Battery" style false-positives
# from spec tables).
_UNIT_WORDS = frozenset({
    "liter", "litre", "liters", "litres", "ml", "cc", "cm", "mm", "m",
    "in", "inch", "inches", "ft", "ftlb", "lb", "kg", "g", "gram", "grams",
    "kw", "hp", "rpm", "nm", "ah", "v", "w", "watt", "watts", "amp",
    "amps", "volt", "volts", "psi", "bar", "kpa", "mpa", "ohm", "ohms",
    "teeth", "links", "deg", "degree", "degrees", "sec", "secs", "min",
    "mins", "hour", "hours", "ratio", "set", "sets", "pcs", "qty", "no",
    "nos", "dia", "dia.", "ea",
})

_ALL_CAPS_HEADING_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9 &/\-+()',]{3,80})\s*[:\-]?\s*$"
)

_PAGE_NUM_RE = re.compile(
    r"^\s*(?:[-–—]\s*)?(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?"
    r"\s*(?:[-–—]\s*)?\s*$",
    flags=re.IGNORECASE,
)

_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", flags=re.UNICODE)

_OCR_AUGMENT_MARKER = "[OCR augment]"

_TABLE_MARKER_RE = re.compile(r"^\s*\[Table\s+\d+\]\s*$")

# `[:\-–—]+` lets us match common combinations like `NOTE :-` or
# `WARNING:--` that show up after OCR on dashed-underline templates.
_CHROME_LABEL_RE = re.compile(
    r"^\s*(?:NOTE|NOTES|CAUTION|WARNING|WARNINGS|REMARKS?|HINT|TIP|TIPS|IMPORTANT|"
    r"DANGER|ATTENTION)\s*[:\-–—\s]*$",
    flags=re.IGNORECASE,
)

# Curated to avoid false positives — every entry is a compound where
# the pre-space half is not a standalone English word.
_LIGATURE_FIXES = {
    "speci c": "specific",
    "speci cally": "specifically",
    "speci cation": "specification",
    "speci cations": "specifications",
    "speci ed": "specified",
    "speci es": "specifies",
    "speci fy": "specify",
    "con rm": "confirm",
    "con rmed": "confirmed",
    "con rms": "confirms",
    "con rmation": "confirmation",
    "con guration": "configuration",
    "con gurations": "configurations",
    "con gured": "configured",
    "con dent": "confident",
    "con dence": "confidence",
    "con dential": "confidential",
    "de ne": "define",
    "de ned": "defined",
    "de nes": "defines",
    "de nition": "definition",
    "de nitions": "definitions",
    "identi cation": "identification",
    "identi ed": "identified",
    "identi es": "identifies",
    "veri ed": "verified",
    "veri es": "verifies",
    "veri cation": "verification",
    "modi cation": "modification",
    "modi cations": "modifications",
    "modi ed": "modified",
    "noti cation": "notification",
    "noti cations": "notifications",
    "noti ed": "notified",
    "classi cation": "classification",
    "classi ed": "classified",
    "quali cation": "qualification",
    "quali ed": "qualified",
    "recti cation": "rectification",
    "recti ed": "rectified",
    "recti er": "rectifier",
    "ampli er": "amplifier",
    "ampli cation": "amplification",
    "puri cation": "purification",
    "puri ed": "purified",
    "magni cation": "magnification",
    "magni ed": "magnified",
    "uni ed": "unified",
    "simpli ed": "simplified",
    "simpli cation": "simplification",
    "intensi ed": "intensified",
    "satis ed": "satisfied",
    "satis es": "satisfies",
    "noti er": "notifier",
    "puri er": "purifier",
    "humidi er": "humidifier",
    "intensi er": "intensifier",
    "ampli ers": "amplifiers",
    "speci cally": "specifically",
    "bene t": "benefit",
    "bene ts": "benefits",
    "bene cial": "beneficial",
    "ef cient": "efficient",
    "ef ciency": "efficiency",
    "suf cient": "sufficient",
    "suf ciently": "sufficiently",
    "dif cult": "difficult",
    "dif culty": "difficulty",
    "ef cacy": "efficacy",
    "of cial": "official",
    "trafc": "traffic",
}

# Compile once: case-insensitive match of any whole-phrase key.
_LIGATURE_FIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_LIGATURE_FIXES, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Table cells extracted by camelot don't go through the upstream
# `_SMART_REPLACEMENTS` pass, so `Speciﬁcation` (with U+FB01) leaks
# into `headers`/`rows` until we normalise it here.
_GLYPH_NORMALISE_TABLE = {
    "ﬀ": "ff",   # ﬀ
    "ﬁ": "fi",   # ﬁ
    "ﬂ": "fl",   # ﬂ
    "ﬃ": "ffi",  # ﬃ
    "ﬄ": "ffl",  # ﬄ
    "ﬅ": "ft",   # ﬅ
    "ﬆ": "st",   # ﬆ
    "'": "'",    # left single quote
    "'": "'",    # right single quote
    """: '"',    # left double quote
    """: '"',    # right double quote
    "–": "-",    # en dash
    "—": "-",    # em dash
    " ": " ",    # non-breaking space
}


def _normalise_glyphs(value: str) -> str:
    if not value:
        return value
    for k, v in _GLYPH_NORMALISE_TABLE.items():
        if k in value:
            value = value.replace(k, v)
    return value


def _drop_ocr_augment(text: str) -> str:
    idx = text.find(_OCR_AUGMENT_MARKER)
    if idx < 0:
        return text
    return text[:idx].rstrip()


def _drop_inline_table_renders(text: str) -> str:
    """Truncate at the first `[Table N]` marker line.

    Tables are emitted as their own `type: "table"` records, so leaving
    the rendered copy in the text chunk would push the same content into
    two embeddings (and through to the LLM twice during retrieval).
    """
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if _TABLE_MARKER_RE.match(ln):
            return "\n".join(lines[:i]).rstrip()
    return text


_PART_NUM_CANDIDATE_RE = re.compile(
    r"\b[A-Z]{1,3}\d[\dA-Z\s]{6,20}",
    flags=re.MULTILINE,
)


def _normalise_part_numbers(text: str) -> str:
    """Fix two recurring OCR errors in BSA-style part numbers:

      1. Internal whitespace splits   `T0208B VF0070N`  ->  `T0208BVF0070N`
                                      `T0208B\\nVF0070N` -> `T0208BVF0070N`
      2. Letter `O` substituted for digit `0`  `017ON`  ->  `0170N`

    BSA part codes never contain the letter `O` — so once we've
    classified a token as a part number, replacing every `O` with `0`
    is safe.
    """
    def _fix(match: re.Match) -> str:
        raw = match.group(0)
        clean = re.sub(r"\s+", "", raw)
        if not (9 <= len(clean) <= 18):
            return raw
        if not re.fullmatch(r"[A-Z]{1,3}\d{3,}[\dA-Z]*", clean):
            return raw
        # Inside a confirmed part code, every `O` is OCR noise for `0`.
        return clean.replace("O", "0")

    return _PART_NUM_CANDIDATE_RE.sub(_fix, text)


def _drop_table_cells_from_text(text: str, tables: list[dict]) -> str:
    """Remove text lines that exactly match a table cell value, and short
    text lines that are wrap fragments of table cells/headers.

    Parts-catalogue pages get extracted with each cell on its own line AND
    parsed structurally into the `tables` array — same data, two formats.
    We keep the structured copy and drop the column-broken lines from the
    text so embeddings don't see the parts list twice.

    Two layers of matching:
      a. Exact match against any cell or header value.
      b. Token match — short lines (<=4 words, <=30 chars) where every
         non-trivial token is also a token of some table header or cell.
         Only triggered when the line is short, so prose like
         "Fault: the sensor reads zero..." is not mis-dropped.
    """
    if not tables:
        return text

    cell_values: set[str] = set()
    cell_tokens: set[str] = set()
    for tbl in tables:
        for row in tbl.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            for v in row.values():
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    cell_values.add(s)
                    # Also store the normalised form so a broken
                    # `T0208B VF0070N` table cell still matches a clean
                    # `T0208BVF0070N` text line (and vice versa).
                    cell_values.add(_normalise_part_numbers(s))
                    for tok in _tokenise_for_match(s):
                        cell_tokens.add(tok)
        for h in tbl.get("headers", []) or []:
            if h:
                hs = str(h).strip()
                cell_values.add(hs)
                for tok in _tokenise_for_match(hs):
                    cell_tokens.add(tok)
    if not cell_values:
        return text

    out_lines: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out_lines.append(ln)
            continue
        if s in cell_values or _normalise_part_numbers(s) in cell_values:
            continue
        if len(s) <= 30:
            line_tokens = _tokenise_for_match(s)
            meaningful = [t for t in line_tokens if len(t) >= 3]
            if (meaningful
                    and len(line_tokens) <= 4
                    and all(t in cell_tokens for t in meaningful)):
                continue
        out_lines.append(ln)
    return "\n".join(out_lines)


_TOKEN_SPLIT_RE = re.compile(r"[\s,;:()/\\\-]+")


def _tokenise_for_match(s: str) -> list[str]:
    if not s:
        return []
    out: list[str] = []
    for tok in _TOKEN_SPLIT_RE.split(s.lower()):
        tok = tok.strip(".'\"")
        if tok:
            out.append(tok)
    return out


def _strip_pua(text: str) -> str:
    return _PUA_RE.sub(" ", text)


def _dehyphenate(text: str) -> str:
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def _fix_ligature_drops(text: str) -> str:
    """Restore high-confidence `fi`/`fl` ligature drops from OCR.

    Uses a curated phrase dictionary so we only fix cases where the
    pre-space half is not a standalone English word — that keeps real
    two-word phrases from being mangled.
    """
    if " " not in text:
        return text

    def _sub(m: re.Match) -> str:
        original = m.group(0)
        fix = _LIGATURE_FIXES.get(original.lower())
        if fix is None:
            return original
        # Preserve case of the leading character.
        if original[0].isupper():
            return fix[0].upper() + fix[1:]
        return fix

    return _LIGATURE_FIX_RE.sub(_sub, text)


def _is_chrome_label(line: str) -> bool:
    return bool(_CHROME_LABEL_RE.match(line))


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False  # blank lines are handled by the blank collapser
    if len(s) == 1 and not s.isalnum():
        return True
    if _PUNCT_ONLY_RE.match(s):
        return True
    return False


def _is_page_number(line: str) -> bool:
    return bool(_PAGE_NUM_RE.match(line))


def _drop_first_or_last_pagenum(lines: list[str]) -> list[str]:
    """Strip a page number appearing as the first or last non-blank line.

    Keeps numeric-ish lines in the middle of content — those are usually
    section numbers or step counts, not folios.
    """
    if not lines:
        return lines
    out = list(lines)
    while out and (not out[0].strip() or _is_page_number(out[0])):
        if out[0].strip():
            out.pop(0)
        else:
            break
    while out and (not out[-1].strip() or _is_page_number(out[-1])):
        if out[-1].strip():
            out.pop()
        else:
            break
    return out


def _collapse_blank_lines(lines: list[str], max_blanks: int = 1) -> list[str]:
    out: list[str] = []
    blanks = 0
    for ln in lines:
        if not ln.strip():
            blanks += 1
            if blanks <= max_blanks:
                out.append("")
        else:
            blanks = 0
            out.append(ln)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _detect_running_header(pages_text: list[str], min_runs: int = 3) -> str | None:
    """Return the first non-blank line that opens `min_runs` consecutive pages.

    Workshop / owner / parts manuals reuse a header line on every page in a
    chapter. When that line shows up at the top of 3+ consecutive pages, treat
    it as page chrome and strip it — it's still preserved as the section title
    via the heading detector.
    """
    firsts: list[str] = []
    for t in pages_text:
        for ln in t.splitlines():
            s = ln.strip()
            if s:
                firsts.append(s)
                break
        else:
            firsts.append("")

    run = 0
    candidate: str | None = None
    best: tuple[int, str] | None = None
    for ln in firsts:
        if ln and ln == candidate:
            run += 1
        else:
            if best is None or run > best[0]:
                if candidate and run >= min_runs:
                    best = (run, candidate)
            candidate = ln
            run = 1
    if best is None or run > best[0]:
        if candidate and run >= min_runs:
            best = (run, candidate)
    return best[1] if best else None


def _strip_header(text: str, header: str) -> str:
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        if ln.strip() == header.strip():
            return "\n".join(lines[:i] + lines[i + 1:])
        return text
    return text


def clean_page_text(text: str,
                    tables: list[dict] | None = None,
                    *,
                    doc_header_tokens: frozenset[str] | None = None) -> str:
    """Run the full cleaning pipeline on one page's text.

    `tables` is the page's parsed table list; cells are subtracted
    line-by-line so a parts-catalogue page isn't represented twice.

    `doc_header_tokens` catches wrap residue on text-only pages where the
    table on a previous page bled fragments into the page text extractor.
    """
    if not text:
        return ""
    text = _drop_ocr_augment(text)
    text = _drop_inline_table_renders(text)
    if tables:
        text = _drop_table_cells_from_text(text, tables)
    text = _strip_pua(text)
    text = _normalise_glyphs(text)
    text = _dehyphenate(text)
    text = _fix_ligature_drops(text)
    text = _normalise_part_numbers(text)

    lines = text.splitlines()
    lines = [ln for ln in lines
             if not _is_noise_line(ln) and not _is_chrome_label(ln)]
    if doc_header_tokens:
        lines = [ln for ln in lines
                 if not _is_doc_header_fragment(ln, doc_header_tokens)]
    lines = _drop_first_or_last_pagenum(lines)
    lines = _collapse_blank_lines(lines, max_blanks=1)
    return "\n".join(lines).strip()


def _is_doc_header_fragment(line: str, header_tokens: frozenset[str]) -> bool:
    """True for short orphan lines that are wrap residue from doc-wide
    table headers/cells.

    Three acceptance shapes:
      a. Short header-word residue: line <=30 chars, <=4 tokens, every
         meaningful (>=3 char) token is a known header token.
      b. Pure-numeric residue: every token is a pure digit string and
         every token appears in `header_tokens`.
      c. Tiny abbreviation residue: line <=6 chars and EVERY token
         (regardless of length) is in `header_tokens`. Catches stray
         `SR.` / `NO.` lines that fall under the >=3-char threshold.
    """
    s = line.strip()
    if not s:
        return False
    line_tokens = _tokenise_for_match(s)
    if not line_tokens:
        return False

    if all(t.isdigit() for t in line_tokens):
        return all(t in header_tokens for t in line_tokens)

    if len(s) <= 6 and all(t in header_tokens for t in line_tokens):
        return True

    if len(s) > 30 or len(line_tokens) > 4:
        return False
    meaningful = [t for t in line_tokens if len(t) >= 3]
    if not meaningful:
        return False
    return all(t in header_tokens for t in meaningful)


def _collect_doc_header_tokens(pages: list[dict]) -> frozenset[str]:
    """Walk every page's tables and return the union of header + numeric-cell tokens.

    Two token classes collected:
      a. Header words — every token of length >=3 from any header.
      b. Numeric cell values — pure-digit tokens from any cell.
         These pick up rows like "6000 12000 18000 24000 30000" that
         leak into text on continuation pages of a maintenance schedule.
         Pure-digit cells are safe to drop on a text-only page because
         a standalone line of joined-up numbers is not real prose.
    """
    tokens: set[str] = set()
    for p in pages:
        for tbl in p.get("tables", []) or []:
            for h in tbl.get("headers", []) or []:
                if not h:
                    continue
                # Glyph-normalise so a header `Rectiﬁcation` (with U+FB01)
                # tokenises identically to a text line `Rectification`.
                # Collect ALL tokens — short abbreviations like `SR.`/`NO.`
                # tokenise to `sr`/`no` and need to be in the set so we
                # can catch them as residue.
                normalised = _normalise_glyphs(_fix_ligature_drops(str(h)))
                for tok in _tokenise_for_match(normalised):
                    if tok:
                        tokens.add(tok)
            for row in tbl.get("rows", []) or []:
                values = (row.values() if isinstance(row, dict)
                          else row if isinstance(row, list) else [])
                for v in values:
                    if v is None:
                        continue
                    s = str(v).strip()
                    # Tokenise inside the cell so a cell like "12000 18000"
                    # contributes both numbers separately.
                    for tok in re.split(r"[\s,/;]+", s):
                        tok = tok.strip()
                        if tok.isdigit() and len(tok) >= 3:
                            tokens.add(tok)
                    # Label-shaped row cells feed the word-token set too —
                    # catches "Technical Specification" style row-header cells
                    # that bleed into text on adjacent pages.
                    if len(s) <= 40 and len(s.split()) <= 4 and not s.endswith("."):
                        normalised = _normalise_glyphs(_fix_ligature_drops(s))
                        for tok in _tokenise_for_match(normalised):
                            if len(tok) >= 3 and not tok.isdigit():
                                tokens.add(tok)
    return frozenset(tokens)


def _parse_heading(line: str, *, isolated: bool = False,
                   next_line: str | None = None
                   ) -> tuple[str, ...] | None:
    """Classify `line` as a heading, returning a tagged tuple:

        ("main", "<number>", "<title>")  numbered chapter/section
        ("main", None,        "<title>") ALL-CAPS isolated section
        ("sub",                "<title>") sub-topic
        None                              not a heading

    `next_line` is the next non-blank line on the page, used to confirm
    short (4–5 letter) ALL-CAPS lines as real sub-topic headings rather
    than spec values like `DOHC`. A short ALL-CAPS line is accepted only
    when the line below reads like descriptive prose.
    """
    s = line.strip()
    if not s or len(s) > 120:
        return None
    if s.endswith((".", ";", ",")):  # full sentences are not headings
        return None

    m = _NUMBERED_HEADING_RE.match(s)
    if m:
        number = m.group(1).strip()
        body = m.group(2).strip().rstrip(":").rstrip("-").strip()
        if _is_unit_body(body):
            return None
        # Reject body text masquerading as a heading. Real headings in
        # BSA manuals are 1-7 words, ~50 chars max. When the PDF extractor
        # merges a heading line with the next sentence, the regex's 80-char
        # limit alone isn't enough — we also reject sentence-shaped
        # candidates by word count and body-text markers.
        if _looks_like_body_sentence(body):
            return None
        return ("main", number, _normalise_heading(body))

    m = _SUBTOPIC_RE.match(s)
    if m:
        body = m.group(1).strip().rstrip(":").rstrip("-").strip(" .,")
        if _looks_like_subtopic_body(body):
            return ("sub", _normalise_heading(body))

    if not isolated:
        return None

    m = _ALL_CAPS_HEADING_RE.match(s)
    if not m:
        return None
    body = m.group(1).strip().rstrip(":").rstrip("-").strip()

    if body.lower() in _CALLOUT_LABELS:
        return None

    if "(" in body and body.endswith(")"):
        return None

    letters = [c for c in body if c.isalpha()]
    if len(letters) < 4:
        return None
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio < 0.90:
        return None

    # Reject lines that are obviously a part-spec value: contain digits
    # AND units / dimension separators (`100 X 83 mm`, `M6X30`, `2.5L`).
    if any(c.isdigit() for c in body):
        if re.search(r"\d.*[xX×*].*\d|\d\s*(mm|cm|kg|nm|rpm|hp|kw|ah|v|w|l|ml)\b",
                     body, flags=re.IGNORECASE):
            return None
        digit_ratio = sum(1 for c in body if c.isdigit()) / len(body)
        if digit_ratio > 0.20:
            return None

    # Short headings (4–5 letters) accepted as SUB-TOPICS only when the
    # next non-blank line reads like prose. This catches real sub-topics
    # like `FUSE` while rejecting one-word spec values like `DOHC`.
    if len(letters) < 6:
        if not _looks_descriptive(next_line):
            return None
        return ("sub", _normalise_heading(body))

    if _looks_like_body_sentence(body):
        return None

    return ("main", None, _normalise_heading(body))


def _looks_descriptive(line: str | None) -> bool:
    if not line:
        return False
    s = line.strip()
    if len(s) < 25:
        return False
    if not any(c.islower() for c in s):
        return False
    if len(s.split()) < 4:
        return False
    return True


def _looks_like_subtopic_body(body: str) -> bool:
    """True when `body` reads like a sub-topic title rather than a
    procedure step. Conservative on purpose — this runs without the
    `prev_blank` isolation check, so false positives are very visible.
    """
    if len(body) < 5 or len(body) > 120:
        return False
    words = body.split()
    if len(words) < 2:
        return False
    if not body[0].isalpha() or not body[0].isupper():
        return False
    first_lower = words[0].lower().strip(".,:;")
    if first_lower in _IMPERATIVE_PREFIXES:
        return False
    if "." in body[:-1] and not re.search(r"[A-Z]\.[A-Z]", body):
        return False
    return True


class _BreadcrumbTracker:
    """Build breadcrumb paths from numbered headings as the document is read.

    BSA workshop manuals reuse local procedure numbers ("3.1", "3.2") inside
    larger numbered sections, so a "3.1" registered on page 154 shouldn't be
    treated as the parent of "3.1.3 SHIM SETTING" 49 pages later.
    `_FRESHNESS_PAGES` controls how far back a prefix can be used as a parent.
    """

    _FRESHNESS_PAGES = 30  # stale parents beyond this many pages are skipped

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, int]] = {}

    def register(self, number: str, title: str, page: int = 0) -> str:
        self._entries[number] = (title, page)
        parts = number.split(".")
        crumbs: list[str] = []
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            entry = self._entries.get(prefix)
            if not entry:
                continue
            t, p = entry
            # Skip parents registered too long ago — almost always a
            # different chapter that just happened to reuse the number.
            if page and p and (page - p) > self._FRESHNESS_PAGES:
                continue
            if not crumbs or crumbs[-1] != t:
                crumbs.append(t)
        return " > ".join(crumbs) if crumbs else title


# PDF extraction occasionally merges a heading line with the next sentence;
# this catches the fused line and rejects it as a heading.
_BODY_TEXT_MARKERS = (
    "needs to be", "need to be", "should be", "must be",
    "following point", "the following", "as follows", "as shown",
    "before assembling", "before installing", "before removing",
    "after installing", "after assembling",
    "is required", "are required", "is recommended", "are recommended",
    "with respect to", "in order to", "in accordance with",
    "such that", " and the ", " or the ", " of the ", " for the ",
)


def _looks_like_body_sentence(body: str) -> bool:
    """True when a heading-shaped candidate is actually a body sentence
    fused with a heading by the PDF extractor.

    Real BSA headings are 1-7 words, ≤55 chars, noun phrase. Body lines
    that get glued onto the heading number have 8+ words, contain
    auxiliary verbs, or include connector phrases like "needs to be".
    """
    s = body.strip()
    if not s:
        return False
    words = s.split()
    if len(words) > 8:
        return True
    if len(s) > 55 and len(words) > 6:
        return True
    low = " " + s.lower() + " "
    for marker in _BODY_TEXT_MARKERS:
        if marker in low:
            return True
    # Trailing connector word indicates the line was cut mid-sentence.
    last = words[-1].lower().strip(",;:.")
    if last in {"and", "or", "with", "for", "to", "of", "the", "a", "an", "in"}:
        return True
    return False


def _is_unit_body(body: str) -> bool:
    """True when the heading body is just a unit / quantity word.

    Catches false positives from spec tables like `2.5 Liter`, `45 TEETH` —
    the regex sees `<number> <Word>` and otherwise treats them as headings.
    """
    words = [w.lower().strip(".:,;") for w in body.split() if w.strip()]
    if not words:
        return True
    if len(words) == 1 and words[0] in _UNIT_WORDS:
        return True
    if len(words) <= 2 and all(w in _UNIT_WORDS for w in words):
        return True
    return False


def _normalise_heading(s: str) -> str:
    out: list[str] = []
    for tok in re.split(r"(\s+|[/&\-])", s):
        if not tok or tok.isspace() or tok in {"&", "/", "-"}:
            out.append(tok)
            continue
        if len(tok) <= 4 and tok.isupper() and any(c.isalpha() for c in tok):
            out.append(tok)  # likely acronym
        else:
            out.append(tok.capitalize())
    return "".join(out).strip()


def split_into_sections(cleaned_text: str,
                        carry_main: str,
                        carry_sub: str,
                        tracker: _BreadcrumbTracker,
                        page: int = 0,
                        ) -> tuple[list[tuple[str, str]], str, str]:
    """Split a cleaned page into (section_path, body) tuples.

    `section_path` is `<main>` or `<main> > <sub>`. A new MAIN heading
    clears the SUB; a new SUB heading replaces the previous one while
    keeping the MAIN. Returns the (main, sub) state to carry into the
    next page.
    """
    lines = cleaned_text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_main = carry_main or ""
    current_sub = carry_sub or ""
    current_body: list[str] = []
    last_was_heading = False  # treat back-to-back headings as isolated

    def _path() -> str:
        if current_main and current_sub:
            return f"{current_main} > {current_sub}"
        return current_main or current_sub

    def _flush():
        if current_body or sections:
            sections.append((_path(), current_body[:]))
        elif _path():
            sections.append((_path(), []))

    for i, ln in enumerate(lines):
        if _TABLE_MARKER_RE.match(ln):
            current_body.append(ln)
            last_was_heading = False
            continue
        # A line qualifies for the loose ALL-CAPS heading path when the
        # line above is blank, the page just started, or the line above
        # was itself a heading (so a chapter title page like
        # `MAINTENANCE AND SERVICING\nFUSE` detects both lines).
        prev_blank = (i == 0) or not lines[i - 1].strip()
        isolated = prev_blank or last_was_heading
        # Find the next non-blank line so the heading detector can use
        # it as context — short ALL-CAPS lines need this confirmation.
        next_non_blank: str | None = None
        for j in range(i + 1, min(i + 5, len(lines))):
            if lines[j].strip():
                next_non_blank = lines[j]
                break
        parsed = _parse_heading(ln, isolated=isolated,
                                next_line=next_non_blank)
        if parsed is None:
            current_body.append(ln)
            last_was_heading = False
            continue

        kind = parsed[0]
        if kind == "main":
            _, number, title = parsed
            _flush()
            current_main = (tracker.register(number, title, page=page)
                            if number else title)
            current_sub = ""  # new section -> clear any sub-topic
            current_body = []
        elif kind == "sub":
            _, title = parsed
            _flush()
            current_sub = title
            current_body = []
        last_was_heading = True

    _flush()

    if not sections:
        sections = [(_path(), [])]

    out = [(path, "\n".join(body).strip()) for path, body in sections]
    return out, current_main, current_sub


def _format_text(section: str, body: str) -> str:
    if section:
        return f"Section: {section}\n{body}".strip()
    return body.strip()


def process(payload: dict, *, source_name: str | None = None) -> list[dict]:
    """Convert the text-extractor payload into clean section records."""
    pages: list[dict] = payload.get("pages", [])
    source = source_name or payload.get("pdf_name") or "unknown"

    # Pre-clean every page so the running-header detector sees real
    # content (not OCR garbage or page numbers). Pass each page's
    # parsed tables in so column cells already captured as table rows
    # are stripped from the text body. Doc-wide header tokens catch
    # fragments on text-only pages where the table on a previous page
    # bled wrap residue into the page text extractor.
    doc_header_tokens = _collect_doc_header_tokens(pages)
    cleaned_pages: list[str] = [
        clean_page_text(p.get("text", ""),
                        p.get("tables", []),
                        doc_header_tokens=doc_header_tokens)
        for p in pages
    ]
    header = _detect_running_header(cleaned_pages, min_runs=3)

    tracker = _BreadcrumbTracker()

    # If the running header is itself a heading-shaped line it's almost
    # always the chapter title. Track per-page whether the header actually
    # appeared at the top so we only seed those pages with `header_seed` —
    # pages with a different first line belong to a different chapter.
    header_seed: str = ""
    header_on_page: list[bool] = [False] * len(cleaned_pages)
    if header:
        parsed_header = _parse_heading(header, isolated=True)
        if parsed_header is not None and parsed_header[0] == "main":
            _, num, title = parsed_header
            header_seed = (tracker.register(num, title) if num else title)
        stripped_pages: list[str] = []
        for i, t in enumerate(cleaned_pages):
            new_t = _strip_header(t, header)
            if new_t != t:
                header_on_page[i] = True
            stripped_pages.append(new_t)
        cleaned_pages = stripped_pages

    records: list[dict] = []
    carry_main: str = ""
    carry_sub: str = ""

    for idx, (page_dict, cleaned) in enumerate(zip(pages, cleaned_pages)):
        page_num = page_dict.get("page")
        method = page_dict.get("method", "text")
        tables = page_dict.get("tables", []) or []
        text_label = _label_for_method(method)

        if header_on_page[idx] and header_seed:
            seed_main = header_seed
        else:
            seed_main = carry_main

        # Carry the sub-topic ONLY when the chapter (main) hasn't changed
        # across the page boundary — a sub-topic from chapter A must not
        # leak into chapter B.
        seed_sub = carry_sub if seed_main == carry_main else ""

        sections, carry_main, carry_sub = split_into_sections(
            cleaned, seed_main, seed_sub, tracker,
            page=int(page_num) if isinstance(page_num, (int, float)) else 0,
        )

        section_idx = 0
        table_idx = 0
        last_section_idx = 0

        for path, body in sections:
            if not body:
                # No text under this heading on this page. Tables (if any)
                # will still be attributed to the heading via `last_path`;
                # we just don't emit an empty stub.
                continue
            section_idx += 1
            last_section_idx = section_idx
            chunk_id = f"p{page_num}_s{section_idx}_c1"
            records.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "section": path,
                "type": "text",
                "text": _format_text(path, body),
                "images": [],
                "metadata": {
                    "source": source,
                    "method": text_label,
                },
            })

        if tables and last_section_idx == 0:
            last_section_idx = 1
        if sections:
            last_path = sections[-1][0]
        elif carry_main and carry_sub:
            last_path = f"{carry_main} > {carry_sub}"
        else:
            last_path = carry_main or carry_sub

        for tbl in tables:
            t_headers, t_rows = _structure_table(tbl)
            if not _is_useful_table(t_headers, t_rows):
                continue  # camelot misfire (diagram pretending to be a table)
            table_idx += 1
            chunk_id = f"p{page_num}_s{last_section_idx}_t{table_idx}"
            records.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "section": last_path,
                "type": "table",
                "text": _table_to_prose(t_headers, t_rows),
                "headers": t_headers,
                "rows": t_rows,
                "images": [],
                "metadata": {
                    "source": source,
                    "method": "structured",
                    "table_index": table_idx,
                    "table_flavor": tbl.get("flavor"),
                },
            })

    return records


def _structure_table(table: dict) -> tuple[list[str], list[list[str]]]:
    """Convert the upstream table dict into positional (headers, rows).

    Two structural fix-ups:
      1. Banner-header promotion — when camelot's header row is really a
         chapter banner, the first data row is the actual header.
      2. Continuation-row merge — camelot lattice splits each multi-line
         cell into N positional rows. Re-glue by picking a key column and
         merging rows whose key column is blank into the previous keyed row.

    Headers longer than 80 chars are treated as "" — camelot occasionally
    absorbs an entire banner row into the first header which is useless as
    a column label.
    """
    raw_headers = table.get("headers") or []
    raw_rows = table.get("rows") or []

    # Determine column order from the first dict row — handles
    # camelot's `col2`, `col3` placeholder keys and any header gaps.
    sample_dict = next((r for r in raw_rows if isinstance(r, dict)), None)
    cols: list[tuple[str, str]] = []
    if sample_dict:
        for i, key in enumerate(sample_dict.keys()):
            raw_label = (str(raw_headers[i]).strip()
                         if i < len(raw_headers) else "")
            label = raw_label if 0 < len(raw_label) <= 80 else ""
            cols.append((label, key))

    headers = [_normalise_glyphs(label) for label, _ in cols]
    rows: list[list[str]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            r: list[str] = []
            for _, key in cols:
                v = row.get(key, "")
                v = "" if v is None else str(v).strip()
                if v:
                    v = _normalise_glyphs(v)
                    v = _fix_ligature_drops(v)
                    v = _normalise_part_numbers(v)
                r.append(v)
            if any(c for c in r):
                rows.append(r)
        else:
            t = str(row).strip()
            if t:
                t = _normalise_glyphs(t)
                t = _fix_ligature_drops(t)
                rows.append([_normalise_part_numbers(t)])

    headers, rows = _promote_header_row(headers, rows)
    rows = _merge_continuation_rows(rows)
    return headers, rows


def _promote_header_row(headers: list[str],
                        rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Promote the first data row to headers when camelot's headers are bogus.

    Two failure modes:
      a. Banner row — camelot dropped a chapter title into header[0] and
         left the rest empty. The real header is row[0].
      b. All blank — camelot found a borderline lattice and assigned no
         header. Same fix: promote row[0] when it looks header-like.
    """
    if not rows:
        return headers, rows

    non_empty_headers = sum(1 for h in headers if h)
    if non_empty_headers >= max(2, len(headers) // 2):
        return headers, rows  # camelot's headers are good enough

    first = rows[0]
    if not _looks_like_header_row(first):
        return headers, rows

    new_headers = [c.strip() for c in first]
    remaining = rows[1:]

    # Stacked-header glue: if row[1] also looks like a header continuation
    # (short, no prose, fills cells that row[0] left empty), merge it in.
    # Continuation rows can have just 1 non-empty cell — `SR.` over `NO.`
    # in parts catalogues is the canonical case — so we use a relaxed check.
    if remaining and _looks_like_header_continuation(remaining[0]):
        cont = remaining[0]
        glued = list(new_headers)
        for i, c in enumerate(cont):
            c = c.strip()
            if not c:
                continue
            if i >= len(glued):
                glued.append(c)
            elif glued[i]:
                glued[i] = f"{glued[i]} {c}".strip()
            else:
                glued[i] = c
        if all(len(c) <= 40 for c in glued):
            new_headers = glued
            remaining = remaining[1:]

    return new_headers, remaining


def _looks_like_header_continuation(row: list[str]) -> bool:
    """True if `row` plausibly is the second line of a 2-row header.

    Relaxed vs `_looks_like_header_row`: only requires >=1 non-empty cell
    because a stacked header like `SR. / NO.` puts a single short word in
    the continuation row.
    """
    if not row:
        return False
    cells = [c.strip() for c in row]
    non_empty = [c for c in cells if c]
    if not non_empty:
        return False
    for c in non_empty:
        if len(c) > 30:
            return False
        if any(p in c for p in (". ", "; ", ": ")):
            return False
        if c.endswith((";", ",")):
            return False
        if c.endswith(".") and len(c.split()) > 1:
            return False
        if len(c.split()) > 3:
            return False
    return True


def _looks_like_header_row(row: list[str]) -> bool:
    """True if `row` looks like a column-label row, not data.

    Trailing-period abbreviations like `SR.` / `NO.` / `REF.` are allowed
    because parts-catalogue header rows lean on them; sentence-style periods
    (multi-word cells ending `.`) are still rejected.
    """
    if not row:
        return False
    cells = [c.strip() for c in row]
    non_empty = [c for c in cells if c]
    if len(non_empty) < 2:
        return False
    for c in non_empty:
        if len(c) > 40:
            return False
        if any(p in c for p in (". ", "; ", ": ")):
            return False
        if c.endswith((";", ",")):
            return False
        if c.endswith(".") and len(c.split()) > 1:
            return False
        if len(c.split()) > 5:
            return False
    return True


def _merge_continuation_rows(rows: list[list[str]]) -> list[list[str]]:
    """Glue back rows that camelot split because cells spanned >1 line.

    Strategy:
      1. Pick a key column — the leftmost column with >=2 distinct non-empty
         values. The "distinct" check rejects columns where the only non-empty
         value is a single repeated label.
      2. A row whose key column is filled starts a new logical row. A row
         whose key column is blank is a continuation: append each non-empty
         cell into the previous logical row, joining with `\\n`.
      3. Orphan rows above the first keyed row are kept as-is.

    If no usable key column exists (1-row table) we return rows unchanged.
    """
    if len(rows) < 2:
        return rows

    n_cols = max(len(r) for r in rows)
    padded = [list(r) + [""] * (n_cols - len(r)) for r in rows]

    key_col = -1
    for c in range(n_cols):
        seen: set[str] = set()
        for r in padded:
            v = r[c].strip()
            if v:
                seen.add(v)
        if len(seen) >= 2:
            key_col = c
            break

    if key_col < 0:
        return rows

    merged: list[list[str]] = []
    have_parent = False
    for row in padded:
        key = row[key_col].strip()
        if key:
            merged.append(list(row))
            have_parent = True
            continue
        if not have_parent:
            merged.append(list(row))
            continue
        parent = merged[-1]
        for i, cell in enumerate(row):
            cell = cell.strip()
            if not cell:
                continue
            if parent[i]:
                parent[i] = f"{parent[i]}\n{cell}"
            else:
                parent[i] = cell

    return [[c.strip() for c in r] for r in merged]


def _is_useful_table(headers: list[str], rows: list[list[str]]) -> bool:
    """True when the structured table has any content worth keeping.

    Preservation-first: never drop a parsed table for being sparse — even
    a wiring-schematic capture is data. Only return False for truly empty
    tables (no non-empty cells at all).
    """
    if not rows:
        return False
    if not any(c for r in rows for c in r):
        return False
    return True


def _table_to_prose(headers: list[str], rows: list[list[str]]) -> str:
    """Render a structured table as readable prose.

    Format: `<intro>: <row1>; <row2>; ...; <rowN>.`
    Multi-line cells (from continuation-row merge) are flattened — newlines
    in rendered prose break sentence parsers.
    """
    if not rows:
        return ""
    rendered: list[str] = []
    for row in rows:
        norm = [_flatten_cell(v) for v in row]
        non_empty = [v for v in norm if v]
        if not non_empty:
            continue
        if len(non_empty) == 1:
            rendered.append(non_empty[0])
        else:
            first, *rest = non_empty
            rendered.append(f"{first} ({', '.join(rest)})")
    if not rendered:
        return ""
    intro = _pick_table_intro(headers)
    return f"{intro}: " + "; ".join(rendered) + "."


_WHITESPACE_RUN_RE = re.compile(r"\s+")


def _flatten_cell(value: str) -> str:
    if not value:
        return ""
    return _WHITESPACE_RUN_RE.sub(" ", str(value)).strip()


def _pick_table_intro(labels: list[str]) -> str:
    """Pick the first label that reads like a column name.

    Skips banners (camelot sometimes puts a section-title row into the
    header), placeholder names like `col2`, and anything with digits.
    """
    for label in labels:
        if not label:
            continue
        if label.startswith("col") and label[3:].isdigit():
            continue
        if any(c.isdigit() for c in label):
            continue
        if len(label) > 30:
            continue
        words = label.split()
        if 1 <= len(words) <= 3:
            return label
    return "Items"


# `structured` is reserved for table records; never assigned here.
_METHOD_LABELS = {
    "text":         "text_clean",
    "ocr_full":     "ocr_clean",
    "ocr_swap":     "ocr_clean",
    "ocr_augment":  "ocr_clean",
    "blank":        "blank",
}


def _label_for_method(method: str) -> str:
    return _METHOD_LABELS.get(method, method or "text_clean")


def _ensure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Iterable[str] | None = None) -> int:
    _ensure_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Path to *.text.json")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output path (default: <input>.clean.json)")
    ap.add_argument("--source", default=None,
                    help="Override the `metadata.source` field "
                         "(default: pdf_name from the input)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = process(payload, source_name=args.source)

    out = args.output or args.input.with_name(
        args.input.stem.replace(".text", "") + ".clean.json"
    )
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {len(records)} records -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
