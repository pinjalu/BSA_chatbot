import argparse
import json
import re
import traceback
from collections import defaultdict
from pathlib import Path

MIN_CHARS = 200            # below this -> try to merge into adjacent chunk
DROP_BELOW_CHARS = 30      # after merging, body shorter than this -> drop
MAX_SECTION_LEN = 100      # sections longer than this are sentence fragments,
                           # not headings -> blank the metadata field


def _match_model(text: str) -> str | None:
    n = (text or "").lower()
    if "bantam" in n:
        return "BSA Bantam"
    if "scrambler" in n:
        return "BSA Scrambler"
    if "goldstar" in n or "gold star" in n or "gold-star" in n:
        return "BSA Gold Star"
    return None


def derive_vehicle_model(pdf_name: str, chunks: list[dict] | None = None) -> str | None:
    m = _match_model(pdf_name)
    if m:
        return m
    # filename didn't say -- fall back to scanning the first chunks of body text
    for c in (chunks or [])[:15]:
        m = _match_model(c.get("text", ""))
        if m:
            return m
    return None


# Priority-ordered classification rules for doc_type. The first regex
# whose pattern matches the joined path wins. Each pattern uses word
# boundaries so short tokens (e.g. "sop") don't substring-match
# unrelated paths ("aesop", "sophisticated"). Adding a new doc type =
# one entry in this tuple — no other code touched.
#
# Order matters: most-specific keywords first. A path that matches
# multiple rules gets the most-specific label (e.g. a "Workshop
# Manual for Owners" path lands on `workshop_manual`, not the looser
# `owners_manual` rule).
DOC_TYPE_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("warranty_booklet",  re.compile(
        r"\bwarrant(?:y|ies)\b|"
        r"\bservice[\s_-]+(?:booklet|schedule|guide)s?\b", re.I,
    )),
    ("accessories_guide", re.compile(
        r"\baccessor(?:y|ies)\b", re.I,
    )),
    ("wiring_diagram",    re.compile(
        r"\bwiring\b|\bschematic\b", re.I,
    )),
    ("parts_catalogue",   re.compile(
        r"\bparts?[\s_-]+(?:catalogue|catalog|list|book)\b|"
        r"\bparts?\s+catalog", re.I,
    )),
    ("workshop_manual",   re.compile(
        r"\bworkshop\b|\bservice[\s_-]+manual\b", re.I,
    )),
    ("owners_manual",     re.compile(
        r"\bowner'?s?\b|\bowners?[\s_-]+manual\b|"
        r"\boperator'?s?[\s_-]+manual\b", re.I,
    )),
    ("sop",               re.compile(
        r"\bsop\b|\bstandard[\s_-]+operating[\s_-]+procedure\b", re.I,
    )),
    ("spec_sheet",        re.compile(
        r"\bspec(?:[\s_-]+sheet|ification)?\b|"
        r"\bdatasheet\b|\bbrochure\b", re.I,
    )),
)


def derive_doc_type(input_path: Path, root: Path) -> str:
    """Classify the source PDF by examining its path (folder names +
    filename). Returns the matching label from DOC_TYPE_RULES, or
    `spec_sheet` as a last-resort fallback (with a warning so
    misclassifications surface during a re-index)."""
    try:
        rel = input_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = input_path
    # Join all path components into one string so regex word-boundaries
    # work across folder/filename boundaries.
    haystack = " ".join(str(p) for p in rel.parts)
    for label, pattern in DOC_TYPE_RULES:
        if pattern.search(haystack):
            return label
    # Nothing matched — emit a visible warning so the operator can spot
    # mis-classified paths in the run log instead of every unknown PDF
    # silently landing in `spec_sheet`.
    print(f"[doc_type] unrecognised path; defaulting to spec_sheet: {input_path}")
    return "spec_sheet"


def is_junk_table(chunk: dict) -> bool:
    """A table is junk (layout artifact) if any of these holds:
      - Empty headers AND <30% non-empty cells, OR
      - Empty headers AND populated cells contain NO letters (e.g.
        rows full of column-position numbers like "1 4 | 5 2 1 3").
        These look real to a cell-counting check but carry no
        searchable content; Camelot occasionally emits them when
        a layout grid is mis-parsed as a table.

    Critical: keeping these would cause `merge_tiny` to promote the
    preceding text chunk to type=table and embed the numeric garbage
    instead of the real part-number prose (Bantam parts page 64,
    front-disc row T1901AB10010N — was being silently dropped).
    """
    if chunk.get("type") != "table":
        return False
    headers = chunk.get("headers") or []
    has_headers = any((h or "").strip() for h in headers)
    rows = chunk.get("rows") or []
    total = non_empty = letter_cells = 0
    for row in rows:
        cells = row.values() if isinstance(row, dict) else row
        for c in cells:
            total += 1
            s = (c or "").strip()
            if s:
                non_empty += 1
                if any(ch.isalpha() for ch in s):
                    letter_cells += 1
    if total == 0:
        return True
    if has_headers:
        return False
    if non_empty / total < 0.3:
        return True
    # No alphabetic content anywhere → layout-grid garbage.
    if letter_cells == 0:
        return True
    return False


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def dedupe_tables(chunks: list[dict]) -> tuple[list[dict], int]:
    """Same page, ≥80% containment of one in the other -> keep one (prefer lattice)."""
    by_page: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        if c.get("type") == "table":
            by_page[c.get("page")].append(i)
    drop: set[int] = set()
    for idxs in by_page.values():
        for i, ia in enumerate(idxs):
            for ib in idxs[i + 1:]:
                if ia in drop or ib in drop:
                    continue
                ta = normalize_for_dedup(chunks[ia].get("text", ""))
                tb = normalize_for_dedup(chunks[ib].get("text", ""))
                if not ta or not tb:
                    continue
                shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
                if len(shorter) / max(len(longer), 1) < 0.8:
                    continue
                if shorter not in longer:
                    continue
                keep, kill = ia, ib
                fa = (chunks[ia].get("metadata") or {}).get("table_flavor")
                fb = (chunks[ib].get("metadata") or {}).get("table_flavor")
                if fb == "lattice" and fa != "lattice":
                    keep, kill = ib, ia
                elif fa == fb and len(chunks[ib].get("text", "")) > len(chunks[ia].get("text", "")):
                    keep, kill = ib, ia
                drop.add(kill)
                chunks[keep].setdefault("images", []).extend(chunks[kill].get("images") or [])
    return [c for i, c in enumerate(chunks) if i not in drop], len(drop)


def light_clean(text: str) -> str:
    text = text or ""
    # U+FFFD (replacement char from PDF extraction) usually stands in for ±
    # in spec tables -- "5.0 ï 0.3" = "5.0 ± 0.3". Conservative replacement.
    text = text.replace("", "±")
    text = re.sub(r"\.{8,}", "", text)
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


_ANY_SECTION_LINE = re.compile(r"^Section:[^\n]*\n+", re.IGNORECASE | re.MULTILINE)


def strip_section_prefix(text: str, section: str) -> str:
    """clean_and_section.py prepends `Section: <name>\\n` to every chunk body.
    We re-add `[Section: <name>]` for embedding context, so strip every
    `Section: ...` line out of the body -- handles:
      - the original prefix
      - mid-body copies created by merge_tiny
      - chunks whose section was blanked by sanitize_section (sentence-fragment
        sections >100 chars) -- the body still carried the original heading."""
    if not text:
        return ""
    return _ANY_SECTION_LINE.sub("", text).strip()


def sanitize_section(section: str) -> str:
    """Sections longer than MAX_SECTION_LEN are sentence fragments mis-detected
    as headings (e.g. 'ABS - Malfunction Indicator Flashes If There Is Any...').
    Blank them out so they don't pollute metadata filtering."""
    s = (section or "").strip()
    if len(s) > MAX_SECTION_LEN:
        return ""
    return s


def chunk_has_content(chunk: dict) -> bool:
    """Does this chunk carry any unique retrievable info?

    Tables: kept if ANY cell has non-empty content -- even short cells like
    'Inner Bush LH' carry searchable terms. Junk-only tables were already
    filtered by `is_junk_table`.

    Text: dropped only when, after stripping the duplicated `Section: ...`
    prefix and any standalone page-number lines, nothing substantive remains.
    These are TOC entries -- the section name is preserved as metadata on
    the real content chunk later in the document.
    """
    if chunk.get("type") == "table":
        for src in (chunk.get("headers") or []):
            if (src or "").strip():
                return True
        for row in chunk.get("rows") or []:
            cells = row.values() if isinstance(row, dict) else row
            for c in cells:
                if (c or "").strip():
                    return True
        # No cells -- fall through to text check (chunk.text may still hold the prose)
    text = chunk.get("text") or ""
    body = strip_section_prefix(text, chunk.get("section", ""))
    body = re.sub(r"^\s*\d{1,4}\s*$", "", body, flags=re.MULTILINE).strip()
    return len(body) >= DROP_BELOW_CHARS


def clean_title(title: str) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", (title or ""))


def is_garbage_title(title: str) -> bool:
    t = clean_title(title)
    if len(t) < 2:
        return True
    alnum = sum(1 for c in t if c.isalnum())
    return alnum / max(len(t), 1) < 0.4


def merge_tiny(chunks: list[dict]) -> tuple[list[dict], int]:
    """Merge a small chunk with its same-page-section sibling.

    Bidirectional merge:
      - If the CURRENT chunk is small, merge it into the PREVIOUS one
        (the original behaviour).
      - If the PREVIOUS chunk is small AND the current one is large,
        prepend the small one's text into the current chunk before
        emitting it. This catches the "tiny header followed by big
        body" pattern that the forward-only merge missed (scanner
        flagged 40 of these in the prior run).

    Cross-type merge (text<->table) is allowed when the small side is a
    stub (<100 chars), because in this corpus tiny `text` chunks are
    usually captions/headers belonging to the table that follows.
    """
    out: list[dict] = []
    merged = 0
    for c in chunks:
        if not out:
            out.append(c)
            continue
        prev = out[-1]
        same_loc = c.get("page") == prev.get("page") and c.get("section") == prev.get("section")
        small = len(c.get("text", "")) < MIN_CHARS
        prev_small = len(prev.get("text", "")) < 100
        same_type = c.get("type") == prev.get("type")
        cross_type_ok = (not same_type) and (len(c.get("text", "")) < 100 or prev_small)

        # Reverse merge: previous chunk is tiny, current is large +
        # same location. Prepend prev's text into c, then replace prev
        # with c. Only triggers when prev is a true stub (<100 chars)
        # so we don't lose substantive content by absorption.
        if (
            same_loc
            and prev_small
            and not small
            and (same_type or cross_type_ok)
        ):
            prev_text = (prev.get("text") or "").strip()
            cur_text = (c.get("text") or "").strip()
            if prev_text and prev_text not in cur_text:
                c["text"] = (prev_text + "\n" + cur_text).strip()
            # Inherit prev's images that aren't already in c.
            existing_imgs = c.setdefault("images", [])
            for img in (prev.get("images") or []):
                if img not in existing_imgs:
                    existing_imgs.append(img)
            # If c is text but prev was a table stub, keep c's type
            # (the substantive chunk's type wins).
            out[-1] = c
            merged += 1
            continue

        if same_loc and small and (same_type or cross_type_ok):
            prev["text"] = (prev.get("text", "").rstrip() + "\n" + (c.get("text") or "")).strip()
            prev.setdefault("images", []).extend(c.get("images") or [])
            # if either side had table structure, preserve it on the merged chunk
            if c.get("type") == "table" and prev.get("type") != "table":
                prev["type"] = "table"
                prev["headers"] = c.get("headers") or []
                prev["rows"] = c.get("rows") or []
                # Mark the merge so build_text_for_embedding knows the
                # prose is INDEPENDENT content (the original text
                # chunk's prose) rather than a re-flattening of the
                # table — and keeps it in the embedding text.
                prev["_merged_prose"] = True
            merged += 1
        else:
            out.append(c)
    return out, merged


def flatten_table(chunk: dict) -> str:
    headers = chunk.get("headers") or []
    rows = chunk.get("rows") or []
    lines: list[str] = []
    if any((h or "").strip() for h in headers):
        lines.append(" | ".join(h or "" for h in headers))
    for row in rows:
        cells = row.values() if isinstance(row, dict) else row
        line = " | ".join((c or "") for c in cells).strip()
        if line.replace("|", "").strip():
            lines.append(line)
    return "\n".join(lines) if lines else (chunk.get("text") or "")


def build_text_for_embedding(chunk: dict) -> str:
    parts: list[str] = []
    section = (chunk.get("section") or "").strip()
    if section:
        parts.append(f"[Section: {section}]")
    if chunk.get("type") == "table":
        prose = strip_section_prefix(chunk.get("text") or "", section)
        rows = chunk.get("rows") or []
        # Decide whether to include prose alongside the flattened table:
        #   - no rows → prose is the only data, keep it
        #   - prose < 200 chars → caption-like preamble, keep both
        #   - _merged_prose flag is set → a text chunk was merged into
        #     a table chunk earlier; the prose is INDEPENDENT data
        #     (e.g. the part-number rows that pdfplumber extracted as
        #     plain text rather than table cells). Without this branch,
        #     long merged prose was being silently dropped — the bug
        #     that caused Bantam front-disc T1901AB10010N to vanish.
        if prose and (
            not rows
            or len(prose) < 200
            or chunk.get("_merged_prose")
        ):
            parts.append(prose)
        parts.append(flatten_table(chunk))
    else:
        parts.append(strip_section_prefix(chunk.get("text") or "", section))
    return "\n".join(p for p in parts if p).strip()


def project_relative_image_path(
    input_path: Path, pdf_stem: str, image_filename: str, project_root: Path
) -> str:
    # Store as "<stem>/images/<file>" — matches the S3 key directly,
    # so api.py can presign without any path rewriting step.
    return f"{pdf_stem}/images/{image_filename}"


def process_one(input_path: Path, root: Path, project_root: Path) -> dict:
    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    pdf_name = None
    if chunks:
        pdf_name = ((chunks[0] or {}).get("metadata", {}) or {}).get("source")
    pdf_name = pdf_name or input_path.stem.replace("_final", "")
    pdf_stem = Path(pdf_name).stem
    vehicle = derive_vehicle_model(pdf_name, chunks)
    doc_type = derive_doc_type(input_path, root)

    report = {
        "input": str(input_path),
        "input_chunks": len(chunks),
        "vehicle_model": vehicle,
        "doc_type": doc_type,
    }

    cleaned = [c for c in chunks if not is_junk_table(c)]
    report["dropped_junk_tables"] = len(chunks) - len(cleaned)

    cleaned, deduped = dedupe_tables(cleaned)
    report["deduped_tables"] = deduped

    for c in cleaned:
        c["text"] = light_clean(c.get("text", ""))
        c["section"] = sanitize_section(c.get("section", ""))
        # Clean table cells too -- U+FFFD often lives in rows, not text
        if c.get("type") == "table":
            c["headers"] = [light_clean(h) for h in (c.get("headers") or [])]
            new_rows = []
            for row in c.get("rows") or []:
                if isinstance(row, dict):
                    new_rows.append({k: light_clean(v) for k, v in row.items()})
                else:
                    new_rows.append([light_clean(v) for v in row])
            c["rows"] = new_rows

    cleaned, merged = merge_tiny(cleaned)
    report["merged_tiny"] = merged

    before_drop = len(cleaned)
    cleaned = [c for c in cleaned if chunk_has_content(c)]
    report["dropped_empty"] = before_drop - len(cleaned)

    out_path = input_path.parent / f"{pdf_stem}_pinecone.jsonl"
    records = 0
    total_chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for c in cleaned:
            text = build_text_for_embedding(c)
            if not text.strip():
                continue
            images = c.get("images") or []
            image_paths = [
                project_relative_image_path(input_path, pdf_stem, i.get("file"), project_root)
                for i in images if i.get("file")
            ]
            image_titles = [
                clean_title(i.get("title")) for i in images
                if i.get("title") and not is_garbage_title(i.get("title"))
            ]
            metadata = {
                "source_pdf": pdf_name,
                "vehicle_model": vehicle,
                "doc_type": doc_type,
                "page": c.get("page"),
                "section": c.get("section", ""),
                "chunk_type": c.get("type"),
                "image_paths": image_paths,
                "image_titles": image_titles,
                "char_count": len(text),
            }
            # Pinecone metadata rejects null/None values -- omit keys instead.
            # Universal docs (warranty, SOPs) end up without `vehicle_model`,
            # which the chatbot can match via {"vehicle_model": {"$exists": False}}.
            metadata = {k: v for k, v in metadata.items() if v is not None}
            record = {
                "id": f"{pdf_stem}::{c.get('chunk_id')}",
                "text": text,
                "metadata": metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1
            total_chars += len(text)

    report["output_chunks"] = records
    report["total_chars"] = total_chars
    report["output"] = str(out_path)
    return report


def main():
    ap = argparse.ArgumentParser(description="Prepare _final.json files for Pinecone.")
    ap.add_argument("--root", default="Data")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    finals = sorted(p for p in root.rglob("*_final.json") if "_pinecone" not in p.name)
    if not finals:
        print(f"no *_final.json under {root}")
        return

    project_root = Path.cwd()
    summary = []
    for f in finals:
        try:
            r = process_one(f, root, project_root)
            summary.append(r)
            print(
                f"[ok] {f.relative_to(root)}: "
                f"{r['input_chunks']}->{r['output_chunks']} chunks "
                f"(junk={r['dropped_junk_tables']}, "
                f"dedup={r['deduped_tables']}, "
                f"merge={r['merged_tiny']}, "
                f"empty={r['dropped_empty']})"
            )
        except Exception as e:
            print(f"[err] {f}: {e}")
            traceback.print_exc()

    summary_path = root / "pinecone_prep_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total_chunks = sum(r["output_chunks"] for r in summary)
    total_chars = sum(r["total_chars"] for r in summary)
    print(
        f"\ntotal: {total_chunks} chunks, ~{total_chars:,} chars "
        f"(~{total_chars // 4:,} tokens)"
    )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
