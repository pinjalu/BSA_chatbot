from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEBUG_LOG_DIR = "debug_logs"

# Serialise writes so concurrent requests don't interleave log blocks
_write_lock = threading.Lock()


def _log_path() -> Path:
    base = Path(__file__).with_name(DEBUG_LOG_DIR)
    base.mkdir(exist_ok=True)
    return base / f"{datetime.now().strftime('%Y-%m-%d')}.txt"


def _fmt_match(idx: int, m: dict) -> list[str]:
    md         = (m.get("metadata") or {}) if isinstance(m, dict) else {}
    score      = m.get("score", 0.0)
    section    = md.get("section") or "(no section)"
    page       = md.get("page", "?")
    source     = md.get("source_pdf", "?")
    vehicle    = md.get("vehicle_model", "—")
    doc_type   = md.get("doc_type", "?")
    chunk_type = md.get("chunk_type", "?")
    images     = md.get("image_paths") or []

    lines = [
        f"  [{idx}] score={float(score):.3f}  page={page}  "
        f"vehicle={vehicle}  type={doc_type}  chunk={chunk_type}",
        f"      section: {section}",
        f"      source : {source}",
    ]
    if images:
        lines.append(f"      images : {len(images)} file(s)")
        for p in images[:5]:
            lines.append(f"               - {p}")
        if len(images) > 5:
            lines.append(f"               (+{len(images) - 5} more)")
    text = md.get("text") or ""
    if text:
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"
        lines.append(f"      preview: {snippet}")
    return lines


def write_debug_entry(
    *,
    question: str,
    retrieval_query: str | None = None,
    answer: str,
    matches: list[dict],
    image_map: dict[str, str] | None = None,
    classification: dict[str, Any] | None = None,
    filter_used: dict | None = None,
    top_k: int | None = None,
    grounding: dict | None = None,
    timings_ms: dict | None = None,
    msg_class: str | None = None,
    user: str | None = None,
    extra: dict | None = None,
    pipeline_steps: list[str] | None = None,
) -> Path | None:
    try:
        path = _log_path()
        ts        = datetime.now().isoformat(timespec="seconds")
        sep_thick = "═" * 90
        sep_thin  = "─" * 90

        lines: list[str] = [
            sep_thick,
            f"[{ts}]  user={user or '—'}  class={msg_class or '—'}",
            sep_thick,
            "",
            f"Q: {question}",
        ]
        if retrieval_query and retrieval_query.strip() != question.strip():
            lines.append(f"   (retrieval query: {retrieval_query})")
        lines.append("")

        if classification:
            lines.append(
                f"Classifier: model={classification.get('model')}  "
                f"variant={classification.get('variant')}  "
                f"intent={classification.get('intent')}  "
                f"confident={classification.get('confident')}"
            )
            fb = classification.get("_fallback")
            if fb:
                lines.append(f"  classifier_path: {fb}")
            err = classification.get("_llm_error")
            if err:
                lines.append(f"  classifier_llm_error: {err}")
            llm_class = classification.get("_llm_class")
            if llm_class:
                lines.append(
                    f"  classifier_llm: model={llm_class.get('model')} "
                    f"variant={llm_class.get('variant')} "
                    f"intent={llm_class.get('intent')} "
                    f"confident={llm_class.get('confident')}"
                )
            regex_class = classification.get("_regex_class")
            if regex_class:
                lines.append(
                    f"  classifier_regex: model={regex_class.get('model')} "
                    f"variant={regex_class.get('variant')} "
                    f"intent={regex_class.get('intent')} "
                    f"confident={regex_class.get('confident')}"
                )
            llm_raw = classification.get("_llm_raw")
            if llm_raw:
                raw_one_line = " ".join(str(llm_raw).split())
                if len(raw_one_line) > 400:
                    raw_one_line = raw_one_line[:400] + "…"
                lines.append(f"  classifier_llm_raw: {raw_one_line}")

        if top_k is not None or filter_used is not None:
            lines.append(f"Filter: {filter_used}    top_k: {top_k}")
        if timings_ms:
            lines.append("Timings: " + "  ".join(f"{k}={v}ms" for k, v in timings_ms.items()))
        lines.append("")

        if pipeline_steps:
            lines.append("Pipeline steps:")
            for i, step in enumerate(pipeline_steps, 1):
                lines.append(f"  {i:>2}. {step}")
            lines.append("")

        lines.append(f"Retrieved chunks ({len(matches)}):")
        if not matches:
            lines.append("  (none)")
        for i, m in enumerate(matches, 1):
            lines.extend(_fmt_match(i, m))
        lines.append("")

        if image_map:
            lines.append(f"Available images ({len(image_map)}):")
            for img_id in sorted(
                image_map.keys(),
                key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0,
            ):
                lines.append(f"  {img_id} → {image_map[img_id]}")
            lines.append("")

        lines.append(sep_thin)
        lines.append("ANSWER")
        lines.append(sep_thin)
        lines.append(answer.rstrip())
        lines.append("")

        if grounding is not None:
            ok = grounding.get("ok", True)
            lines.append(f"Grounding: {'OK' if ok else 'FLAGGED'}")
            if not ok:
                for u in grounding.get("unsupported", []) or []:
                    lines.append(f"  ! {u.get('kind')}: {u.get('value')}")
            lines.append("")

        if extra:
            lines.append("Extra:")
            for k, v in extra.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        block = "\n".join(lines) + "\n"
        with _write_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(block)
        return path
    except Exception as e:  # noqa: BLE001
        log.warning("debug log write failed: %s", e)
        return None
