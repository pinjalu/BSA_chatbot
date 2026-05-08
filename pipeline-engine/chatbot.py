from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv(Path(__file__).with_name(".env"))

# Windows console UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


DEFAULT_INDEX = "bsa-manuals"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5-mini"           # technical-manual RAG: preserves verbatim
                                # safety warnings, reliable numbered steps,
                                # ~2s latency, ~$0.014/question
DEFAULT_TOP_K = 6
DEFAULT_MIN_SCORE = 0.05       # vector search is now just a pre-filter;
                               # the cross-encoder reranker is what decides
                               # final relevance. Letting almost everything
                               # through (only filtering out noise like
                               # "image-only" chunks with score < 0.05)
                               # gives rerank a richer pool to pick from.
DEFAULT_PROMPT_FILE = "system_prompt.txt"
DEFAULT_USER = "default"
DEFAULT_HISTORY_TURNS = 5       # prior Q/A pairs sent to the LLM for
                                # follow-up coherence; bounds token cost
LOG_DIR = "chat_logs"           # one .txt per user, appended forever
# gpt-5 family reasoning controls. "medium" (default) makes gpt-5-mini
# spend 30-60s thinking before emitting tokens — overkill for RAG.
# "minimal" is near-instant but turns out to be too literal: it
# pattern-matches "ASK FIRST" rules and loops on clarifications. "low"
# is the sweet spot — fast (~2s TTFT) yet thoughtful enough to follow
# the conversation context (recognises when a user has already
# answered, doesn't re-ask).
GPT5_REASONING_EFFORT = "low"
# verbosity tames gpt-5's tendency to write long answers; "low" keeps
# replies tight, "medium" is the default, "high" rambles. We use
# "medium" for technical-manual RAG: instructions need enough room
# for numbered steps + safety notes without padding.
GPT5_VERBOSITY = "medium"
RELEVANCE_CHECK_MODEL = "gpt-4o-mini"  # fast/cheap judge — not gpt-5
RELEVANCE_CHECK_TIMEOUT_S = 8.0        # fail-open if the judge is slow


def _gpt5_extra(model: str, intent: str | None = None) -> dict:
    """Return reasoning_effort + verbosity kwargs only for gpt-5 models;
    other models reject these fields.

    NOTE: do NOT set max_completion_tokens on gpt-5 reasoning models
    — that cap includes reasoning tokens, so even a generous budget
    can leave the visible answer empty.
    """
    if not model.startswith("gpt-5"):
        return {}
    try:
        from accuracy import intent_llm_settings
        if intent is None:
            return {
                "reasoning_effort": GPT5_REASONING_EFFORT,
                "verbosity": GPT5_VERBOSITY,
            }
        return intent_llm_settings({"intent": intent})
    except Exception:  # noqa: BLE001
        # If accuracy.py can't be imported (partial deploy), fall back
        # to the module-level defaults so the chat path still works.
        return {
            "reasoning_effort": GPT5_REASONING_EFFORT,
            "verbosity": GPT5_VERBOSITY,
        }


def load_system_prompt(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).with_name(str(path))
    return p.read_text(encoding="utf-8")


def safe_user_id(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned or DEFAULT_USER


def log_path_for(user: str) -> Path:
    log_dir = Path(__file__).with_name(LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    return log_dir / f"{safe_user_id(user)}.txt"


def append_log(user: str, question: str, answer_text: str,
               matches: list[dict]) -> None:
    path = log_path_for(user)
    ts = datetime.now().isoformat(timespec="seconds")
    lines = [
        "═" * 67,
        f"[{ts}] User: {user}",
        "═" * 67,
        "",
        f"Q: {question}",
        "",
        f"A: {answer_text}",
        "",
    ]
    if matches:
        lines.append("Retrieved chunks:")
        for i, m in enumerate(matches, 1):
            md = m["metadata"]
            lines.append(
                f"  [{i}] score={m['score']:.3f}  page={md.get('page', '?')}  "
                f"vehicle={md.get('vehicle_model', '—')}  "
                f"type={md.get('doc_type', '?')}"
            )
            lines.append(f"      section: {md.get('section') or '?'}")
            lines.append(f"      source : {md.get('source_pdf', '?')}")
        lines.append("")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_log_history(user: str, last_n: int) -> list[tuple[str, str]]:
    """Pull the last N (question, answer) pairs from the user's log file.

    Used by --resume. Reads the same Q:/A: blocks that append_log writes;
    if the format changes, this will return fewer pairs (or none) but
    never crash.
    """
    path = log_path_for(user)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    turns = re.split(r"\n=+\n", text.replace("═", "="))
    pairs: list[tuple[str, str]] = []
    for turn in turns:
        q_match = re.search(r"^Q:\s*(.*?)(?=\n[A-Z][a-z]*:|\nA:)",
                            turn, re.S | re.M)
        a_match = re.search(r"^A:\s*(.*?)(?=\nRetrieved chunks:|\Z)",
                            turn, re.S | re.M)
        if q_match and a_match:
            q = q_match.group(1).strip()
            a = a_match.group(1).strip()
            if q and a:
                pairs.append((q, a))
    return pairs[-last_n:] if last_n > 0 else []


# In-process LRU cache for query embeddings. text-embedding-3-large adds
# 150-300 ms per call; on a chatbot with repeat questions (REPL retries,
# learned-answer lookup + main retrieval using the same query, common FAQs
# across users) hits drop that to ~0 ms.
#
# Sized at 512 entries. Each 3072-dim float32 vector ≈ 12 KB → ~6 MB max.
# Process-local: in CLI it survives the REPL session; in api.py it
# persists for the life of the uvicorn worker.
_EMBED_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()
_EMBED_CACHE_MAX = 512
_EMBED_CACHE_STATS = {"hits": 0, "misses": 0}


def _embed_cache_key(text: str) -> str:
    # Whitespace + case normalisation. Don't strip punctuation —
    # "10W-50" vs "10W50" are different tokens to the embedder.
    return " ".join(text.lower().split())


def embed_query(client: OpenAI, text: str, use_cache: bool = True) -> list[float]:
    """Embed `text` with EMBED_MODEL using a small in-process LRU cache.

    Pass use_cache=False to force a fresh embed (e.g. when comparing
    embedding versions).
    """
    if use_cache:
        key = _embed_cache_key(text)
        cached = _EMBED_CACHE.get(key)
        if cached is not None:
            _EMBED_CACHE.move_to_end(key)
            _EMBED_CACHE_STATS["hits"] += 1
            return cached

    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = resp.data[0].embedding

    if use_cache:
        _EMBED_CACHE[key] = vec
        _EMBED_CACHE.move_to_end(key)
        if len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
            _EMBED_CACHE.popitem(last=False)
        _EMBED_CACHE_STATS["misses"] += 1

    return vec


def embed_cache_stats() -> dict:
    return {
        "size": len(_EMBED_CACHE),
        "max": _EMBED_CACHE_MAX,
        "hits": _EMBED_CACHE_STATS["hits"],
        "misses": _EMBED_CACHE_STATS["misses"],
    }


# Comparative / multi-entity signals — these almost always need wider
# retrieval to surface both sides of a comparison or a procedure spanning
# multiple components. Kept tight on purpose; common conjunctions like
# "and"/"or" appear in plenty of single-fact queries and would over-trigger.
_COMPLEX_QUERY_RX = re.compile(
    r"\b(vs|versus|compare|comparison|difference|differ|differs|"
    r"between|both)\b",
    re.I,
)


def dynamic_top_k(question: str, default: int | None = None) -> int:
    """Pick top_k based on query characteristics.

      - Short / single-fact (≤ 6 words)        → 4   (faster LLM)
      - Comparative or long (≥ 15 words / vs)  → 10  (more recall)
      - Everything else                         → `default` (6)

    `default` lets a caller override the medium tier.
    """
    text = question.strip()
    if not text:
        return default if default is not None else DEFAULT_TOP_K

    # Comparative keywords win even on short queries — "compare X" is
    # 2 words but still needs wide retrieval to surface both sides.
    if _COMPLEX_QUERY_RX.search(text):
        return 10

    word_count = len(text.split())
    if word_count <= 6:
        return 4
    if word_count >= 15:
        return 10
    return default if default is not None else DEFAULT_TOP_K


def build_filter(vehicle: str | None, doc_type: str | None) -> dict | None:
    """Build a Pinecone metadata filter.

    When a vehicle is named, also let through "universal" docs with no
    vehicle_model field — warranty booklets, SOPs, accessories guides
    apply to all bikes and would otherwise be invisible when the user
    mentions a model.
    """
    f: dict = {}
    if vehicle:
        f["$or"] = [
            {"vehicle_model": vehicle},
            {"vehicle_model": {"$exists": False}},
        ]
    if doc_type:
        f["doc_type"] = doc_type
    return f or None


def retrieve(index, vector: list[float], top_k: int, min_score: float,
             flt: dict | None) -> list[dict]:
    res = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=flt,
    )
    return [m for m in res["matches"] if m["score"] >= min_score]


def rerank(pc, query: str, matches: list[dict], keep: int = 5,
           model: str = "bge-reranker-v2-m3") -> list[dict]:
    """Re-score Pinecone matches with a cross-encoder and return the top `keep`.

    Vector retrieval scores query and document independently. A cross-encoder
    looks at the query and the chunk together, producing a much sharper
    relevance score. Standard upgrade: over-fetch (e.g. 20), rerank down
    to a small precise set (e.g. 5).

    Falls back to the original retrieval order if the reranker errors, so
    chat never breaks because of a transient inference failure.
    """
    if not matches:
        return matches
    if keep >= len(matches):
        return matches  # nothing to trim, skip the API hop

    docs = [
        {"text": (m.get("metadata") or {}).get("text", "") or ""}
        for m in matches
    ]
    try:
        result = pc.inference.rerank(
            model=model,
            query=query,
            documents=docs,
            top_n=keep,
            rank_fields=["text"],
            return_documents=False,
            # bge-reranker caps each query+doc pair at 1024 tokens. Long
            # chunks easily exceed this and cause a 400. truncate="END"
            # clips the tail of the document instead of erroring.
            parameters={"truncate": "END"},
        )
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "rerank failed, using vector order: %s", e,
        )
        return matches[:keep]

    reranked: list[dict] = []
    for item in result.data:
        idx = getattr(item, "index", None)
        score = getattr(item, "score", None)
        if idx is None or not (0 <= idx < len(matches)):
            continue
        original = matches[idx]
        # Preserve all metadata; only swap in the rerank score so
        # downstream code (citations, top_score, image_map) keeps working.
        reranked.append({
            "id": original.get("id"),
            "score": float(score) if score is not None else 0.0,
            "vector_score": float(original.get("score", 0.0)),
            "metadata": original.get("metadata", {}),
        })
    return reranked


def _normalize_chunk_text(text: str) -> str:
    """Convert pipe-soup table chunks into prose-like lines.

    Many parsed-PDF tables land in Pinecone as a single line of
    `Key | Value | Key | Value | ...` separators. Cross-encoders and LLMs
    both score and quote this poorly compared to natural language.

    Triggers only when pipe density is above a line-relative threshold
    (real prose rarely exceeds ~0.5 pipes/line; flattened tables run 5+),
    so prose chunks are passed through unchanged. Does NOT alter cell
    contents — torque values, part numbers, units stay verbatim.
    """
    if not text or "|" not in text:
        return text
    lines = text.splitlines() or [text]
    pipe_count = text.count("|")
    if pipe_count / max(len(lines), 1) < 3:
        return text
    out_lines: list[str] = []
    for line in lines:
        if line.count("|") < 2:
            out_lines.append(line)
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        out_lines.append("; ".join(cells))
    rebuilt = "\n".join(out_lines)
    import re as _re
    return _re.sub(r"[ \t]{2,}", " ", rebuilt)


MAX_IMAGES_PER_CHUNK = 3   # images surfaced to LLM per chunk
MAX_IMAGES_IN_CONTEXT = 6  # hard cap across all chunks — beyond this the LLM ignores them


def render_context(matches: list[dict]) -> str:
    """Format retrieved chunks for the LLM prompt.

    Image references are inlined as "Images: IMG_N, ..." using globally
    unique IDs so the model can emit [[SHOW_IMAGE: IMG_N]] tags.
    Use build_image_map(matches) to get the matching id→path map.
    """
    blocks = []
    img_id = 0
    total_imgs = 0
    for i, m in enumerate(matches, 1):
        md = m["metadata"]
        section = md.get("section") or "(no section)"
        page = md.get("page", "?")
        src = md.get("source_pdf", "(unknown)")
        body = _normalize_chunk_text(md.get("text", ""))
        block = (
            f"[CTX {i}]  Section: {section}  |  Source: {src}  |  Page {page}\n"
            f"{body}"
        )
        paths = md.get("image_paths") or []
        if paths and total_imgs < MAX_IMAGES_IN_CONTEXT:
            ids = []
            for _ in paths[:MAX_IMAGES_PER_CHUNK]:
                if total_imgs >= MAX_IMAGES_IN_CONTEXT:
                    break
                img_id += 1
                total_imgs += 1
                ids.append(f"IMG_{img_id}")
            if ids:
                block += f"\nImages: {', '.join(ids)}"
        elif paths:
            # Still advance img_id so build_image_map stays in sync
            img_id += len(paths)
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def build_image_map(matches: list[dict]) -> dict[str, str]:
    """Return {IMG_N: relative_path} matching what render_context emitted.

    IDs are assigned in the same order as render_context.
    """
    out: dict[str, str] = {}
    img_id = 0
    for m in matches:
        md = m["metadata"]
        for path in (md.get("image_paths") or []):
            img_id += 1
            out[f"IMG_{img_id}"] = path
    return out


def build_chat_messages(question: str, matches: list[dict],
                        system_prompt: str,
                        history: list[tuple[str, str]] | None = None,
                        image_b64: str | None = None) -> list[dict]:
    """Assemble the OpenAI message list for one Q&A turn.

    Prior turns are included as bare Q/A pairs (no RAG context block) so
    the model has conversational continuity for follow-ups without
    re-paying for stale context tokens.

    If `image_b64` is supplied (raw base64, no data: prefix), the current
    user turn becomes a multimodal message.
    """
    user_text = (
        f"Context passages:\n\n{render_context(matches)}\n\nQuestion: {question}"
    )
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for prev_q, prev_a in (history or []):
        msgs.append({"role": "user", "content": prev_q})
        msgs.append({"role": "assistant", "content": prev_a})

    if image_b64:
        msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "auto",
                    },
                },
            ],
        })
    else:
        msgs.append({"role": "user", "content": user_text})
    return msgs


def _intent_directive(intent: str | None) -> str:
    """Per-intent nudge prepended to the user message.

    The system prompt already instructs full-procedure reproduction, but
    gpt-5-mini sometimes still summarizes when chunk content is dense.
    A short user-voice directive at the head of the message gives the
    model a second, harder-to-ignore signal — and only when the
    classifier flagged this as a depth-sensitive question.
    """
    if intent == "procedure":
        return (
            "[INSTRUCTION: This is a procedural question. Reproduce the "
            "ENTIRE workshop procedure from the [CTX] blocks — not just "
            "the steps, but EVERY accompanying section. Workshop procedures "
            "in BSA manuals are organised as multi-section blocks; you "
            "must preserve that structure.\n\n"
            "MANDATORY OUTPUT STRUCTURE (use these section headings as "
            "they appear in [CTX] — keep the manual's order):\n"
            "  1. **General Information / Introduction** — what this "
            "     procedure is, why it matters, what shims/parts/components "
            "     do. If [CTX] has an introductory block, reproduce it.\n"
            "  2. **Tools & Materials** — list EVERY tool/instrument named "
            "     in [CTX] (Allen keys with sizes, feeler gauges, "
            "     micrometer, torque wrench, sealant, gaskets, etc.). "
            "     Don't fall back to 'basic hand tools' — name each.\n"
            "  3. **Procedure / Process** — every numbered step from [CTX], "
            "     in order, with all sub-steps and torque values. Reproduce "
            "     WARNING / CAUTION / NOTE callouts VERBATIM next to the "
            "     step they apply to.\n"
            "  4. **Reassembly** — if [CTX] has a separate reassembly "
            "     sub-section, reproduce it as its own section.\n"
            "  5. **Troubleshooting** — REQUIRED if [CTX] contains ANY "
            "     troubleshooting / common-issues / remedies content, "
            "     even if not explicitly labelled 'Troubleshooting'. "
            "     Look for sections titled 'Troubleshooting', 'Common "
            "     Issues', 'Issues and Remedies', 'If problems persist', "
            "     or any 'Issue: ... Remedy: ...' style entries. "
            "     Reproduce EVERY item as a dedicated section with each "
            "     issue + its remedy. Never collapse into one generic "
            "     line. If 7 issues are listed in [CTX], output all 7.\n"
            "  6. **Final warnings / notes** — any closing WARNING from "
            "     [CTX] reproduced verbatim at the end.\n\n"
            "PAGE ANCHORS: Append a page reference in parentheses next to "
            "each major section heading you emit, e.g. "
            "'**Procedure / Process** (p. 204–206)'. Pull the page numbers "
            "from the [CTX] block headers. Use ranges when steps span "
            "multiple pages. Do NOT inline-cite mid-prose — anchors go "
            "ONLY next to the section headings.\n\n"
            "SOURCES LINE: End the entire answer with a single 'Sources:' "
            "line listing every PDF the answer drew from (semicolon "
            "separated, cleaned of '.pdf', '_REV-N', date stamps).\n\n"
            "IMAGES (step-wise, no duplicates): up to 3 figures per "
            "procedure, INTERLEAVED with the steps. Place each "
            "[[SHOW_IMAGE: IMG_N]] tag on its OWN line, IMMEDIATELY "
            "AFTER the step it illustrates and BEFORE the next step. "
            "Do NOT group images at the top, do NOT group images at "
            "the end as a gallery, do NOT put two markers on one line, "
            "do NOT embed mid-sentence. Each IMG_N appears AT MOST "
            "ONCE in the answer — if a figure could illustrate two "
            "steps, pick the FIRST step where the part / mark / "
            "location is acted on. If a step doesn't have a directly-"
            "illustrative [CTX] image, leave it image-free; do NOT "
            "reuse a loosely-related figure as filler.\n\n"
            "CORRECT example (note: image is INTERLEAVED, not collected):\n"
            "    3. Remove the drain plug.\n"
            "    [[SHOW_IMAGE: IMG_2]]\n"
            "\n"
            "    4. Let the oil drain.\n\n"
            "DO NOT summarize, abbreviate, paraphrase, or skip 'obvious' "
            "sub-steps. Do NOT flatten the manual's section structure into "
            "one long numbered list. Length and structure follow the source.]"
            "\n\n"
        )
    if intent == "safety":
        return (
            "[INSTRUCTION: Safety-critical question. Lead with the "
            "immediate safety action (stop, isolate, ventilate). Then "
            "reproduce the full diagnostic / remediation procedure from "
            "[CTX] verbatim — every warning, every step.]\n\n"
        )
    if intent == "diagnostic":
        return (
            "[INSTRUCTION: Diagnostic / troubleshooting question. The "
            "user is describing a symptom and wants the cause + how to "
            "confirm it. Structure the answer in this order:\n\n"
            "  1. **Most likely causes** — a SHORT ranked list, "
            "     most-likely first, drawn from the [CTX] troubleshooting "
            "     tables / Issues & Remedies / Faults & Causes content. "
            "     RANK by relevance to the symptom in the question and "
            "     the conditions it mentions (speed, load, temperature, "
            "     gear, recent service). For example: high-speed "
            "     vibration ranks wheels / suspension / wheel balance / "
            "     tyre / drive chain ABOVE bottom-end engine wear; "
            "     hard-cold-start ranks battery / fuel pressure / IACV "
            "     ABOVE crankshaft wear; rough-idle-after-service ranks "
            "     valve clearance / IACV / sensor adaptation ABOVE "
            "     piston wear. Cite the source page for each cause. "
            "     Do NOT dump every cause as an equal-weight flat list.\n"
            "  2. **Triage / checks** — numbered sequence of checks, "
            "     CHEAPEST and QUICKEST first: visual inspection → "
            "     measurement / feeler-gauge / multimeter → scan tool / "
            "     DTC read / freeze-frame → teardown. Use [CTX] check "
            "     wording verbatim where given.\n"
            "  3. **Remedies** — for each likely cause, the [CTX] "
            "     remedy reproduced verbatim. Don't paraphrase.\n"
            "  4. **Escalate** — when to stop and hand off to an "
            "     authorised BSA service centre, using the [CTX] wording "
            "     where present.\n\n"
            "If [CTX] contains a Faults / Causes / Action table, "
            "REPRODUCE every relevant row — do not collapse multiple "
            "rows into one summary.\n\n"
            "IMAGES (be conservative, step-wise, no duplicates): "
            "default to ZERO. Embed at most ONE figure, and ONLY if "
            "it directly shows the specific fault, location, or "
            "inspection point the answer is discussing. Do NOT add an "
            "'Images to help' or 'For reference' block at the end "
            "with multiple figures — that is decoration, not help. If "
            "you do embed, place [[SHOW_IMAGE: IMG_N]] on its OWN "
            "line, IMMEDIATELY AFTER the specific cause or check it "
            "illustrates (interleaved with the content, not at the "
            "end), and use that IMG_N AT MOST ONCE in the entire "
            "answer. End with a Sources line listing every PDF "
            "cited.]\n\n"
        )
    if intent == "wiring":
        return (
            "[INSTRUCTION: Wiring/electrical question. Reproduce all "
            "relevant pin assignments, colour codes, connector IDs, and "
            "torque/voltage values verbatim from [CTX]. Do not paraphrase "
            "table data.]\n\n"
        )
    if intent == "warranty":
        return (
            "[INSTRUCTION: Warranty / Service Booklet question. Reproduce "
            "the relevant clauses, lists, and tables from [CTX] AS WRITTEN "
            "— same wording, same enumeration, same conditions and "
            "exclusions. Do NOT summarize, paraphrase, or condense legal/"
            "policy language. If the booklet lists 7 service intervals, "
            "list all 7 with the same fields. If a clause has multiple "
            "conditions, reproduce every condition verbatim.]\n\n"
        )
    return ""


def answer(client: OpenAI, question: str, matches: list[dict],
           system_prompt: str, history: list[tuple[str, str]] | None = None,
           model: str = CHAT_MODEL, image_b64: str | None = None,
           intent: str | None = None) -> str:
    # Empty matches still go to the LLM so the prompt's greeting/closing
    # rules can fire — and so an unanswerable question gets the prompt's
    # "retrieved passages don't cover this" reply, not a hardcoded fallback.
    directive = _intent_directive(intent)
    augmented_question = directive + question if directive else question
    msgs = build_chat_messages(augmented_question, matches, system_prompt,
                                history, image_b64=image_b64)
    # gpt-5 family only supports the default temperature; do not pass one.
    resp = client.chat.completions.create(
        model=model,
        messages=msgs,

        **_gpt5_extra(model, intent=intent),
    )
    return resp.choices[0].message.content.strip()


def stream_answer(client: OpenAI, question: str, matches: list[dict],
                  system_prompt: str,
                  history: list[tuple[str, str]] | None = None,
                  model: str = CHAT_MODEL,
                  image_b64: str | None = None,
                  intent: str | None = None):
    """Yield content tokens as the model produces them.

    `intent` (from the classifier) gates two depth controls: the user-message
    directive (procedure/safety/wiring nudge) and verbosity=high on gpt-5
    calls. Without it, gpt-5-mini caps procedural output and summarizes
    8-step procedures into 4.
    """
    directive = _intent_directive(intent)
    augmented_question = directive + question if directive else question
    msgs = build_chat_messages(augmented_question, matches, system_prompt,
                                history, image_b64=image_b64)
    # gpt-5 family only supports the default temperature; do not pass one.
    stream = client.chat.completions.create(
        model=model,
        messages=msgs,
        stream=True,
        **_gpt5_extra(model, intent=intent),
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


_RELEVANCE_JUDGE_PROMPT = (
    "You are a strict relevance judge for a motorcycle service-manual chatbot.\n"
    "Given a user question and a generated answer, decide:\n"
    "  • Does the answer DIRECTLY address what the user asked?\n"
    "  • Is the content on-topic (not about a different system, component, or vehicle)?\n\n"
    "Reply with ONLY valid JSON — no markdown fences, no extra text:\n"
    '{"relevant": true, "reason": "one short sentence"}\n\n'
    "Mark relevant=false for: vague non-answers, answers about the wrong topic, "
    "answers that repeat the question without actually answering, or clearly "
    "hallucinated content that doesn't match the question context."
)

_RELEVANCE_FALLBACK = (
    "I wasn't able to find a direct answer to your question in the BSA service "
    "manuals. Try rephrasing your question, or ask about a specific component, "
    "procedure, or part number."
)


def check_relevance(client: OpenAI, question: str, answer_text: str) -> dict:
    """Second LLM call: judge whether the generated answer addresses the question.

    Returns {"relevant": bool, "reason": str, "skipped": bool}.
    skipped=True means the check timed out or errored — callers treat the
    answer as passing (fail-open) so the main chat path is never blocked.
    """
    user_msg = (
        f"Question: {question}\n\n"
        f"Answer:\n{answer_text[:2500]}"  # cap to avoid token cost blow-up
    )

    def _call() -> dict:
        resp = client.chat.completions.create(
            model=RELEVANCE_CHECK_MODEL,
            messages=[
                {"role": "system", "content": _RELEVANCE_JUDGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=80,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences some models add despite the instruction.
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        parsed = json.loads(raw)
        return {
            "relevant": bool(parsed.get("relevant", True)),
            "reason": str(parsed.get("reason", "")),
            "skipped": False,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            return fut.result(timeout=RELEVANCE_CHECK_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            return {"relevant": True, "reason": "timed out", "skipped": True}
        except Exception:  # noqa: BLE001 — JSON error, API error, etc.
            return {"relevant": True, "reason": "", "skipped": True}


def print_citations(matches: list[dict], show_text: bool) -> None:
    print("\nRetrieved chunks:")
    for i, m in enumerate(matches, 1):
        md = m["metadata"]
        section = md.get("section") or "?"
        page = md.get("page", "?")
        src = md.get("source_pdf", "?")
        vehicle = md.get("vehicle_model", "—")
        doc_type = md.get("doc_type", "?")
        score = m["score"]
        print(f"  [{i}] score={score:.3f}  page={page}  vehicle={vehicle}  "
              f"type={doc_type}")
        print(f"      section: {section}")
        print(f"      source : {src}")
        if md.get("image_paths"):
            print(f"      images : {len(md['image_paths'])} file(s)")
            for p in md["image_paths"][:3]:
                print(f"               - {p}")
            if len(md["image_paths"]) > 3:
                print(f"               (+{len(md['image_paths']) - 3} more)")
        if show_text:
            snippet = (md.get("text", "")[:300]).replace("\n", " ")
            print(f"      text   : {snippet}...")


# Delegates to accuracy.VEHICLES so adding a new model is a one-line
# config edit there — no need to update patterns in two files.
def detect_vehicle(question: str) -> str | None:
    from accuracy import detect_vehicle_in
    return detect_vehicle_in(question)


def ask(client: OpenAI, index, question: str, args, system_prompt: str,
        history: list[tuple[str, str]]) -> None:
    # Lazy import so chatbot.py stays importable even if accuracy.py is
    # missing (e.g., during partial deployments).
    from accuracy import (
        classifier_to_filter_vehicle, classify_query,
        diversify_by_doc_type, filter_by_rerank_floor,
        filter_by_section_relevance,
        intent_doc_types, intent_fetch_k, intent_top_k,
        merge_unique_matches, page_neighbor_ids,
        verify_grounding,
    )

    flt = build_filter(args.vehicle, args.doc_type)

    if not args.vehicle and not args.no_auto_filter:
        guess = detect_vehicle(question)
        if guess:
            flt = build_filter(guess, args.doc_type)
            print(f"[auto-filter] detected vehicle: {guess}")

    classification = classify_query(client, question)
    print(f"[classifier] model={classification['model']}  "
          f"variant={classification['variant']}  "
          f"intent={classification['intent']}  "
          f"confident={classification['confident']}")

    if not args.vehicle and (flt is None or "$or" not in flt):
        cv = classifier_to_filter_vehicle(classification)
        if cv:
            flt = build_filter(cv, args.doc_type)
            print(f"[auto-filter] classifier vehicle: {cv}")

    if args.top_k is not None:
        top_k = args.top_k
    else:
        base = dynamic_top_k(question)
        top_k = intent_top_k(classification, base)
        print(f"[auto-top-k] {top_k} chunks (intent-adjusted from {base})")

    vec = embed_query(client, question)

    fetch_k = intent_fetch_k(classification, top_k)
    matches = retrieve(index, vec, fetch_k, args.min_score, flt)

    matches = filter_by_section_relevance(matches, question, classification)

    matches = filter_by_rerank_floor(matches, classification)

    if len(matches) > top_k:
        matches = matches[:top_k]

    pref_types = intent_doc_types(classification)
    if matches:
        if len(matches) > top_k:
            matches = diversify_by_doc_type(matches, keep=top_k,
                                             preferred_types=pref_types or None)
        elif pref_types:
            matches = diversify_by_doc_type(matches, keep=len(matches),
                                             preferred_types=pref_types)

    from accuracy import intent_long_form
    long_form_intent = intent_long_form(classification)
    if long_form_intent and matches:
        neighbour_ids = page_neighbor_ids(matches[0], span=2)
        if neighbour_ids:
            try:
                fetch_res = index.fetch(ids=neighbour_ids)
                seed_vehicle = (matches[0].get("metadata") or {}).get("vehicle_model")
                extra_neighbours: list[dict] = []
                for vid, vec_obj in (fetch_res.vectors or {}).items():
                    md_n = getattr(vec_obj, "metadata", None) or {}
                    nv = md_n.get("vehicle_model")
                    if seed_vehicle and nv and nv != seed_vehicle:
                        continue
                    extra_neighbours.append({
                        "id": vid,
                        "score": 0.0,
                        "metadata": md_n,
                    })
                if extra_neighbours:
                    before = len(matches)
                    matches = merge_unique_matches(matches, extra_neighbours)
                    print(f"[page-neighbour] +{len(matches) - before} chunks "
                          f"around page {(matches[0].get('metadata') or {}).get('page')}")
            except Exception as e:  # noqa: BLE001
                print(f"[page-neighbour] fetch failed: {e}")

    recent = history[-args.history_turns:] if args.history_turns > 0 else []
    answer_text = answer(client, question, matches, system_prompt,
                         history=recent, model=args.model,
                         intent=classification.get("intent"))

    relevance = check_relevance(client, question, answer_text)
    answer_irrelevant = False
    if relevance["skipped"]:
        print("[relevance-check] skipped (timeout or error) — showing answer as-is")
    elif not relevance["relevant"]:
        print(f"[relevance-check] FAIL — {relevance['reason']}")
        answer_irrelevant = True

    if answer_irrelevant:
        print(f"\nAnswer:\n{_RELEVANCE_FALLBACK}")
        # Don't add to history: a "not found" reply is not useful context for
        # follow-up turns and would pollute the conversation memory.
    else:
        ctx_text = " ".join(
            (m.get("metadata") or {}).get("text", "") or "" for m in matches
        )
        ctx_pages = [(m.get("metadata") or {}).get("page") for m in matches]
        grounding = verify_grounding(answer_text, ctx_text, ctx_pages=ctx_pages)

        print(f"\nAnswer:\n{answer_text}")
        if not grounding["ok"]:
            print("\n[grounding] WARNING — unsupported atoms in answer:")
            for u in grounding["unsupported"]:
                print(f"  - {u['kind']}: {u['value']}")

        history.append((question, answer_text))

    if matches:
        print_citations(matches, args.show_text)
    else:
        print("\n(No matches above min-score threshold. Try --top-k 10 or "
              "--min-score 0.20 to widen the search.)")
    if not args.no_log:
        append_log(args.user, question, answer_text, matches)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("question", nargs="*", help="Question (omit for REPL)")
    ap.add_argument("--index", default=DEFAULT_INDEX,
                    help=f"Pinecone index (default: {DEFAULT_INDEX})")
    ap.add_argument("--vehicle", default=None,
                    help='Filter by vehicle_model, e.g. "BSA Bantam"')
    ap.add_argument("--doc-type", default=None,
                    help="Filter by doc_type, e.g. owners_manual")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Number of chunks to retrieve. Omit for auto: "
                         f"4 for short queries, 10 for complex, else "
                         f"{DEFAULT_TOP_K}. Pass an integer to override.")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                    help=f"Drop matches below this score (default: {DEFAULT_MIN_SCORE})")
    ap.add_argument("--show-text", action="store_true",
                    help="Print a snippet of each retrieved chunk")
    ap.add_argument("--no-auto-filter", action="store_true",
                    help="Don't auto-detect vehicle from the question text")
    ap.add_argument("--model", default=CHAT_MODEL,
                    help=f"OpenAI chat model (default: {CHAT_MODEL}). "
                         "Try gpt-4.1, gpt-4o-mini, or gpt-5 to compare.")
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE,
                    help=f"Path to system prompt text file "
                         f"(default: {DEFAULT_PROMPT_FILE}, resolved next "
                         "to chatbot.py if relative)")
    ap.add_argument("--user", default=DEFAULT_USER,
                    help=f"Session/user id; logs go to "
                         f"{LOG_DIR}/<user>.txt (default: {DEFAULT_USER})")
    ap.add_argument("--history-turns", type=int, default=DEFAULT_HISTORY_TURNS,
                    help=f"Prior Q/A pairs sent to the LLM for follow-up "
                         f"context (default: {DEFAULT_HISTORY_TURNS}; "
                         "0 disables)")
    ap.add_argument("--resume", type=int, default=0, metavar="N",
                    help="At REPL start, rehydrate the last N Q/A pairs "
                         "from this user's log file (default: 0)")
    ap.add_argument("--no-log", action="store_true",
                    help="Don't append this session's Q/A to the log file")
    args = ap.parse_args()

    try:
        system_prompt = load_system_prompt(args.prompt_file)
    except FileNotFoundError:
        print(f"ERROR: prompt file not found: {args.prompt_file}",
              file=sys.stderr)
        return 1

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not pinecone_key or not openai_key:
        print("ERROR: set PINECONE_API_KEY and OPENAI_API_KEY in .env",
              file=sys.stderr)
        return 1

    pc = Pinecone(api_key=pinecone_key)
    if args.index not in {ix["name"] for ix in pc.list_indexes()}:
        print(f"ERROR: index {args.index!r} does not exist. "
              f"Run embed_and_upsert.py first.", file=sys.stderr)
        return 1
    index = pc.Index(args.index)
    client = OpenAI(api_key=openai_key)

    history: list[tuple[str, str]] = []
    if args.resume > 0:
        history = parse_log_history(args.user, args.resume)
        if history:
            print(f"[resume] loaded {len(history)} prior turn(s) for "
                  f"user {args.user!r}")

    question = " ".join(args.question).strip()
    if question:
        ask(client, index, question, args, system_prompt, history)
        return 0

    print(f"RAG chat over {args.index!r}. Type your question (Ctrl-C to exit).")
    print(f"  user : {args.user}  (log: {LOG_DIR}/{safe_user_id(args.user)}.txt"
          f"{', disabled' if args.no_log else ''})")
    top_k_label = "auto" if args.top_k is None else str(args.top_k)
    print(f"  model: {args.model}  |  embed: {EMBED_MODEL}  |  "
          f"top_k: {top_k_label}  |  history: {args.history_turns} turn(s)")
    if args.vehicle:
        print(f"  vehicle filter: {args.vehicle}")
    if args.doc_type:
        print(f"  doc_type filter: {args.doc_type}")
    print("  REPL commands: /clear  /history  /cache  exit|quit")
    print()
    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                return 0
            if q == "/clear":
                history.clear()
                print("[history cleared]\n")
                continue
            if q == "/cache":
                s = embed_cache_stats()
                total = s["hits"] + s["misses"]
                hit_rate = (s["hits"] / total * 100) if total else 0.0
                print(f"[embed cache] size={s['size']}/{s['max']}  "
                      f"hits={s['hits']}  misses={s['misses']}  "
                      f"hit_rate={hit_rate:.1f}%\n")
                continue
            if q == "/history":
                if not history:
                    print("[no history yet]\n")
                else:
                    for i, (pq, pa) in enumerate(history, 1):
                        print(f"  [{i}] Q: {pq}")
                        print(f"      A: {pa[:120]}"
                              f"{'...' if len(pa) > 120 else ''}")
                    print()
                continue
            ask(client, index, q, args, system_prompt, history)
            print()
    except (KeyboardInterrupt, EOFError):
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
