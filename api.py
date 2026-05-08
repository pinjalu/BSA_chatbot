"""FastAPI wrapper around chatbot.py — streaming RAG endpoint for Node.js.

The Node Express layer (server/) handles sessions, persistence, dealer
sites, and the admin panel; this service does ONE thing — given a
question + history + filters, stream back the answer.

Endpoints
  GET  /health              — liveness probe
  POST /chat                — Server-Sent Events stream of answer tokens
  POST /retrieve            — non-streaming: just the retrieved chunks
                              (used by the Node layer to compute a
                              confidence signal for human handoff)

Run
  uvicorn api:app --host 127.0.0.1 --port 8000

Env (.env, same file as chatbot.py uses)
  OPENAI_API_KEY
  PINECONE_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bsa-api")

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field

# boto3 is optional — only needed for presigning S3 image URLs. If the
# package isn't installed or AWS credentials aren't configured, the
# presigner falls back to returning the original URL (which works only
# if the bucket is public). Imports are wrapped so api.py stays
# importable in environments without boto3.
try:
    import boto3
    from botocore.config import Config as _BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO_AVAILABLE = True
except ImportError:
    _BOTO_AVAILABLE = False
    BotoCoreError = ClientError = Exception  # noqa: N806

from urllib.parse import urlparse, unquote

from chatbot import (
    CHAT_MODEL,
    DEFAULT_INDEX,
    DEFAULT_MIN_SCORE,
    DEFAULT_PROMPT_FILE,
    DEFAULT_TOP_K,
    build_filter,
    build_image_map,
    detect_vehicle,
    dynamic_top_k,
    embed_cache_stats,
    embed_query,
    load_system_prompt,
    render_context,
    rerank,
    retrieve,
    stream_answer,
)
from accuracy import (
    classifier_to_filter_vehicle,
    classify_query_async,
    detect_dominant_section,
    diversify_by_doc_type,
    filter_by_rerank_floor,
    filter_by_section_relevance,
    get_executor,
    intent_doc_types,
    intent_fetch_k,
    intent_long_form,
    intent_top_k,
    merge_unique_matches,
    page_neighbor_ids,
    regex_classify,
    verify_grounding,
)
from debug_log import write_debug_entry

load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI(title="BSA RAG core", version="1.0")

# Learned-answer namespace inside the same Pinecone index. When a human
# admin resolves a handoff with a useful reply, we upsert (question
# embedding → saved answer) here. Future similar questions get the
# saved answer directly, no LLM round-trip.
LEARNED_NS = os.environ.get("LEARNED_NAMESPACE", "learned")
# Cosine-similarity threshold for "this is the same question we've
# answered before". 0.85 is conservative — we want high precision so
# we don't reuse an unrelated stored answer.
LEARNED_THRESHOLD = float(os.environ.get("LEARNED_THRESHOLD", "0.85"))


# ─────────────────────── singletons ───────────────────────
# These are expensive to construct (network handshake, key validation).
# Build once at startup and reuse for every request.
_openai: Optional[OpenAI] = None
_pinecone_client: Optional[Pinecone] = None  # for inference.rerank
_pinecone_index = None
_system_prompt: Optional[str] = None

# Rerank config — over-fetch from Pinecone with a bi-encoder, then
# re-score with a cross-encoder to keep only the most relevant chunks
# before sending to the LLM. This trades ~150 ms for noticeably
# better answer quality and a smaller LLM context.
RERANK_FETCH_K = int(os.environ.get("RERANK_FETCH_K", "20"))
RERANK_KEEP = int(os.environ.get("RERANK_KEEP", "5"))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "bge-reranker-v2-m3")
USE_RERANK = os.environ.get("USE_RERANK", "1").strip().lower() not in (
    "0", "false", "off", "no",
)

# LLM query rewrite (pre-retrieval). Keep it fast + safe: strict timeout,
# deterministic fallback.
USE_QUERY_REWRITE = os.environ.get("USE_QUERY_REWRITE", "0").strip().lower() not in (
    "0", "false", "off", "no",
)
QUERY_REWRITE_MODEL = os.environ.get("QUERY_REWRITE_MODEL", "gpt-4o-mini")
QUERY_REWRITE_TIMEOUT_S = float(os.environ.get("QUERY_REWRITE_TIMEOUT_S", "3.0"))

# Cap chunks sent to the LLM so answers stay complete but readable.
# Retrieval can still stay wide for accuracy; we trim only at the
# generation boundary.
ANSWER_MAX_CHUNKS = int(os.environ.get("ANSWER_MAX_CHUNKS", "10"))
ANSWER_MAX_CHUNKS_FULL = int(os.environ.get("ANSWER_MAX_CHUNKS_FULL", "18"))


# S3 client — initialised lazily at startup. None means signing is
# disabled (boto3 missing or no AWS creds), and image URLs pass through
# unchanged (which works only if the bucket is public).
_s3_client = None
_S3_PRESIGN_TTL = int(os.environ.get("S3_PRESIGN_TTL", "3600"))  # 1h default


@app.on_event("startup")
def _startup() -> None:
    global _openai, _pinecone_client, _pinecone_index, _system_prompt
    global _s3_client

    openai_key = os.environ.get("OPENAI_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        raise RuntimeError(
            "OPENAI_API_KEY and PINECONE_API_KEY must be set in .env"
        )

    _openai = OpenAI(api_key=openai_key)
    _pinecone_client = Pinecone(api_key=pinecone_key)
    index_name = os.environ.get("PINECONE_INDEX", DEFAULT_INDEX)
    if index_name not in {ix["name"] for ix in _pinecone_client.list_indexes()}:
        raise RuntimeError(
            f"Pinecone index {index_name!r} does not exist. "
            "Run embed_and_upsert.py first."
        )
    _pinecone_index = _pinecone_client.Index(index_name)

    prompt_file = os.environ.get("PROMPT_FILE", DEFAULT_PROMPT_FILE)
    _system_prompt = load_system_prompt(prompt_file)

    # Initialise the S3 client used to sign image URLs at request time.
    # If boto3 isn't installed or AWS credentials aren't configured, we
    # leave _s3_client = None and image URLs pass through unsigned. We
    # MUST pin the region + endpoint to ap-southeast-2: without them
    # boto3 falls back to the global s3.amazonaws.com endpoint and the
    # generated SigV4 signature won't match the regional URL the browser
    # ends up calling, producing a 403 SignatureDoesNotMatch.
    if _BOTO_AVAILABLE:
        try:
            region = (
                os.environ.get("AWS_DEFAULT_REGION")
                or os.environ.get("AWS_REGION")
                or "ap-southeast-2"
            )
            _s3_client = boto3.client(
                "s3",
                region_name=region,
                endpoint_url=f"https://s3.{region}.amazonaws.com",
                config=_BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "virtual"},
                ),
            )
            log.info(
                "[startup] s3 presigner ready (region=%s TTL=%ds)",
                region, _S3_PRESIGN_TTL,
            )
        except (BotoCoreError, ClientError, Exception) as e:  # noqa: BLE001
            log.warning(
                "[startup] s3 client init failed (%s); image URLs will pass "
                "through unsigned and may 403 if the bucket is private", e,
            )
            _s3_client = None
    else:
        log.warning(
            "[startup] boto3 not installed; image URLs will pass through "
            "unsigned and may 403 if the bucket is private",
        )

    log.info(
        f"[startup] index={index_name} rerank={'on' if USE_RERANK else 'off'} "
        f"fetch_k={RERANK_FETCH_K} keep={RERANK_KEEP} model={RERANK_MODEL}"
    )


def _parse_s3_url(url: str) -> tuple[str, str] | None:
    """Best-effort parser for both addressing styles boto3/S3 produce:

      virtual-hosted-style (most common):
        https://<bucket>.s3.<region>.amazonaws.com/<key>
        https://<bucket>.s3.amazonaws.com/<key>
      path-style (legacy / some regions):
        https://s3.<region>.amazonaws.com/<bucket>/<key>
        https://s3.amazonaws.com/<bucket>/<key>

    Returns (bucket, key) on success, None if the URL doesn't look like
    an S3 HTTPS URL (e.g. a local relative path stored before the S3
    rewrite). The key is URL-decoded so spaces and parens come back to
    the literal form expected by S3 GetObject.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return None
    host = parsed.netloc.lower()
    path = parsed.path.lstrip("/")
    if not host.endswith("amazonaws.com") or not path:
        return None
    # Virtual-hosted: <bucket>.s3[.region].amazonaws.com
    if ".s3." in host or host.startswith("s3."):
        # path-style if host begins with "s3." or "s3-"
        if host.startswith("s3.") or host.startswith("s3-"):
            parts = path.split("/", 1)
            if len(parts) != 2:
                return None
            return parts[0], unquote(parts[1])
        # virtual-hosted: split on first ".s3"
        bucket = host.split(".s3", 1)[0]
        return bucket, unquote(path)
    return None


def _normalize_s3_key(key: str) -> str:
    """Strip stray "Data/" path segments left over from an older
    indexing run. The current S3 bucket layout has no "Data/" folder
    (top-level prefixes are model/document names), but Pinecone records
    upserted before the S3 reshuffle still carry the obsolete prefix
    inline (e.g. "Workshop Manuals/Data/BSA GOLDSTAR.../images/...").
    Removing every literal "Data/" segment maps those stale paths back
    to the actual object keys. Safe for this bucket: no top-level or
    intermediate folder is named "Data".
    """
    parts = [p for p in key.split("/") if p and p != "Data"]
    return "/".join(parts)


def _sign_image_url(url: str) -> str:
    """Return a presigned GET URL for an S3 object, or the original URL
    if signing isn't possible (no client, not an S3 URL, error). Always
    returns SOMETHING — the widget can still try the unsigned URL as a
    last resort if the bucket happens to be public."""
    if _s3_client is None:
        return url
    parsed = _parse_s3_url(url)
    if parsed is None:
        return url
    bucket, key = parsed
    key = _normalize_s3_key(key)
    try:
        return _s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=_S3_PRESIGN_TTL,
        )
    except (BotoCoreError, ClientError, Exception) as e:  # noqa: BLE001
        log.warning("presign failed for %s: %s", url, e)
        return url


def _sign_image_map(image_map: dict[str, str]) -> dict[str, str]:
    """Apply _sign_image_url to every URL in {IMG_N: url} map."""
    if not image_map or _s3_client is None:
        return image_map
    return {k: _sign_image_url(v) for k, v in image_map.items()}


# ─────────────────────── request models ───────────────────────
class HistoryTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[HistoryTurn] = Field(default_factory=list)
    vehicle: Optional[str] = None      # e.g. "BSA Bantam"
    doc_type: Optional[str] = None     # e.g. "owners_manual"
    # top_k=None → server picks dynamically from question shape.
    # The Node layer can still pass an int to force a specific value.
    top_k: Optional[int] = None
    min_score: float = DEFAULT_MIN_SCORE
    auto_filter: bool = True           # detect vehicle from question text
    model: str = CHAT_MODEL
    # Optional user-uploaded image (raw base64, no data: prefix). When
    # present, the current turn is sent to the LLM as a multimodal
    # message: text + image. Discarded after the request.
    image_b64: Optional[str] = None


class RetrieveRequest(BaseModel):
    message: str = Field(..., min_length=1)
    vehicle: Optional[str] = None
    doc_type: Optional[str] = None
    top_k: Optional[int] = None        # None → dynamic
    min_score: float = DEFAULT_MIN_SCORE
    auto_filter: bool = True


class LearnRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    dealer_id: Optional[str] = None
    session_id: Optional[str] = None


# ─────────────────────── helpers ───────────────────────
def _resolve_filter(message: str, vehicle: Optional[str],
                    doc_type: Optional[str], auto_filter: bool,
                    history: list[tuple[str, str]] | None = None) -> dict | None:
    """Build the Pinecone metadata filter.

    Auto-detection looks at the current message AND the most recent prior
    user question, so a short clarification like "gold star euro-5" still
    locks the filter to the right vehicle if the original question
    didn't mention one.
    """
    if vehicle:
        return build_filter(vehicle, doc_type)
    if auto_filter:
        guess = detect_vehicle(message)
        if not guess and history:
            # Walk back through prior user turns; the most recent named
            # model wins.
            for prev_q, _ in reversed(history):
                guess = detect_vehicle(prev_q)
                if guess:
                    break
        if guess:
            return build_filter(guess, doc_type)
    return build_filter(None, doc_type)


_FULL_DETAIL_RX = re.compile(
    r"\b("
    r"all steps?|all details?|full (?:steps?|details?|procedure|process|list|table)"
    r"|complete (?:steps?|details?|procedure|process|list|table)"
    r"|detailed|step[- ]?by[- ]?step|in detail|end[- ]?to[- ]?end|exhaustive"
    r")\b",
    re.I,
)


def _wants_full_detail(text: str) -> bool:
    """True when the user explicitly asks for exhaustive detail."""
    if not text:
        return False
    return bool(_FULL_DETAIL_RX.search(text))


def _enforce_vehicle_purity(matches: list[dict], vehicle: str | None) -> list[dict]:
    """Keep only chunks for the selected vehicle (plus universal chunks)."""
    if not matches or not vehicle:
        return matches
    kept: list[dict] = []
    for m in matches:
        md = m.get("metadata") or {}
        vm = md.get("vehicle_model")
        if not vm or vm == vehicle:
            kept.append(m)
    return kept or matches


_BATTERY_CHARGING_RX = re.compile(
    r"\b(battery|charging|charge|regulator|rectifier|rr unit|alternator|"
    r"magneto|voltage|ocv|fuse|vbatt)\b",
    re.I,
)
_OFFTOPIC_DIAG_RX = re.compile(
    r"\b(iacv|throttle|injector|spark plug|clutch|gearbox|tyre|brake)\b",
    re.I,
)


def _filter_diagnostic_offtopic(
    matches: list[dict], question: str, classification: dict,
) -> list[dict]:
    """Drop obviously off-subsystem diagnostic chunks for focused faults."""
    if not matches or classification.get("intent") != "diagnostic":
        return matches
    if not _BATTERY_CHARGING_RX.search(question or ""):
        return matches
    kept: list[dict] = []
    for m in matches:
        md = m.get("metadata") or {}
        hay = f"{md.get('section') or ''}\n{md.get('text') or ''}".lower()
        if _OFFTOPIC_DIAG_RX.search(hay):
            continue
        kept.append(m)
    return kept or matches


def _extract_vehicle_from_filter(flt: dict | None) -> str | None:
    """Best-effort vehicle extraction from build_filter() output."""
    if not isinstance(flt, dict):
        return None
    ors = flt.get("$or")
    if isinstance(ors, list):
        for item in ors:
            if isinstance(item, dict) and isinstance(item.get("vehicle_model"), str):
                return item["vehicle_model"]
    vm = flt.get("vehicle_model")
    return vm if isinstance(vm, str) else None


_PROC_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "be", "as", "by", "from", "at", "into", "your", "you",
    "how", "do", "i", "we", "it", "this", "that", "these", "those",
    "step", "steps", "procedure", "process", "method", "guide", "show",
    "please", "need", "want", "help",
}


def _key_terms(text: str) -> set[str]:
    """Extract simple keyword set from a user question (deterministic)."""
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]{3,}", text.lower())
    terms = {w for w in words if w not in _PROC_STOPWORDS}
    # Add a couple of useful composed terms if present.
    if "wi" in terms and "fi" in terms:
        terms.discard("wi"); terms.discard("fi")
        terms.add("wifi")
    return terms


def _key_phrases(text: str) -> set[str]:
    """Extract 2-3 word phrases from the question (deterministic).

    Helps separate nearby SOP sub-flows:
      "data sync" ≠ "data collection"
      "software version" ≠ generic "software"
    """
    if not text:
        return set()
    toks = re.findall(r"[a-z0-9]{2,}", text.lower())
    toks = [t for t in toks if t not in _PROC_STOPWORDS]
    phrases: set[str] = set()
    for n in (2, 3):
        for i in range(0, len(toks) - n + 1):
            p = " ".join(toks[i:i + n])
            # Avoid trivial phrases like "diagnostic tool" that appear everywhere.
            if p in {"diagnostic tool", "service manual", "workshop manual"}:
                continue
            phrases.add(p)
    return phrases


def _filter_procedure_offtopic(
    matches: list[dict], question: str, classification: dict,
) -> list[dict]:
    """Generic off-topic trimming for procedure answers.

    Keep chunks that share key terms with the user's question, and drop
    chunks that look like adjacent sub-flows not requested. This is
    query-driven (no static feature words).
    """
    if not matches or classification.get("intent") != "procedure":
        return matches

    q = question or ""
    terms = _key_terms(q)
    phrases = _key_phrases(q)
    if not terms and not phrases:
        return matches

    # Always protect the top 2 chunks (highest rank after rerank/floor),
    # so we don't accidentally drop the core evidence on short queries.
    protected_ids = {m.get("id") for m in matches[:2] if m.get("id")}

    kept: list[dict] = []
    for m in matches:
        if m.get("id") in protected_ids:
            kept.append(m)
            continue
        md = m.get("metadata") or {}
        hay = f"{md.get('section') or ''}\n{md.get('text') or ''}".lower()
        # Prefer phrase matches (more specific), fall back to term overlap.
        phrase_hit = 0
        for p in phrases:
            if p and p in hay:
                phrase_hit += 1
                break

        term_hit = 0
        for t in terms:
            if t in hay:
                term_hit += 1
                if term_hit >= 3:
                    break

        # Keep if it matches a specific phrase OR multiple terms.
        if phrase_hit >= 1 or term_hit >= 2:
            kept.append(m)

    # Never return empty; if we were too aggressive, fall back to original.
    return kept if kept else matches


# Heuristic threshold for "this looks like a clarification reply" —
# a real clarification is typically a 1-3 word reply (a model name,
# a variant, "yes/no") meant to resolve the bot's prior question.
# A multi-word technical question that just happens not to start
# with "how/what/..." is NOT a clarification, even if short.
_CLARIFICATION_MAX_LEN = 50
_CLARIFICATION_MAX_WORDS = 4         # >4 words → treat as a fresh query
_QUESTION_WORD_RX = re.compile(
    r"\b(how|what|where|why|which|when|who|do|does|is|are|can|should|"
    r"could|would|tell|show|explain|walk)\b",
    re.I,
)

# A trailing "?" plus 3+ words almost always signals a fresh question
# even without a leading wh-word. "Front disc part number for the
# Bantam?" was being misclassified as a clarification because none
# of the wh-words appear in it; this catches that case.
_LIKELY_QUESTION_RX = re.compile(r"\?\s*$")

# Concrete-noun signals — if any of these appear, the message is
# almost certainly a fresh technical question, not a clarification
# reply to the bot. New domains can extend this without touching
# logic. Matched as substrings; lowercase.
_TECHNICAL_NOUN_HINTS = (
    "part number", "part no", "part #", "p/n",
    "torque", "capacity", "displacement", "compression ratio",
    "clearance", "spec", "specification", "diagram", "schematic",
    "warranty", "interval", "schedule",
    "disc", "brake", "caliper", "clutch", "valve", "shim",
    "spark plug", "filter", "coolant", "battery", "wiring",
    "wheel", "tyre", "tire", "fork", "shock", "swingarm",
    "harness", "connector", "fuse", "relay", "ecu",
    "fuel pump", "injector", "carburettor", "carburetor",
    "exhaust", "muffler", "catalyst",
    # accessories / availability queries
    "saddle", "bag", "bags", "accessory", "accessories",
    "cover", "seat", "grip", "mirror", "luggage", "storage",
    "price", "cost", "available",
    # symptom / diagnostic triggers
    "smell", "noise", "sound", "vibration", "leak", "leaking",
    "starting", "overheating", "temperature", "engine",
    "oil", "smoke", "burning",
)

# Phrases that signal "this question relies on the previous turn for
# its referent" — i.e. follow-ups like "what about the rear?" or
# "and for the Scrambler?". For these we keep ONE prior Q/A pair so
# the model knows the topic; everything else gets sent with NO
# history so the model focuses fully on the current question.
_FOLLOWUP_RX = re.compile(
    r"^\s*("
    r"and\b|"
    r"also\b|"
    r"what about\b|how about\b|how about you\b|"
    r"what'?s about\b|"
    r"and what\b|and how\b|and the\b|and for\b|"
    r"same (?:for|on|with)\b|"
    r"on (?:it|that|this|the same)\b|"
    r"for (?:it|that|this|the same)\b"
    r")",
    re.I,
)


def _is_clarification_reply(text: str) -> bool:
    """Short message with no question word — almost certainly an answer
    to a clarification question, not a new query.

    Guarded against three false-positive cases:
      1. Messages with 6+ words → fresh question (multi-concept).
      2. Messages ending in "?" with 3+ words → fresh question.
      3. Messages containing a technical noun (part number, torque,
         clutch, etc.) → fresh technical question.

    Without these guards, "Front disc part number for the Bantam?"
    gets misclassified as a clarification (39 chars, no wh-word),
    then merged with the prior question — producing a polluted
    retrieval query that misses the actual answer in Pinecone.
    """
    t = text.strip()
    if not t:
        return False
    if len(t) >= _CLARIFICATION_MAX_LEN:
        return False
    word_count = len(t.split())
    if word_count > _CLARIFICATION_MAX_WORDS:
        return False
    # "...?" with 3+ words — almost certainly a question, not a
    # one-word clarification reply.
    if _LIKELY_QUESTION_RX.search(t) and word_count >= 3:
        return False
    # Technical noun present — fresh domain question.
    low = t.lower()
    for hint in _TECHNICAL_NOUN_HINTS:
        if hint in low:
            return False
    # Has a wh-word — clearly a question, not a clarification.
    if _QUESTION_WORD_RX.search(t):
        return False
    return True


# Tokens that carry zero retrieval signal but pollute the embedding
# when concatenated to a substantive question. Examples: a user
# replies "yes please" to the bot's clarification, the merger glues
# that onto the prior question and embeds the whole thing — the
# resulting vector drifts away from the docs because "yes please"
# is closer to small-talk than to "valve clearance procedure".
# Stripped from the RETRIEVAL query only; the LLM still sees the
# original message so politeness/intent isn't lost.
_AFFIRMATIVE_NOISE_RX = re.compile(
    r"\b(?:yes\s+please|yes\s+thanks?|yes\s+go\s+on|yes|yeah|yep|"
    r"yup|sure|ok|okay|please|go\s+on|go\s+ahead|continue|"
    r"sounds\s+good|that\s+works|perfect|great)\b"
    r"[!.,]*",
    re.I,
)


def _strip_affirmative_noise(text: str) -> str:
    """Remove pure-affirmative tokens. Returns the cleaned text if
    something substantive remains, else the original (so a bare
    "yes please" still passes through to the clarification merger).
    """
    cleaned = _AFFIRMATIVE_NOISE_RX.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?")
    return cleaned if cleaned else text


# ─────────────────────── query refinement ───────────────────────
#
# Keep this conservative and deterministic: we want better retrieval for
# messy English shorthand without introducing extra latency or
# brittle LLM-dependent rewriting.
_NON_CONTENT_RX = re.compile(r"[^\w\s\-/+.:]", re.UNICODE)
_WHITESPACE_RX = re.compile(r"\s+")


def _refine_query_for_retrieval(text: str) -> str:
    """Light normalization + minimal keyword expansion for retrieval.

    - Normalizes whitespace and strips noisy punctuation.
    - Adds a few English synonyms for common service queries.

    Returns the original string if refinement wouldn't add signal.
    """
    raw = (text or "").strip()
    if not raw:
        return text

    # Normalize just enough to reduce embedding drift from random punctuation.
    cleaned = _NON_CONTENT_RX.sub(" ", raw)
    cleaned = _WHITESPACE_RX.sub(" ", cleaned).strip()

    low = cleaned.lower()
    additions: list[str] = []

    # Quantity / how-much phrasing (English).
    if any(w in low for w in ("how much", "quantity", "amount", "capacity", "volume")):
        additions.append("how much quantity amount capacity volume")
    if any(w in low for w in ("fill", "refill", "top up", "add", "pour")):
        additions.append("fill refill top up add pour")
    if any(w in low for w in ("after", "post")):
        additions.append("after")

    # Service topics with high exact-intent risk.
    if ("oil" in low and ("change" in low or "replace" in low)) or any(
        w in low for w in ("engine oil", "oil capacity", "oil quantity")
    ):
        additions.append("engine oil capacity refill quantity specification")

    if not additions:
        return cleaned if cleaned != raw else text

    extra = " ".join(dict.fromkeys(" ".join(additions).split()))
    refined = f"{cleaned} ({extra})"

    # Avoid runaway query growth.
    if len(refined) > 320:
        refined = refined[:320].rsplit(" ", 1)[0].rstrip(" ,.;:-") + ")"

    return refined if refined != raw else text


_QUERY_REWRITE_SYSTEM = (
    "You rewrite user questions for semantic search over technical manuals.\n"
    "Output ONLY JSON: {\"query\": \"...\"}.\n"
    "Rules:\n"
    "- Preserve model names/variants exactly if present (e.g., Gold Star, Bantam, Scrambler).\n"
    "- Convert to clear, specific English.\n"
    "- Keep it a single sentence query; do NOT answer.\n"
    "- Do NOT add credentials, emails, passwords, or sensitive data.\n"
    "- Keep it under 25 words.\n"
)


def _rewrite_query_llm(client: OpenAI, text: str) -> str:
    """LLM rewrite for retrieval, with strict timeout at call site."""
    raw = (text or "").strip()
    if not raw:
        return text
    try:
        resp = client.chat.completions.create(
            model=QUERY_REWRITE_MODEL,
            messages=[
                {"role": "system", "content": _QUERY_REWRITE_SYSTEM},
                {"role": "user", "content": raw[:800]},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="minimal",
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content else {}
        q = str(data.get("query") or "").strip()
        # Keep it conservative: if empty or too short, fall back.
        if len(q.split()) < 3:
            return text
        return q
    except Exception:  # noqa: BLE001
        return text


def _is_followup(text: str) -> bool:
    """Heuristic: this message references the prior turn (pronouns,
    'and the X', 'what about Y', 'same for Z'). Such messages need
    one prior Q/A pair to be answerable."""
    return bool(_FOLLOWUP_RX.search(text))


def _classify_message(text: str) -> str:
    """Return 'clarification' | 'followup' | 'standalone'.

    Drives how much history we send to the LLM:
      clarification → 0 turns (history is folded into the message)
      followup      → 1 turn  (just enough to anchor the topic)
      standalone    → 0 turns (don't let prior topics bias the answer)

    Order matters: check followup BEFORE clarification, otherwise
    short follow-ups like "And the Scrambler?" get mis-classified as
    one-word clarifications (no question word in our regex) and get
    merged into the prior question instead of paired with history.
    """
    if _is_followup(text):
        return "followup"
    if _is_clarification_reply(text):
        return "clarification"
    return "standalone"


def _resolve_clarification(
    message: str,
    history: list[tuple[str, str]] | None,
) -> tuple[str, str, list[tuple[str, str]]]:
    """If the user's latest message is a clarification reply (short, no
    question word), merge it together with the most recent SUBSTANTIVE
    question and return:

      (retrieval_query, llm_message, pruned_history)

    retrieval_query — a CLEAN sentence used for the embedding/Pinecone
                      lookup. Just the original question + clarifications,
                      no instructional boilerplate.
    llm_message     — the same content plus a strong directive in the
                      user's voice telling the model not to re-ask. This
                      is what the LLM sees as the user message.
    pruned_history  — history with the bot's "ask again" turn(s) removed
                      so the model doesn't pattern-match into looping.

    For non-clarifications or empty history, retrieval_query and
    llm_message are both the original message.
    """
    if not history or not _is_clarification_reply(message):
        return message, message, list(history or [])

    msg = message.strip()
    clarifications: list[str] = [msg]
    substantive_q: Optional[str] = None
    cut_idx = 0

    # Walk newest → oldest, accumulating any prior clarifications
    # until we find a real question.
    for i in range(len(history) - 1, -1, -1):
        prev_q = (history[i][0] or "").strip()
        if not prev_q:
            continue
        if _is_clarification_reply(prev_q):
            clarifications.insert(0, prev_q)
            continue
        substantive_q = prev_q
        cut_idx = i  # this turn is now folded into the message;
                     # keep only the turns before it
        break

    if substantive_q is None:
        # All history is clarifications — just space-join them.
        joined = " ".join(clarifications)
        return joined, joined, []

    # Strip any leftover qualifiers from the substantive question so
    # we don't accumulate prefixes / suffixes turn after turn:
    #   "For the Bantam: For the Bantam: How do I ..."
    #   "How do I ... (For the Bantam.) (For the Bantam.)"
    base = substantive_q.strip()
    base = re.sub(r"^\s*For the [^:]+:\s*", "", base, flags=re.I)
    base = re.sub(r"\s*\(For the [^)]+\.?\)\s*$", "", base, flags=re.I)
    base = base.rstrip(" .!?")

    # Deduplicate clarifications case-insensitively while keeping the
    # first occurrence's spelling. Repeats happen when the user
    # repeats themselves; complementary qualifiers (e.g. "Gold Star"
    # then "Euro-5+") stay separate. The LLM's prompt rules tell it
    # to treat the LAST qualifier as authoritative if any conflict.
    seen: set[str] = set()
    unique_clarifs: list[str] = []
    for c in clarifications:
        c_clean = c.strip()
        if not c_clean:
            continue
        key = c_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_clarifs.append(c_clean)

    qualifiers = ", ".join(unique_clarifs)

    # Clean form for retrieval: the original question + the qualifier(s)
    # only. No directive text — boilerplate would dilute the embedding
    # and miss real chunks. Also strip pure-affirmative tokens
    # ("yes please", "ok", etc.) — when the user just said "yes" to a
    # clarification, those words are noise in the embedding space.
    qualifiers_for_retrieval = _strip_affirmative_noise(qualifiers)
    if qualifiers_for_retrieval:
        retrieval_query = f"{base} {qualifiers_for_retrieval}".strip()
    else:
        # Pure affirmative reply — the substantive question alone IS
        # the retrieval intent. Don't pollute it with "yes".
        retrieval_query = base.strip()

    # Directive-laden form for the LLM: harder to ignore than a bare
    # parenthetical. Phrased as the user speaking, so the model treats
    # it as user intent rather than a system override.
    llm_message = (
        f"{base}? "
        f"(I've already told you: I'm asking about the {qualifiers}. "
        f"Do not ask the model/variant again — answer the question "
        f"directly for the {qualifiers}.)"
    )
    # Send NO history for a clarification: the merged message already
    # contains the substantive question + the user's qualifier, which
    # IS the complete intent. Any history we add here is unrelated
    # older conversation that biases the LLM toward off-topic chunks
    # and pattern-matches into asking the same clarification again.
    return retrieval_query, llm_message, []


def _serialize_match(m: dict) -> dict:
    """Strip Pinecone's match object down to JSON-safe metadata for the
    Node layer to persist alongside each conversation."""
    md = m["metadata"]
    return {
        "score": float(m["score"]),
        "section": md.get("section"),
        "page": md.get("page"),
        "source_pdf": md.get("source_pdf"),
        "vehicle_model": md.get("vehicle_model"),
        "doc_type": md.get("doc_type"),
        "image_paths": md.get("image_paths") or [],
    }


# ─────────────────────── endpoints ───────────────────────
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": CHAT_MODEL,
        "embed_cache": embed_cache_stats(),
    }


# ─────────────────────── learned answers ───────────────────────
def _stream_learned_answer(learned: dict, lookup_ms: float) -> StreamingResponse:
    """Stream a saved admin reply back through the same SSE protocol
    the LLM path uses, so the widget treats it identically — but
    flagged as `learned: true` so the UI can show a 'Saved reply
    from our team' label."""
    answer_text = str(learned["answer"])

    def gen():
        # 1. Meta event — no Pinecone matches, no images, just the
        #    learned-answer flag and a trivial timing breakdown.
        meta = {
            "matches": [],
            "top_score": float(learned["score"]),
            "filter": None,
            "images": {},
            "timings_ms": {
                "embed": 0,
                "pinecone": 0,
                "rerank": 0,
                "learned_lookup": round(lookup_ms),
            },
            "learned": {
                "match_score": float(learned["score"]),
                "created_at": learned.get("created_at"),
                "dealer_id": learned.get("dealer_id"),
            },
        }
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 2. Stream the saved answer in small chunks so the widget's
        #    typewriter still feels alive (rather than dumping the
        #    whole text in one event).
        chunk_size = 12
        for i in range(0, len(answer_text), chunk_size):
            piece = answer_text[i:i + chunk_size]
            yield f"event: token\ndata: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"

        # 3. Done event — total latency is just the lookup time.
        done_payload = {
            "finish_reason": "stop",
            "learned": True,
            "timings_ms": {
                "embed": 0,
                "pinecone": 0,
                "rerank": 0,
                "learned_lookup": round(lookup_ms),
                "llm_ttft": 0,
                "llm_total": 0,
                "total": round(lookup_ms),
            },
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



def _lookup_learned_answer(query: str) -> Optional[dict]:
    """Search the 'learned' namespace for a stored Q&A whose question
    is semantically very close to the current query. Returns a dict
    with 'answer', 'question', 'created_at', 'score', or None."""
    if _openai is None or _pinecone_index is None:
        return None
    try:
        vec = embed_query(_openai, query)
        res = _pinecone_index.query(
            vector=vec,
            top_k=1,
            include_metadata=True,
            namespace=LEARNED_NS,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("learned lookup failed: %s", e)
        return None
    matches = res.get("matches") or []
    if not matches:
        return None
    top = matches[0]
    score = float(top.get("score", 0.0))
    if score < LEARNED_THRESHOLD:
        return None
    md = top.get("metadata") or {}
    answer_text = md.get("answer")
    if not answer_text:
        return None
    return {
        "answer": answer_text,
        "question": md.get("question") or "",
        "created_at": md.get("created_at"),
        "dealer_id": md.get("dealer_id"),
        "score": score,
    }


@app.post("/learn")
def learn(req: LearnRequest) -> dict:
    """Store an admin-resolved Q&A so future similar questions get the
    same answer instantly. Idempotent-ish: each call adds a vector,
    duplicates can pile up but only the closest match wins at lookup."""
    if _openai is None or _pinecone_index is None:
        raise HTTPException(503, "service not ready")

    import time as _time
    import uuid

    vec = embed_query(_openai, req.question)
    vec_id = f"learned-{int(_time.time()*1000)}-{uuid.uuid4().hex[:8]}"
    metadata = {
        "question": req.question[:4000],   # Pinecone metadata size cap
        "answer": req.answer[:38000],
        "is_learned": True,
        "dealer_id": req.dealer_id or "",
        "session_id": req.session_id or "",
        "created_at": int(_time.time() * 1000),
    }
    _pinecone_index.upsert(
        vectors=[{"id": vec_id, "values": vec, "metadata": metadata}],
        namespace=LEARNED_NS,
    )
    log.info(f"[learn] saved learned Q&A id={vec_id} dealer={req.dealer_id} "
             f"q_len={len(req.question)} a_len={len(req.answer)}")
    return {"ok": True, "id": vec_id}


@app.delete("/learned/{item_id}")
def delete_learned(item_id: str) -> dict:
    if _pinecone_index is None:
        raise HTTPException(503, "service not ready")
    _pinecone_index.delete(ids=[item_id], namespace=LEARNED_NS)
    log.info(f"[learn] deleted id={item_id}")
    return {"ok": True}


@app.put("/learned/{item_id}")
def update_learned(item_id: str, req: LearnRequest) -> dict:
    if _openai is None or _pinecone_index is None:
        raise HTTPException(503, "service not ready")
    import time as _time
    _pinecone_index.delete(ids=[item_id], namespace=LEARNED_NS)
    vec = embed_query(_openai, req.question)
    metadata = {
        "question": req.question[:4000],
        "answer": req.answer[:38000],
        "is_learned": True,
        "dealer_id": req.dealer_id or "",
        "session_id": req.session_id or "",
        "created_at": int(_time.time() * 1000),
    }
    _pinecone_index.upsert(
        vectors=[{"id": item_id, "values": vec, "metadata": metadata}],
        namespace=LEARNED_NS,
    )
    log.info(f"[learn] updated id={item_id}")
    return {"ok": True, "id": item_id}


@app.post("/retrieve")
def retrieve_only(req: RetrieveRequest) -> dict:
    """Run embed + retrieve (+ optional rerank) and return the matches.
    Used by the Node layer to compute a confidence signal (top score)
    before deciding whether to call /chat or recommend human handoff."""
    if _openai is None or _pinecone_index is None:
        raise HTTPException(503, "service not ready")

    retrieval_query = _refine_query_for_retrieval(
        _strip_affirmative_noise(req.message)
    )
    if USE_QUERY_REWRITE:
        # Best-effort rewrite; fallback to the current retrieval_query.
        t0 = time.perf_counter()
        rewritten = _rewrite_query_llm(_openai, retrieval_query)
        if rewritten != retrieval_query:
            retrieval_query = rewritten
        _ = time.perf_counter() - t0  # keep local; /retrieve stays minimal
    flt = _resolve_filter(retrieval_query, req.vehicle, req.doc_type,
                          req.auto_filter)
    top_k = req.top_k if req.top_k is not None else dynamic_top_k(retrieval_query)
    vec = embed_query(_openai, retrieval_query)
    fetch_k = max(top_k, RERANK_FETCH_K) if USE_RERANK else top_k
    matches = retrieve(_pinecone_index, vec, fetch_k, req.min_score, flt)
    if USE_RERANK and matches and _pinecone_client is not None:
        keep = min(RERANK_KEEP, top_k)
        matches = rerank(_pinecone_client, retrieval_query, matches,
                         keep=keep, model=RERANK_MODEL)
    return {
        "matches": [_serialize_match(m) for m in matches],
        "top_score": float(matches[0]["score"]) if matches else 0.0,
        "filter": flt,
        "retrieval_query": retrieval_query,
        "top_k": top_k,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Streaming protocol — one event per record, blank line between events:
      event: status         data: {"text": "Looking at your question..."}
      event: meta           data: {"matches": [...], "top_score": 0.81}
      event: token          data: {"text": "The "}
      event: token          data: {"text": "recommended "}
      ...
      event: done           data: {"finish_reason": "stop"}
      event: error          data: {"message": "..."}

    The HTTP response opens immediately so the widget gets a `status` frame
    within ~100ms, instead of staring at typing dots until the LLM produces
    its first token. Pre-LLM pipeline (classify -> retrieve -> rerank ->
    filter -> diversify) runs INSIDE sse_gen, with status frames between
    steps so the user sees real progress.
    """
    if _openai is None or _pinecone_index is None or _system_prompt is None:
        raise HTTPException(503, "service not ready")

    request_started = time.perf_counter()
    raw_history = [(h.question, h.answer) for h in req.history]

    # Step trace — every gate the request passes through gets one line
    # appended here so the per-day debug log shows the full pipeline,
    # not just the final classifier values.
    steps: list[str] = []
    steps.append(
        f"request: msg_len={len(req.message)} "
        f"history={len(raw_history)} vehicle={req.vehicle or '-'} "
        f"doc_type={req.doc_type or '-'} top_k={req.top_k or 'auto'}"
    )

    # Message classification — clarification / followup / standalone.
    msg_class = _classify_message(req.message)
    steps.append(f"msg_class: {msg_class}")

    if msg_class == "clarification":
        retrieval_query, llm_message, history = _resolve_clarification(
            req.message, raw_history,
        )
    elif msg_class == "followup":
        retrieval_query = _strip_affirmative_noise(req.message)
        llm_message = req.message
        history = raw_history[-1:] if raw_history else []
    else:  # standalone
        retrieval_query = _strip_affirmative_noise(req.message)
        llm_message = req.message
        history = raw_history[-2:] if raw_history else []
    steps.append(f"history_used: {len(history)} turn(s)")

    # Conservative deterministic refinement before learned lookup + retrieval.
    pre_refine = retrieval_query
    retrieval_query = _refine_query_for_retrieval(retrieval_query)
    if retrieval_query.strip() != req.message.strip():
        steps.append(f"retrieval_query_rewritten: {retrieval_query!r}")
    if retrieval_query != pre_refine:
        steps.append(f"retrieval_query_refined: {retrieval_query!r}")

    # LLM rewrite (fast) before classifier + retrieval.
    if USE_QUERY_REWRITE and _openai is not None:
        # Run the rewrite with a strict budget so production latency doesn't spike.
        t_rw_start = time.perf_counter()
        rewritten = retrieval_query
        try:
            # OpenAI call itself can't be force-killed; timeout is enforced
            # by limiting how long we wait for the thread result.
            from concurrent.futures import TimeoutError as _FutTimeout  # local import
            fut = get_executor().submit(_rewrite_query_llm, _openai, retrieval_query)
            rewritten = fut.result(timeout=QUERY_REWRITE_TIMEOUT_S)
        except Exception:  # noqa: BLE001
            rewritten = retrieval_query
        t_rw_ms = (time.perf_counter() - t_rw_start) * 1000
        if rewritten and rewritten != retrieval_query:
            steps.append(
                f"retrieval_query_llm_rewrite: {rewritten!r} ({t_rw_ms:.0f}ms)"
            )
            retrieval_query = rewritten
        else:
            steps.append(f"retrieval_query_llm_rewrite: skipped ({t_rw_ms:.0f}ms)")

    # Learned-answers fast path — returns its own StreamingResponse.
    t_learned_start = time.perf_counter()
    learned = _lookup_learned_answer(retrieval_query)
    t_learned_ms = (time.perf_counter() - t_learned_start) * 1000
    if learned:
        log.info(
            f"[chat] q={req.message[:60]!r} -> LEARNED HIT "
            f"score={learned['score']:.3f} lookup={t_learned_ms:.0f}ms"
        )
        return _stream_learned_answer(learned, t_learned_ms)

    # Helper for SSE status frames. Newlines are real (\n) so the SSE
    # parser sees correct field separators.
    def _status(text: str) -> str:
        return (
            "event: status\n"
            "data: " + json.dumps({"text": text}, ensure_ascii=False) + "\n\n"
        )

    def sse_gen():
        # 2 KB SSE padding comment — defeats proxy/CDN/browser buffering.
        # Browsers (Chrome, Safari) and reverse proxies buffer SSE until they
        # accumulate ≥1 KB; without this, status events and the first token
        # are held invisible for 3-4 s even though the server has already
        # yielded them. The `:` prefix marks this as an SSE comment that
        # browsers silently ignore.
        yield ": " + (" " * 2048) + "\n\n"

        # === Status 1 — fires immediately when the response opens ===
        yield _status("Looking at your question...")

        # Classifier dispatch (runs in background thread)
        t_classify_start = time.perf_counter()
        classify_future = classify_query_async(
            get_executor(), _openai, retrieval_query,
        )
        steps.append("classifier: dispatched async (gpt-5-mini)")

        # Filter resolution (regex auto-detect, fast)
        flt = _resolve_filter(
            retrieval_query, req.vehicle, req.doc_type,
            req.auto_filter, history=raw_history,
        )
        steps.append(f"filter_initial: {flt}")

        base_top_k = (
            req.top_k if req.top_k is not None
            else dynamic_top_k(retrieval_query)
        )
        steps.append(
            f"top_k_base: {base_top_k} "
            f"({'caller-override' if req.top_k is not None else 'dynamic'})"
        )

        # === Status 2 — searching ===
        yield _status("Searching the manuals...")

        t_embed_start = time.perf_counter()
        vec = embed_query(_openai, retrieval_query)
        t_embed_ms = (time.perf_counter() - t_embed_start) * 1000
        steps.append(f"embed: {t_embed_ms:.0f}ms")

        # Classifier wait + regex fallback on timeout. The future keeps
        # running on timeout so its eventual result still populates the
        # LRU cache for the next identical question.
        has_vehicle = bool(req.vehicle) or (flt is not None and "$or" in flt)
        classify_timeout = 4.0 if has_vehicle else 5.5
        timed_out = False
        try:
            classification = classify_future.result(timeout=classify_timeout)
        except Exception as e:  # noqa: BLE001
            err_label = str(e) or type(e).__name__
            log.warning(
                "classifier timeout/error (%.1fs), using regex fallback: %s",
                classify_timeout, err_label,
            )
            timed_out = True
            classification = regex_classify(retrieval_query)
            classification["_fallback"] = "regex_only_timeout"
            classification["_llm_error"] = (
                f"{err_label} (no LLM response within {classify_timeout}s)"
            )
        t_classify_ms = (time.perf_counter() - t_classify_start) * 1000
        steps.append(
            f"classifier_resolved: path={classification.get('_fallback')} "
            f"timeout={classify_timeout}s "
            f"{'(TIMED OUT - used regex)' if timed_out else '(LLM completed)'} "
            f"in {t_classify_ms:.0f}ms"
        )
        steps.append(
            f"classifier_final: model={classification.get('model')} "
            f"variant={classification.get('variant')} "
            f"intent={classification.get('intent')} "
            f"confident={classification.get('confident')}"
        )
        full_detail = _wants_full_detail(req.message)
        steps.append(
            "detail_mode: "
            + ("full (explicit user request)" if full_detail else "concise-default")
        )
        llm_message_for_answer = llm_message
        if not full_detail:
            # Keep default answers complete but compact unless the user
            # explicitly asks for full/exhaustive detail.
            is_long_or_step_answer = classification.get("intent") in {
                "procedure", "diagnostic",
            }
            focus_terms = sorted(_key_terms(retrieval_query))[:10]
            focus_line = (
                (" Focus strictly on these query terms: " + ", ".join(focus_terms) + ".")
                if focus_terms else ""
            )
            if is_long_or_step_answer:
                # Strong relevance shaping only for long/step-style answers.
                llm_message_for_answer = (
                    f"{llm_message}\n\n"
                    "Response style for this reply: concise-default. "
                    "Include all required and safety-critical steps for the asked task, "
                    "but keep wording short and merge duplicate lines. "
                    + focus_line
                    + " Do not include steps or sub-flows that are not directly required "
                    "to answer the user’s question. "
                    "Do not output manual-style sections like "
                    "'General Information', 'Reassembly', or long "
                    "'Final warnings/notes' blocks unless explicitly requested. "
                    "Use only: Quick setup (optional, max 3 bullets), Steps, "
                    "and If it fails (optional, max 3 bullets)."
                )
            else:
                # Short/static answers should stay lightweight; avoid
                # over-constraining them with heavy step-focused directives.
                llm_message_for_answer = (
                    f"{llm_message}\n\n"
                    "Response style for this reply: concise-default. "
                    "Answer directly in short, clear wording and avoid unrelated details."
                )

        # Promote a confident classifier vehicle to a hard filter when
        # regex auto-detect didn't bind one.
        classifier_vehicle = classifier_to_filter_vehicle(classification)
        if (
            classifier_vehicle
            and not req.vehicle
            and (flt is None or "$or" not in flt)
        ):
            flt = build_filter(classifier_vehicle, req.doc_type)
            steps.append(
                f"filter_promoted: classifier vehicle={classifier_vehicle}"
            )
        elif classifier_vehicle:
            steps.append(
                "filter_promoted: skipped "
                f"(req.vehicle={req.vehicle!r}, regex_or has $or)"
            )

        # Hard guard: for model-sensitive intents, ask model first if unknown.
        active_vehicle = (
            req.vehicle
            or classifier_vehicle
            or _extract_vehicle_from_filter(flt)
        )
        model_sensitive_intents = {
            "procedure", "diagnostic", "spec", "parts", "wiring", "safety",
        }
        if classification.get("intent") in model_sensitive_intents and not active_vehicle:
            steps.append(
                "model_guard: vehicle unknown for model-sensitive intent; "
                "asking clarification"
            )
            q = (
                "Which model are you using: "
                "BSA Bantam, BSA Gold Star, or BSA Scrambler?"
            )
            # Keep this as the entire assistant response (no partial answer).
            yield f"event: token\ndata: {json.dumps({'text': q}, ensure_ascii=False)}\n\n"
            yield (
                "event: done\ndata: "
                + json.dumps({"finish_reason": "stop", "timings_ms": {
                    "classify": round(t_classify_ms), "embed": round(t_embed_ms),
                }}, ensure_ascii=False)
                + "\n\n"
            )
            return

        # Intent-based top_k.
        top_k = (
            base_top_k if req.top_k is not None
            else intent_top_k(classification, base_top_k)
        )
        if top_k != base_top_k:
            steps.append(
                f"top_k_intent_adjusted: {top_k} (from base {base_top_k}, "
                f"intent={classification.get('intent')})"
            )
        else:
            steps.append(f"top_k_final: {top_k}")

        base_fetch_k = (
            max(top_k, RERANK_FETCH_K) if USE_RERANK else top_k
        )
        fetch_k = (
            intent_fetch_k(classification, base_fetch_k)
            if USE_RERANK else top_k
        )
        steps.append(f"fetch_k: {fetch_k} (rerank_enabled={USE_RERANK})")
        steps.append(f"filter_used_for_retrieval: {flt}")

        t_pine_start = time.perf_counter()
        matches = retrieve(
            _pinecone_index, vec, fetch_k, req.min_score, flt,
        )
        t_pine_ms = (time.perf_counter() - t_pine_start) * 1000
        steps.append(
            f"pinecone_retrieve: {len(matches)} chunks in {t_pine_ms:.0f}ms "
            f"(min_score={req.min_score})"
        )

        # === Status 3 — reading ===
        yield _status(f"Reading {len(matches)} relevant pages...")

        # Rerank with a cross-encoder.
        t_rerank_ms = 0.0
        fetched_count = len(matches)
        if USE_RERANK and matches and _pinecone_client is not None:
            rerank_keep = max(RERANK_KEEP, top_k)
            t_rerank_start = time.perf_counter()
            matches = rerank(
                _pinecone_client, retrieval_query, matches,
                keep=rerank_keep, model=RERANK_MODEL,
            )
            t_rerank_ms = (time.perf_counter() - t_rerank_start) * 1000
            steps.append(
                f"rerank: {fetched_count} -> {len(matches)} chunks "
                f"via {RERANK_MODEL} in {t_rerank_ms:.0f}ms "
                f"(keep={rerank_keep})"
            )

            # Section-locked expansion (long-form intents only).
            long_form = intent_long_form(classification) and full_detail
            if long_form:
                dominant = detect_dominant_section(
                    matches, top_n=3, question=retrieval_query,
                )
                if dominant:
                    section_filter = (
                        dict(flt) if isinstance(flt, dict) else {}
                    )
                    section_filter["section"] = dominant
                    try:
                        extra = retrieve(
                            _pinecone_index, vec, fetch_k,
                            req.min_score, section_filter,
                        )
                    except Exception as e:  # noqa: BLE001
                        log.warning(
                            "section-locked expansion failed: %s", e,
                        )
                        extra = []
                    if extra:
                        before = len(matches)
                        matches = merge_unique_matches(matches, extra)
                        log.info(
                            f"[chat] section-lock '{dominant}' added "
                            f"{len(matches) - before} chunks"
                        )
                        steps.append(
                            f"section_lock: '{dominant}' added "
                            f"{len(matches) - before} chunks"
                        )
                    else:
                        steps.append(
                            f"section_lock: '{dominant}' "
                            "found no extra chunks"
                        )
                else:
                    steps.append("section_lock: skipped (no dominant section)")
            else:
                steps.append(
                    f"section_lock: skipped (intent="
                    f"{classification.get('intent')} not full-detail mode)"
                )
        elif not USE_RERANK:
            steps.append("rerank: disabled (USE_RERANK=False)")
        elif not matches:
            steps.append("rerank: skipped (no Pinecone matches)")

        # === Status 4 — picking the most relevant ===
        yield _status("Picking the most relevant...")

        # Section-relevance soft filter.
        before_section = len(matches)
        matches = filter_by_section_relevance(
            matches, retrieval_query, classification,
        )
        if len(matches) != before_section:
            log.info(
                f"[chat] section filter dropped "
                f"{before_section - len(matches)}/{before_section} "
                f"off-subsystem chunks"
            )
            steps.append(
                f"section_relevance: dropped "
                f"{before_section - len(matches)}, kept {len(matches)}"
            )
        else:
            steps.append(
                f"section_relevance: no chunks dropped "
                f"({before_section} kept)"
            )

        # Rerank-floor filter.
        before_floor = len(matches)
        matches = filter_by_rerank_floor(matches, classification)
        if len(matches) != before_floor:
            log.info(
                f"[chat] rerank floor dropped {before_floor - len(matches)}"
                f"/{before_floor} chunks (intent={classification['intent']})"
            )
            steps.append(
                f"rerank_floor: dropped {before_floor - len(matches)}, "
                f"kept {len(matches)} (intent={classification.get('intent')})"
            )
        else:
            steps.append(
                f"rerank_floor: no chunks dropped ({before_floor} kept)"
            )

        # Enforce same-model evidence in final answer context whenever
        # a target vehicle is known (explicit user filter or confident classifier).
        answer_vehicle = req.vehicle or classifier_vehicle
        before_vehicle = len(matches)
        matches = _enforce_vehicle_purity(matches, answer_vehicle)
        if len(matches) != before_vehicle:
            steps.append(
                f"vehicle_purity: dropped {before_vehicle - len(matches)}, "
                f"kept {len(matches)} (vehicle={answer_vehicle})"
            )
        else:
            steps.append(
                f"vehicle_purity: no chunks dropped (vehicle={answer_vehicle or '-'})"
            )

        # Focus diagnostic answers on the requested subsystem (e.g. battery/charging)
        # to avoid unrelated remedies leaking into final output.
        before_diag = len(matches)
        matches = _filter_diagnostic_offtopic(matches, retrieval_query, classification)
        if len(matches) != before_diag:
            steps.append(
                f"diagnostic_focus: dropped {before_diag - len(matches)}, "
                f"kept {len(matches)}"
            )
        else:
            steps.append("diagnostic_focus: no chunks dropped")

        # Procedure-focused off-topic trimming (e.g. Job Card/IUPR steps)
        # unless the user explicitly asked for those sub-flows.
        before_proc = len(matches)
        matches = _filter_procedure_offtopic(matches, retrieval_query, classification)
        if len(matches) != before_proc:
            steps.append(
                f"procedure_focus: dropped {before_proc - len(matches)}, "
                f"kept {len(matches)}"
            )
        else:
            steps.append("procedure_focus: no chunks dropped")

        # Doc-type diversification.
        pref_types = intent_doc_types(classification)
        before_div = len(matches)
        if matches and len(matches) > top_k:
            matches = diversify_by_doc_type(
                matches, keep=top_k,
                preferred_types=pref_types or None,
            )
            steps.append(
                f"diversify: trimmed {before_div} -> {len(matches)} "
                f"(preferred_types={list(pref_types) or '-'})"
            )
        elif matches and pref_types:
            matches = diversify_by_doc_type(
                matches, keep=len(matches),
                preferred_types=pref_types,
            )
            steps.append(
                f"diversify: reordered {len(matches)} chunks "
                f"(preferred_types={list(pref_types)})"
            )
        else:
            steps.append(
                f"diversify: skipped ({len(matches)} chunks, no pref_types)"
            )

        # Page-neighbour expansion (long-form intents only).
        if intent_long_form(classification) and full_detail and matches:
            neighbour_ids = page_neighbor_ids(matches[0], span=2)
            if neighbour_ids:
                try:
                    fetch_res = _pinecone_index.fetch(ids=neighbour_ids)
                    seed_vehicle = (
                        (matches[0].get("metadata") or {}).get("vehicle_model")
                    )
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
                        matches = merge_unique_matches(
                            matches, extra_neighbours,
                        )
                        page = (matches[0].get("metadata") or {}).get("page")
                        log.info(
                            f"[chat] page-neighbour added "
                            f"{len(matches) - before} chunks around "
                            f"page {page}"
                        )
                        steps.append(
                            f"page_neighbour: added "
                            f"{len(matches) - before} chunks "
                            f"around page {page}"
                        )
                    else:
                        steps.append(
                            "page_neighbour: no extra chunks "
                            "(vehicle filter or non-existent IDs)"
                        )
                except Exception as e:  # noqa: BLE001
                    log.warning("page-neighbour fetch failed: %s", e)
                    steps.append(f"page_neighbour: fetch failed ({e})")
            else:
                steps.append(
                    "page_neighbour: no candidate IDs from top match"
                )
        else:
            steps.append(
                f"page_neighbour: skipped "
                f"(intent={classification.get('intent')} not full-detail mode, "
                "or no matches)"
            )

        # Final generation cap: keep prompt compact by default while
        # allowing larger context when the user explicitly asks for full detail.
        answer_cap = (
            ANSWER_MAX_CHUNKS_FULL if full_detail else ANSWER_MAX_CHUNKS
        )
        if len(matches) > answer_cap:
            before_cap = len(matches)
            matches = matches[:answer_cap]
            steps.append(
                f"answer_cap: trimmed {before_cap} -> {len(matches)} "
                f"(cap={answer_cap}, mode={'full' if full_detail else 'concise'})"
            )
        else:
            steps.append(
                f"answer_cap: kept {len(matches)} "
                f"(cap={answer_cap}, mode={'full' if full_detail else 'concise'})"
            )
        steps.append(f"final_chunks: {len(matches)}")

        image_map = build_image_map(matches)
        # Sign each S3 image URL with a short-lived presigned URL so the
        # widget can fetch the image even though the bucket is private.
        # Falls back to the unsigned URL if boto3 isn't configured or
        # the URL doesn't match an S3 pattern.
        image_map = _sign_image_map(image_map)
        preview = req.message[:60].replace("\n", " ")
        top_score = float(matches[0]["score"]) if matches else 0.0
        log.info(
            f"[chat] q={preview!r} class={msg_class} hist={len(history)} "
            f"intent={classification['intent']} "
            f"model={classification['model']} "
            f"variant={classification['variant']} "
            f"conf={classification['confident']} "
            f"classify={t_classify_ms:.0f}ms "
            f"top_k={top_k}"
            f"{'(auto)' if req.top_k is None else ''} "
            f"embed={t_embed_ms:.0f}ms pinecone={t_pine_ms:.0f}ms "
            f"rerank={t_rerank_ms:.0f}ms fetched={fetched_count} "
            f"kept={len(matches)} top={top_score:.2f} model={req.model}"
        )

        ctx_text_for_verify = render_context(matches) if matches else ""
        ctx_pages_for_verify = [
            (m.get("metadata") or {}).get("page") for m in matches
        ]

        # Send the meta event WITHOUT images so it stays small (~1 KB).
        # Images are delivered in a dedicated `images` event right after the
        # first token, so they never sit in the pipe before the answer starts.
        meta = {
            "matches": [_serialize_match(m) for m in matches],
            "top_score": float(matches[0]["score"]) if matches else 0.0,
            "filter": flt,
            "classification": classification,
            "timings_ms": {
                "embed": round(t_embed_ms),
                "pinecone": round(t_pine_ms),
                "rerank": round(t_rerank_ms),
                "classify": round(t_classify_ms),
            },
            "rerank": {
                "enabled": USE_RERANK,
                "fetched": fetched_count,
                "kept": len(matches),
                "model": RERANK_MODEL if USE_RERANK else None,
            },
        }
        yield (
            "event: meta\ndata: "
            + json.dumps(meta, ensure_ascii=False)
            + "\n\n"
        )

        # === Status 5 — writing ===
        yield _status("Writing answer...")

        # LLM stream + finally block to guarantee debug log entry.
        t_llm_start = time.perf_counter()
        t_first_token = None
        token_count = 0
        answer_buf: list[str] = []
        verify_result: dict = {"ok": True, "unsupported": []}
        _images_sent = False  # images event is deferred until after first token
        t_verify_ms = 0.0
        ttft_ms = 0.0
        llm_total_ms = 0.0
        total_ms = 0.0
        full_answer = ""
        llm_error: Exception | None = None
        steps.append(
            f"llm_dispatch: model={req.model} "
            f"intent={classification.get('intent')} matches={len(matches)}"
        )
        try:
            try:
                for piece in stream_answer(
                    _openai, llm_message_for_answer, matches, _system_prompt,
                    history=history, model=req.model,
                    image_b64=req.image_b64,
                    intent=classification.get("intent"),
                ):
                    if t_first_token is None and piece and piece.strip():
                        t_first_token = time.perf_counter()
                        ttft_now = (t_first_token - request_started) * 1000
                        log.info(
                            f"[chat] first-token (user sees answer) "
                            f"after {ttft_now:.0f}ms"
                        )
                        steps.append(f"llm_first_token: at {ttft_now:.0f}ms")
                        # Send image map immediately after the first real token
                        # so the frontend can resolve [[SHOW_IMAGE]] tags as
                        # they stream in. Sending it here (not in meta) means
                        # the meta event stays tiny and the first word appears
                        # with zero buffering delay.
                        if not _images_sent:
                            _images_sent = True
                            yield (
                                "event: images\ndata: "
                                + json.dumps(
                                    {"images": image_map}, ensure_ascii=False
                                )
                                + "\n\n"
                            )
                    token_count += 1
                    answer_buf.append(piece)
                    payload = json.dumps({"text": piece}, ensure_ascii=False)
                    yield f"event: token\ndata: {payload}\n\n"
            except Exception as e:  # noqa: BLE001
                llm_error = e
                err_label = (
                    f"{type(e).__name__}: "
                    f"{str(e) or '(empty error message)'}"
                )
                steps.append(f"llm_error: {err_label}")
                log.warning("[chat] LLM stream raised: %s", err_label)
                err = json.dumps(
                    {"message": str(e) or type(e).__name__},
                    ensure_ascii=False,
                )
                yield f"event: error\ndata: {err}\n\n"

            # Guarantee images event is sent even when LLM errors before
            # producing a real token (so the frontend never waits forever).
            if not _images_sent:
                _images_sent = True
                yield (
                    "event: images\ndata: "
                    + json.dumps({"images": image_map}, ensure_ascii=False)
                    + "\n\n"
                )

            full_answer = "".join(answer_buf)
            if llm_error is None and full_answer:
                t_verify_start = time.perf_counter()
                try:
                    verify_result = verify_grounding(
                        full_answer, ctx_text_for_verify,
                        ctx_pages=ctx_pages_for_verify,
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("grounding verifier failed: %s", e)
                    verify_result = {"ok": True, "unsupported": []}
                t_verify_ms = (time.perf_counter() - t_verify_start) * 1000

                if not verify_result["ok"]:
                    log.warning(
                        "[chat] grounding flag - unsupported atoms: %s",
                        verify_result["unsupported"],
                    )

            t_llm_end = time.perf_counter()
            ttft_ms = ((t_first_token or t_llm_end) - t_llm_start) * 1000
            llm_total_ms = (t_llm_end - t_llm_start) * 1000
            total_ms = (t_llm_end - request_started) * 1000

            if llm_error is None:
                log.info(
                    f"[chat] done  ttft={ttft_ms:.0f}ms  "
                    f"llm_total={llm_total_ms:.0f}ms  "
                    f"out_chunks={token_count}  total={total_ms:.0f}ms  "
                    f"grounding="
                    f"{'ok' if verify_result['ok'] else 'FLAGGED'}"
                )
                done_payload = {
                    "finish_reason": "stop",
                    "grounding": verify_result,
                    "timings_ms": {
                        "embed": round(t_embed_ms),
                        "pinecone": round(t_pine_ms),
                        "rerank": round(t_rerank_ms),
                        "classify": round(t_classify_ms),
                        "verify": round(t_verify_ms),
                        "llm_ttft": round(ttft_ms),
                        "llm_total": round(llm_total_ms),
                        "total": round(total_ms),
                    },
                }
                yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
        finally:
            try:
                write_debug_entry(
                    question=req.message,
                    retrieval_query=retrieval_query,
                    answer=full_answer or (
                        f"[LLM ERROR - no answer streamed: "
                        f"{type(llm_error).__name__}: {llm_error}]"
                        if llm_error else ""
                    ),
                    matches=matches,
                    image_map=image_map,
                    classification=classification,
                    filter_used=flt,
                    top_k=top_k,
                    grounding=verify_result,
                    msg_class=msg_class,
                    timings_ms={
                        "classify": round(t_classify_ms),
                        "embed": round(t_embed_ms),
                        "pinecone": round(t_pine_ms),
                        "rerank": round(t_rerank_ms),
                        "llm_ttft": round(ttft_ms),
                        "llm_total": round(llm_total_ms),
                        "verify": round(t_verify_ms),
                        "total": round(total_ms),
                    },
                    pipeline_steps=steps,
                )
            except Exception as log_err:  # noqa: BLE001
                log.warning(
                    "debug log write failed in finally: %s", log_err,
                )

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
