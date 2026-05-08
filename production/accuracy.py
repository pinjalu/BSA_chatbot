from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

log = logging.getLogger(__name__)


CLASSIFIER_MODEL = os.environ.get("ACCURACY_CLASSIFIER_MODEL", "gpt-5-mini")


@dataclass(frozen=True)
class Vehicle:
    id: str                     # canonical name returned by classifier
    pinecone_filter: str        # the value stored in Pinecone metadata
    aliases: tuple[str, ...]    # lowercase normalised aliases (no spaces/dashes)
    pattern: re.Pattern         # regex matched against question text


VEHICLES: list[Vehicle] = [
    Vehicle(
        id="Bantam",
        pinecone_filter="BSA Bantam",
        aliases=("bantam",),
        pattern=re.compile(r"\bbantam\b", re.I),
    ),
    Vehicle(
        id="Goldstar",
        pinecone_filter="BSA Gold Star",
        aliases=("goldstar", "gs"),
        pattern=re.compile(r"\b(?:gold[\s-]?star|gs)\b", re.I),
    ),
    Vehicle(
        id="Scrambler",
        pinecone_filter="BSA Scrambler",
        aliases=("scrambler",),
        pattern=re.compile(r"\bscrambler\b", re.I),
    ),
]


@dataclass(frozen=True)
class Variant:
    id: str
    aliases: tuple[str, ...]    # lowercase normalised forms
    pattern: re.Pattern         # regex matched against question text


# Listed longest-pattern-first so "Euro-5+" beats "Euro-5".
# Note: the trailing `\+` doesn't need a `\b` because `+` is itself a
# non-word character and acts as a delimiter; appending `\b` actually
# breaks the match when followed by punctuation/whitespace.
VARIANTS: list[Variant] = [
    Variant(
        id="Euro-5+",
        aliases=("euro-5+", "euro5+", "euro-5plus", "euro5plus"),
        pattern=re.compile(r"\beuro[\s-]?5\s*(?:\+|plus\b)", re.I),
    ),
    Variant(
        id="Euro-5",
        aliases=("euro-5", "euro5"),
        pattern=re.compile(r"\beuro[\s-]?5\b", re.I),
    ),
]


@dataclass(frozen=True)
class IntentProfile:
    """All retrieval + generation knobs for one intent label.

    Adding a new intent = one entry. Logic below reads these fields,
    so there are no `if intent == "..."` checks scattered around.
    """
    label: str
    top_k: int
    fetch_k: int
    rerank_floor: float          # drop chunks below floor*top_score
    min_keep: int                # never trim below this regardless of floor
    preferred_doc_types: tuple[str, ...] = ()
    long_form: bool = False      # triggers procedure-style depth in LLM
    llm_verbosity: str = "medium"
    llm_reasoning_effort: str = "low"
    regex: Optional[re.Pattern] = None  # pattern for regex fallback


# Default profile applied to ANY intent label not explicitly listed —
# including new labels a future classifier might emit. Means "we
# tolerate unknown intents without crashing or losing recall".
DEFAULT_INTENT_PROFILE = IntentProfile(
    label="general",
    top_k=6,
    fetch_k=20,
    rerank_floor=0.20,
    min_keep=3,
    preferred_doc_types=(),
    long_form=False,
    llm_verbosity="medium",
    llm_reasoning_effort="low",
)


INTENT_PROFILES: list[IntentProfile] = [
    IntentProfile(
        label="parts",
        top_k=4, fetch_k=20, rerank_floor=0.40, min_keep=2,
        preferred_doc_types=("parts_catalogue",),
        regex=re.compile(
            r"\b(part (?:no|number|#)|p/n|partno|part code)\b", re.I,
        ),
    ),
    IntentProfile(
        label="spec",
        top_k=5, fetch_k=20, rerank_floor=0.40, min_keep=2,
        preferred_doc_types=("workshop_manual", "owners_manual", "spec_sheet"),
        regex=re.compile(
            r"\b(torque (?:spec|value)|gap (?:spec|setting)|"
            r"compression ratio|capacit(?:y|ies)|displacement|"
            r"bore [x×] stroke|spark plug (?:gap|spec|model)|"
            r"oil grade|oil capacity|recommended)\b", re.I,
        ),
    ),
    IntentProfile(
        label="warranty",
        top_k=8, fetch_k=24, rerank_floor=0.20, min_keep=4,
        preferred_doc_types=(),  # warranty booklet is universal
        long_form=True,
        llm_verbosity="high",
        llm_reasoning_effort="minimal",
        regex=re.compile(
            r"\b(warranty|warrant(?:y|ies)|claim|coverage|covered under|"
            r"transferable|emission certificate|service interval|"
            r"service schedule|maintenance schedule|"
            r"(?:when|how often) (?:should|do|to) (?:i|you|the)?\s*"
            r"(?:replace|change|service|inspect|clean|check)|"
            r"(?:should|do|does) [\w\s]+ be "
            r"(?:serviced|replaced|inspected|cleaned|changed))\b",
            re.I,
        ),
    ),
    IntentProfile(
        label="procedure",
        top_k=16, fetch_k=30, rerank_floor=0.08, min_keep=8,
        preferred_doc_types=("workshop_manual", "owners_manual", "sop"),
        long_form=True,
        llm_verbosity="medium",
        llm_reasoning_effort="minimal",
        regex=re.compile(
            r"\b(walk me through|step[- ]?by[- ]?step|step\s*\d|"
            r"procedure|how (?:do|to|can) (?:i|you)|how should i|"
            r"reflash|adjust|replace|remove|install|disassembl|"
            r"reassembl|service the|change the|set the|tighten|torque|"
            r"bleed the|drain the|fit the|fitment|overhaul)\b", re.I,
        ),
    ),
    IntentProfile(
        label="safety",
        top_k=10, fetch_k=24, rerank_floor=0.15, min_keep=5,
        preferred_doc_types=("owners_manual", "workshop_manual"),
        long_form=True,
        llm_verbosity="high",
        llm_reasoning_effort="minimal",
        regex=re.compile(
            r"\b(fire|smok(?:e|ing)|leak(?:ing)?|spill|burn(?:ing)?|"
            r"overheat(?:ing)?|smell|fume|emergency|"
            r"brake (?:fail|gone))\b", re.I,
        ),
    ),
    IntentProfile(
        # Diagnostic / troubleshooting — symptom-first questions
        # ("why is X happening", "what causes Y", "won't start",
        # "noise / vibration / rattle / smoke / DTC / warning light").
        # Tuned similarly to procedure (long-form, multi-source) but
        # leans on workshop-manual troubleshooting tables and SOPs.
        label="diagnostic",
        top_k=12, fetch_k=28, rerank_floor=0.10, min_keep=6,
        preferred_doc_types=("workshop_manual", "sop", "owners_manual"),
        long_form=True,
        llm_verbosity="medium",
        llm_reasoning_effort="minimal",
        regex=re.compile(
            r"\b("
            r"why\s+(?:does|is|won['']?t|won|can['']?t|cannot|"
            r"doesn['']?t|does\s+not|isn['']?t|is\s+not)|"
            r"what\s+(?:could\s+be|is\s+causing|causes|might\s+cause|"
            r"would\s+cause)|"
            r"how\s+come|"
            r"how\s+do\s+i\s+(?:fix|diagnose|troubleshoot)|"
            r"vibrat(?:e|es|ing|ion|ions)|"
            r"nois(?:e|y|es)|rattl(?:e|es|ing)|"
            r"knock(?:ing|s)?|squeal(?:ing|s)?|"
            r"grind(?:ing|s)?|clunk(?:ing|s)?|"
            r"shake(?:s|n|ing)?|wobbl(?:e|es|ing|y)|"
            r"shudder(?:s|ing)?|hesitat(?:e|es|ion|ing)|"
            r"won['']?t\s+(?:start|run|crank|idle|move|shift|engage|fire)|"
            r"will\s+not\s+(?:start|run|crank|idle)|"
            r"(?:doesn['']?t|does\s+not)\s+"
            r"(?:start|run|crank|idle|engage|shift)|"
            r"hard\s+to\s+start|rough\s+idle|"
            r"stall(?:s|ing|ed)?|misfire(?:s|d)?|"
            r"loss\s+of\s+power|low\s+power|losing\s+power|"
            r"overheat(?:s|ing|ed)?|"
            r"\bdtc\b|trouble\s+code|warning\s+light|"
            r"check\s+engine|\bmil\b|service\s+light|"
            r"diagnos(?:e|tic|tics|ing|tics\s+tool)|"
            r"fault|abnormal|unusual|strange|"
            r"symptom|"
            r"(?:issue|problem|trouble|fault)\s+with|"
            r"(?:not|isn['']?t|aren['']?t)\s+working|"
            r"fails?\s+to|failure"
            r")\b",
            re.I,
        ),
    ),
    IntentProfile(
        label="wiring",
        top_k=10, fetch_k=24, rerank_floor=0.20, min_keep=5,
        preferred_doc_types=("wiring_diagram", "workshop_manual"),
        long_form=True,
        llm_verbosity="high",
        llm_reasoning_effort="minimal",
        regex=re.compile(
            r"\b(wiring (?:diagram|schematic)|harness|ecu pin|pinout|"
            r"connector\s+\w+|ground point|fuse box|relay box)\b", re.I,
        ),
    ),
    DEFAULT_INTENT_PROFILE,  # label="general", catch-all
]


# Topic clusters used by section-relevance filtering. A chunk's section is
# "off-subsystem" if it matches a cluster the question doesn't, AND the
# question matches some other cluster. Adding a new subsystem = one entry.
TOPIC_CLUSTERS: dict[str, tuple[str, ...]] = {
    "valve_train":        ("valve", "shim", "cam", "camshaft", "tappet",
                           "rocker", "cylinder head"),
    "fuel_system":        ("fuel", "petrol", "carburettor", "carburetor",
                           "injector", "fuel pump", "fuel line",
                           "fuel tank", "throttle"),
    "ignition":           ("ignition", "spark plug", "magneto",
                           "ecu reflash",
                           # Diagnostic-symptom keywords so a question like
                           # "engine misfiring at idle" lands in this cluster
                           # and section-relevance filtering can drop chunks
                           # from unrelated subsystems.
                           "misfire", "misfiring", "rough idle",
                           "won't start", "hard to start"),
    "lubrication":        ("engine oil", "lubricat", "oil pump",
                           "oil filter", "oil change",
                           "oil leak", "oil consumption"),
    "cooling":            ("coolant", "radiator", "thermostat", "cooling",
                           # Symptom phrasings users actually type — without
                           # these, a question like "my bike overheats" fails
                           # to bind to any cluster and the section filter is
                           # bypassed entirely.
                           "overheat", "radiator fan", "engine temperature",
                           "coolant temp"),
    "transmission":       ("clutch", "gearbox", "transmission", "shift",
                           "primary drive", "drive chain",
                           "slipping clutch", "gear noise",
                           "grinding gears"),
    "brakes":             ("brake", "abs", "pad", "caliper", "rotor",
                           "disc",
                           "brake squeal", "brake noise", "brake fail"),
    "suspension_chassis": ("suspension", "fork", "shock", "swingarm",
                           "frame", "chassis", "wheel", "tyre", "tire",
                           # High-speed vibration / wobble most commonly
                           # belongs to this subsystem (wheel balance, tyre,
                           # suspension) — listing here so symptom-only
                           # questions route correctly.
                           "vibration", "vibrat", "wobble", "wobbl",
                           "shake", "shudder", "rattle", "rattl"),
    "electrical":         ("wiring", "harness", "battery", "starter motor",
                           "alternator", "headlamp", "indicator", "fuse",
                           "relay", "switch", "ecu pin",
                           "warning light", "check engine", "mil",
                           "dtc", "trouble code"),
    "exhaust_emission":   ("exhaust", "muffler", "emission", "catalyst",
                           "lambda",
                           "smoke", "smoking exhaust"),
    "warranty_admin":     ("warranty", "service interval",
                           "service schedule", "claim", "certificate",
                           "ownership"),
}


# Build O(1) lookups from the config so each call is constant-time.
_PROFILE_BY_LABEL: dict[str, IntentProfile] = {
    p.label: p for p in INTENT_PROFILES
}
_VALID_INTENTS: frozenset[str] = frozenset(_PROFILE_BY_LABEL.keys())
_ALIAS_TO_VEHICLE: dict[str, Vehicle] = {
    a: v for v in VEHICLES for a in v.aliases
}
_ALIAS_TO_VARIANT: dict[str, Variant] = {
    a: vt for vt in VARIANTS for a in vt.aliases
}


def _profile_for(intent: str | None) -> IntentProfile:
    """Look up the IntentProfile, falling back to the default for any
    unknown label. This is what makes new intent labels safe."""
    if intent is None:
        return DEFAULT_INTENT_PROFILE
    return _PROFILE_BY_LABEL.get(intent, DEFAULT_INTENT_PROFILE)


_CLASS_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CLASS_CACHE_MAX = int(os.environ.get("ACCURACY_CLASS_CACHE_MAX", "256"))


def _build_classifier_system_prompt() -> str:
    """Generate the classifier system prompt FROM the config so the
    enums stay in sync with VEHICLES / VARIANTS / INTENT_PROFILES."""
    vehicle_ids = " | ".join(f'"{v.id}"' for v in VEHICLES) + ' | "unknown"'
    variant_ids = " | ".join(f'"{v.id}"' for v in VARIANTS) + ' | "any" | "unknown"'
    intent_ids = " | ".join(f'"{p.label}"' for p in INTENT_PROFILES)

    return f"""You are a query router for a motorcycle technical RAG system.
Return ONE compact JSON object — no prose, no markdown.

Schema:
  {{
    "model":   {vehicle_ids},
    "variant": {variant_ids},
    "intent":  {intent_ids},
    "confident": true | false
  }}

Rules:
- "model" — set ONLY if the question or recent context clearly names a
  known vehicle. Match aliases case-insensitively. Otherwise "unknown".
- "variant" — pick the most specific match if present; "any" if cross-
  variant; "unknown" if missing AND the answer would differ.
- "intent" — pick the best label for the dominant question type:
    parts      → asking for a part number / part identifier
    spec       → torque, capacity, ratio, gap, dimension, oil grade, plug
    procedure  → "how do I", "walk me through", step-by-step
    warranty   → warranty terms, coverage, transfer, claim, schedules
    safety     → fire, leak, brake fail, smoke (act-now situations)
    wiring     → wiring diagram, ECU pin, connector
    diagnostic → symptom-first troubleshooting: "why is X happening",
                 "what causes Y", noise / vibration / rattle / knock /
                 won't start / hard to start / rough idle / stalling /
                 misfire / loss of power / overheating / DTC / warning
                 light / fault / abnormal behaviour. Choose this — NOT
                 'safety' — when the user is asking about a symptom and
                 wants the cause + how to confirm it.
    general    → anything else / brand info
- "confident": true if the question alone makes the model+intent
  unambiguous; false if you had to guess.
"""


_CLASSIFIER_SYSTEM = _build_classifier_system_prompt()


def _class_cache_key(text: str) -> str:
    return " ".join(text.lower().split())


def _normalise_alias(s: str) -> str:
    """Lowercase + strip spaces/underscores/dashes — same form used in
    the alias tables."""
    return s.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _norm_model(raw) -> str:
    if not isinstance(raw, str):
        return "unknown"
    normed = _normalise_alias(raw)
    v = _ALIAS_TO_VEHICLE.get(normed)
    return v.id if v else "unknown"


def _norm_variant(raw) -> str:
    if not isinstance(raw, str):
        return "unknown"
    normed = _normalise_alias(raw)
    if normed in {"any", "all"}:
        return "any"
    vt = _ALIAS_TO_VARIANT.get(normed)
    return vt.id if vt else "unknown"


def _norm_intent(raw) -> str:
    if isinstance(raw, str) and raw.strip().lower() in _VALID_INTENTS:
        return raw.strip().lower()
    return "general"


def _regex_classify(question: str) -> dict:
    """Cheap, deterministic classifier — no API call.

    Returns the same shape as classify_query. Used as a fallback when the
    LLM classifier fails or returns low-confidence defaults for a query
    whose surface form is actually unambiguous.
    """
    intent = "general"
    for profile in INTENT_PROFILES:
        if profile.regex and profile.regex.search(question):
            intent = profile.label
            break

    model_hit = "unknown"
    for v in VEHICLES:
        if v.pattern.search(question):
            model_hit = v.id
            break

    # List is ordered longest-first so "Euro-5+" beats "Euro-5".
    variant = "unknown"
    for vt in VARIANTS:
        if vt.pattern.search(question):
            variant = vt.id
            break

    confident = (intent != "general") or (model_hit != "unknown")
    return {
        "model": model_hit,
        "variant": variant,
        "intent": intent,
        "confident": confident,
        "_source": "regex",
    }


def _merge_classifications(llm: dict, regex: dict) -> dict:
    """Fill any 'unknown' / 'general' field from the LLM with a
    definite hit from regex. Confidence is the OR of both signals."""
    out = dict(llm)
    if out.get("model") == "unknown" and regex.get("model") != "unknown":
        out["model"] = regex["model"]
    if (
        out.get("variant") in (None, "unknown")
        and regex.get("variant") not in (None, "unknown")
    ):
        out["variant"] = regex["variant"]
    if out.get("intent") == "general" and regex.get("intent") != "general":
        out["intent"] = regex["intent"]
    out["confident"] = bool(out.get("confident") or regex.get("confident"))
    return out


def regex_classify(question: str) -> dict:
    """Public wrapper for the deterministic regex classifier. Used by the
    request handler as a synchronous fallback when the LLM classifier
    times out — keeps the chat path off the all-unknown path that drops
    every signal."""
    out = _regex_classify(question)
    out.pop("_source", None)
    return out


def classify_query(client: OpenAI, question: str,
                   model: str = CLASSIFIER_MODEL) -> dict:
    """Return {model, variant, intent, confident, _fallback, ...diagnostics}.

    Cached by normalised question text. Two-tier:
      1. LLM classifier (gpt-5-mini, JSON output) — best quality.
      2. Regex fallback — runs ALWAYS; fills any 'unknown' / 'general'
         the LLM left behind, and is the sole signal if the LLM call
         errors. Without this, a transient API failure silently drops
         specific questions into the 'general' bucket.
    """
    key = _class_cache_key(question)
    cached = _CLASS_CACHE.get(key)
    if cached is not None:
        _CLASS_CACHE.move_to_end(key)
        return dict(cached)

    regex_class = _regex_classify(question)
    regex_class.pop("_source", None)

    llm_class: dict | None = None
    llm_error: str | None = None
    llm_raw: str | None = None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": question.strip()[:1000]},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="minimal",
            verbosity="low",
        )
        llm_raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(llm_raw)
        llm_class = {
            "model": _norm_model(data.get("model")),
            "variant": _norm_variant(data.get("variant")),
            "intent": _norm_intent(data.get("intent")),
            "confident": bool(data.get("confident", False)),
        }
    except Exception as e:  # noqa: BLE001
        llm_error = str(e)
        log.warning("classifier LLM call failed, using regex only: %s", e)

    if llm_class is None:
        out = dict(regex_class)
        out["_fallback"] = "regex_only"
        out["_llm_error"] = llm_error
    else:
        out = _merge_classifications(llm_class, regex_class)
        out["_fallback"] = "llm+regex"
    out["_llm_raw"] = llm_raw
    out["_llm_class"] = dict(llm_class) if llm_class else None
    out["_regex_class"] = dict(regex_class)

    _CLASS_CACHE[key] = out
    _CLASS_CACHE.move_to_end(key)
    if len(_CLASS_CACHE) > _CLASS_CACHE_MAX:
        _CLASS_CACHE.popitem(last=False)
    return dict(out)


def classify_query_async(executor: ThreadPoolExecutor, client: OpenAI,
                         question: str) -> Future:
    """Submit classify_query to a thread pool so it runs in parallel
    with embedding/Pinecone queries. Caller `.result()`s when needed."""
    return executor.submit(classify_query, client, question)


def classifier_to_filter_vehicle(classification: dict) -> Optional[str]:
    """Map classifier output to the Pinecone vehicle_model field, or
    None if we shouldn't apply a hard filter (low confidence / unknown)."""
    if not classification.get("confident"):
        return None
    vid = classification.get("model")
    for v in VEHICLES:
        if v.id == vid:
            return v.pinecone_filter
    return None


def intent_doc_types(classification: dict) -> list[str]:
    return list(_profile_for(classification.get("intent")).preferred_doc_types)


def intent_top_k(classification: dict, base_top_k: int) -> int:
    """Adjust top_k by intent. Don't shrink below the caller's choice
    for already-wide queries (compare/long → base_top_k=10+)."""
    target = _profile_for(classification.get("intent")).top_k
    return max(target, base_top_k) if base_top_k >= 10 else target


def intent_fetch_k(classification: dict, base_fetch_k: int) -> int:
    """Per-intent over-fetch before rerank. Never shrinks below the
    caller's base."""
    target = _profile_for(classification.get("intent")).fetch_k
    return max(target, base_fetch_k)


def intent_min_keep(classification: dict) -> int:
    return _profile_for(classification.get("intent")).min_keep


def intent_long_form(classification: dict) -> bool:
    """True for intents that need verbatim, full-section reproduction
    (procedure, safety, wiring, warranty). Generation layer reads this
    to bump verbosity and prepend a structure directive."""
    return _profile_for(classification.get("intent")).long_form


def intent_llm_settings(classification: dict) -> dict:
    """Generation knobs (verbosity, reasoning_effort) for this intent.
    Returned as a dict for unpacking into the OpenAI client call."""
    p = _profile_for(classification.get("intent"))
    return {
        "verbosity": p.llm_verbosity,
        "reasoning_effort": p.llm_reasoning_effort,
    }


def _classify_topics(text: str) -> set[str]:
    if not text:
        return set()
    low = text.lower()
    hits: set[str] = set()
    for cluster, keywords in TOPIC_CLUSTERS.items():
        for kw in keywords:
            if kw in low:
                hits.add(cluster)
                break
    return hits


def filter_by_section_relevance(matches: list[dict], question: str,
                                 classification: dict) -> list[dict]:
    """Soft filter: drop chunks whose subsystem differs from the question's.

    Topic detection considers BOTH the chunk's section label AND its body
    text. Many BSA chunks carry an umbrella section name (e.g. "Technical
    Specifications") that says nothing about the actual subsystem. Falling
    back to body-text keywords lets us surface the real topic instead of
    dropping the chunk because its label is generic.

    Preserves at least `intent_min_keep` chunks.
    """
    if not matches:
        return matches
    q_topics = _classify_topics(question)
    if not q_topics:
        return matches  # can't identify question topic; don't filter

    min_keep = intent_min_keep(classification)
    survivors: list[dict] = []
    rejects: list[dict] = []
    for m in matches:
        md = (m.get("metadata") or {})
        section = md.get("section") or ""
        body = md.get("text") or ""
        # Body scan capped to keep it cheap; the first few hundred chars are
        # enough to identify the chunk's dominant subsystem.
        chunk_topics = _classify_topics(section) | _classify_topics(body[:600])
        if chunk_topics and not (chunk_topics & q_topics):
            rejects.append(m)
        else:
            survivors.append(m)

    if len(survivors) >= min_keep:
        return survivors
    return (survivors + rejects)[:max(min_keep, len(survivors))]


def detect_dominant_section(matches: list[dict],
                              top_n: int = 3,
                              min_share: float = 0.5,
                              dominance_ratio: float = 2.0,
                              question: str | None = None) -> str | None:
    """Return the section name to expand on, or None if no signal.

    Two ways a section can be 'dominant':
      (A) Majority — `min_share` (default 50%) of the top `top_n` chunks
          share the same section.
      (B) Score gap — the TOP chunk's score is `dominance_ratio`× higher
          than the median of the remaining matches. Without this branch,
          single-strong-match queries miss adjacent continuation content.

    When `question` is supplied AND its topic clusters are identifiable,
    the score-gap branch additionally requires that the candidate section
    shares a topic cluster with the question — guards against locking on
    a section that happens to score high but isn't about the user's subject.
    """
    if not matches:
        return None

    q_topics = _classify_topics(question or "")

    # (C) Question-topic-aligned section (highest priority):
    # Prefer the section whose name BEST aligns with the question topic —
    # measured by how many topic-cluster keywords appear in the section name.
    # Without this guard, the lock can fire on a high-scoring chunk whose
    # section is in the right subsystem but whose leaf name is about an
    # unrelated symptom (e.g. "Radiator > Symptom 10: Misfiring" beating
    # "Radiator > Symptom 1: Overheating" on an overheat question).
    if q_topics:
        topic_keywords: set[str] = set()
        for cluster in q_topics:
            for kw in TOPIC_CLUSTERS.get(cluster, ()):
                topic_keywords.add(kw)
        # Also add the question's own non-trivial words so that "overheats"
        # matches a section literally named "Overheating".
        for w in (question or "").lower().split():
            w = w.strip(".,?!:;()'\"")
            if len(w) >= 4:
                topic_keywords.add(w)

        best_section: str | None = None
        best_overlap = 0
        best_score = -1.0
        for m in matches:
            section = ((m.get("metadata") or {}).get("section") or "").strip()
            if not section:
                continue
            section_low = section.lower()
            overlap = sum(1 for kw in topic_keywords if kw in section_low)
            if overlap == 0:
                continue
            score = float(m.get("score") or 0.0)
            if overlap > best_overlap or (
                overlap == best_overlap and score > best_score
            ):
                best_section = section
                best_overlap = overlap
                best_score = score
        if best_section is not None:
            return best_section

    # (B) Score-gap dominance
    if len(matches) >= 2:
        top_score = float(matches[0].get("score") or 0.0)
        rest_scores = sorted(
            (float(m.get("score") or 0.0) for m in matches[1:]),
            reverse=True,
        )
        # Use the median of the rest, not the second-best; one accidental
        # near-duplicate shouldn't suppress this signal.
        if rest_scores:
            mid = rest_scores[len(rest_scores) // 2]
            if mid > 0 and top_score >= mid * dominance_ratio:
                top_md = matches[0].get("metadata") or {}
                top_section = (top_md.get("section") or "").strip()
                if top_section:
                    if not q_topics:
                        return top_section
                    section_topics = (
                        _classify_topics(top_section)
                        | _classify_topics((top_md.get("text") or "")[:600])
                    )
                    if section_topics & q_topics:
                        return top_section

    # (A) Majority dominance
    window = matches[:top_n]
    sections = [
        ((m.get("metadata") or {}).get("section") or "").strip()
        for m in window
    ]
    sections = [s for s in sections if s]
    if not sections:
        return None
    counts: dict[str, int] = {}
    for s in sections:
        counts[s] = counts.get(s, 0) + 1
    top_section, top_count = max(counts.items(), key=lambda kv: kv[1])
    if top_count / len(sections) >= min_share:
        return top_section
    return None


def page_neighbor_ids(top_match: dict, span: int = 2) -> list[str]:
    """Return Pinecone IDs for chunks within ±`span` pages of the top match.

    Generates candidate IDs across pages in [top_page - span, top_page + span],
    section indices 1..3, and chunk slots c1, c2, c3, t1, t2.
    Pinecone's fetch() ignores IDs that don't exist — over-generating is
    cheap and gets us all page-continuation chunks the rerank-by-score
    pipeline misses.
    """
    md = top_match.get("metadata") or {}
    page = md.get("page")
    full_id = top_match.get("id") or ""
    if not full_id or "::" not in full_id:
        return []
    source = full_id.split("::", 1)[0]
    try:
        page_num = int(float(page))
    except (TypeError, ValueError):
        return []
    if page_num <= 0:
        return []

    # Don't include the top match itself; merge_unique_matches dedupes
    # but skipping it keeps the candidate list lean.
    own_short = full_id.split("::", 1)[1]

    out: list[str] = []
    for p in range(max(1, page_num - span), page_num + span + 1):
        for s in (1, 2, 3):
            for chunk in ("c1", "c2", "c3", "t1", "t2"):
                short = f"p{p}_s{s}_{chunk}"
                if short == own_short:
                    continue
                out.append(f"{source}::{short}")
    return out


def merge_unique_matches(primary: list[dict],
                          secondary: list[dict]) -> list[dict]:
    """Append matches from `secondary` not already in `primary`,
    de-duped by Pinecone match id; preserves primary's order."""
    seen: set = set()
    out: list[dict] = []
    for m in primary + secondary:
        mid = m.get("id")
        if mid is not None and mid in seen:
            continue
        if mid is not None:
            seen.add(mid)
        out.append(m)
    return out


def filter_by_rerank_floor(matches: list[dict],
                            classification: dict) -> list[dict]:
    """Drop matches whose rerank score is below `floor * top_score`,
    BUT always preserve at least `intent_min_keep` matches.

    The floor suppresses noise; the minimum guarantees the LLM has enough
    context to reproduce a multi-section answer.
    """
    if not matches:
        return matches
    top = float(matches[0].get("score") or 0.0)
    if top <= 0.0:
        return matches
    profile = _profile_for(classification.get("intent"))
    cutoff = top * profile.rerank_floor
    kept = [m for m in matches if float(m.get("score") or 0.0) >= cutoff]
    if len(kept) >= profile.min_keep:
        return kept
    return matches[:max(profile.min_keep, len(kept))]


def diversify_by_doc_type(matches: list[dict], keep: int,
                          preferred_types: list[str] | None = None) -> list[dict]:
    """Re-order/trim matches so the kept top-N spans multiple doc_types
    where possible. Round-robin one pick per bucket; preserves rerank
    order within each bucket."""
    if not matches or keep >= len(matches):
        return matches[:keep] if matches else matches

    buckets: "OrderedDict[str, list[dict]]" = OrderedDict()
    for m in matches:
        dt = (m.get("metadata") or {}).get("doc_type") or "_other"
        buckets.setdefault(dt, []).append(m)

    ordered_keys: list[str] = []
    if preferred_types:
        for t in preferred_types:
            if t in buckets and t not in ordered_keys:
                ordered_keys.append(t)
    for k in buckets:
        if k not in ordered_keys:
            ordered_keys.append(k)

    out: list[dict] = []
    while len(out) < keep:
        progressed = False
        for k in ordered_keys:
            if buckets[k]:
                out.append(buckets[k].pop(0))
                progressed = True
                if len(out) >= keep:
                    break
        if not progressed:
            break

    return out


# Atomic factual tokens the LLM is most likely to hallucinate.
_ATOMIC_TOKENS: list[tuple[str, re.Pattern]] = [
    ("part_number", re.compile(r"\b[A-Z]\d{3,4}[A-Z]{2,4}\d{4,6}[A-Z0-9]*\b")),
    ("torque",      re.compile(r"\b\d+(?:\.\d+)?\s*(?:Nm|N·m|kgf[·.]?m|lb[-·.]?ft)\b", re.I)),
    ("page_ref",    re.compile(r"\bp(?:age|p)?\.?\s*(\d{1,4}(?:\.\d+)?)\b", re.I)),
    ("version",     re.compile(r"\b(?:REV[-\s]?\d+|v\d+\.\d+)\b", re.I)),
]


def _normalise_for_match(text: str) -> str:
    return re.sub(r"[\s.\-_/]+", "", text.lower())


def verify_grounding(answer: str, ctx_text: str,
                     ctx_pages: list | None = None) -> dict:
    """Tier-1 regex grounding check. `ok=True` iff every atomic claim
    in `answer` appears in `ctx_text` (or, for page references, in
    `ctx_pages`). Returns `{ok, unsupported}`."""
    norm_ctx = _normalise_for_match(ctx_text)

    page_atoms: set[str] = set()
    for p in ctx_pages or []:
        if p is None:
            continue
        s = str(p).strip()
        if not s:
            continue
        page_atoms.add(_normalise_for_match(s))
        try:
            page_atoms.add(_normalise_for_match(str(int(float(s)))))
        except (TypeError, ValueError):
            pass

    # Skip the Sources: line — we don't want to flag PDF names there.
    answer_body = re.split(r"\n\s*Sources?:", answer, maxsplit=1)[0]

    unsupported: list[dict] = []
    for kind, rx in _ATOMIC_TOKENS:
        for m in rx.finditer(answer_body):
            val = m.group(0)
            if _normalise_for_match(val) in norm_ctx:
                continue
            if kind == "page_ref":
                num_match = re.search(r"\d+(?:\.\d+)?", val)
                if num_match:
                    norm_num = _normalise_for_match(num_match.group(0))
                    if norm_num in page_atoms or norm_num in norm_ctx:
                        continue
            unsupported.append({"kind": kind, "value": val})

    return {"ok": not unsupported, "unsupported": unsupported}


_executor: Optional[ThreadPoolExecutor] = None


def get_executor() -> ThreadPoolExecutor:
    """Lazily-built thread pool used to run the classifier alongside
    embedding/Pinecone queries. Bounded at 4 workers — never need more
    than a handful of concurrent classifier calls per uvicorn worker."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=4,
                                        thread_name_prefix="accuracy")
    return _executor


def detect_vehicle_in(text: str) -> Optional[str]:
    """Return the Pinecone vehicle_model filter value if `text` names a
    known vehicle, else None. Centralised here so chatbot.py / api.py
    don't duplicate the patterns."""
    for v in VEHICLES:
        if v.pattern.search(text):
            return v.pinecone_filter
    return None
