"""FastAPI front layer — replaces server/server.js.

Serves:
  - Chat widget SSE proxy  →  Python RAG core (api.py, port 8000)
  - Pipeline SSE proxy     →  Python pipeline (pipeline_api.py, port 8001)
  - PostgreSQL persistence →  sessions, messages, handoffs, learned Q&A
  - React admin SPA        →  server/public/app/
  - Static assets          →  server/public/

Run:
  uvicorn main:app --host 127.0.0.1 --port 3000

Env (server/.env):
  PORT, PYTHON_API_URL, PIPELINE_API_URL, ADMIN_PASSWORD,
  DATABASE_URL, DATABASE_SSL, CORS_ORIGINS, HISTORY_TURNS,
  CONFIDENCE_THRESHOLD
"""
from __future__ import annotations

import logging
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# database.py lives in the same directory
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import database as db

load_dotenv(HERE / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bsa-server")

# ── config ────────────────────────────────────────────

PORT               = int(os.environ.get("PORT", 3000))
PYTHON_API_URL     = os.environ.get("PYTHON_API_URL",    "http://127.0.0.1:8000")
PIPELINE_API_URL   = os.environ.get("PIPELINE_API_URL",  "http://127.0.0.1:8001")
ADMIN_PASSWORD     = os.environ.get("ADMIN_PASSWORD", "")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
HISTORY_TURNS      = int(os.environ.get("HISTORY_TURNS", "5"))
DATABASE_URL       = os.environ.get("DATABASE_URL", "")
DATABASE_SSL       = os.environ.get("DATABASE_SSL", "").lower() == "true"
CORS_ORIGINS_RAW   = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS       = [s.strip() for s in CORS_ORIGINS_RAW.split(",") if s.strip()]

PROJECT_ROOT = HERE.parent
PUBLIC_DIR   = HERE / "public"

if not ADMIN_PASSWORD:
    log.warning("[warn] ADMIN_PASSWORD is empty — admin panel is OPEN")

NO_ANSWER_RX = re.compile(
    "|".join([
        "couldn'?t find",
        "could not find",
        "don'?t have (?:that|this|the answer|enough info)",
        "do not have (?:that|this|the answer)",
        "passages? (?:don'?t|do not) cover",
        "not (?:in|covered in) (?:the )?(?:manuals?|docs?|documentation|context|excerpts?)",
        "no (?:relevant )?information",
        "i'?m not finding",
        "i can'?t find",
        "outside (?:the )?scope",
    ]),
    re.IGNORECASE,
)

ALLOWED_IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# ── app ───────────────────────────────────────────────

app = FastAPI(title="BSA chat server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in CORS_ORIGINS else CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)


# ── startup / shutdown ────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL must be set in server/.env")
    await db.init(DATABASE_URL, ssl=DATABASE_SSL)
    log.info(f"BSA chat server ready on port {PORT}")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if db._pool:
        await db._pool.close()


# ── auth dependency ───────────────────────────────────

def require_admin(authorization: str = Header(default="")) -> None:
    if not ADMIN_PASSWORD or authorization != f"Bearer {ADMIN_PASSWORD}":
        raise HTTPException(401, "unauthorized")


# ── SSE helpers ───────────────────────────────────────

def _sse(event: str, data: object) -> str:
    import json
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_sse(raw: str) -> Optional[dict]:
    import json
    event = "message"
    data_lines: list[str] = []
    for line in raw.split("\n"):
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return None
    combined = "\n".join(data_lines)
    try:
        return {"event": event, "data": json.loads(combined)}
    except (ValueError, Exception):
        return {"event": event, "data": combined}


def _remap_images(raw_images: dict) -> dict:
    """Convert image paths from the RAG core to widget-loadable URLs."""
    images: dict[str, str] = {}
    for img_id, p in raw_images.items():
        if not isinstance(p, str) or not p:
            continue
        if re.match(r"^https?://", p, re.IGNORECASE):
            images[img_id] = p
        else:
            # Encode each path segment separately (same as Express encodeURIComponent).
            parts = re.split(r"[/\\]", p)
            encoded = "/".join(quote(seg, safe="") for seg in parts if seg)
            images[img_id] = f"/api/images/{encoded}"
    return images


# ── Pydantic models ───────────────────────────────────

class ChatBody(BaseModel):
    sessionId: Optional[str] = None
    dealerId:  Optional[str] = None
    message:   str
    vehicle:   Optional[str] = None
    docType:   Optional[str] = None
    imageB64:  Optional[str] = None


class HandoffBody(BaseModel):
    sessionId: str
    name:   Optional[str] = None
    email:  Optional[str] = None
    reason: Optional[str] = None


class AdminLoginBody(BaseModel):
    password: str


class AdminReplyBody(BaseModel):
    sessionId: str
    message:   str
    resolve:   bool = False


class AdminResolveBody(BaseModel):
    sessionId: str


class RetryBody(BaseModel):
    skip:  Optional[str] = None
    force: Optional[str] = None


class ContentBody(BaseModel):
    question:  str
    answer:    str
    dealer_id: Optional[str] = None


class ContentUpdateBody(BaseModel):
    question: str
    answer:   str


class KBSearchBody(BaseModel):
    query:   str
    vehicle: Optional[str] = None


class KBHandoffBody(BaseModel):
    name:     Optional[str] = None
    email:    str
    question: Optional[str] = None


# ── static pages ──────────────────────────────────────

@app.get("/")
async def root() -> Response:
    return Response(
        "BSA chat server running.  Admin: /admin   Widget: /widget.js",
        media_type="text/plain",
    )


@app.get("/widget.js")
async def widget_js() -> FileResponse:
    return FileResponse(
        PUBLIC_DIR / "widget.js",
        media_type="application/javascript",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma":        "no-cache",
            "Expires":       "0",
        },
    )


@app.get("/admin")
async def admin_page()          -> FileResponse: return FileResponse(PUBLIC_DIR / "admin.html")

@app.get("/admin/analytics")
async def admin_analytics_html() -> FileResponse: return FileResponse(PUBLIC_DIR / "analytics.html")

@app.get("/admin/content")
async def admin_content_html()   -> FileResponse: return FileResponse(PUBLIC_DIR / "content.html")

@app.get("/admin/pipeline")
async def admin_pipeline_html()  -> FileResponse: return FileResponse(PUBLIC_DIR / "pipeline.html")

@app.get("/kb")
async def kb_page()              -> FileResponse: return FileResponse(PUBLIC_DIR / "kb.html")


# ── React SPA ─────────────────────────────────────────
# Serve the Vite build at /app/*. Real assets (JS/CSS) are served directly;
# client-side navigation routes fall back to index.html.

@app.get("/app")
async def react_app_root() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "app" / "index.html")


@app.get("/app/{path:path}")
async def react_app(path: str) -> FileResponse:
    candidate = PUBLIC_DIR / "app" / path
    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(PUBLIC_DIR / "app" / "index.html")


# ── image serving ─────────────────────────────────────

@app.get("/api/images/{path:path}")
async def serve_image(path: str) -> FileResponse:
    ext = Path(path).suffix.lower()
    if not path or ext not in ALLOWED_IMG_EXT:
        raise HTTPException(400, "bad path")
    target = (PROJECT_ROOT / path).resolve()
    # Prevent path traversal
    if not str(target).startswith(str(PROJECT_ROOT) + os.sep) and target != PROJECT_ROOT:
        raise HTTPException(403, "out of bounds")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "not found")
    mime = mimetypes.types_map.get(ext, "application/octet-stream")
    return FileResponse(target, media_type=mime, headers={"Cache-Control": "public, max-age=86400"})


# ── chat (streaming SSE) ──────────────────────────────

@app.post("/api/chat")
async def api_chat(body: ChatBody) -> StreamingResponse:
    if not body.message or not body.message.strip():
        raise HTTPException(400, "message is required")

    session_id = await db.get_or_create_session(body.sessionId, body.dealerId)
    await db.add_message(session_id, "user", body.message)
    history = await db.get_history_pairs(session_id, HISTORY_TURNS)
    history = [p for p in history if p["question"] != body.message or p["answer"]]

    async def generate() -> AsyncGenerator[str, None]:
        yield _sse("session", {"sessionId": session_id})

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{PYTHON_API_URL}/chat",
                    json={
                        "message":   body.message,
                        "history":   history,
                        "vehicle":   body.vehicle  or None,
                        "doc_type":  body.docType  or None,
                        "image_b64": body.imageB64 or None,
                    },
                ) as resp:
                    if resp.status_code >= 400:
                        err = await resp.aread()
                        yield _sse("error", {"message": f"RAG core {resp.status_code}: {err.decode()}"})
                        return

                    assistant_text = ""
                    top_score      = 0.0
                    citations      = None
                    timings        = None
                    learned_meta   = None
                    buf            = ""

                    async for chunk in resp.aiter_text():
                        buf += chunk
                        while "\n\n" in buf:
                            raw, buf = buf.split("\n\n", 1)
                            ev = _parse_sse(raw)
                            if not ev:
                                continue
                            event, data = ev["event"], ev["data"]

                            if event == "token":
                                assistant_text += data.get("text", "")
                                yield _sse("token", {"text": data.get("text", "")})

                            elif event == "meta":
                                top_score  = data.get("top_score", 0)
                                citations  = data.get("matches", [])
                                images     = _remap_images(data.get("images", {}))
                                if data.get("learned"):
                                    learned_meta = data["learned"]
                                yield _sse("meta", {
                                    "topScore":  top_score,
                                    "citations": citations,
                                    "images":    images,
                                    "learned":   data.get("learned") or None,
                                })

                            elif event == "images":
                                yield _sse("images", {"images": _remap_images(data.get("images", {}))})

                            elif event == "status":
                                yield _sse("status", data)

                            elif event == "error":
                                yield _sse("error", data)

                            elif event == "done":
                                if isinstance(data, dict):
                                    if data.get("timings_ms"):
                                        timings = data["timings_ms"]
                                    if data.get("learned"):
                                        learned_meta = learned_meta or {"match_score": top_score}

                    # Persist assistant turn after stream completes
                    if assistant_text:
                        await db.add_message(
                            session_id, "assistant", assistant_text,
                            citations=citations, top_score=top_score,
                        )

                    low_conf    = top_score < CONFIDENCE_THRESHOLD
                    no_ans      = bool(NO_ANSWER_RX.search(assistant_text or ""))
                    already_open = await db.has_open_handoff(session_id)

                    if timings:
                        log.info(
                            "[chat] total=%sms embed=%sms pinecone=%sms ttft=%sms llm=%sms",
                            timings.get("total"), timings.get("embed"),
                            timings.get("pinecone"), timings.get("llm_ttft"),
                            timings.get("llm_total"),
                        )

                    yield _sse("done", {
                        "topScore":      top_score,
                        "suggestHandoff": (low_conf or no_ans) and not already_open,
                        "handoffOpen":   already_open,
                        "timings":       timings,
                        "learned":       learned_meta,
                    })

        except httpx.ConnectError as exc:
            yield _sse("error", {"message": f"RAG core unreachable: {exc}"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── handoff ───────────────────────────────────────────

@app.post("/api/handoff")
async def api_handoff(body: HandoffBody) -> dict:
    session = await db.get_or_create_session(body.sessionId)
    if body.name or body.email:
        await db.set_session_contact(session, body.name, body.email)
    handoff = await db.open_handoff(session, body.reason or "user_requested")
    parts = []
    if body.name:  parts.append(f"by {body.name}")
    if body.email: parts.append(f"({body.email})")
    msg = "Handoff requested" + (" " + " ".join(parts) if parts else "") + "."
    await db.add_message(session, "system", msg)
    return {"ok": True, "handoff": handoff}


# ── message polling ───────────────────────────────────

@app.get("/api/messages/{session_id}")
async def api_messages(session_id: str, after: int = Query(default=0)) -> dict:
    messages     = await db.get_messages_after(session_id, after)
    handoff_open = await db.has_open_handoff(session_id)
    return {"messages": messages, "handoffOpen": handoff_open}


# ── admin auth ────────────────────────────────────────

@app.post("/api/admin/login")
async def admin_login(body: AdminLoginBody) -> dict:
    if not ADMIN_PASSWORD or body.password != ADMIN_PASSWORD:
        raise HTTPException(401, "wrong password")
    return {"token": ADMIN_PASSWORD}


@app.get("/api/admin/handoffs")
async def admin_list_handoffs(_: None = Depends(require_admin)) -> dict:
    return {"handoffs": await db.list_open_handoffs()}


@app.get("/api/admin/session/{session_id}")
async def admin_session(session_id: str, _: None = Depends(require_admin)) -> dict:
    return {"sessionId": session_id, "messages": await db.get_messages(session_id)}


@app.post("/api/admin/reply")
async def admin_reply(body: AdminReplyBody, _: None = Depends(require_admin)) -> dict:
    if not body.sessionId or not body.message:
        raise HTTPException(400, "sessionId and message required")
    await db.add_message(body.sessionId, "admin", body.message)

    learned = None
    if body.resolve:
        await db.resolve_handoff(body.sessionId)
        user_q = await db.get_last_user_question(body.sessionId)
        if user_q:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(
                        f"{PYTHON_API_URL}/learn",
                        json={
                            "question":   user_q,
                            "answer":     body.message,
                            "session_id": body.sessionId,
                        },
                    )
                    if r.status_code < 400:
                        learned = r.json()
                        log.info(
                            "[learn] saved admin reply (session=%s, id=%s)",
                            body.sessionId, learned.get("id"),
                        )
                    else:
                        log.warning("[learn] failed: %s", r.status_code)
            except Exception as exc:
                log.warning("[learn] error reaching Python: %s", exc)

    return {"ok": True, "learned": learned}


@app.post("/api/admin/resolve")
async def admin_resolve(body: AdminResolveBody, _: None = Depends(require_admin)) -> dict:
    if not body.sessionId:
        raise HTTPException(400, "sessionId required")
    await db.resolve_handoff(body.sessionId)
    return {"ok": True}


# ── pipeline (PDF ingestion) ───────────────────────────

@app.post("/api/admin/pipeline/upload")
async def pipeline_upload(
    file:     UploadFile         = File(...),
    skip:     Optional[str]      = Form(default=None),
    force:    Optional[str]      = Form(default=None),
    category: Optional[str]      = Form(default=None),
    _:        None               = Depends(require_admin),
) -> Response:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")
    content = await file.read()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            files = {"file": (file.filename, content, "application/pdf")}
            data: dict[str, str] = {}
            if skip:     data["skip"]     = skip
            if force:    data["force"]    = force
            if category: data["category"] = category
            r = await client.post(
                f"{PIPELINE_API_URL}/pipeline/upload", files=files, data=data,
            )
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/json"),
            )
    except httpx.ConnectError:
        raise HTTPException(502, "pipeline service unreachable")


@app.get("/api/admin/pipeline/jobs")
async def pipeline_jobs(_: None = Depends(require_admin)) -> Response:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{PIPELINE_API_URL}/pipeline/jobs")
            return Response(content=r.content, status_code=r.status_code, media_type="application/json")
    except httpx.ConnectError:
        raise HTTPException(502, "pipeline service unreachable")


@app.get("/api/admin/pipeline/jobs/{job_id}")
async def pipeline_job(job_id: str, _: None = Depends(require_admin)) -> Response:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{PIPELINE_API_URL}/pipeline/jobs/{job_id}")
            return Response(content=r.content, status_code=r.status_code, media_type="application/json")
    except httpx.ConnectError:
        raise HTTPException(502, "pipeline service unreachable")


@app.delete("/api/admin/pipeline/jobs/{job_id}")
async def pipeline_job_delete(job_id: str, _: None = Depends(require_admin)) -> Response:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.delete(f"{PIPELINE_API_URL}/pipeline/jobs/{job_id}")
            return Response(content=r.content, status_code=r.status_code, media_type="application/json")
    except httpx.ConnectError:
        raise HTTPException(502, "pipeline service unreachable")


@app.post("/api/admin/pipeline/jobs/{job_id}/retry")
async def pipeline_job_retry(
    job_id: str,
    body:   RetryBody = Body(default_factory=RetryBody),
    _:      None      = Depends(require_admin),
) -> Response:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            data: dict[str, str] = {}
            if body.skip:  data["skip"]  = body.skip
            if body.force: data["force"] = body.force
            r = await client.post(
                f"{PIPELINE_API_URL}/pipeline/jobs/{job_id}/retry", data=data,
            )
            return Response(content=r.content, status_code=r.status_code, media_type="application/json")
    except httpx.ConnectError:
        raise HTTPException(502, "pipeline service unreachable")


@app.post("/api/admin/pipeline/jobs/{job_id}/cancel")
async def pipeline_job_cancel(job_id: str, _: None = Depends(require_admin)) -> Response:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{PIPELINE_API_URL}/pipeline/jobs/{job_id}/cancel")
            return Response(content=r.content, status_code=r.status_code, media_type="application/json")
    except httpx.ConnectError:
        raise HTTPException(502, "pipeline service unreachable")


# Pipeline SSE stream — EventSource can't send custom headers, so auth
# arrives as a query param instead of Bearer header.
@app.get("/api/admin/pipeline/jobs/{job_id}/stream")
async def pipeline_job_stream(
    job_id: str, token: str = Query(default=""),
) -> StreamingResponse:
    if not ADMIN_PASSWORD or token != ADMIN_PASSWORD:
        raise HTTPException(401)

    async def generate() -> AsyncGenerator[bytes, None]:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "GET", f"{PIPELINE_API_URL}/pipeline/jobs/{job_id}/stream",
                ) as resp:
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        yield _sse("error", {"status": resp.status_code, "body": body.decode()}).encode()
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except httpx.ConnectError as exc:
            yield _sse("error", {"message": f"pipeline service unreachable: {exc}"}).encode()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ── analytics ─────────────────────────────────────────

@app.get("/api/admin/analytics")
async def admin_analytics(
    days: int   = Query(default=7),
    _:    None  = Depends(require_admin),
) -> dict:
    return await db.get_analytics(days)


# ── content management ────────────────────────────────

@app.get("/api/admin/content")
async def admin_content_list(_: None = Depends(require_admin)) -> dict:
    return {"items": await db.list_learned_qa()}


@app.post("/api/admin/content")
async def admin_content_create(
    body: ContentBody, _: None = Depends(require_admin),
) -> dict:
    if not body.question or not body.answer:
        raise HTTPException(400, "question and answer required")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{PYTHON_API_URL}/learn",
                json={
                    "question":  body.question,
                    "answer":    body.answer,
                    "dealer_id": body.dealer_id or "",
                },
            )
            if r.status_code >= 400:
                raise HTTPException(502, "python error")
            data = r.json()
            await db.add_learned_qa(data["id"], body.question, body.answer, body.dealer_id)
            return {"ok": True, "id": data["id"]}
    except httpx.ConnectError:
        raise HTTPException(502, "python api unreachable")


@app.put("/api/admin/content/{item_id}")
async def admin_content_update(
    item_id: str, body: ContentUpdateBody, _: None = Depends(require_admin),
) -> dict:
    if not body.question or not body.answer:
        raise HTTPException(400, "question and answer required")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.put(
                f"{PYTHON_API_URL}/learned/{item_id}",
                json={"question": body.question, "answer": body.answer},
            )
            if r.status_code >= 400:
                raise HTTPException(502, "python error")
            await db.update_learned_qa(item_id, body.question, body.answer)
            return {"ok": True}
    except httpx.ConnectError:
        raise HTTPException(502, "python api unreachable")


@app.delete("/api/admin/content/{item_id}")
async def admin_content_delete(item_id: str, _: None = Depends(require_admin)) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.delete(f"{PYTHON_API_URL}/learned/{item_id}")
    except Exception:
        pass
    await db.delete_learned_qa(item_id)
    return {"ok": True}


# ── knowledge base ────────────────────────────────────

@app.post("/api/kb/search")
async def kb_search(body: KBSearchBody) -> StreamingResponse:
    if not body.query:
        raise HTTPException(400, "query required")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{PYTHON_API_URL}/chat",
                    json={"message": body.query, "history": [], "vehicle": body.vehicle or None},
                ) as resp:
                    if resp.status_code >= 400:
                        yield _sse("error", {"message": f"RAG core error {resp.status_code}"})
                        return
                    buf = ""
                    async for chunk in resp.aiter_text():
                        buf += chunk
                        while "\n\n" in buf:
                            raw, buf = buf.split("\n\n", 1)
                            ev = _parse_sse(raw)
                            if not ev:
                                continue
                            event, data = ev["event"], ev["data"]
                            if event == "token":
                                yield _sse("token", data)
                            elif event == "status":
                                yield _sse("status", data)
                            elif event in ("meta", "images"):
                                key = "matches" if event == "meta" else "images"
                                img_key = "images" if event == "meta" else "images"
                                images = _remap_images(data.get(img_key, {}))
                                if event == "meta":
                                    yield _sse("meta", {"citations": data.get("matches", []), "images": images})
                                else:
                                    yield _sse("images", {"images": images})
                            elif event == "done":
                                yield _sse("done", {"timings": (data or {}).get("timings_ms") or None})
                            elif event == "error":
                                yield _sse("error", data)
        except httpx.ConnectError as exc:
            yield _sse("error", {"message": f"RAG core unreachable: {exc}"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/kb/handoff")
async def kb_handoff(body: KBHandoffBody) -> dict:
    if not body.email:
        raise HTTPException(400, "email required")
    session_id = await db.create_session("kb")
    if body.name or body.email:
        await db.set_session_contact(session_id, body.name, body.email)
    if body.question:
        await db.add_message(session_id, "user", body.question)
    await db.open_handoff(session_id, "kb_search")
    name_part  = f" from {body.name}"  if body.name  else ""
    email_part = f" ({body.email})"    if body.email else ""
    await db.add_message(
        session_id, "system",
        f"KB handoff{name_part}{email_part}. Question: \"{body.question or '—'}\"",
    )
    return {"ok": True, "sessionId": session_id}


# ── remaining public static files ─────────────────────
# Catch-all for any file in server/public/ not matched above
# (chart.min.js, test.html, legacy HTML pages, etc.)
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=False), name="public")
