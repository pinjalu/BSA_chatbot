from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv(Path(__file__).with_name(".env"))

HERE          = Path(__file__).resolve().parent
DATA_ROOT     = Path(os.getenv("PIPELINE_DATA_ROOT", "Data/uploads"))
JOBS_FILE     = HERE / "pipeline_jobs.json"
MAX_JOBS      = 200
MAX_LOG_LINES = 500
MAX_FILE_MB   = 200
SSE_POLL_S    = 0.4
SSE_TIMEOUT_S = 3600

_SKIP_DEFAULT = [s.strip() for s in
                 os.getenv("PIPELINE_SKIP", "").split(",") if s.strip()]


@dataclass
class JobRecord:
    job_id:      str
    pdf_name:    str
    pdf_path:    str
    status:      str  = "queued"  # queued | running | done | error | cancelled
    created_at:  str  = ""
    started_at:  str  = ""
    finished_at: str  = ""
    error:       str  = ""
    skip_steps:  list = field(default_factory=list)
    force:       bool = False
    stages:      dict = field(default_factory=dict)
    events:      list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("events", None)
        return d

    def to_dict_full(self) -> dict:
        d = asdict(self)
        d["events"] = d["events"][-MAX_LOG_LINES:]
        return d


_jobs:    dict[str, JobRecord]        = {}
_threads: dict[str, threading.Thread] = {}
_engines: dict[str, Any]              = {}
_lock     = threading.Lock()


def _save_jobs() -> None:
    try:
        with _lock:
            records = [j.to_dict_full() for j in list(_jobs.values())[-MAX_JOBS:]]
        JOBS_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_jobs() -> None:
    if not JOBS_FILE.exists():
        return
    try:
        for d in json.loads(JOBS_FILE.read_text(encoding="utf-8")):
            valid = {k: v for k, v in d.items()
                     if k in JobRecord.__dataclass_fields__}
            jr = JobRecord(**valid)
            if jr.status in ("queued", "running"):
                jr.status = "error"
                jr.error  = "Server restarted while job was active"
            _jobs[jr.job_id] = jr
    except Exception:
        pass


def _run_job(job_id: str) -> None:
    from pipeline_engine import PipelineEngine

    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return

    job.status     = "running"
    job.started_at = datetime.now(timezone.utc).isoformat()
    _append_event(job_id, "job_start", {"message": f"Starting pipeline for {job.pdf_name}"})
    _save_jobs()

    def on_event(evt: dict) -> None:
        stage = evt.get("stage", "")
        event = evt.get("event", "")
        with _lock:
            j = _jobs.get(job_id)
            if j is None:
                return
            ss = evt.get("stage_state")
            if ss and stage:
                j.stages[stage] = ss

        label   = evt.get("stage_state", {}).get("label", stage) if evt.get("stage_state") else stage
        elapsed = evt.get("stage_state", {}).get("elapsed", 0) if evt.get("stage_state") else 0
        msg_map = {
            "start":     f"[{label}] Starting …",
            "done":      f"[{label}] Done  ({elapsed:.1f}s)",
            "error":     f"[{label}] ERROR: {evt.get('error', '')}",
            "skipped":   f"[{label}] Skipped — {evt.get('reason', '')}",
            "cancelled": f"[{label}] Cancelled",
            "log":       evt.get("message", ""),
        }
        msg = msg_map.get(event, "")
        if msg:
            _append_event(job_id, event, {
                "stage":       stage,
                "message":     msg,
                "stage_state": evt.get("stage_state"),
            })

    skip   = job.skip_steps if job.skip_steps else _SKIP_DEFAULT
    engine = PipelineEngine(
        job.pdf_path,
        job_id=job_id,
        progress_callback=on_event,
        skip_steps=skip,
        force=job.force,
    )
    with _lock:
        _engines[job_id] = engine

    try:
        result = engine.run()
        with _lock:
            j             = _jobs[job_id]
            j.status      = result.get("status", "done")
            j.finished_at = datetime.now(timezone.utc).isoformat()
            j.stages      = result.get("stages", {})
        _append_event(job_id, "job_done", {
            "message": (f"Pipeline {j.status.upper()} — "
                        f"{result.get('total_elapsed', 0):.1f}s total"),
            "status": j.status,
        })
    except Exception as exc:
        with _lock:
            j             = _jobs[job_id]
            j.status      = "error"
            j.error       = str(exc)
            j.finished_at = datetime.now(timezone.utc).isoformat()
        _append_event(job_id, "job_error", {"message": f"Unhandled error: {exc}"})
    finally:
        with _lock:
            _engines.pop(job_id, None)
        _save_jobs()


def _append_event(job_id: str, etype: str, data: dict[str, Any]) -> None:
    with _lock:
        j = _jobs.get(job_id)
        if j is None:
            return
        evt = {"type": etype,
               "timestamp": datetime.now(timezone.utc).isoformat(),
               **data}
        j.events.append(evt)
        if len(j.events) > MAX_LOG_LINES:
            j.events = j.events[-MAX_LOG_LINES:]


app = FastAPI(
    title="BSA PDF Ingestion Pipeline",
    description="Upload a PDF — watch it flow through 8 processing stages.",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _load_jobs()


@app.get("/health")
async def health():
    running = sum(1 for j in _jobs.values() if j.status == "running")
    return {"status": "ok", "total_jobs": len(_jobs), "running": running}


@app.post("/pipeline/upload")
async def upload_pdf(
    file:  UploadFile = File(...),
    skip:  str        = Form(""),
    force: bool       = Form(False),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File too large (max {MAX_FILE_MB} MB)")

    dest = DATA_ROOT / Path(file.filename).name
    if dest.exists():
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = DATA_ROOT / f"{Path(file.filename).stem}_{ts}.pdf"

    dest.write_bytes(content)

    job_id    = str(uuid.uuid4())
    skip_list = [s.strip() for s in skip.split(",") if s.strip()]
    job       = JobRecord(
        job_id     = job_id,
        pdf_name   = file.filename,
        pdf_path   = str(dest.resolve()),
        status     = "queued",
        created_at = datetime.now(timezone.utc).isoformat(),
        skip_steps = skip_list,
        force      = force,
    )
    with _lock:
        _jobs[job_id] = job

    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    _threads[job_id] = t
    t.start()

    _save_jobs()
    return {"job_id": job_id, "pdf_name": file.filename, "status": "queued"}


@app.get("/pipeline/jobs")
async def list_jobs(limit: int = 50):
    with _lock:
        jobs = list(_jobs.values())
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [j.to_dict() for j in jobs[:limit]]


@app.get("/pipeline/jobs/{job_id}")
async def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict_full()


@app.get("/pipeline/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    with _lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

    async def generate():
        cursor   = 0
        deadline = time.time() + SSE_TIMEOUT_S
        while time.time() < deadline:
            with _lock:
                j      = _jobs.get(job_id)
                events = (j.events[cursor:] if j else [])
                status = j.status if j else "error"
            for evt in events:
                yield f"data: {json.dumps(evt)}\n\n"
                cursor += 1
            if status in ("done", "error", "cancelled"):
                end_evt = json.dumps({"type": "stream_end", "status": status})
                yield f"data: {end_evt}\n\n"
                return
            await asyncio.sleep(SSE_POLL_S)
        yield f"data: {json.dumps({'type': 'stream_timeout'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.post("/pipeline/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    with _lock:
        job    = _jobs.get(job_id)
        engine = _engines.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("queued", "running"):
        return {"job_id": job_id, "status": job.status, "message": "Nothing to cancel"}

    if engine:
        engine.stop()  # sets stop event + terminates active subprocess

    with _lock:
        _jobs[job_id].status = "cancelled"
    _append_event(job_id, "job_cancelled", {"message": "Cancelled by user"})
    _save_jobs()
    return {"job_id": job_id, "status": "cancelled"}


@app.post("/pipeline/jobs/{job_id}/retry")
async def retry_job(job_id: str, skip: str = Form(""), force: bool = Form(False)):
    with _lock:
        old = _jobs.get(job_id)
    if old is None:
        raise HTTPException(status_code=404, detail="Job not found")

    new_id    = str(uuid.uuid4())
    skip_list = [s.strip() for s in skip.split(",") if s.strip()]
    job = JobRecord(
        job_id     = new_id,
        pdf_name   = old.pdf_name,
        pdf_path   = old.pdf_path,
        status     = "queued",
        created_at = datetime.now(timezone.utc).isoformat(),
        skip_steps = skip_list,
        force      = force,
    )
    with _lock:
        _jobs[new_id] = job
    t = threading.Thread(target=_run_job, args=(new_id,), daemon=True)
    _threads[new_id] = t
    t.start()
    _save_jobs()
    return {"job_id": new_id, "pdf_name": old.pdf_name, "status": "queued"}


@app.delete("/pipeline/jobs/{job_id}")
async def delete_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in ("queued", "running"):
        raise HTTPException(status_code=400,
                            detail="Cancel the job before deleting it")
    with _lock:
        _jobs.pop(job_id, None)
    _save_jobs()
    return {"job_id": job_id, "deleted": True}
