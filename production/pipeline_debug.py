from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_lock = threading.Lock()
LOG_DIR = Path(__file__).parent / "debug_logs"


def _log_path() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    return LOG_DIR / f"pipeline_{datetime.now():%Y-%m-%d}.txt"


def _write(lines: list[str]) -> None:
    path = _log_path()
    block = "\n".join(lines) + "\n"
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(block)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_pipeline_start(job_id: str, pdf_path: str) -> None:
    _write([
        "",
        "=" * 70,
        f"[PIPELINE START]  {_ts()}",
        f"  job_id : {job_id}",
        f"  pdf    : {pdf_path}",
        "=" * 70,
    ])


def log_pipeline_end(job_id: str, pdf_path: str, steps: dict) -> None:
    total = sum(s.get("elapsed", 0) for s in steps.values() if isinstance(s, dict))
    statuses = {name: s.get("status", "?") for name, s in steps.items()}
    _write([
        "",
        f"[PIPELINE END]  {_ts()}  job={job_id}  total={total:.1f}s",
        f"  pdf    : {pdf_path}",
        f"  steps  : {statuses}",
        "=" * 70,
    ])


def log_step_start(job_id: str, step_name: str, step_label: str,
                   input_path: str = "") -> None:
    _write([
        f"  [{_ts()}] START  {step_label}  (job={job_id})",
        f"    input : {input_path}" if input_path else "",
    ])


def log_step_end(job_id: str, step_name: str, step_label: str,
                 elapsed: float, output_path: str = "") -> None:
    _write([
        f"  [{_ts()}] DONE   {step_label}  ({elapsed:.1f}s)  (job={job_id})",
        f"    output: {output_path}" if output_path else "",
    ])


def log_step_skipped(job_id: str, step_name: str, step_label: str,
                     reason: str) -> None:
    _write([
        f"  [{_ts()}] SKIP   {step_label}  (job={job_id})",
        f"    reason: {reason}",
    ])


def log_step_error(job_id: str, step_name: str, step_label: str,
                   error: str, tb: str = "") -> None:
    lines = [
        f"  [{_ts()}] ERROR  {step_label}  (job={job_id})",
        f"    error : {error}",
    ]
    if tb:
        for tbline in tb.splitlines():
            lines.append(f"    {tbline}")
    _write(lines)


def log_event(job_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
    parts = [f"  [{_ts()}] {event_type}  (job={job_id})"]
    if data:
        for k, v in data.items():
            parts.append(f"    {k}: {v}")
    _write(parts)
