from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pipeline_debug as dbg

HERE = Path(__file__).resolve().parent

STAGE_ORDER = [
    "extract_text",
    "extract_images",
    "clean_text",
    "clean_images",
    "merge_chunks",
    "prepare_pinecone",
    "embed_upsert",
    "upload_s3",
]

STAGE_LABELS = {
    "extract_text":     "Extract Text",
    "extract_images":   "Extract Images",
    "clean_text":       "Clean & Section Text",
    "clean_images":     "Clean Images (GPT-4o mini)",
    "merge_chunks":     "Merge Text + Images",
    "prepare_pinecone": "Prepare Pinecone JSON",
    "embed_upsert":     "Embed + Upsert to Pinecone",
    "upload_s3":        "Upload Images to S3",
}

# False = failure is logged but pipeline continues
STAGE_CRITICAL = {
    "extract_text":     True,
    "extract_images":   False,
    "clean_text":       True,
    "clean_images":     False,
    "merge_chunks":     True,
    "prepare_pinecone": True,
    "embed_upsert":     True,
    "upload_s3":        False,
}


@dataclass
class StageState:
    name:        str
    label:       str
    status:      str   = "pending"  # pending | running | done | error | skipped
    start_time:  float = 0.0
    end_time:    float = 0.0
    output_path: str   = ""
    error:       str   = ""
    logs:        list  = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elapsed"] = self.elapsed
        return d


class PipelineEngine:

    def __init__(
        self,
        pdf_path: str | Path,
        *,
        job_id:            str = "",
        progress_callback: Callable[[dict], None] | None = None,
        force:             bool = False,
        skip_steps:        list[str] | None = None,
    ):
        self.pdf    = Path(pdf_path).resolve()
        self.job_id = job_id or f"job_{int(time.time())}"
        self.cb     = progress_callback
        self.force  = force
        self.skip   = set(skip_steps or [])

        self.stem    = self.pdf.stem
        self.pdf_dir = self.pdf.parent

        self.text_json      = self.pdf.with_suffix(".text.json")
        self.images_dir     = self.pdf_dir / "Data" / self.stem
        self.images_folder  = self.images_dir / "images"
        self.images_json    = self.images_dir / f"{self.stem}_images.json"
        self.clean_json     = self.pdf.with_name(self.stem + ".clean.json")
        self.final_json     = self.pdf.with_name(self.stem + "_final.json")
        self.pinecone_jsonl = self.pdf.with_name(self.stem + "_pinecone.jsonl")

        self.stages: dict[str, StageState] = {
            name: StageState(name=name, label=STAGE_LABELS[name])
            for name in STAGE_ORDER
        }
        self._stop = threading.Event()
        self._proc = None

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self) -> dict:
        dbg.log_pipeline_start(self.job_id, str(self.pdf))
        try:
            return self._run_all()
        finally:
            dbg.log_pipeline_end(
                self.job_id, str(self.pdf),
                {n: s.to_dict() for n, s in self.stages.items()},
            )

    def get_state(self) -> dict:
        return {
            "job_id": self.job_id,
            "pdf":    str(self.pdf),
            "stages": {n: s.to_dict() for n, s in self.stages.items()},
        }

    def _run_all(self) -> dict:
        if not self._should_skip("extract_text", self.text_json):
            ok = self._run("extract_text",
                           [sys.executable, str(HERE / "step1_extract_text.py"), str(self.pdf)],
                           output=self.text_json)
            if not ok:
                return self._summary()

        if not self._should_skip("extract_images", self.images_json):
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self._run("extract_images",
                      [sys.executable, str(HERE / "step2_extract_images.py"), str(self.pdf)],
                      output=self.images_dir, critical=False)

        if not self.text_json.exists():
            self._skip("clean_text", "No text JSON (step 1 failed)")
        elif not self._should_skip("clean_text", self.clean_json):
            ok = self._run("clean_text",
                           [sys.executable, str(HERE / "step3_clean_text.py"), str(self.text_json)],
                           output=self.clean_json)
            if not ok:
                return self._summary()

        cleaned_dir = self.images_folder / "cleaned"
        if "clean_images" in self.skip:
            self._skip("clean_images", "Skipped by caller")
        elif not self.images_folder.exists():
            self._skip("clean_images", "No images folder (step 2 skipped/failed)")
        elif cleaned_dir.exists() and any(cleaned_dir.iterdir()) and not self.force:
            self._skip("clean_images",
                       f"cleaned/ already exists ({sum(1 for _ in cleaned_dir.iterdir())} images)")
        else:
            self._run("clean_images",
                      [sys.executable, str(HERE / "step4_clean_images.py"), str(self.images_folder)],
                      output=cleaned_dir, critical=False)

        if not self.clean_json.exists():
            self._skip("merge_chunks", "No clean JSON (step 3 failed)")
        elif not self._should_skip("merge_chunks", self.final_json):
            if not self.images_json.exists():
                self.images_dir.mkdir(parents=True, exist_ok=True)
                self.images_json.write_text('{"pages": []}', encoding="utf-8")
                self._log("merge_chunks", "[info] Created empty images stub")
            ok = self._run("merge_chunks",
                           [sys.executable, str(HERE / "step5_merge_chunks.py"),
                            str(self.clean_json), str(self.images_json), str(self.final_json)],
                           output=self.final_json)
            if not ok:
                return self._summary()

        if not self.final_json.exists():
            self._skip("prepare_pinecone", "No final JSON (step 5 failed)")
        elif not self._should_skip("prepare_pinecone", self.pinecone_jsonl):
            ok = self._run("prepare_pinecone",
                           [sys.executable, str(HERE / "step6_prepare_pinecone.py"),
                            "--root", str(self.final_json.parent)],
                           output=self.pinecone_jsonl)
            if not ok:
                return self._summary()

        if "embed_upsert" in self.skip:
            self._skip("embed_upsert", "Skipped by caller")
        elif not self.pinecone_jsonl.exists():
            self._skip("embed_upsert", "No JSONL (step 6 failed)")
        else:
            self._run("embed_upsert",
                      [sys.executable, str(HERE / "step7_embed_upsert.py"),
                       "--files", str(self.pinecone_jsonl)])

        if "upload_s3" in self.skip:
            self._skip("upload_s3", "Skipped by caller")
        elif not self.images_dir.exists():
            self._skip("upload_s3", "No images folder")
        else:
            # strip_leading removes everything up to (not including) the stem,
            # so S3 keys become "<stem>/images/<file>" with no Data/ prefixes.
            strip_leading = (
                self.images_dir.parent.resolve()
                .relative_to(HERE.resolve())
                .as_posix() + "/"
            )
            self._run("upload_s3",
                      [sys.executable, str(HERE / "step8_upload_s3.py"),
                       "--root", str(self.images_dir),
                       "--strip-leading", strip_leading],
                      critical=False)

        return self._summary()

    def _should_skip(self, name: str, output: Path | None) -> bool:
        if name in self.skip:
            self._skip(name, "Skipped by caller")
            return True
        if output and output.exists() and not self.force:
            self._skip(name, f"{output.name} already exists")
            return True
        return False

    def _run(self, name: str, cmd: list,
             output: Path | None = None,
             critical: bool | None = None) -> bool:
        if critical is None:
            critical = STAGE_CRITICAL.get(name, True)

        stage = self.stages[name]
        stage.status     = "running"
        stage.start_time = time.time()
        self._emit(name, "start")
        dbg.log_step_start(self.job_id, name, stage.label, str(cmd[-1]) if cmd else "")

        try:
            proc = subprocess.Popen(
                [str(c) for c in cmd],
                cwd=str(HERE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            self._proc = proc

            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self._log(name, line)
                if self._stop.is_set():
                    proc.terminate()
                    stage.status   = "error"
                    stage.error    = "Cancelled"
                    stage.end_time = time.time()
                    self._emit(name, "cancelled")
                    return False

            proc.wait()
            stage.end_time = time.time()
            self._proc = None

            if proc.returncode == 0:
                stage.status      = "done"
                stage.output_path = str(output) if output else ""
                self._emit(name, "done")
                dbg.log_step_end(self.job_id, name, stage.label,
                                 stage.elapsed, stage.output_path)
                return True

            stage.status = "error"
            stage.error  = f"Exit code {proc.returncode}"
            self._emit(name, "error", {"error": stage.error})
            dbg.log_step_error(self.job_id, name, stage.label, stage.error)
            return not critical

        except Exception as exc:
            stage.status   = "error"
            stage.error    = str(exc)
            stage.end_time = time.time()
            self._proc = None
            self._emit(name, "error", {"error": stage.error})
            dbg.log_step_error(self.job_id, name, stage.label, stage.error)
            return not critical

    def _skip(self, name: str, reason: str) -> None:
        stage = self.stages[name]
        stage.status     = "skipped"
        stage.start_time = stage.end_time = time.time()
        self._log(name, f"[skip] {reason}")
        self._emit(name, "skipped", {"reason": reason})
        dbg.log_step_skipped(self.job_id, name, stage.label, reason)

    def _log(self, name: str, line: str) -> None:
        if name in self.stages:
            self.stages[name].logs.append(line)
        self._emit(name, "log", {"message": line})

    def _emit(self, name: str, event: str, data: dict | None = None) -> None:
        if self.cb:
            payload = {
                "job_id":      self.job_id,
                "stage":       name,
                "event":       event,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "stage_state": self.stages[name].to_dict() if name in self.stages else {},
                **(data or {}),
            }
            self.cb(payload)

    def _summary(self) -> dict:
        total_elapsed = sum(s.elapsed for s in self.stages.values())
        overall = "done"
        for name in STAGE_ORDER:
            s = self.stages[name]
            if s.status == "error" and STAGE_CRITICAL.get(name, True):
                overall = "error"
                break
        return {
            "job_id":        self.job_id,
            "pdf":           str(self.pdf),
            "status":        overall,
            "total_elapsed": total_elapsed,
            "stages":        {n: s.to_dict() for n, s in self.stages.items()},
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="PDF file to process")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], metavar="STAGE")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"ERROR: {pdf} is not a file", file=sys.stderr)
        return 1

    def on_event(evt: dict) -> None:
        stage = evt.get("stage", "")
        event = evt.get("event", "")
        msg   = evt.get("message", "")
        if event == "start":
            label = evt.get("stage_state", {}).get("label", stage)
            print(f"\n[{stage}] {label} — STARTING")
        elif event in ("done", "error", "skipped", "cancelled"):
            elapsed = evt.get("stage_state", {}).get("elapsed", 0)
            print(f"[{stage}] {event.upper()}  ({elapsed:.1f}s)")
        elif event == "log" and msg:
            print(f"  {msg}")

    engine = PipelineEngine(pdf, force=args.force, skip_steps=args.skip,
                            progress_callback=on_event)
    result = engine.run()

    print("\n" + "=" * 60)
    print(f"STATUS: {result['status'].upper()}  ({result['total_elapsed']:.1f}s total)")
    for name, s in result["stages"].items():
        icon = {"done": "✓", "error": "✗", "skipped": "—",
                "pending": "·", "running": "…"}.get(s["status"], "?")
        print(f"  {icon}  {STAGE_LABELS[name]:<35} {s['status']}"
              + (f"  ({s['elapsed']:.1f}s)" if s["elapsed"] else ""))
    return 0 if result["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
