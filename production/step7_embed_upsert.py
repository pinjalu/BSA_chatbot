from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv(Path(__file__).with_name(".env"))

DEFAULT_INDEX_NAME = "bsa-manuals"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

DEFAULT_BATCH = 100        # OpenAI lets you batch up to 2048; 100 keeps
                           # individual request payloads small + retryable
PROGRESS_FILENAME = ".embed_progress.json"

MAX_INPUT_CHARS = 30_000   # ≈7.5K tokens, well under 8191 limit
PRICE_PER_M_TOKENS = 0.13  # text-embedding-3-large


def collect_jsonl(args) -> list[Path]:
    paths: list[Path] = []
    if args.files:
        for f in args.files:
            p = Path(f)
            if p.is_file() and p.suffix == ".jsonl":
                paths.append(p)
            else:
                print(f"[warn] not a .jsonl file: {f}", file=sys.stderr)
    if args.root:
        root = Path(args.root)
        if root.is_dir():
            paths.extend(sorted(root.rglob("*_pinecone.jsonl")))
    if not args.files and not args.root:
        paths.extend(sorted(Path("Data").rglob("*_pinecone.jsonl")))
    seen, uniq = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def iter_records(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] {path}:{line_no} bad JSON: {e}", file=sys.stderr)


def make_context_header(meta: dict) -> str:
    """Prepended to the chunk text BEFORE embedding so the resulting vector
    captures source/section/page context. This significantly improves
    recall on short chunks whose body text alone (e.g. 'Damaged Rollers')
    is too generic to retrieve cleanly.
    """
    parts = []
    pdf = meta.get("source_pdf", "")
    if pdf:
        parts.append(f"Doc: {Path(pdf).stem}")
    if meta.get("vehicle_model"):
        parts.append(f"Vehicle: {meta['vehicle_model']}")
    if meta.get("doc_type"):
        parts.append(f"Type: {meta['doc_type']}")
    if meta.get("section"):
        parts.append(f"Section: {meta['section']}")
    if meta.get("page") is not None:
        parts.append(f"Page {meta['page']}")
    return "[" + " | ".join(parts) + "]" if parts else ""


def text_for_embedding(record: dict) -> str:
    header = make_context_header(record["metadata"])
    body = record.get("text", "")
    combined = f"{header}\n{body}" if header else body
    return combined[:MAX_INPUT_CHARS]


def build_pinecone_metadata(record: dict) -> dict:
    md = dict(record["metadata"])
    md["text"] = record.get("text", "")
    return md


def progress_path(root: Path) -> Path:
    return root / PROGRESS_FILENAME


def load_progress(p: Path) -> set[str]:
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        print(f"[warn] could not parse {p}; ignoring it.")
        return set()


def append_progress(p: Path, ids: list[str]) -> None:
    existing = load_progress(p)
    existing.update(ids)
    p.write_text(json.dumps(sorted(existing)), encoding="utf-8")


def ensure_index(pc: Pinecone, name: str) -> object:
    names = {ix["name"] for ix in pc.list_indexes()}
    if name not in names:
        print(f"[setup] creating index {name!r} ({EMBED_DIM}-dim, cosine, "
              f"{PINECONE_CLOUD}/{PINECONE_REGION})")
        pc.create_index(
            name=name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
        print(f"[setup] index {name!r} ready")
    desc = pc.describe_index(name)
    if desc.dimension != EMBED_DIM:
        raise SystemExit(
            f"existing index {name!r} is {desc.dimension}-dim but "
            f"{EMBED_MODEL!r} produces {EMBED_DIM}-dim. "
            f"Use --index <new_name> or delete the old one."
        )
    return pc.Index(name)


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def run(records: list[dict], index, client: OpenAI, batch_size: int,
        progress_p: Path) -> tuple[int, int]:
    done_ids = load_progress(progress_p)
    pending = [r for r in records if r["id"] not in done_ids]

    if done_ids:
        print(f"[resume] {len(done_ids)} already done; "
              f"{len(pending)} pending")

    upserted = failed = 0
    started = time.time()

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [text_for_embedding(r) for r in batch]

        try:
            vectors_data = embed_batch(client, texts)
        except Exception as e:
            print(f"[error] embed batch failed: {e}", file=sys.stderr)
            failed += len(batch)
            continue

        vectors = [
            {
                "id": r["id"],
                "values": emb,
                "metadata": build_pinecone_metadata(r),
            }
            for r, emb in zip(batch, vectors_data)
        ]

        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            print(f"[error] upsert failed: {e}", file=sys.stderr)
            failed += len(batch)
            continue

        upserted += len(batch)
        append_progress(progress_p, [r["id"] for r in batch])

        elapsed = time.time() - started
        rate = upserted / elapsed if elapsed > 0 else 0
        remaining = len(pending) - upserted
        eta = remaining / rate if rate > 0 else 0
        print(f"  upserted {upserted:>5}/{len(pending):<5}  "
              f"({rate:>5.1f}/s, ETA {eta:>4.0f}s)")

    return upserted, failed


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default=None,
                    help="Folder to scan (default: Data/)")
    ap.add_argument("--files", nargs="*",
                    help="Explicit list of .jsonl files")
    ap.add_argument("--index", default=DEFAULT_INDEX_NAME,
                    help=f"Pinecone index name (default: {DEFAULT_INDEX_NAME})")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"Batch size (default: {DEFAULT_BATCH})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count + cost estimate only, no API calls")
    ap.add_argument("--wipe", action="store_true",
                    help="Delete ALL vectors from the index before upserting")
    ap.add_argument("--restart", action="store_true",
                    help="Ignore progress file and reprocess every chunk")
    args = ap.parse_args()

    paths = collect_jsonl(args)
    if not paths:
        print("no *_pinecone.jsonl files found", file=sys.stderr)
        return 1

    print(f"[scan] {len(paths)} file(s):")
    for p in paths:
        print(f"   {p}")

    records = list(iter_records(paths))
    total_chars = sum(len(r.get("text", "")) for r in records)
    approx_tokens = total_chars // 4
    cost = approx_tokens / 1_000_000 * PRICE_PER_M_TOKENS

    print(f"\n[stats]  {len(records):,} records, ~{total_chars:,} chars "
          f"(~{approx_tokens:,} tokens)")
    print(f"[model]  {EMBED_MODEL}  ({EMBED_DIM} dims, cosine)")
    print(f"[index]  {args.index}")
    print(f"[cost]   estimated: ${cost:.4f}")

    if args.dry_run:
        print("\n--dry-run: stopping before any API call")
        return 0

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not pinecone_key or not openai_key:
        print("ERROR: set PINECONE_API_KEY and OPENAI_API_KEY in .env",
              file=sys.stderr)
        return 1

    pc = Pinecone(api_key=pinecone_key)
    index = ensure_index(pc, args.index)
    client = OpenAI(api_key=openai_key)

    if args.wipe:
        print("[wipe] deleting all vectors...")
        index.delete(delete_all=True)
        time.sleep(2)

    progress_p = progress_path(Path(args.root) if args.root else Path("Data"))
    if args.restart and progress_p.exists():
        progress_p.unlink()
        print(f"[restart] removed {progress_p}")

    upserted, failed = run(records, index, client, args.batch, progress_p)

    print(f"\n[done] upserted {upserted}, failed {failed}")
    print("\n[index stats]")
    print(index.describe_index_stats())
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
