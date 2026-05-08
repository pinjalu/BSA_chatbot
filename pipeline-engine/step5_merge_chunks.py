import json
import re
import sys
from pathlib import Path


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def cells_of(chunk: dict) -> list[str]:
    out = list(chunk.get("headers") or [])
    for row in chunk.get("rows") or []:
        out.extend(row.values() if isinstance(row, dict) else row)
    return out


def find_chunk(img: dict, page_chunks: list[dict]) -> dict | None:
    title = norm(img.get("title"))
    if not title:
        return page_chunks[0] if page_chunks else None

    # 1. table header/cell match
    for c in page_chunks:
        if c.get("type") == "table" and any(norm(v) == title for v in cells_of(c)):
            return c

    # 2. section name match
    for c in page_chunks:
        if norm(c.get("section")) == title:
            return c

    # 3. fallback: first chunk on page
    return page_chunks[0] if page_chunks else None


def main(clean_path: str, images_path: str, out_path: str) -> None:
    chunks     = json.loads(Path(clean_path).read_text(encoding="utf-8"))
    images_doc = json.loads(Path(images_path).read_text(encoding="utf-8"))

    by_page: dict[int, list[dict]] = {}
    for c in chunks:
        by_page.setdefault(int(c["page"]), []).append(c)

    total = 0
    for page_entry in images_doc.get("pages", []):
        page_chunks = by_page.get(int(page_entry["page"]), [])
        for img in page_entry.get("images", []):
            target = find_chunk(img, page_chunks)
            if target is None:
                continue
            target.setdefault("images", []).append({
                "file":  img.get("file"),
                "path":  img.get("path"),
                "title": img.get("title"),
                "bbox":  img.get("bbox"),
            })
            total += 1

    Path(out_path).write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {out_path} ({total} images mapped)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
