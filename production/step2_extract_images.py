from __future__ import annotations

import os

# PaddlePaddle 3.x on Windows crashes in the OneDNN instruction converter for
# some kernels — must be set BEFORE any paddle / paddleocr import.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_set_to_1d", "0")
os.environ.setdefault("FLAGS_use_cudnn", "0")

import argparse
import hashlib
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import fitz
import cv2
import numpy as np
import pandas as pd


__all__ = [
    "extract_images",
    "extract_page_images",
    "find_nearby_title",
    "paddle_detect_title",
    "cluster_drawing_regions",
]


log = logging.getLogger("only_image")


def _setup_logging(level: str | int) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="[IMG] %(asctime)s %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
        )
    else:
        logging.getLogger().setLevel(level)


# PaddleOCR / tesseract are not thread-safe — serialise all OCR calls.
_OCR_LOCK = threading.Lock()
# Guards the lazy-init `_paddle_ocr` module global from concurrent
# "init or mark FAILED" decisions.
_PADDLE_INIT_LOCK = threading.Lock()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    _atomic_write_bytes(path, text.encode("utf-8"))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_paddle_ocr = None


def get_paddle_ocr():
    """Construct and cache a PaddleOCR instance. Returns None if paddleocr
    isn't installed or initialisation fails.

    Thread-safe: the lazy init runs at most once across all worker threads,
    guarded by `_PADDLE_INIT_LOCK`.  After init the module global is read
    without locking (only the inference call needs `_OCR_LOCK`).
    """
    global _paddle_ocr
    if _paddle_ocr == "FAILED":
        return None
    if _paddle_ocr is not None:
        return _paddle_ocr

    with _PADDLE_INIT_LOCK:
        # Re-check inside the lock — another thread may have raced to init.
        if _paddle_ocr == "FAILED":
            return None
        if _paddle_ocr is not None:
            return _paddle_ocr

        # OneDNN causes a ConvertPirAttribute2RuntimeAttribute crash on
        # Windows for some paddlepaddle builds — disable it before importing.
        import os as _os
        _os.environ.setdefault("FLAGS_use_mkldnn", "0")
        _os.environ.setdefault("FLAGS_use_cudnn", "0")

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            log.warning("PaddleOCR not installed: %s — falling back to tesseract", exc)
            _paddle_ocr = "FAILED"
            return None
        except Exception as exc:  # noqa: BLE001 — paddle import can raise anything
            log.warning("PaddleOCR import failed: %s", exc)
            _paddle_ocr = "FAILED"
            return None

        # Try progressively simpler kwargs until one succeeds. Prefer CPU +
        # no-MKL paths since that sidesteps the Windows OneDNN issue.
        for kwargs in (
            {"lang": "en", "enable_mkldnn": False, "use_gpu": False},
            {"lang": "en"},
            {"lang": "en", "use_angle_cls": True},
            {"lang": "en", "use_angle_cls": True, "show_log": False},
            {},
        ):
            try:
                _paddle_ocr = PaddleOCR(**kwargs)
                log.debug("PaddleOCR ready — kwargs=%s", kwargs)
                return _paddle_ocr
            except Exception as exc:  # noqa: BLE001 — paddle init varies
                log.debug("PaddleOCR init kwargs=%s failed: %s", kwargs, exc)
                continue

        log.warning("PaddleOCR init failed across all known signatures")
        _paddle_ocr = "FAILED"
        return None


def _text_blocks(fitz_page) -> list[tuple[float, float, float, float, str]]:
    raw = fitz_page.get_text("blocks")
    out = []
    for b in raw:
        if len(b) < 7 or b[6] != 0:  # skip image/drawing blocks
            continue
        text = (b[4] or "").strip()
        if not text:
            continue
        out.append((b[0], b[1], b[2], b[3], text))
    return out


def _pick_caption(text: str) -> str:
    """From a multi-line block, pick the line most likely to be a caption.

    Captions are short, often upper-case, and sit on their own line. Prefer
    the line with the highest upper-case ratio; fall back to the first line.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    scored = []
    for ln in lines:
        letters = [c for c in ln if c.isalpha()]
        if not letters:
            continue
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        score = upper_ratio * 2 + (1 if len(ln) <= 60 else 0)
        scored.append((score, ln))
    if not scored:
        return lines[0]
    scored.sort(reverse=True)
    return scored[0][1]


def find_nearby_title(fitz_page, bbox,
                      max_distance: float = 60.0) -> Optional[str]:
    x0, y0, x1, y1 = bbox
    img_w = max(x1 - x0, 1)

    above: list[tuple[float, str]] = []
    below: list[tuple[float, str]] = []

    for bx0, by0, bx1, by1, text in _text_blocks(fitz_page):
        h_overlap = max(0.0, min(bx1, x1) - max(bx0, x0))
        if h_overlap / img_w < 0.2:
            continue

        if by1 <= y0 + 2:  # above
            d = y0 - by1
            if 0 <= d <= max_distance:
                above.append((d, text))
        elif by0 >= y1 - 2:  # below
            d = by0 - y1
            if 0 <= d <= max_distance:
                below.append((d, text))

    for candidates in (above, below):
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0])
        caption = _pick_caption(candidates[0][1])
        if caption:
            return caption
    return None


def preprocess_for_ocr(image_bytes: bytes,
                       min_short_side: int = 720) -> Optional[np.ndarray]:
    """Decode bytes → grayscale OpenCV BGR array, upsampled if small.

    `min_short_side` controls the upsampling target; PaddleOCR is noticeably
    more accurate on ≥720 px short sides but the default is overridable for
    speed-vs-quality tuning.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    short = min(h, w)
    if short < min_short_side and short > 0:
        scale = min_short_side / short
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    # Keep 3 channels for PaddleOCR input
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _flatten_paddle_result(result):
    items: list[tuple[list, str, float]] = []
    if result is None:
        return items

    # PaddleOCR 3.x returns a list of dict-like objects with rec_texts/rec_scores/boxes
    if isinstance(result, list) and result and isinstance(result[0], dict):
        for r in result:
            texts = r.get("rec_texts") or r.get("texts") or []
            scores = r.get("rec_scores") or r.get("scores") or []
            boxes = r.get("rec_boxes") or r.get("boxes") or r.get("dt_polys") or []
            for text, score, bbox in zip(texts, scores, boxes):
                items.append((bbox, str(text), float(score)))
        return items

    # PaddleOCR 2.x returns [[ [box, (text, conf)], ... ]]
    if isinstance(result, list) and result and isinstance(result[0], list):
        for line in result[0]:
            try:
                bbox = line[0]
                text, conf = line[1]
                items.append((bbox, str(text), float(conf)))
            except Exception:
                continue
    return items


def _tesseract_detect_title(arr) -> Optional[str]:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return None
    try:
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(pil) or ""
    except Exception:
        return None
    candidates: list[tuple[float, str]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not (3 <= len(ln) <= 80):
            continue
        letters = [c for c in ln if c.isalpha()]
        if not letters:
            continue
        upper = sum(1 for c in letters if c.isupper()) / len(letters)
        score = upper * 2 + (1 if len(ln) <= 60 else 0)
        candidates.append((score, ln))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def paddle_detect_title(image_bytes: bytes,
                        min_conf: float = 0.6,
                        ocr_short_side: int = 720) -> Optional[str]:
    """Run PaddleOCR on the image bytes and return the topmost high-confidence
    line. Falls back to pytesseract if Paddle isn't available or crashes.

    Thread-safe: the inference call is serialised via `_OCR_LOCK` because
    PaddleOCR / tesseract aren't safe to invoke from multiple threads at once.
    """
    arr = preprocess_for_ocr(image_bytes, min_short_side=ocr_short_side)
    if arr is None:
        return None

    with _OCR_LOCK:
        ocr = get_paddle_ocr()
        if ocr is None:
            return _tesseract_detect_title(arr)

        # Try both inference APIs — PaddleOCR 3.x uses predict(), 2.x uses ocr().
        # On Windows, predict() can crash with an OneDNN error; if that happens
        # fall back to ocr().
        result = None
        errors: list[str] = []
        for call in ("predict", "ocr"):
            fn = getattr(ocr, call, None)
            if fn is None:
                continue
            try:
                result = fn(arr)
                break
            except Exception as e:
                errors.append(f"{call}: {e}")
                continue
        if result is None:
            if errors:
                log.warning(
                    "PaddleOCR inference failed, falling back to tesseract: %s",
                    errors[-1].splitlines()[0],
                )
            # Disable paddle for subsequent calls and use tesseract.
            global _paddle_ocr
            _paddle_ocr = "FAILED"
            return _tesseract_detect_title(arr)

    items = _flatten_paddle_result(result)
    items = [
        (bbox, text.strip(), conf)
        for bbox, text, conf in items
        if conf >= min_conf and 3 <= len(text.strip()) <= 80
    ]
    if not items:
        return None

    def _ykey(item):
        bbox = item[0]
        try:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                # xyxy
                return (bbox[1] + bbox[3]) / 2
            return sum(p[1] for p in bbox) / len(bbox)
        except Exception:
            return 0.0

    items.sort(key=lambda it: (_ykey(it), -len(it[1])))
    return items[0][1]


def _is_white_or_invisible(drawing: dict) -> bool:
    """True if the drawing is a no-op (no stroke and fill is white/none).

    Many PDFs sprinkle white background rects and invisible clip paths; these
    explode the drawing count and cover the whole page if not filtered out.
    """
    stroke = drawing.get("color") or drawing.get("stroke")
    fill = drawing.get("fill")
    if not stroke and not fill:
        return True
    if not stroke and fill:
        try:
            comps = fill if isinstance(fill, (list, tuple)) else (fill,)
            if all(isinstance(c, (int, float)) and c >= 0.98 for c in comps):
                return True
        except Exception:
            pass
    return False


def _text_block_rects(fitz_page) -> list:
    out: list[fitz.Rect] = []
    for b in fitz_page.get_text("blocks"):
        if len(b) < 7 or b[6] != 0:  # type 0 = text block
            continue
        if not (b[4] or "").strip():
            continue
        out.append(fitz.Rect(b[0], b[1], b[2], b[3]))
    return out


def _get_table_rects(fitz_page,
                     table_text_ratio: float = 0.20) -> list:
    """Rects of regions that look like real text tables on this page.

    Uses PyMuPDF's `find_tables()`. A region is treated as a table only when
    text blocks cover more than `table_text_ratio` of its area — otherwise
    it might just be a pictorial-index grid we DO want rendered as images.
    """
    table_rects: list[fitz.Rect] = []
    try:
        finder = fitz_page.find_tables()
    except Exception:
        return table_rects

    text_rects = _text_block_rects(fitz_page)
    for t in finder:
        bbox = getattr(t, "bbox", None)
        if not bbox:
            continue
        tab_rect = fitz.Rect(bbox)
        tab_area = max(tab_rect.width * tab_rect.height, 1.0)

        text_coverage = 0.0
        for tr in text_rects:
            inter = tab_rect & tr
            if not inter.is_empty:
                text_coverage += inter.width * inter.height

        if (text_coverage / tab_area) > table_text_ratio:
            table_rects.append(tab_rect)
    return table_rects


def _overlaps_any(a: fitz.Rect, others: list,
                  threshold: float = 0.60) -> bool:
    a_area = max(a.width * a.height, 1e-6)
    for o in others:
        inter = fitz.Rect(a) & o
        if inter.is_empty:
            continue
        if (inter.width * inter.height) / a_area > threshold:
            return True
    return False


def cluster_drawing_regions(fitz_page,
                            min_area_ratio: float = 0.02,
                            max_area_ratio: float = 0.88,
                            pad: float = 10.0,
                            min_primitives: int = 5,
                            max_text_coverage: float = 0.55,
                            exclude_tables: bool = True,
                            table_text_ratio: float = 0.20,
                            table_overlap_threshold: float = 0.60) -> list:
    """Group vector drawings into diagram-sized bounding rects.

    Returns a list of fitz.Rect. Each rect is a cluster of connected/nearby
    drawing primitives that together look like a figure — NOT a single
    border rect, a text box with a frame, a table, or a background fill.

    Filters that kill the common false-positive patterns:
      * `min_primitives` — a diagram is several strokes; a box around a
        heading is one or two.
      * `max_text_coverage` — if text blocks cover most of the cluster's
        area, it's a framed text block (table row, parts-list heading),
        not a picture.
      * aspect clamp — degenerate slivers (aspect > 12:1) are rulings.
      * `max_area_ratio` — page-sized fills are background.
      * `exclude_tables` — drop clusters that overlap a region detected as
        a real text table by `find_tables()`. Without this, a parts-list
        page can be captured as one big PNG.
    """
    drawings = fitz_page.get_drawings()
    if not drawings:
        return []

    page_rect = fitz_page.rect
    page_area = max(page_rect.width * page_rect.height, 1.0)

    boxes: list[fitz.Rect] = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        if _is_white_or_invisible(d):
            continue
        w = r.width
        h = r.height
        if w < 3 or h < 3:
            continue
        if (w * h) / page_area > max_area_ratio:
            continue  # page-sized background fill
        boxes.append(fitz.Rect(r))

    if not boxes:
        return []

    # Each cluster remembers how many drawing primitives it absorbed.
    clusters = [{"rect": fitz.Rect(b), "count": 1} for b in boxes]
    changed = True
    while changed:
        changed = False
        new_clusters: list[dict] = []
        for c in clusters:
            placed = False
            for n in new_clusters:
                n_exp = fitz.Rect(n["rect"].x0 - pad, n["rect"].y0 - pad,
                                  n["rect"].x1 + pad, n["rect"].y1 + pad)
                if n_exp.intersects(c["rect"]):
                    n["rect"].include_rect(c["rect"])
                    n["count"] += c["count"]
                    placed = True
                    changed = True
                    break
            if not placed:
                new_clusters.append({"rect": fitz.Rect(c["rect"]),
                                     "count": c["count"]})
        clusters = new_clusters

    text_rects = _text_block_rects(fitz_page)
    table_rects = (_get_table_rects(fitz_page,
                                    table_text_ratio=table_text_ratio)
                   if exclude_tables else [])

    out: list[fitz.Rect] = []
    for c in clusters:
        m = c["rect"]
        count = c["count"]
        w, h = m.width, m.height
        area = w * h

        if area / page_area < min_area_ratio:
            continue
        if min(w, h) < 28:
            continue
        # Kill thin rulings / horizontal separators.
        aspect = max(w, h) / max(min(w, h), 1.0)
        if aspect > 12:
            continue
        # A figure is many primitives; a framed heading is 1-2.
        if count < min_primitives:
            continue
        # If the cluster is mostly text, it's a text box not a picture.
        text_overlap = 0.0
        for tr in text_rects:
            inter = fitz.Rect(m) & tr
            if not inter.is_empty:
                text_overlap += inter.width * inter.height
        if area > 0 and (text_overlap / area) > max_text_coverage:
            continue
        # Drop clusters that mostly overlap a real text-table region —
        # otherwise a parts list / spec table gets captured as a PNG.
        if table_rects and _overlaps_any(m, table_rects,
                                         table_overlap_threshold):
            continue

        out.append(m)
    return out


def _clean_title(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip(" .:;,-_|")
    return s


def _rects_overlap_ratio(a, b) -> float:
    a_area = max(a.width * a.height, 1e-6)
    inter = fitz.Rect(a) & b
    if inter.is_empty:
        return 0.0
    return (inter.width * inter.height) / a_area


def extract_page_images(fitz_page, doc, page_num: int, images_dir: Path,
                        *,
                        use_ocr: bool = True,
                        include_vectors: bool = True,
                        min_image_pixels: int = 24,
                        min_render_pt: float = 24.0,
                        render_dpi: int = 144,
                        ocr_min_conf: float = 0.6,
                        ocr_short_side: int = 720,
                        title_distance: float = 60.0,
                        cluster_kwargs: Optional[dict] = None) -> list[dict]:
    """Extract every picture on this page — embedded rasters AND vector
    diagrams — and resolve a title for each.

    All thresholds are explicit keyword arguments so callers can override
    them per-PDF / per-environment.  `cluster_kwargs` is forwarded to
    `cluster_drawing_regions()` (e.g. table-overlap settings).

    `min_render_pt` rejects placements that occupy less than that many
    PDF points on either axis, regardless of the underlying image's
    pixel resolution.  A 12 × 9 pt placement of a 32 × 32 px icon glyph
    (high-beam indicator, Bluetooth logo, etc.) is dropped here even
    though the pixel filter would let it through.  Without this, a
    single icon placed inline 5 times on a manual page produces 5
    "images" whose titles are unrelated body sentences picked up by
    `find_nearby_title`.
    """
    cluster_kwargs = cluster_kwargs or {}
    results: list[dict] = []
    embedded_rects: list[fitz.Rect] = []
    img_idx = 0

    seen_xrefs: set[int] = set()
    for img_info in fitz_page.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            base = doc.extract_image(xref)
        except Exception as exc:  # noqa: BLE001 — fitz raises diverse errors
            log.debug("page %d xref %d: extract_image failed: %s",
                      page_num, xref, exc)
            continue

        width = base.get("width") or 0
        height = base.get("height") or 0
        if width < min_image_pixels or height < min_image_pixels:
            # Almost always a 1-2px spacer / ruling line / icon glyph.
            continue

        rects = fitz_page.get_image_rects(xref)
        if not rects:
            # Referenced but not placed; still save (no bbox → OCR-only title).
            rects = [None]

        # When the same image is placed multiple times on the page (a
        # decorative icon repeated next to body sentences, a banner
        # logo at every section header), keep only the largest
        # placement.  Saving N identical files plus N unrelated titles
        # adds noise without information — for the rare case of
        # before/after using the same source image, the user still
        # gets the larger of the two.  Placements without a bbox stay
        # as-is.
        sized_rects = [r for r in rects if r is not None]
        if len(sized_rects) > 1:
            best = max(sized_rects,
                       key=lambda r: float(r.x1 - r.x0) * float(r.y1 - r.y0))
            rects = [best]

        ext = base.get("ext", "png")
        image_bytes = base["image"]
        for rect in rects:
            # Drop icon-sized placements regardless of pixel resolution.
            # Stops 12 × 9 pt status-icon glyphs from being captured as
            # full-fledged "images" with text titles attached.
            if rect is not None:
                rw = float(rect.x1 - rect.x0)
                rh = float(rect.y1 - rect.y0)
                if rw < min_render_pt or rh < min_render_pt:
                    continue

            img_idx += 1
            filename = f"page_{page_num}_img_{img_idx}.{ext}"
            try:
                _atomic_write_bytes(images_dir / filename, image_bytes)
            except OSError as exc:
                log.error("page %d: failed to write %s: %s",
                          page_num, filename, exc)
                continue

            bbox = tuple(rect) if rect is not None else None
            if rect is not None:
                embedded_rects.append(fitz.Rect(rect))

            title = (find_nearby_title(fitz_page, bbox, title_distance)
                     if bbox else None)
            if not title and use_ocr:
                title = paddle_detect_title(image_bytes,
                                            min_conf=ocr_min_conf,
                                            ocr_short_side=ocr_short_side)

            results.append({
                "title": _clean_title(title or ""),
                "file": filename,
                "_bbox": list(bbox) if bbox else None,
                "_width": width,
                "_height": height,
                "_ext": ext,
                "_xref": xref,
                "_source": "embedded",
                "_hash": hashlib.md5(image_bytes).hexdigest()[:16],
            })

    if include_vectors:
        for diag_rect in cluster_drawing_regions(fitz_page, **cluster_kwargs):
            if any(_rects_overlap_ratio(diag_rect, er) > 0.7
                   for er in embedded_rects):
                continue

            try:
                mat = fitz.Matrix(render_dpi / 72.0, render_dpi / 72.0)
                pix = fitz_page.get_pixmap(matrix=mat, clip=diag_rect,
                                           alpha=False)
            except Exception as exc:  # noqa: BLE001 — pixmap raises diverse errors
                log.debug("page %d: get_pixmap failed: %s", page_num, exc)
                continue
            if pix.width < min_image_pixels or pix.height < min_image_pixels:
                continue
            img_bytes = pix.tobytes("png")

            img_idx += 1
            filename = f"page_{page_num}_img_{img_idx}.png"
            try:
                _atomic_write_bytes(images_dir / filename, img_bytes)
            except OSError as exc:
                log.error("page %d: failed to write %s: %s",
                          page_num, filename, exc)
                continue

            bbox = tuple(diag_rect)
            title = find_nearby_title(fitz_page, bbox, title_distance)
            if not title and use_ocr:
                title = paddle_detect_title(img_bytes,
                                            min_conf=ocr_min_conf,
                                            ocr_short_side=ocr_short_side)

            results.append({
                "title": _clean_title(title or ""),
                "file": filename,
                "_bbox": list(bbox),
                "_width": pix.width,
                "_height": pix.height,
                "_ext": "png",
                "_xref": None,
                "_source": "rendered",
                "_hash": hashlib.md5(img_bytes).hexdigest()[:16],
            })

    return results


def _filter_recurring_decorations(
        results_by_page: dict[int, tuple[list, float]],
        images_dir: Path,
        *,
        recurrence_threshold: int = 3,
        max_decorative_pt: float = 80.0) -> int:
    """Drop images whose byte-content repeats across many pages AND
    whose placement is small.  Targets decorative logos / status icons
    / footer banners that the PDF embeds on every page — same image
    bytes, different xrefs, so per-page xref dedup can't see them.

    A hash counts as decorative when:
      * it appears on more than `recurrence_threshold` pages, AND
      * every observed placement of that image is smaller than
        `max_decorative_pt` on its longer side (a real cross-referenced
        diagram is usually rendered large somewhere).

    Files corresponding to dropped images are deleted from `images_dir`
    so the manifest matches what's on disk.  Returns the count dropped.
    """
    if not results_by_page:
        return 0

    from collections import Counter, defaultdict

    hash_pages: dict[str, set[int]] = defaultdict(set)
    hash_max_dim: dict[str, float] = defaultdict(float)

    for page_num, (imgs, _dt) in results_by_page.items():
        for it in imgs:
            h = it.get("_hash")
            if not h:
                continue
            hash_pages[h].add(page_num)
            bbox = it.get("_bbox")
            if bbox:
                long_side = max(float(bbox[2] - bbox[0]),
                                float(bbox[3] - bbox[1]))
                hash_max_dim[h] = max(hash_max_dim[h], long_side)
            else:
                # No bbox — fall back to pixel dimensions converted
                # roughly to points (1 pt ≈ 1.33 px at 96 DPI).
                w = it.get("_width") or 0
                h2 = it.get("_height") or 0
                hash_max_dim[h] = max(hash_max_dim[h], max(w, h2) / 1.33)

    decorative: set[str] = set()
    for h, pages in hash_pages.items():
        if len(pages) > recurrence_threshold and hash_max_dim[h] < max_decorative_pt:
            decorative.add(h)

    if not decorative:
        return 0

    dropped = 0
    for page_num in list(results_by_page.keys()):
        imgs, dt = results_by_page[page_num]
        kept: list[dict] = []
        for it in imgs:
            if it.get("_hash") in decorative:
                fp = images_dir / it.get("file", "")
                if fp.is_file():
                    try:
                        fp.unlink()
                    except OSError:
                        pass
                dropped += 1
                continue
            kept.append(it)
        results_by_page[page_num] = (kept, dt)

    log.info("Cross-page dedup: dropped %d decorative repeats "
             "(%d unique hashes appeared on > %d pages each)",
             dropped, len(decorative), recurrence_threshold)
    return dropped


def _accumulate_page(page_num: int, total: int, imgs: list, dt: float,
                     pages_output: list, rows: list) -> None:
    if not imgs:
        log.info("  [%4d/%d]  0 images  %4.1fs", page_num, total, dt)
        return

    clean_imgs = [
        {"title": it["title"], "file": it["file"],
         "path": f"images/{it['file']}",
         "bbox": it.get("_bbox")}
        for it in imgs
    ]
    pages_output.append({"page": page_num, "images": clean_imgs})

    for it in imgs:
        rows.append({
            "page": page_num,
            "file": it["file"],
            "title": it["title"],
            "width": it["_width"],
            "height": it["_height"],
            "has_title": bool(it["title"]),
            "source": it.get("_source", "embedded"),
        })

    titled = sum(1 for it in imgs if it["title"])
    rendered = sum(1 for it in imgs if it.get("_source") == "rendered")
    log.info("  [%4d/%d]  %2d images  (%d rendered)  %2d titled  %4.1fs",
             page_num, total, len(imgs), rendered, titled, dt)


def extract_images(pdf_path: Path, output_dir: Path,
                   *,
                   use_ocr: bool = True,
                   include_vectors: bool = True,
                   workers: int = 1,
                   min_image_pixels: int = 24,
                   min_render_pt: float = 24.0,
                   render_dpi: int = 144,
                   ocr_min_conf: float = 0.6,
                   ocr_short_side: int = 720,
                   title_distance: float = 60.0,
                   table_text_ratio: float = 0.20,
                   table_overlap_threshold: float = 0.60,
                   exclude_tables: bool = True) -> dict:
    """Extract every image from `pdf_path` into `output_dir`.

    All tuning thresholds are explicit keyword arguments so the function is
    safe to call as a library on a server with per-tenant config.
    """
    pdf_path = pdf_path.resolve()
    output_dir = output_dir.resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    cluster_kwargs = {
        "exclude_tables": exclude_tables,
        "table_text_ratio": table_text_ratio,
        "table_overlap_threshold": table_overlap_threshold,
    }
    page_kwargs = dict(
        use_ocr=use_ocr,
        include_vectors=include_vectors,
        min_image_pixels=min_image_pixels,
        min_render_pt=min_render_pt,
        render_dpi=render_dpi,
        ocr_min_conf=ocr_min_conf,
        ocr_short_side=ocr_short_side,
        title_distance=title_distance,
        cluster_kwargs=cluster_kwargs,
    )

    t_start = time.time()
    log.info("Opening %s", pdf_path.name)

    pages_output: list[dict] = []
    rows: list[dict] = []
    workers = max(1, int(workers))

    # Buffer all pages first so we can run a cross-page dedup pass over
    # raw results before final accumulation. The dedup needs visibility
    # of every image's content hash across every page.
    results_by_page: dict[int, tuple[list, float]] = {}

    with fitz.open(str(pdf_path)) as doc:
        total = doc.page_count
        log.info("%d pages | ocr=%s vectors=%s workers=%d render_dpi=%d",
                 total, use_ocr, include_vectors, workers, render_dpi)

        if workers <= 1:
            for i in range(total):
                page_num = i + 1
                t0 = time.time()
                fitz_page = doc[i]
                try:
                    imgs = extract_page_images(
                        fitz_page, doc, page_num, images_dir, **page_kwargs,
                    )
                except Exception as exc:  # noqa: BLE001 — never let one page kill the run
                    log.error("page %d: extraction crashed: %s", page_num, exc,
                              exc_info=log.isEnabledFor(logging.DEBUG))
                    imgs = []
                results_by_page[page_num] = (imgs, time.time() - t0)
        else:
            # Parallel — each worker thread holds its own fitz.Document
            # (Document is not thread-safe).  PaddleOCR / tesseract calls
            # are serialised inside `paddle_detect_title` via `_OCR_LOCK`.
            tlocal = threading.local()
            opened_docs: list = []
            opened_lock = threading.Lock()

            def _open_thread_doc():
                d = getattr(tlocal, "doc", None)
                if d is None:
                    d = fitz.open(str(pdf_path))
                    tlocal.doc = d
                    with opened_lock:
                        opened_docs.append(d)
                return d

            def _process_one(p_num: int):
                t0 = time.time()
                tdoc = _open_thread_doc()
                fp = tdoc[p_num - 1]
                try:
                    out = extract_page_images(
                        fp, tdoc, p_num, images_dir, **page_kwargs,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error("page %d: extraction crashed: %s", p_num, exc,
                              exc_info=log.isEnabledFor(logging.DEBUG))
                    out = []
                return p_num, out, time.time() - t0

            try:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_process_one, i + 1)
                               for i in range(total)]
                    try:
                        for fut in as_completed(futures):
                            p_num, imgs, dt = fut.result()
                            results_by_page[p_num] = (imgs, dt)
                    except KeyboardInterrupt:
                        log.warning("Interrupted — cancelling pending pages")
                        for f in futures:
                            f.cancel()
                        raise
            finally:
                # Release fds promptly instead of waiting for GC.
                for d in opened_docs:
                    try:
                        d.close()
                    except Exception:  # noqa: BLE001
                        pass

    _filter_recurring_decorations(results_by_page, images_dir)

    for p_num in sorted(results_by_page):
        imgs, dt = results_by_page[p_num]
        _accumulate_page(p_num, total, imgs, dt, pages_output, rows)

    total_dt = time.time() - t_start

    df = pd.DataFrame(rows)
    summary: dict = {
        "total_images": int(len(df)),
        "pages_with_images": int(len(pages_output)),
        "images_with_titles": int(df["has_title"].sum()) if not df.empty else 0,
    }
    if not df.empty:
        summary["avg_images_per_page"] = round(
            df.groupby("page").size().mean(), 2
        )
        summary["embedded_count"] = int((df["source"] == "embedded").sum())
        summary["rendered_count"] = int((df["source"] == "rendered").sum())

    output = {
        "pdf_path": str(pdf_path),
        "pdf_name": pdf_path.name,
        "total_pages": total,
        "duration_seconds": round(total_dt, 2),
        **summary,
        "pages": pages_output,
    }

    out_json = output_dir / f"{pdf_path.stem}_images.json"
    _atomic_write_json(out_json, output)

    emb = summary.get("embedded_count", 0)
    rnd = summary.get("rendered_count", 0)
    log.info(
        "DONE in %.1fs  images=%d  (embedded=%d rendered=%d)  "
        "titled=%d/%d  pages_with_images=%d",
        total_dt, summary["total_images"], emb, rnd,
        summary["images_with_titles"], summary["total_images"],
        summary["pages_with_images"],
    )
    log.info("Manifest → %s", out_json)
    log.info("Images   → %s", images_dir)
    return output


def _ensure_utf8_console() -> None:
    """Force UTF-8 on stdout/stderr (Windows defaults to cp1252).

    Without this, `--help` and log lines containing box-drawing or arrow
    characters crash on Windows servers.  Python 3.7+ supports `reconfigure`.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_utf8_console()

    default_workers = _env_int("ONLY_IMAGE_WORKERS",
                               min(8, (os.cpu_count() or 1)))
    default_dpi = _env_int("ONLY_IMAGE_RENDER_DPI", 144)
    default_min_pix = _env_int("ONLY_IMAGE_MIN_PIXELS", 24)
    default_min_conf = _env_float("ONLY_IMAGE_OCR_MIN_CONF", 0.6)
    default_title_dist = _env_float("ONLY_IMAGE_TITLE_DISTANCE", 60.0)
    default_table_text = _env_float("ONLY_IMAGE_TABLE_TEXT_RATIO", 0.20)
    default_table_overlap = _env_float("ONLY_IMAGE_TABLE_OVERLAP", 0.60)
    default_log_level = os.environ.get("ONLY_IMAGE_LOG_LEVEL", "INFO")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pdf", type=Path, help="Path to the PDF")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Output folder (default: <pdf_parent>/Data/<pdf_stem>/)")

    ap.add_argument("--no-ocr", action="store_true",
                    help="Skip OCR title fallback (faster)")
    ap.add_argument("--no-vectors", action="store_true",
                    help="Skip vector-diagram rendering")
    ap.add_argument("--keep-tables", action="store_true",
                    help="Render text-heavy table regions as images "
                         "(default: skip them)")

    ap.add_argument("--workers", type=int, default=default_workers,
                    help=f"Parallel page workers (default: {default_workers})")

    ap.add_argument("--render-dpi", type=int, default=default_dpi,
                    help=f"DPI for rendering vector diagrams "
                         f"(default: {default_dpi})")
    ap.add_argument("--min-image-pixels", type=int, default=default_min_pix,
                    help=f"Minimum image side length in pixels "
                         f"(default: {default_min_pix})")
    ap.add_argument("--ocr-min-conf", type=float, default=default_min_conf,
                    help=f"Minimum PaddleOCR confidence for a title line "
                         f"(default: {default_min_conf})")
    ap.add_argument("--title-distance", type=float, default=default_title_dist,
                    help=f"Caption search radius in PDF points "
                         f"(default: {default_title_dist})")
    ap.add_argument("--table-text-ratio", type=float,
                    default=default_table_text,
                    help=f"Text coverage required to call a region a table "
                         f"(default: {default_table_text})")
    ap.add_argument("--table-overlap", type=float,
                    default=default_table_overlap,
                    help=f"Cluster-vs-table overlap above which a cluster is "
                         f"dropped (default: {default_table_overlap})")

    ap.add_argument("--log-level", default=default_log_level,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help=f"Log level (default: {default_log_level})")
    ap.add_argument("--quiet", action="store_true",
                    help="Shorthand for --log-level WARNING")

    args = ap.parse_args(argv)

    _setup_logging("WARNING" if args.quiet else args.log_level)

    if not args.pdf.exists():
        log.error("PDF not found: %s", args.pdf)
        return 1
    if not args.pdf.is_file():
        log.error("Not a file: %s", args.pdf)
        return 1

    out_dir = args.output_dir or (args.pdf.parent / "Data" / args.pdf.stem)

    try:
        extract_images(
            args.pdf, out_dir,
            use_ocr=not args.no_ocr,
            include_vectors=not args.no_vectors,
            workers=args.workers,
            min_image_pixels=args.min_image_pixels,
            render_dpi=args.render_dpi,
            ocr_min_conf=args.ocr_min_conf,
            title_distance=args.title_distance,
            table_text_ratio=args.table_text_ratio,
            table_overlap_threshold=args.table_overlap,
            exclude_tables=not args.keep_tables,
        )
    except KeyboardInterrupt:
        log.warning("Cancelled.")
        return 130
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface anything else with a clean exit
        log.exception("Fatal: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
