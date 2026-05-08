from __future__ import annotations

import argparse
import base64
import json
import re as _re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))


# CV pre-filter tunables — a connected component is TEXT if ALL hold:
_CV_TEXT_MIN_H   = 5      # noise floor
_CV_TEXT_MAX_H   = 60     # catches large bold headings
_CV_TEXT_MAX_ASP = 12.0   # very wide thin rules
_CV_TEXT_MIN_ASP = 0.15   # very tall thin leader lines
_CV_TEXT_MAX_AREA = 4500  # too big to be a single glyph

# Minimum spatial spread of non-text pixels to count as a real diagram
_CV_MIN_BBOX_W  = 80
_CV_MIN_BBOX_H  = 80
# Minimum graphic-pixel density inside that bbox (prevents TOC-bullet false positives)
_CV_MIN_DENSITY = 0.004
# Dilate text mask by this many px so adjacent glyphs don't leave ink islands
_CV_DILATE_PX   = 4


def _cv_ink_mask(gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return m


def _cv_text_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if hh < _CV_TEXT_MIN_H or hh > _CV_TEXT_MAX_H:
            continue
        asp = ww / max(hh, 1)
        if asp > _CV_TEXT_MAX_ASP or asp < _CV_TEXT_MIN_ASP:
            continue
        if area > _CV_TEXT_MAX_AREA:
            continue
        mask[labels == i] = 255
    if _CV_DILATE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (_CV_DILATE_PX * 2 + 1, _CV_DILATE_PX * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


def _cv_prefilter(raw: bytes) -> bytes | None:
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return raw  # undecodable — let LLM decide

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink     = _cv_ink_mask(gray)
    text    = _cv_text_mask(gray)
    graphic = cv2.bitwise_and(ink, cv2.bitwise_not(text))

    graphic_px = int((graphic > 0).sum())
    if graphic_px < 200:
        return None

    ys, xs = np.where(graphic > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    if (x1 - x0 + 1) < _CV_MIN_BBOX_W or (y1 - y0 + 1) < _CV_MIN_BBOX_H:
        return None

    if graphic_px / float((x1 - x0 + 1) * (y1 - y0 + 1)) < _CV_MIN_DENSITY:
        return None  # scattered TOC bullets, not a diagram

    return raw


VISION_MODEL          = "gpt-4o-mini"
RETRY_ATTEMPTS        = 5
RETRY_DELAY_S         = 3.0
RETRY_BUFFER_S        = 1.0
BATCH_SIZE            = 10
BATCH_PAUSE_S         = 6.0
REQUEST_DELAY_S       = 0.5
IMAGE_EXTS            = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PRECISION_CROP_MODE   = True
SPLIT_RATIO_THRESHOLD = 2.5

_SYSTEM_PROMPT = """\
You are a quality filter for images extracted from BSA motorcycle documents.
Documents include: service manuals, workshop manuals, owner's manuals,
parts catalogues, wiring diagrams, AND Standard Operating Procedures (SOPs)
for diagnostic software tools.

Your job: decide if each image is USEFUL or NOT USEFUL to show users in a chat.

USEFUL — keep ALL of these:
• Technical diagrams with numbered callouts (motorcycle parts)
• Photos of the motorcycle, engine, components, or physical tools
• Schematic / wiring drawings
• Screenshots of diagnostic software / apps (e.g. CLPL Diagnostic Tester,
  Flash ECU screens, DTC screens, parameter screens) — these are STEP
  instructions, always keep them even if they look like "just text"
• Photos of a phone/tablet/device showing a software interface
• UI screenshots showing menus, buttons, selection screens, or settings
• Partial screenshots showing a file name, firmware version, or specific
  item to select — these are specific step instructions, always keep them
• Instrument panel diagrams or warning symbol diagrams

CRITICAL RULE: If ANY part of the image contains a real diagram, photo,
software screenshot, or device photo — even small — classify as USEFUL.
Only classify as NOT USEFUL when there is ZERO visual content.

NOT USEFUL — only skip these (100% sure cases):
• Table of contents pages (plain bullet list of chapter names, zero visuals)
• Pages with ONLY plain text paragraphs and absolutely no image/screenshot/diagram
• Specification tables (rows of numbers only: torque, dimensions, nothing else)
• VIN/engine-number decode grids (plain columns of letters/numbers, no visuals)
• Solid colour backgrounds with no content
• Single decorative icons alone: small arrows, bullet dots, brand logos on their own

Respond with ONLY valid JSON — no markdown, no extra text:
{"useful": true | false, "reason": "one short sentence"}
"""

_PRECISION_SYSTEM_PROMPT = """\
You are a diagram extractor for images from BSA motorcycle service manuals.
Find the tightest bounding box that covers ONLY the visual diagram or photo —
the actual drawing/photograph itself, nothing else.

EXCLUDE from the bounding box:
• Heading text above the diagram
• Explanatory text paragraphs beside the diagram
• Numbered label lists BELOW the diagram

INCLUDE in the bounding box:
• The actual drawing, photo or schematic itself
• Callout arrows and circled numbers overlaid on the diagram
• The diagram border/box if it has one

If the whole image is just a diagram with no text, return the full image dimensions.
If there is no diagram at all, set useful=false.

Respond with ONLY valid JSON — no markdown:
{"useful": true|false, "x1": int, "y1": int, "x2": int, "y2": int, "reason": "one sentence"}
x1,y1 = top-left, x2,y2 = bottom-right (pixel coordinates).
"""

_USER_PROMPT           = "Classify this image. JSON only."
_PRECISION_USER_PROMPT = "Image size: {w}x{h}px. Extract the diagram bounding box. JSON only."


def _with_retry(fn, label: str = "", verbose: bool = False):
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == RETRY_ATTEMPTS:
                raise
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                m = _re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", err)
                if m:
                    val  = float(m.group(1))
                    wait = (val / 1000.0 if m.group(2) == "ms" else val) + RETRY_BUFFER_S
                else:
                    wait = RETRY_DELAY_S * attempt
                if verbose:
                    print(f"  [429]{' ' + label if label else ''} — "
                          f"waiting {wait:.1f}s (attempt {attempt}/{RETRY_ATTEMPTS})")
                time.sleep(wait)
            else:
                time.sleep(RETRY_DELAY_S)


def _call_llm(client: OpenAI, raw: bytes, filename: str) -> dict:
    b64  = base64.b64encode(raw).decode("ascii")
    ext  = Path(filename).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp"}.get(ext, "image/png")
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}},
                {"type": "text", "text": _USER_PROMPT},
            ]},
        ],
        max_tokens=80, temperature=0,
    )
    raw_text = (resp.choices[0].message.content or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(raw_text)
    return {"useful": bool(parsed.get("useful", False)),
            "reason": str(parsed.get("reason", ""))}


def _call_llm_precision(client: OpenAI, raw: bytes, filename: str,
                        img_w: int, img_h: int) -> dict:
    b64  = base64.b64encode(raw).decode("ascii")
    ext  = Path(filename).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp"}.get(ext, "image/png")
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": _PRECISION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                {"type": "text",
                 "text": _PRECISION_USER_PROMPT.format(w=img_w, h=img_h)},
            ]},
        ],
        max_tokens=120, temperature=0,
    )
    raw_text = (resp.choices[0].message.content or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(raw_text)
    return {"useful": bool(parsed.get("useful", True)),
            "reason": str(parsed.get("reason", "")),
            "x1": int(parsed.get("x1", 0)),     "y1": int(parsed.get("y1", 0)),
            "x2": int(parsed.get("x2", img_w)), "y2": int(parsed.get("y2", img_h))}


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return m


def _text_mask_dilated(gray: np.ndarray) -> np.ndarray:
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    mask = np.zeros_like(gray)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if hh < 5 or hh > 38:
            continue
        asp = ww / max(hh, 1)
        if asp > 12.0 or asp < 0.15:
            continue
        if area > 4500:
            continue
        mask[labels == i] = 255
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))


def _graphic_density(region: np.ndarray, text_mask: np.ndarray) -> float:
    total = region.shape[0] * region.shape[1]
    if total == 0:
        return 0.0
    graphic = cv2.bitwise_and(_ink_mask(region), cv2.bitwise_not(text_mask))
    return float((graphic > 0).sum()) / total


def _trim_text_border(raw: bytes) -> bytes:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return raw
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h       = gray.shape[0]
    ink     = _ink_mask(gray)
    text    = _text_mask_dilated(gray)
    graphic = cv2.bitwise_and(ink, cv2.bitwise_not(text))

    bottom = h
    for y in range(h - 1, h // 3, -1):
        row_ink = int((ink[y] > 0).sum())
        if row_ink < 3:
            continue
        if int((graphic[y] > 0).sum()) / max(row_ink, 1) > 0.12:
            bottom = y + 1
            break

    top = 0
    for y in range(0, h * 2 // 3):
        row_ink = int((ink[y] > 0).sum())
        if row_ink < 3:
            continue
        if int((graphic[y] > 0).sum()) / max(row_ink, 1) > 0.12:
            top = max(0, y - 4)
            break

    if top == 0 and bottom == h:
        return raw
    margin  = 6
    cropped = img[max(0, top - margin):min(h, bottom + margin), :]
    ok, buf = cv2.imencode(".png", cropped, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    return buf.tobytes() if ok else raw


def smart_crop(raw: bytes) -> bytes:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return raw
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if float((_ink_mask(gray) > 0).sum()) / (h * w) > 0.35:
        return raw  # colour photo — keep as-is

    text_mask  = _text_mask_dilated(gray)
    best_crop: tuple[int, int, int, int] | None = None
    best_ratio = SPLIT_RATIO_THRESHOLD

    for frac in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        sx = int(w * frac)
        if sx < 40 or sx > w - 40:
            continue
        dl = _graphic_density(gray[:, :sx], text_mask[:, :sx])
        dr = _graphic_density(gray[:, sx:], text_mask[:, sx:])
        if dl > 0 and dr > 0:
            if dl / dr >= best_ratio:
                best_ratio, best_crop = dl / dr, (0, 0, sx, h)
            elif dr / dl >= best_ratio:
                best_ratio, best_crop = dr / dl, (sx, 0, w, h)

    for frac in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        sy = int(h * frac)
        if sy < 40 or sy > h - 40:
            continue
        dt = _graphic_density(gray[:sy, :], text_mask[:sy, :])
        db = _graphic_density(gray[sy:, :], text_mask[sy:, :])
        if dt > 0 and db > 0:
            if dt / db >= best_ratio:
                best_ratio, best_crop = dt / db, (0, 0, w, sy)
            elif db / dt >= best_ratio:
                best_ratio, best_crop = db / dt, (0, sy, w, h)

    if best_crop is None:
        return raw
    x0, y0, x1, y1 = best_crop
    margin = 8
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(w, x1 + margin), min(h, y1 + margin)
    cropped = img[y0:y1, x0:x1]
    ok, buf = cv2.imencode(".png", cropped, [cv2.IMWRITE_PNG_COMPRESSION, 5])
    return buf.tobytes() if ok else raw


def filter_image(
    client: OpenAI,
    raw: bytes,
    filename: str = "image.png",
    verbose: bool = False,
) -> bytes | None:
    if not raw:
        return None

    # Stage 0: OpenCV pre-filter (free, instant)
    pre = _cv_prefilter(raw)
    if pre is None:
        if verbose:
            print(f"  [SKIP-CV] {filename} — OpenCV: no diagram content")
        return None

    # Stage 1: LLM semantic classification
    try:
        decision = _with_retry(
            lambda: _call_llm(client, raw, filename),
            label="classify", verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"  [warn] classify failed: {e} — keeping image")
        return raw

    if not decision["useful"]:
        if verbose:
            print(f"  [SKIP-LLM] {filename} — {decision['reason']}")
        return None

    if verbose:
        print(f"  [KEEP] {filename} — {decision['reason']}")

    # Stage 2: Precision crop
    if PRECISION_CROP_MODE:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            img_h, img_w = img.shape[:2]
            try:
                box = _with_retry(
                    lambda: _call_llm_precision(client, raw, filename, img_w, img_h),
                    label="precision", verbose=verbose,
                )
                if box["useful"]:
                    x1, y1 = max(0, box["x1"]), max(0, box["y1"])
                    x2, y2 = min(img_w, box["x2"]), min(img_h, box["y2"])
                    if x2 > x1 + 10 and y2 > y1 + 10:
                        cropped = img[y1:y2, x1:x2]
                        ok, buf = cv2.imencode(".png", cropped,
                                              [cv2.IMWRITE_PNG_COMPRESSION, 5])
                        if ok:
                            raw = buf.tobytes()
                            raw = _trim_text_border(raw)
            except Exception as e:
                if verbose:
                    print(f"  [warn] precision crop failed: {e} — using smart_crop")
                raw = smart_crop(raw)
    else:
        raw = smart_crop(raw)

    return raw


def process_folder(
    client: OpenAI,
    images_dir: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    images_dir = Path(images_dir)
    if output_dir is None:
        output_dir = images_dir / "cleaned"

    imgs = sorted(p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        print(f"No images found in {images_dir}")
        return {}

    est_cost  = len(imgs) * 135 * (0.000150 / 1000)
    n_batches = (len(imgs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nProcessing {len(imgs)} images in {images_dir}")
    print(f"Model : {VISION_MODEL}  |  Stage 0: OpenCV pre-filter (free)")
    print(f"Batches: {n_batches} x {BATCH_SIZE}  |  Est. cost: ${est_cost:.4f}\n")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(imgs), "kept": 0,
             "skipped_cv": 0, "skipped_llm": 0,
             "cropped": 0, "errors": 0}

    for i, img_path in enumerate(imgs, 1):
        raw = img_path.read_bytes()

        pre = _cv_prefilter(raw)
        if pre is None:
            print(f"[{i}/{len(imgs)}] SKIP-CV   {img_path.name}")
            stats["skipped_cv"] += 1
            continue

        try:
            result = filter_image(client, raw, img_path.name, verbose=verbose)
        except Exception as e:
            print(f"[{i}/{len(imgs)}] ERROR     {img_path.name}: {e}")
            stats["errors"] += 1
        else:
            if result is None:
                if not verbose:
                    print(f"[{i}/{len(imgs)}] SKIP-LLM  {img_path.name}")
                stats["skipped_llm"] += 1
            else:
                if not verbose:
                    print(f"[{i}/{len(imgs)}] KEEP      {img_path.name}")
                stats["kept"] += 1
                if result != raw:
                    stats["cropped"] += 1
                if not dry_run:
                    (output_dir / (img_path.stem + ".png")).write_bytes(result)

        if i < len(imgs):
            if i % BATCH_SIZE == 0:
                print(f"\n  [batch {i // BATCH_SIZE}/{n_batches} done] "
                      f"pausing {BATCH_PAUSE_S}s ...\n")
                time.sleep(BATCH_PAUSE_S)
            else:
                time.sleep(REQUEST_DELAY_S)

    total_skipped = stats["skipped_cv"] + stats["skipped_llm"]
    print(f"\nDone — kept: {stats['kept']}  (cropped: {stats['cropped']})"
          f"  skipped: {total_skipped}"
          f"  (CV: {stats['skipped_cv']}, LLM: {stats['skipped_llm']})"
          f"  errors: {stats['errors']}  total: {stats['total']}")
    if not dry_run and stats["kept"] > 0:
        print(f"Output: {output_dir}")
    return stats


def main() -> int:
    global PRECISION_CROP_MODE, VISION_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--model",        default=VISION_MODEL)
    ap.add_argument("--quiet",        action="store_true")
    ap.add_argument("--no-precision", action="store_true")
    args = ap.parse_args()

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in .env", file=sys.stderr)
        return 1

    if args.no_precision:
        PRECISION_CROP_MODE = False
    VISION_MODEL = args.model

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    target = Path(args.path)

    if target.is_dir():
        process_folder(client, target, Path(args.output) if args.output else None,
                       dry_run=args.dry_run, verbose=not args.quiet)
    elif target.is_file():
        raw    = target.read_bytes()
        result = filter_image(client, raw, target.name, verbose=True)
        if result is None:
            print(f"[SKIP] {target.name}")
        else:
            out = (Path(args.output) if args.output
                   else target.with_stem(target.stem + "_clean").with_suffix(".png"))
            if not args.dry_run:
                out.write_bytes(result)
                print(f"Saved to {out}")
    else:
        print(f"ERROR: {target} is not a file or directory", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
