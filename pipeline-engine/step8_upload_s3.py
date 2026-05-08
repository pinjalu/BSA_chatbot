from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def find_image_folders(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("images") if p.is_dir())


def iter_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def s3_key_for(file_path: Path, project_root: Path, prefix: str,
               strip_leading: str = "") -> str:
    """Project-relative POSIX path, optionally prefixed.

    If `strip_leading` is set (e.g. "Data/"), it's removed from the start
    of the relative path BEFORE the prefix is prepended. Use this to
    collapse the on-disk Data/Data/... layout down to Data/... in S3.
    """
    rel = file_path.resolve().relative_to(project_root.resolve()).as_posix()
    if strip_leading:
        sl = strip_leading.lstrip("/").rstrip("/") + "/"
        if rel.startswith(sl):
            rel = rel[len(sl):]
    if prefix:
        return f"{prefix.rstrip('/')}/{rel}"
    return rel


def key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def upload_folder(
    s3, bucket: str, folder: Path, project_root: Path, prefix: str,
    overwrite: bool, public: bool, dry_run: bool,
    strip_leading: str = "",
    remaining: int | None = None,
) -> tuple[int, int, int]:
    """`remaining` caps how many uploads this call may perform across all
    files; once hit, the rest of the folder is skipped silently.
    """
    files = iter_images(folder)
    if not files:
        print(f"  (no images in {folder})")
        return 0, 0, 0

    rel_folder = folder.resolve().relative_to(project_root.resolve()).as_posix()
    print(f"\n[folder] {rel_folder}  ({len(files)} files)")

    uploaded = skipped = failed = 0
    for f in files:
        if remaining is not None and uploaded >= remaining:
            break
        key = s3_key_for(f, project_root, prefix, strip_leading)

        if not overwrite and not dry_run and key_exists(s3, bucket, key):
            skipped += 1
            print(f"  - skip (exists)  {key}")
            continue

        if dry_run:
            print(f"  + would upload   s3://{bucket}/{key}")
            uploaded += 1
            continue

        ctype, _ = mimetypes.guess_type(f.name)
        extra = {"ContentType": ctype} if ctype else {}
        if public:
            extra["ACL"] = "public-read"

        try:
            s3.upload_file(str(f), bucket, key, ExtraArgs=extra)
        except ClientError as e:
            failed += 1
            print(f"  ! failed         {key}: {e}", file=sys.stderr)
            continue

        uploaded += 1
        print(f"  + uploaded       s3://{bucket}/{key}")

    return uploaded, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default="Data",
                    help="Folder to scan for images/ subfolders (default: Data)")
    ap.add_argument("--folder", default=None,
                    help="Upload one specific images/ folder (skips scan)")
    ap.add_argument("--file", default=None,
                    help="Upload one specific image file (skips scan)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Upload at most N files in total (useful for testing)")
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET"),
                    help="S3 bucket name (default: $S3_BUCKET from .env)")
    ap.add_argument("--region", default=None,
                    help="Override AWS region (default: from `aws configure`)")
    ap.add_argument("--prefix", default="",
                    help="Optional S3 key prefix, e.g. 'manuals/'")
    ap.add_argument("--strip-leading", default="Data/",
                    help="Strip this prefix from the relative path so on-disk "
                         "Data/Data/... becomes Data/... in S3 (default: 'Data/'). "
                         "Pass '' to disable.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-upload even if the key already exists")
    ap.add_argument("--public", action="store_true",
                    help="Set ACL=public-read on uploaded objects")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be uploaded, no API calls")
    args = ap.parse_args()

    if not args.bucket:
        print("ERROR: set S3_BUCKET in .env or pass --bucket", file=sys.stderr)
        return 1

    project_root = Path(__file__).parent.resolve()

    if args.file:
        f = Path(args.file).resolve()
        if not f.is_file():
            print(f"ERROR: not a file: {args.file}", file=sys.stderr)
            return 1
        s3 = boto3.client("s3", region_name=args.region) if args.region \
            else boto3.client("s3")
        key = s3_key_for(f, project_root, args.prefix, args.strip_leading)
        if args.dry_run:
            print(f"+ would upload   s3://{args.bucket}/{key}")
            return 0
        ctype, _ = mimetypes.guess_type(f.name)
        extra = {"ContentType": ctype} if ctype else {}
        if args.public:
            extra["ACL"] = "public-read"
        s3.upload_file(str(f), args.bucket, key, ExtraArgs=extra)
        print(f"+ uploaded       s3://{args.bucket}/{key}")
        return 0

    if args.folder:
        folders = [Path(args.folder).resolve()]
        if not folders[0].is_dir():
            print(f"ERROR: not a directory: {args.folder}", file=sys.stderr)
            return 1
    else:
        scan_root = (project_root / args.root) if not Path(args.root).is_absolute() \
            else Path(args.root)
        if not scan_root.is_dir():
            print(f"ERROR: scan root not found: {scan_root}", file=sys.stderr)
            return 1
        folders = find_image_folders(scan_root)

    if not folders:
        print("no images/ folders found")
        return 0

    print(f"[scan] {len(folders)} images/ folder(s)")
    for f in folders:
        print(f"   {f.resolve().relative_to(project_root)}")

    s3 = boto3.client("s3", region_name=args.region) if args.region \
        else boto3.client("s3")

    total_up = total_skip = total_fail = 0
    for folder in folders:
        if args.limit is not None and total_up >= args.limit:
            print(f"\n[limit] reached --limit={args.limit}, stopping")
            break
        u, s, f = upload_folder(
            s3, args.bucket, folder, project_root, args.prefix,
            overwrite=args.overwrite, public=args.public, dry_run=args.dry_run,
            strip_leading=args.strip_leading,
            remaining=(args.limit - total_up) if args.limit is not None else None,
        )
        total_up += u
        total_skip += s
        total_fail += f

    print(
        f"\n[done] uploaded={total_up}  skipped={total_skip}  "
        f"failed={total_fail}  bucket={args.bucket}"
        f"{'  (dry-run)' if args.dry_run else ''}"
    )
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
