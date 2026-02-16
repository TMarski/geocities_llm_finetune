#!/usr/bin/env python3
"""
Extract all .htm/.html files from a directory of .tar archives into a single
destination folder, renaming each extracted file to a random 32-character name
while preserving its original extension (.html or .htm).

- Processes .tar files in alphabetical order.
- Skips partial/multi-part archives like *.001, *.002, etc. (and anything not a .tar).
- Extracts .htm/.html from any subdirectories inside each tar.
- Does NOT preserve original paths, filenames, timestamps, or permissions.
"""

import os
import tarfile
import secrets
from pathlib import Path


ARCHIVE_DIR = r"D:\Geocities_Archive"
DEST_DIR = r"D:\Extracted Geocities Text"


def random_name_32() -> str:
    # 32 hex chars = 16 random bytes -> 32-character lowercase hex string
    return secrets.token_hex(16)


def is_html_path(name: str) -> bool:
    # name inside the tar is always a forward-slash-ish path; treat it as a string.
    lower = name.lower()
    return lower.endswith(".html") or lower.endswith(".htm")


def safe_ext(name: str) -> str:
    # Preserve original extension exactly as .html or .htm (lowercased).
    lower = name.lower()
    if lower.endswith(".html"):
        return ".html"
    if lower.endswith(".htm"):
        return ".htm"
    return ""


def iter_tar_files_sorted(archive_dir: Path):
    # Only real ".tar" files. This inherently skips *.tar.001 etc.
    # (Those end with ".001", not ".tar".)
    for p in sorted(archive_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() == ".tar":
            yield p


def write_member_to_dest(tf: tarfile.TarFile, member: tarfile.TarInfo, dest_dir: Path) -> Path | None:
    # Skip non-regular files (directories, symlinks, devices, etc.)
    if not member.isreg():
        return None

    if not is_html_path(member.name):
        return None

    ext = safe_ext(member.name)
    if not ext:
        return None

    src_f = tf.extractfile(member)
    if src_f is None:
        return None

    # Generate a unique random filename; extremely unlikely to collide, but loop anyway.
    while True:
        out_path = dest_dir / f"{random_name_32()}{ext}"
        if not out_path.exists():
            break

    # Stream copy to disk (no metadata preserved).
    with src_f, open(out_path, "wb") as out_f:
        # Chunked copy keeps memory usage low.
        for chunk in iter(lambda: src_f.read(1024 * 1024), b""):
            out_f.write(chunk)

    return out_path


def main():
    archive_dir = Path(ARCHIVE_DIR)
    dest_dir = Path(DEST_DIR)

    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    tar_paths = list(iter_tar_files_sorted(archive_dir))
    if not tar_paths:
        print(f"No .tar files found in: {archive_dir}")
        return

    extracted_count = 0
    tar_count = 0
    errors = 0

    for tar_path in tar_paths:
        tar_count += 1
        print(f"[{tar_count}/{len(tar_paths)}] Processing: {tar_path.name}")

        try:
            # Use streaming mode to avoid reading tar index into memory (works for regular tar files).
            # If some tars are random-access-friendly, tarfile can still handle it.
            with tarfile.open(tar_path, mode="r:*") as tf:
                for member in tf:
                    try:
                        out_path = write_member_to_dest(tf, member, dest_dir)
                        if out_path:
                            extracted_count += 1
                            # Print occasionally to reduce spam; change frequency as desired.
                            if extracted_count % 500 == 0:
                                print(f"  Extracted so far: {extracted_count}")
                    except (tarfile.TarError, OSError) as e:
                        errors += 1
                        # Continue past individual file errors
                        print(f"  ! Error extracting member {member.name!r} from {tar_path.name}: {e}")
        except (tarfile.TarError, OSError) as e:
            errors += 1
            print(f"  !! Error opening/reading tar {tar_path.name}: {e}")

    print("\nDone.")
    print(f"  Tars processed:   {tar_count}")
    print(f"  Files extracted:  {extracted_count}")
    print(f"  Errors:           {errors}")
    print(f"  Output directory: {dest_dir}")


if __name__ == "__main__":
    main()
