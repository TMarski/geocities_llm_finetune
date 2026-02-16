#!/usr/bin/env python3
"""
Geocities HTML -> clean English text JSONL chunker (~2000 *approx* tokens).

Pipeline:
- Replace <br> variants with '\n' BEFORE parsing so line breaks are preserved.
- Strip all HTML tags -> plain text only.
- Remove URLs, image filenames, and email addresses by deleting any "token"
  (contiguous text bounded by whitespace/newlines) matching:
    - contains: http://, https://
    - contains any of: .gif .bmp .jpg .jpeg .tiff .png  (case-insensitive)
    - matches email-like: (text)@(text).(2-4 chars)
- Filter non-English pages using lingua-language-detector (module name: 'lingua').
- Chunk into ~2000 *approx* tokens using character-based approximation:
    1 token ≈ token_char_ratio chars (default 3.75)
  With ±2% elasticity to avoid chopping "sentences":
    - Prefer forward search up to +2% for '.' or '\n'
    - Else backward down to -2%
    - Else hard cut at target boundary
- Output JSONL entries: {"text": "..."} with embedded newlines preserved.

Requires:
  pip install beautifulsoup4 lxml lingua-language-detector

Example:
  python geocities_make_jsonl.py ^
    --input-dir "D:\\Extracted Geocities Text" ^
    --output-jsonl "D:\\geocities_english_clean.jsonl"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterator, Optional, List

from bs4 import BeautifulSoup  # type: ignore
from lingua import Language, LanguageDetectorBuilder  # type: ignore


# Matches <br>, <br/>, <br /> (any case), across messy HTML
BR_RE = re.compile(r"(?is)<\s*br\s*/?\s*>")

# Token-level removals (operate on whitespace-bounded "tokens")
# Any token containing one of these substrings is removed (case-insensitive handling is done in code).
BAD_SUBSTRINGS = (
    "http://",
    "https://",
    ".gif",
    ".bmp",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".png",
)

# Whitespace tokenization preserving separators:
# group 1 = whitespace, group 2 = non-whitespace token
SPLIT_KEEP_WS_RE = re.compile(r"(\s+)|(\S+)")

# Email-like token: something@something.tld (tld 2-4 chars)
# This is intentionally simple and token-scoped per your requirement.
EMAIL_TOKEN_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[A-Za-z]{2,4}$")


def decode_html_bytes(data: bytes) -> str:
    """
    Best-effort decode without extra deps.
    - Try UTF-8
    - Fallback to latin-1 (never fails)
    """
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def html_to_text(html: str) -> str:
    """
    - Replace <br> with newline before parsing.
    - Remove script/style/template/noscript content.
    - Extract plain text only (no HTML tags remain).
    - Preserve line-ish structure and normalize whitespace/newlines.
    """
    html = BR_RE.sub("\n", html)

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "template", "noscript"]):
        tag.decompose()

    # Separator '\n' retains “line breaks” reasonably well
    text = soup.get_text(separator="\n")

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim per-line trailing whitespace, keep newlines
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse huge blank-line runs (keep some structure)
    text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

    return text.strip()


def remove_bad_tokens(text: str) -> str:
    """
    Remove any whitespace-bounded token that:
      - contains URL substrings (http:// or https://)
      - contains common image extensions (.gif, .bmp, .jpg, .jpeg, .tiff, .png)
      - matches a simple email pattern: something@something.tld (2-4 alpha TLD)

    Keeps whitespace separators so line structure isn't destroyed, then normalizes.
    """
    parts = []
    for m in SPLIT_KEEP_WS_RE.finditer(text):
        ws = m.group(1)
        tok = m.group(2)

        if ws is not None:
            parts.append(ws)
            continue

        if tok is None:
            continue

        low = tok.lower()

        # Substring-based removal
        if any(sub in low for sub in BAD_SUBSTRINGS):
            continue

        # Email removal (token-scoped)
        if EMAIL_TOKEN_RE.match(tok):
            continue

        parts.append(tok)

    cleaned = "".join(parts)

    # Cleanup: collapse runs of spaces/tabs, keep newlines
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # Cleanup: trim trailing spaces per line
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))

    # Collapse huge blank-line runs again (token removal may create them)
    cleaned = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", cleaned)

    return cleaned.strip()


def _get_language_enum(name: str) -> Optional[Language]:
    key = name.strip().upper()
    return getattr(Language, key, None)


def build_language_detector(extra_langs: Optional[str] = None):
    """
    Version-safe language list (NORWEGIAN varies across lingua versions).
    Smaller list = faster and often more accurate for web text.
    """
    default_names = [
        "ENGLISH",
        "SPANISH",
        "FRENCH",
        "GERMAN",
        "ITALIAN",
        "PORTUGUESE",
        "DUTCH",
    ]

    langs: List[Language] = []
    for nm in default_names:
        lang = _get_language_enum(nm)
        if lang is not None:
            langs.append(lang)

    if extra_langs:
        for nm in extra_langs.split(","):
            lang = _get_language_enum(nm)
            if lang is not None and lang not in langs:
                langs.append(lang)

    if Language.ENGLISH not in langs:
        raise RuntimeError("Your lingua installation does not expose Language.ENGLISH")

    return LanguageDetectorBuilder.from_languages(*langs).build()


def is_english(detector, text: str, min_chars: int = 200) -> bool:
    """
    Detect English after HTML is stripped and token junk is removed.
    """
    t = text.strip()
    if len(t) < min_chars:
        return False

    sample = t[:8000]

    lang = detector.detect_language_of(sample)
    if lang != Language.ENGLISH:
        return False

    # Optional confidence threshold if available
    try:
        confs = detector.compute_language_confidence_values(sample)
        top = confs[0]
        return getattr(top, "language", None) == Language.ENGLISH and getattr(top, "value", 0.0) >= 0.70
    except Exception:
        return True


def chunk_text_with_elasticity(text: str, target_chars: int, fuzz_chars: int) -> Iterator[str]:
    """
    Chunk by characters with your boundary rule:
      - aim at target_chars
      - search forward up to target+fuzz for '.' or '\n'
      - else search backward down to target-fuzz
      - else hard cut at target_chars
    """
    i = 0
    n = len(text)

    while i < n:
        remaining = n - i
        if remaining <= target_chars:
            chunk = text[i:].strip()
            if chunk:
                yield chunk
            return

        base = i + target_chars
        cut: Optional[int] = None

        # forward search
        forward_limit = min(base + fuzz_chars, n - 1)
        for j in range(base, forward_limit + 1):
            if text[j] == "." or text[j] == "\n":
                cut = j + 1
                break

        # backward search
        if cut is None:
            backward_start = max(i + target_chars - fuzz_chars, i + 1)
            for j in range(base - 1, backward_start - 1, -1):
                if text[j] == "." or text[j] == "\n":
                    cut = j + 1
                    break

        if cut is None:
            cut = base

        chunk = text[i:cut].strip()
        if chunk:
            yield chunk

        i = cut


def iter_html_files(input_dir: Path) -> Iterator[Path]:
    for p in sorted(input_dir.rglob("*"), key=lambda x: x.as_posix().lower()):
        if p.is_file() and p.suffix.lower() in (".html", ".htm"):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Folder containing extracted .html/.htm files")
    ap.add_argument("--output-jsonl", required=True, help="Destination .jsonl file path")
    ap.add_argument("--min-chars", type=int, default=200, help="Min cleaned chars to attempt language detection")
    ap.add_argument("--target-tokens", type=int, default=2000, help="Target 'tokens' per chunk (approx)")
    ap.add_argument("--token-char-ratio", type=float, default=3.75, help="Approx chars per token (used for chunking)")
    ap.add_argument("--extra-langs", default="", help="Comma-separated lingua Language enum names to add (optional)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output_jsonl)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert the user's desired token rule into chars
    target_chars = int(round(args.target_tokens * args.token_char_ratio))
    fuzz_chars = int(round(target_chars * 0.02))  # 2%

    detector = build_language_detector(args.extra_langs)

    files = list(iter_html_files(input_dir))
    if not files:
        print(f"No .html/.htm found under {input_dir}")
        return

    total_files = 0
    kept_files = 0
    total_chunks = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out_f:
        for fp in files:
            total_files += 1

            try:
                raw = fp.read_bytes()
            except OSError as e:
                print(f"READ ERROR: {fp} -> {e}")
                continue

            html = decode_html_bytes(raw)
            text = html_to_text(html)

            # Remove URLs/images/emails as requested
            text = remove_bad_tokens(text)

            # Re-check: token removal may make it empty/too short
            if not is_english(detector, text, min_chars=args.min_chars):
                continue

            kept_files += 1

            for chunk in chunk_text_with_elasticity(text, target_chars=target_chars, fuzz_chars=fuzz_chars):
                out_f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                total_chunks += 1

            if kept_files % 500 == 0:
                print(f"Processed {total_files} files; kept {kept_files}; wrote {total_chunks} chunks")

    print("Done.")
    print(f"Target chars/chunk:  {target_chars} (≈ {args.target_tokens} tokens @ {args.token_char_ratio:.2f} chars/token)")
    print(f"Fuzz chars:          {fuzz_chars} (≈ 2%)")
    print(f"Total files scanned: {total_files}")
    print(f"English pages kept:  {kept_files}")
    print(f"JSONL chunks wrote:  {total_chunks}")
    print(f"Output JSONL:        {out_path}")


if __name__ == "__main__":
    main()
