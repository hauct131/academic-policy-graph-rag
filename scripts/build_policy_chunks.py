#!/usr/bin/env python3
"""
scripts/01_build_policy_chunks.py

First chunking pipeline for the Academic Policy Graph RAG project.

Reads cleaned Markdown files from data/raw/cleaned/, parses YAML frontmatter,
splits each document into policy-section chunks, and writes JSONL output.

Usage:
    python scripts/01_build_policy_chunks.py [--input-dir ...] [--output-file ...] [--min-chars ...]
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML-like frontmatter delimited by '---' lines.

    Returns (metadata_dict, body_text).  Fields with simple scalar values
    (string, no special YAML constructs) are supported.  Multi-line values
    and complex YAML types are returned as raw strings.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, text

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:])

    meta: dict[str, Any] = {}
    current_key: str | None = None
    current_val_parts: list[str] = []

    def flush_key() -> None:
        if current_key is not None:
            val = " ".join(current_val_parts).strip()
            # Strip surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            meta[current_key] = val

    for line in fm_lines:
        # Match "key: value" or "key:" patterns
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)', line)
        if m:
            flush_key()
            current_key = m.group(1)
            current_val_parts = [m.group(2)]
        elif current_key is not None and (line.startswith("  ") or line.startswith("\t")):
            current_val_parts.append(line.strip())
        else:
            flush_key()
            current_key = None
            current_val_parts = []

    flush_key()
    return meta, body


# ---------------------------------------------------------------------------
# ASCII-safe slug helpers
# ---------------------------------------------------------------------------

def _remove_accents(text: str) -> str:
    """Remove Vietnamese and Latin diacritics using stdlib only.

    Strategy:
    1. Handle đ/Đ explicitly (NFD decomposition does not strip these).
    2. NFD-decompose the rest so that combining diacritical marks become
       separate code-points with Unicode category 'Mn'.
    3. Drop all 'Mn' (non-spacing mark) characters.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def slugify(text: str) -> str:
    """Convert arbitrary text to a lowercase ASCII slug with underscores."""
    text = _remove_accents(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


# ---------------------------------------------------------------------------
# Heading detection helpers
# ---------------------------------------------------------------------------

# Matches: "### Điều 9. ..." or "## Điều 9. ..."
_DIEU_RE = re.compile(r'^(#{2,3})\s+Điều\s+(\d+)[\.\.]?\s*(.*)', re.UNICODE)

# Matches chapter headings at ## level
_CHAPTER_RE = re.compile(
    r'^##\s+(Chương\s+[IVXLC\d]+[\.\.]?\s*.+)',
    re.UNICODE | re.IGNORECASE
)

# Matches "## Phụ lục ..." (appendix)
_PHU_LUC_RE = re.compile(r'^##\s+(Phụ lục\s*\w*[\.\s]*.+)', re.UNICODE | re.IGNORECASE)

# Appendix sub-headings to split out individually
_APPENDIX_SUB_RE = re.compile(
    r'^###\s+(Nơi cấp chứng chỉ|Thời hạn chứng chỉ|Cách tính điểm thi|Bảng quy đổi .+)',
    re.UNICODE | re.IGNORECASE
)


def _extract_dieu_number(heading_text: str) -> str | None:
    """Extract numeric part from a '### Điều N. ...' heading."""
    m = _DIEU_RE.match(heading_text)
    if m:
        return m.group(2)
    return None


# ---------------------------------------------------------------------------
# Chunk ID generators
# ---------------------------------------------------------------------------

def make_dieu_chunk_id(doc_id: str, number: str) -> str:
    return f"{doc_id}__dieu_{number}"


def make_appendix_chunk_id(doc_id: str, appendix_label: str, sub: str | None = None) -> str:
    """
    appendix_label: e.g. 'I', 'II', or raw heading text
    sub: optional sub-section name
    """
    label_slug = slugify(appendix_label)
    base = f"{doc_id}__phu_luc_{label_slug}"
    if sub:
        return f"{base}__{slugify(sub)}"
    return base


# ---------------------------------------------------------------------------
# Document splitter
# ---------------------------------------------------------------------------

def _count_words(text: str) -> int:
    return len(text.split())


def split_document_into_chunks(
    body: str,
    meta: dict[str, Any],
    source_path: str,
    min_chars: int,
) -> list[dict[str, Any]]:
    """
    Split a policy document body into chunks.

    Splitting strategy:
    - Track current Chương (chapter) heading for chapter_title.
    - Each "### Điều N." or "## Điều N." heading starts a new Điều chunk.
    - "## Phụ lục ..." starts a new appendix chunk.
    - Appendix sub-headings ("### Nơi cấp chứng chỉ", "### Thời hạn chứng chỉ", etc.)
      start sub-chunks within the appendix.
    - The heading line is kept inside the chunk text.
    - Chunks shorter than min_chars (based on text content) are merged upward
      into the previous chunk unless they are the first chunk.
    """
    doc_id: str = meta.get("doc_id", slugify(source_path))
    title: str = meta.get("title", "")
    decision_no: str = meta.get("decision_no", "")
    issued_date: str = meta.get("issued_date", "")
    institution: str = meta.get("institution", "")
    education_mode: str = meta.get("education_mode", "")
    source_pdf: str = meta.get("source_pdf", meta.get("parsed_source", meta.get("ocr_source", "")))

    lines = body.split("\n")

    # We build a list of raw "segments": (chunk_type, heading_line, content_lines, chapter_title)
    # then post-process to merge short ones.

    segments: list[dict[str, Any]] = []

    current_chapter_title: str = ""
    current_lines: list[str] = []
    current_heading: str = ""
    current_type: str = "preamble"
    current_sub: str | None = None   # appendix sub label
    in_appendix: bool = False
    appendix_label: str = ""

    def flush_segment() -> None:
        nonlocal current_lines, current_heading, current_type, current_sub
        text_block = "\n".join(current_lines).strip()
        if not text_block and not current_heading:
            current_lines = []
            return
        segments.append({
            "chunk_type": current_type,
            "heading": current_heading,
            "chapter_title": current_chapter_title,
            "text_lines": list(current_lines),
            "appendix_label": appendix_label,
            "appendix_sub": current_sub,
        })
        current_lines = []

    for line in lines:
        stripped = line.strip()

        # --- Chapter heading ---
        m_chap = _CHAPTER_RE.match(stripped)
        if m_chap:
            flush_segment()
            current_chapter_title = m_chap.group(1).strip()
            current_heading = stripped
            current_type = "chapter_header"
            current_lines = [stripped]
            in_appendix = False
            continue

        # --- Phụ lục heading ---
        m_pl = _PHU_LUC_RE.match(stripped)
        if m_pl:
            flush_segment()
            # Extract appendix label (e.g. "I" from "Phụ lục I. ...")
            full = m_pl.group(1).strip()
            # Try to get roman numeral or letter label
            m_label = re.search(r'Phụ lục\s+([IVXLC\d]+)', full, re.IGNORECASE | re.UNICODE)
            appendix_label = m_label.group(1) if m_label else slugify(full)
            current_heading = stripped
            current_type = "appendix"
            current_sub = None
            current_lines = [stripped]
            in_appendix = True
            continue

        # --- Appendix sub-heading ---
        if in_appendix:
            m_sub = _APPENDIX_SUB_RE.match(stripped)
            if m_sub:
                flush_segment()
                current_sub = m_sub.group(1).strip()
                current_heading = stripped
                current_type = "appendix_sub"
                current_lines = [stripped]
                continue

        # --- Điều heading ---
        m_dieu = _DIEU_RE.match(stripped)
        if m_dieu:
            flush_segment()
            current_heading = stripped
            current_type = "dieu"
            current_sub = None
            in_appendix = False
            current_lines = [stripped]
            continue

        # Regular content line
        current_lines.append(line)

    # Flush last segment
    flush_segment()

    # -----------------------------------------------------------------------
    # Convert segments to chunk objects; merge tiny segments upward
    # -----------------------------------------------------------------------
    chunks: list[dict[str, Any]] = []

    for seg in segments:
        text = "\n".join(seg["text_lines"]).strip()
        if not text:
            continue

        heading = seg["heading"]
        chapter_title = seg["chapter_title"]
        chunk_type = seg["chunk_type"]
        seg_appendix_label = seg["appendix_label"]
        appendix_sub = seg["appendix_sub"]

        # Build section_title and section_number
        section_title = ""
        section_number = ""

        if chunk_type == "dieu":
            m_dieu = _DIEU_RE.match(heading)
            if m_dieu:
                section_number = m_dieu.group(2)
                section_title = f"Điều {m_dieu.group(2)}. {m_dieu.group(3)}".strip()
                section_title = re.sub(r'\.\s*$', '', section_title)
            chunk_id = make_dieu_chunk_id(doc_id, section_number)

        elif chunk_type == "appendix":
            section_title = heading.lstrip("#").strip()
            chunk_id = make_appendix_chunk_id(doc_id, seg_appendix_label)

        elif chunk_type == "appendix_sub":
            section_title = appendix_sub or heading.lstrip("#").strip()
            chunk_id = make_appendix_chunk_id(doc_id, seg_appendix_label, sub=section_title)

        elif chunk_type == "chapter_header":
            section_title = heading.lstrip("#").strip()
            chunk_id = f"{doc_id}__chuong_{slugify(section_title)}"

        else:  # preamble / misc
            section_title = heading.lstrip("#").strip() if heading else ""
            chunk_id = f"{doc_id}__preamble"

        char_count = len(text)

        # If below min_chars, merge into previous chunk (if any)
        if char_count < min_chars and chunks:
            prev = chunks[-1]
            prev["text"] = prev["text"].rstrip() + "\n\n" + text
            prev["char_count"] = len(prev["text"])
            prev["word_count"] = _count_words(prev["text"])
            continue

        chunk: dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "title": title,
            "decision_no": decision_no,
            "issued_date": issued_date,
            "institution": institution,
            "education_mode": education_mode,
            "chapter_title": chapter_title,
            "section_title": section_title,
            "section_number": section_number,
            "chunk_type": chunk_type,
            "source_path": source_path,
            "source_pdf": source_pdf,
            "text": text,
            "char_count": char_count,
            "word_count": _count_words(text),
        }
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_chunks(
    input_dir: str,
    output_file: str,
    min_chars: int,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_path.glob("*.md"))
    if not md_files:
        print(f"[WARN] No .md files found in {input_dir}", file=sys.stderr)
        return

    all_chunks: list[dict[str, Any]] = []
    per_doc: dict[str, int] = {}

    for md_file in md_files:
        rel_path = str(md_file)
        raw_text = md_file.read_text(encoding="utf-8")

        meta, body = parse_frontmatter(raw_text)
        chunks = split_document_into_chunks(body, meta, rel_path, min_chars)

        doc_id = meta.get("doc_id", md_file.stem)
        per_doc[doc_id] = len(chunks)
        all_chunks.extend(chunks)

    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Summary
    print("=" * 60)
    print("Policy chunk build complete")
    print("=" * 60)
    print(f"  Source files : {len(md_files)}")
    print(f"  Total chunks : {len(all_chunks)}")
    print(f"  Output       : {output_path.resolve()}")
    print()
    print("  Chunks per document:")
    for doc_id, count in per_doc.items():
        print(f"    {doc_id}: {count}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build policy chunks JSONL from cleaned Markdown files."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/cleaned",
        help="Directory containing cleaned .md files (default: data/raw/cleaned)",
    )
    parser.add_argument(
        "--output-file",
        default="data/chunks/policy_chunks.jsonl",
        help="Output JSONL file path (default: data/chunks/policy_chunks.jsonl)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Minimum characters for a chunk to be kept standalone (default: 80)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    build_chunks(
        input_dir=args.input_dir,
        output_file=args.output_file,
        min_chars=args.min_chars,
    )
