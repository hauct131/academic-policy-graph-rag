"""
tests/test_policy_chunking.py

Unit tests for the policy chunking pipeline.

Run with:
    python -m pytest
"""

import json
import textwrap
from pathlib import Path

import pytest

# Import the module under test
import sys
import os

# Ensure the project root is on sys.path so the script can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import build_policy_chunks as _mod

parse_frontmatter = _mod.parse_frontmatter
split_document_into_chunks = _mod.split_document_into_chunks
make_dieu_chunk_id = _mod.make_dieu_chunk_id
make_appendix_chunk_id = _mod.make_appendix_chunk_id
slugify = _mod.slugify
build_chunks = _mod.build_chunks


# ---------------------------------------------------------------------------
# Frontmatter parsing tests
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        text = textwrap.dedent("""\
            ---
            doc_id: test_doc
            title: Test Title
            issued_date: 2023-01-01
            ---
            Body text here.
        """)
        meta, body = parse_frontmatter(text)
        assert meta["doc_id"] == "test_doc"
        assert meta["title"] == "Test Title"
        assert meta["issued_date"] == "2023-01-01"
        assert "Body text here." in body

    def test_quoted_value(self):
        text = textwrap.dedent("""\
            ---
            notes: "Some quoted note with spaces"
            ---
            Content
        """)
        meta, body = parse_frontmatter(text)
        assert meta["notes"] == "Some quoted note with spaces"

    def test_no_frontmatter(self):
        text = "Just a document body."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_missing_closing_delimiter(self):
        text = "---\nkey: value\nNo closing delimiter"
        meta, body = parse_frontmatter(text)
        assert meta == {}

    def test_all_required_fields(self):
        text = textwrap.dedent("""\
            ---
            doc_id: ou_fulltime_credit_training_regulation_2016
            title: Quy chế đào tạo
            decision_no: 2026/QĐ-ĐHM
            issued_date: 2016-12-20
            institution: Trường Đại học Mở
            education_mode: full_time
            source_pdf: data/raw/documents/foo.pdf
            parsed_source: data/raw/parsed/foo.txt
            cleaning_status: text_extracted
            ---
            Body
        """)
        meta, body = parse_frontmatter(text)
        for field in ("doc_id", "title", "decision_no", "issued_date",
                      "institution", "education_mode", "source_pdf",
                      "parsed_source", "cleaning_status"):
            assert field in meta, f"Field '{field}' missing"

    def test_ocr_source_field(self):
        text = textwrap.dedent("""\
            ---
            doc_id: test
            ocr_source: data/raw/parsed/test.ocr_raw.txt
            ---
        """)
        meta, _ = parse_frontmatter(text)
        assert meta["ocr_source"] == "data/raw/parsed/test.ocr_raw.txt"


# ---------------------------------------------------------------------------
# Slugify tests
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_phu_luc_appendix_heading(self):
        """Required: slugify("Phụ lục I. Danh mục chứng chỉ") must be lowercase ASCII."""
        s = slugify("Phụ lục I. Danh mục chứng chỉ")
        assert s == s.lower(), "result must be lowercase"
        assert all(c.isascii() for c in s), "result must be ASCII-safe"
        assert " " not in s, "result must have no spaces"
        # Should contain recognisable fragments
        assert "phu" in s
        assert "luc" in s

    def test_lowercase_output(self):
        assert slugify("HELLO WORLD") == "hello_world"

    def test_underscores_replace_spaces(self):
        assert " " not in slugify("some text here")

    def test_ascii_safe(self):
        s = slugify("Điều 9 - Đăng ký khối lượng")
        assert all(c.isascii() for c in s)

    def test_d_stroke_replaced(self):
        """đ and Đ must be converted to 'd'/'D' before lowercasing."""
        s = slugify("đại học")
        assert s.startswith("d")

    def test_no_leading_trailing_underscores(self):
        s = slugify("  hello  ")
        assert not s.startswith("_")
        assert not s.endswith("_")

    def test_numeric_preserved(self):
        s = slugify("Chương II. Tổ chức đào tạo")
        assert "ii" in s

    def test_consecutive_non_alnum_collapsed(self):
        """Multiple non-alnum characters should collapse to a single underscore."""
        s = slugify("a -- b")
        assert "__" not in s


# ---------------------------------------------------------------------------
# Splitting by Điều tests
# ---------------------------------------------------------------------------

class TestSplitByDieu:
    def _make_meta(self, doc_id="test_doc"):
        return {
            "doc_id": doc_id,
            "title": "Test",
            "decision_no": "001/QĐ",
            "issued_date": "2023-01-01",
            "institution": "OU",
            "education_mode": "full_time",
            "source_pdf": "test.pdf",
        }

    def test_splits_by_dieu_heading(self):
        body = textwrap.dedent("""\
            ## Điều 1. First article
            Content of article one.

            ## Điều 2. Second article
            Content of article two.
        """)
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        dieu_chunks = [c for c in chunks if c["chunk_type"] == "dieu"]
        assert len(dieu_chunks) == 2
        assert dieu_chunks[0]["section_number"] == "1"
        assert dieu_chunks[1]["section_number"] == "2"

    def test_heading_included_in_text(self):
        body = textwrap.dedent("""\
            ### Điều 5. Some article
            Body text.
        """)
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        dieu_chunks = [c for c in chunks if c["chunk_type"] == "dieu"]
        assert len(dieu_chunks) == 1
        assert "Điều 5" in dieu_chunks[0]["text"]
        assert "Body text." in dieu_chunks[0]["text"]

    def test_chapter_title_preserved(self):
        body = textwrap.dedent("""\
            ## Chương I. Những quy định chung

            ### Điều 1. Phạm vi điều chỉnh
            Content one.

            ## Chương II. Tổ chức đào tạo

            ### Điều 7. Thời gian đào tạo
            Content seven.
        """)
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        dieu_chunks = [c for c in chunks if c["chunk_type"] == "dieu"]
        assert len(dieu_chunks) == 2

        ch1 = next(c for c in dieu_chunks if c["section_number"] == "1")
        assert "Chương I" in ch1["chapter_title"]

        ch7 = next(c for c in dieu_chunks if c["section_number"] == "7")
        assert "Chương II" in ch7["chapter_title"]

    def test_section_title_extracted(self):
        body = "## Điều 9. Đăng ký khối lượng học tập\nContent."
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        dieu_chunks = [c for c in chunks if c["chunk_type"] == "dieu"]
        assert dieu_chunks[0]["section_title"].startswith("Điều 9")

    def test_char_and_word_counts(self):
        body = "## Điều 3. Test\nHello world this is a test."
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        dieu_chunks = [c for c in chunks if c["chunk_type"] == "dieu"]
        c = dieu_chunks[0]
        assert c["char_count"] == len(c["text"])
        assert c["word_count"] > 0

    def test_min_chars_merge(self):
        """Short chunks below min_chars should be merged into the previous."""
        body = textwrap.dedent("""\
            ## Điều 1. Article one
            This has plenty of content to satisfy the minimum character count.

            ## Điều 2. Article two
            Short.
        """)
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", min_chars=100)
        dieu_chunks = [c for c in chunks if c["chunk_type"] == "dieu"]
        # Điều 2 body is short so it should be merged into Điều 1
        assert len(dieu_chunks) == 1
        assert "Short." in dieu_chunks[0]["text"]

    def test_required_chunk_fields(self):
        body = "## Điều 1. Test article\nContent here."
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        required = {
            "chunk_id", "doc_id", "title", "decision_no", "issued_date",
            "institution", "education_mode", "chapter_title", "section_title",
            "section_number", "chunk_type", "source_path", "source_pdf",
            "text", "char_count", "word_count",
        }
        for chunk in chunks:
            missing = required - set(chunk.keys())
            assert not missing, f"Missing fields: {missing}"


# ---------------------------------------------------------------------------
# Chunk ID generation tests
# ---------------------------------------------------------------------------

class TestChunkIdGeneration:
    def test_dieu_chunk_id_format(self):
        chunk_id = make_dieu_chunk_id("ou_fulltime_credit_training_regulation_2016", "9")
        assert chunk_id == "ou_fulltime_credit_training_regulation_2016__dieu_9"

    def test_dieu_chunk_id_two_digits(self):
        chunk_id = make_dieu_chunk_id("test_doc", "33")
        assert chunk_id == "test_doc__dieu_33"

    def test_appendix_chunk_id_roman(self):
        chunk_id = make_appendix_chunk_id("test_doc", "I")
        assert chunk_id == "test_doc__phu_luc_i"

    def test_appendix_sub_chunk_id(self):
        chunk_id = make_appendix_chunk_id("test_doc", "I", sub="Nơi cấp chứng chỉ")
        assert "phu_luc" in chunk_id
        assert "noi" in chunk_id

    def test_appendix_sub_time_limit(self):
        chunk_id = make_appendix_chunk_id("test_doc", "I", sub="Thời hạn chứng chỉ")
        assert "phu_luc" in chunk_id
        assert "thoi" in chunk_id

    def test_slug_lowercase_ascii(self):
        s = slugify("Điều 9 - Đăng ký khối lượng")
        assert s == s.lower()
        assert all(c.isascii() for c in s)
        assert " " not in s

    def test_dieu_chunk_id_from_split(self):
        body = "## Điều 9. Đăng ký khối lượng học tập\nSome content here."
        meta = {
            "doc_id": "ou_fulltime_credit_training_regulation_2016",
            "title": "T", "decision_no": "D", "issued_date": "2016-01-01",
            "institution": "OU", "education_mode": "full_time", "source_pdf": "x.pdf",
        }
        chunks = split_document_into_chunks(body, meta, "test.md", 10)
        dieu = next(c for c in chunks if c["chunk_type"] == "dieu")
        assert dieu["chunk_id"] == "ou_fulltime_credit_training_regulation_2016__dieu_9"


# ---------------------------------------------------------------------------
# Appendix chunk generation tests
# ---------------------------------------------------------------------------

class TestAppendixChunks:
    def _make_meta(self, doc_id="test_doc"):
        return {
            "doc_id": doc_id,
            "title": "Test",
            "decision_no": "001/QĐ",
            "issued_date": "2023-01-01",
            "institution": "OU",
            "education_mode": "full_time",
            "source_pdf": "test.pdf",
        }

    def test_appendix_chunk_generated(self):
        body = textwrap.dedent("""\
            ## Phụ lục I. Danh mục các chứng chỉ tiếng Anh được xét miễn

            | Col A | Col B |
            |---|---|
            | val1 | val2 |
        """)
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        appendix = [c for c in chunks if c["chunk_type"] == "appendix"]
        assert len(appendix) == 1
        assert "phu_luc" in appendix[0]["chunk_id"]

    def test_appendix_sub_noi_cap(self):
        body = textwrap.dedent("""\
            ## Phụ lục I. Danh mục chứng chỉ

            Main appendix content.

            ### Nơi cấp chứng chỉ

            | Cert | Issuer |
            |---|---|
            | IELTS | British Council |
        """)
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        sub_chunks = [c for c in chunks if c["chunk_type"] == "appendix_sub"]
        assert len(sub_chunks) == 1
        assert "noi" in sub_chunks[0]["chunk_id"]

    def test_appendix_sub_thoi_han(self):
        body = textwrap.dedent("""\
            ## Phụ lục I. Danh mục chứng chỉ

            Main content.

            ### Thời hạn chứng chỉ

            | Group | Validity |
            |---|---|
            | IELTS | 2 years |
        """)
        chunks = split_document_into_chunks(body, self._make_meta(), "test.md", 10)
        sub_chunks = [c for c in chunks if c["chunk_type"] == "appendix_sub"]
        assert len(sub_chunks) == 1
        assert "thoi" in sub_chunks[0]["chunk_id"]

    def test_appendix_chunk_id_contains_phu_luc(self):
        body = textwrap.dedent("""\
            ## Phụ lục I. Some appendix

            Content here.
        """)
        meta = self._make_meta("my_doc")
        chunks = split_document_into_chunks(body, meta, "test.md", 10)
        appendix = [c for c in chunks if "appendix" in c["chunk_type"]]
        assert len(appendix) >= 1
        for c in appendix:
            assert "phu_luc" in c["chunk_id"]

    def test_full_appendix_flow(self):
        """Simulate the full Phụ lục I from the foreign language regulation."""
        body = textwrap.dedent("""\
            ## Phụ lục I. Danh mục các chứng chỉ tiếng Anh được xét miễn

            | STT | Aptis | TOEIC | IELTS |
            |---|---|---|---|
            | 1 | >=168 | >=675 | >=6.0 |

            ### Nơi cấp chứng chỉ

            | Chứng chỉ | Nơi cấp |
            |---|---|
            | IELTS | British Council |

            ### Thời hạn chứng chỉ

            | Nhóm | Thời hạn |
            |---|---|
            | TOEIC | 2 năm |
        """)
        meta = self._make_meta("ou_non_major_foreign_language_regulation_2023")
        chunks = split_document_into_chunks(body, meta, "test.md", 10)

        types = {c["chunk_type"] for c in chunks}
        assert "appendix" in types
        assert "appendix_sub" in types

        ids = [c["chunk_id"] for c in chunks]
        assert any("phu_luc" in cid and "noi" in cid for cid in ids)
        assert any("phu_luc" in cid and "thoi" in cid for cid in ids)


# ---------------------------------------------------------------------------
# Integration: run on actual data/raw/cleaned if available
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests that only run when the actual data files are present."""

    _CLEANED_DIR = Path(__file__).parent.parent / "data" / "raw" / "cleaned"

    def _files_present(self) -> bool:
        return self._CLEANED_DIR.exists() and bool(list(self._CLEANED_DIR.glob("*.md")))

    def test_real_files_produce_chunks(self, tmp_path):
        if not self._files_present():
            pytest.skip("No cleaned source files available")

        output = tmp_path / "chunks.jsonl"
        build_chunks(
            input_dir=str(self._CLEANED_DIR),
            output_file=str(output),
            min_chars=80,
        )
        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0

        for line in lines:
            obj = json.loads(line)
            # Verify required fields
            for field in ("chunk_id", "doc_id", "text", "chunk_type",
                          "char_count", "word_count"):
                assert field in obj, f"Field '{field}' missing in chunk"
            assert obj["char_count"] == len(obj["text"])

    def test_dieu_chunks_in_real_files(self, tmp_path):
        if not self._files_present():
            pytest.skip("No cleaned source files available")

        output = tmp_path / "chunks.jsonl"
        build_chunks(
            input_dir=str(self._CLEANED_DIR),
            output_file=str(output),
            min_chars=80,
        )
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        chunks = [json.loads(l) for l in lines]
        dieu_chunks = [c for c in chunks if c["chunk_type"] == "dieu"]
        assert len(dieu_chunks) > 0, "Expected at least one Điều chunk"

    def test_appendix_chunks_in_foreign_language_doc(self, tmp_path):
        if not self._files_present():
            pytest.skip("No cleaned source files available")

        output = tmp_path / "chunks.jsonl"
        build_chunks(
            input_dir=str(self._CLEANED_DIR),
            output_file=str(output),
            min_chars=80,
        )
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        chunks = [json.loads(l) for l in lines]
        appendix_chunks = [
            c for c in chunks
            if "appendix" in c.get("chunk_type", "")
            and "non_major_foreign" in c.get("doc_id", "")
        ]
        assert len(appendix_chunks) > 0, (
            "Expected appendix chunks for ou_non_major_foreign_language_regulation_2023"
        )
