"""
tests/test_policy_retrieval.py

Unit and integration tests for the lightweight policy chunk retrieval pipeline.

Run with:
    python -m pytest tests/test_policy_retrieval.py
"""

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module

_mod = import_module("05_retrieve_policy_chunks")

normalize_text = _mod.normalize_text
tokenize = _mod.tokenize
read_jsonl = _mod.read_jsonl
score_chunk = _mod.score_chunk
filter_chunks = _mod.filter_chunks
retrieve_chunks = _mod.retrieve_chunks


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, section_title: str = "", **kwargs) -> dict:
    """Build a minimal chunk dict for retrieval testing."""
    base = {
        "chunk_id": "test_chunk",
        "doc_id": "test_doc",
        "section_number": "1",
        "section_title": section_title,
        "chapter_title": "",
        "text": text,
        "policy_area": [],
        "action_tags": [],
        "requirement_tags": [],
        "procedure_tags": [],
        "risk_tags": [],
        "evidence_groups": [],
        "time_tags": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_removes_accents(self):
        assert normalize_text("Điều kiện xét tốt nghiệp") == "dieu kien xet tot nghiep"
        assert normalize_text("Giáo dục thể chất") == "giao duc the chat"
        assert normalize_text("Đường lối cách mạng") == "duong loi cach mang"

    def test_handles_empty_and_none(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""


# ---------------------------------------------------------------------------
# 2. tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_returns_lowercase_normalized_tokens(self):
        tokens = tokenize("Điều 1. Phạm vi điều chỉnh!")
        assert tokens == ["dieu", "1", "pham", "vi", "dieu", "chinh"]

    def test_handles_punctuation(self):
        tokens = tokenize("A, B; C? D.")
        assert tokens == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# 3. Scoring weight: Section title vs Body text
# ---------------------------------------------------------------------------

class TestScoringWeights:
    def test_section_title_match_scores_higher_than_body_only(self):
        # Match in section title
        chunk_title_match = _make_chunk(
            text="Nội dung khác hoàn toàn không liên quan.",
            section_title="Quy định học vượt"
        )
        # Match in text body only
        chunk_body_match = _make_chunk(
            text="Sinh viên có nguyện vọng học vượt nộp đơn tại Phòng QLĐT.",
            section_title="Điều 1. Phạm vi"
        )

        query = "học vượt"
        tokens = tokenize(query)
        phrase = " ".join(tokens)

        score_title = score_chunk(chunk_title_match, tokens, phrase)
        score_body = score_chunk(chunk_body_match, tokens, phrase)

        # Title match (2 tokens in query "học vượt" -> title has both "học vượt" -> 2 * 2 = 4 points.
        # Plus title exact phrase match bonus -> +5 = 9. Total = 9)
        # Body match (2 tokens -> body has both -> 2 * 1 = 2 points.
        # Plus body exact phrase match bonus -> +3 = 5. Total = 5)
        assert score_title > score_body, f"Title match ({score_title}) should score higher than body match ({score_body})"


# ---------------------------------------------------------------------------
# 4. Exact phrase match bonus
# ---------------------------------------------------------------------------

class TestExactPhraseMatchBonus:
    def test_exact_phrase_bonus_is_applied(self):
        # Chunk with exact phrase "xét tốt nghiệp"
        chunk_exact = _make_chunk(
            text="Sinh viên đủ điều kiện xét tốt nghiệp sẽ được công nhận.",
            section_title="Quy định chung"
        )
        # Chunk with individual tokens but not in sequence
        chunk_no_exact = _make_chunk(
            text="Hội đồng xét hồ sơ của sinh viên tốt nghiệp kỳ trước.",
            section_title="Quy định chung"
        )

        query = "xét tốt nghiệp"
        tokens = tokenize(query)
        phrase = " ".join(tokens)

        score_exact = score_chunk(chunk_exact, tokens, phrase)
        score_no_exact = score_chunk(chunk_no_exact, tokens, phrase)

        # Both contain "xét" and "tốt nghiệp" (2 tokens overlap, giving 2 * 1.0 = 2.0).
        # chunk_exact gets the exact phrase bonus of +3.0 (total 5.0).
        # chunk_no_exact does not get the phrase bonus (total 2.0).
        assert score_exact > score_no_exact
        assert score_exact - score_no_exact == 3.0


# ---------------------------------------------------------------------------
# 5. policy_area filter
# ---------------------------------------------------------------------------

class TestPolicyAreaFilter:
    def test_filter_policy_area(self):
        chunks = [
            _make_chunk("Text A", policy_area=["graduation"]),
            _make_chunk("Text B", policy_area=["course_exemption"]),
        ]
        res = filter_chunks(chunks, policy_area="graduation")
        assert len(res) == 1
        assert res[0]["text"] == "Text A"


# ---------------------------------------------------------------------------
# 6. action_tag filter
# ---------------------------------------------------------------------------

class TestActionTagFilter:
    def test_filter_action_tag(self):
        chunks = [
            _make_chunk("Text A", action_tags=["study_ahead"]),
            _make_chunk("Text B", action_tags=["register_course"]),
        ]
        res = filter_chunks(chunks, action_tag="study_ahead")
        assert len(res) == 1
        assert res[0]["text"] == "Text A"


# ---------------------------------------------------------------------------
# 7. requirement_tag filter
# ---------------------------------------------------------------------------

class TestRequirementTagFilter:
    def test_filter_requirement_tag(self):
        chunks = [
            _make_chunk("Text A", requirement_tags=["english_exit_requirement"]),
            _make_chunk("Text B", requirement_tags=["minimum_gpa_required"]),
        ]
        res = filter_chunks(chunks, requirement_tag="english_exit_requirement")
        assert len(res) == 1
        assert res[0]["text"] == "Text A"


# ---------------------------------------------------------------------------
# 8. Sorting by score descending
# ---------------------------------------------------------------------------

class TestSortingOrder:
    def test_results_sorted_by_score_descending(self):
        chunks = [
            _make_chunk("Sinh viên thi tiếng Anh đầu ra để tốt nghiệp", section_title="Điều 1", chunk_id="chunk_1"),
            _make_chunk("Không liên quan gì cả", section_title="Điều 2", chunk_id="chunk_2"),
            _make_chunk("Điều kiện tốt nghiệp tiếng Anh đầu ra là đạt chuẩn", section_title="Điều 3. Tiếng Anh đầu ra", chunk_id="chunk_3"),
        ]
        query = "tiếng Anh đầu ra"
        results = retrieve_chunks(chunks, query, top_k=5)

        scores = [score for chunk, score in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0][0]["chunk_id"] == "chunk_3"  # Higher score due to more matches/overlap


# ---------------------------------------------------------------------------
# 9. Unknown filter returns empty list
# ---------------------------------------------------------------------------

class TestUnknownFilter:
    def test_unknown_filter_returns_empty_list(self):
        chunks = [
            _make_chunk("Text A", policy_area=["graduation"]),
            _make_chunk("Text B", policy_area=["course_exemption"]),
        ]
        # unknown policy area
        res = filter_chunks(chunks, policy_area="nonexistent_area")
        assert res == []


# ---------------------------------------------------------------------------
# 10. Integration test against real data/chunks/policy_chunks.annotated.jsonl
# ---------------------------------------------------------------------------

class TestIntegrationRetrieval:
    _INPUT = Path(__file__).parent.parent / "data" / "chunks" / "policy_chunks.annotated.jsonl"

    def test_real_query_returns_matches(self):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl is not built")

        # Load real chunks
        chunks = read_jsonl(self._INPUT)

        # Run query
        query = "xét tốt nghiệp"
        results = retrieve_chunks(chunks, query, top_k=5)

        # Assert at least one match
        assert len(results) >= 1
        # Highest rank should have positive score
        assert results[0][1] > 0.0
        # Verify graduation related metadata is present
        top_chunk = results[0][0]
        assert (
            "graduation" in top_chunk.get("policy_area", [])
            or "graduation_audit" in top_chunk.get("action_tags", [])
            or "xét tốt nghiệp" in top_chunk.get("text", "").lower()
        )

