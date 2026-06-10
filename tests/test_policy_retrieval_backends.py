"""
tests/test_policy_retrieval_backends.py

Tests for policy_retrieval_backends module:
  - PolicyRetrievalBackend Protocol
  - LexicalPolicyRetrievalBackend
  - get_default_retrieval_backend factory
"""

import sys
from pathlib import Path
import pytest

# Ensure scripts directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from policy_retrieval_backends import (
    LexicalPolicyRetrievalBackend,
    get_default_retrieval_backend,
)


# ---------------------------------------------------------------------------
# Synthetic chunk fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_CHUNKS = [
    {
        "chunk_id": "doc_a__dieu_1",
        "doc_id": "doc_a",
        "text": "Điều kiện xét tốt nghiệp bao gồm tích lũy đủ số tín chỉ.",
        "section_title": "Điều 1. Điều kiện tốt nghiệp",
        "section_number": "1",
        "chapter_title": "Chương I",
        "chunk_type": "article",
        "policy_area": ["graduation"],
        "action_tags": ["xét tốt nghiệp"],
        "requirement_tags": ["tín chỉ"],
        "procedure_tags": [],
        "risk_tags": [],
        "evidence_groups": [],
        "time_tags": [],
    },
    {
        "chunk_id": "doc_a__dieu_2",
        "doc_id": "doc_a",
        "text": "Sinh viên cảnh báo học vụ khi điểm trung bình dưới 1.0.",
        "section_title": "Điều 2. Cảnh báo học vụ",
        "section_number": "2",
        "chapter_title": "Chương I",
        "chunk_type": "article",
        "policy_area": ["academic_warning"],
        "action_tags": ["cảnh báo học vụ"],
        "requirement_tags": [],
        "procedure_tags": [],
        "risk_tags": ["đình chỉ học"],
        "evidence_groups": [],
        "time_tags": [],
    },
    {
        "chunk_id": "doc_b__dieu_5",
        "doc_id": "doc_b",
        "text": "Hồ sơ miễn môn học gồm đơn xin miễn và bảng điểm.",
        "section_title": "Điều 5. Hồ sơ",
        "section_number": "5",
        "chapter_title": "Chương II",
        "chunk_type": "article",
        "policy_area": ["course_exemption"],
        "action_tags": ["miễn môn"],
        "requirement_tags": ["hồ sơ"],
        "procedure_tags": [],
        "risk_tags": [],
        "evidence_groups": [],
        "time_tags": [],
    },
]


# ---------------------------------------------------------------------------
# Test 1: get_default_retrieval_backend returns lexical_v0
# ---------------------------------------------------------------------------


def test_get_default_retrieval_backend_returns_lexical_v0():
    backend = get_default_retrieval_backend()
    assert backend.name == "lexical_v0"


# ---------------------------------------------------------------------------
# Test 2: LexicalPolicyRetrievalBackend.retrieve returns a list
# ---------------------------------------------------------------------------


def test_lexical_backend_retrieve_returns_list():
    backend = LexicalPolicyRetrievalBackend()
    results = backend.retrieve(
        chunks=SYNTHETIC_CHUNKS,
        query="tốt nghiệp",
        top_k=5,
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test 3: Backend retrieves matching text with positive score
# ---------------------------------------------------------------------------


def test_lexical_backend_retrieves_with_positive_score():
    backend = LexicalPolicyRetrievalBackend()
    results = backend.retrieve(
        chunks=SYNTHETIC_CHUNKS,
        query="điều kiện xét tốt nghiệp",
        top_k=5,
    )
    assert len(results) > 0
    # All returned scores should be positive
    for chunk, score in results:
        assert score > 0.0
    # Top result should be the graduation chunk
    top_chunk, top_score = results[0]
    assert "graduation" in top_chunk.get("policy_area", [])


# ---------------------------------------------------------------------------
# Test 4: Backend respects policy_area filter
# ---------------------------------------------------------------------------


def test_lexical_backend_respects_policy_area_filter():
    backend = LexicalPolicyRetrievalBackend()
    # Use a broad query that could match multiple chunks, but filter to graduation
    results = backend.retrieve(
        chunks=SYNTHETIC_CHUNKS,
        query="điều kiện",
        top_k=5,
        policy_area="graduation",
    )
    for chunk, score in results:
        assert "graduation" in chunk.get("policy_area", [])


def test_lexical_backend_policy_area_filter_excludes_others():
    backend = LexicalPolicyRetrievalBackend()
    # Filter strictly to course_exemption; academic_warning chunks should be excluded
    results = backend.retrieve(
        chunks=SYNTHETIC_CHUNKS,
        query="hồ sơ miễn môn",
        top_k=5,
        policy_area="course_exemption",
    )
    for chunk, _ in results:
        assert "course_exemption" in chunk.get("policy_area", [])
        assert "academic_warning" not in chunk.get("policy_area", [])


# ---------------------------------------------------------------------------
# Test 5: Backend accepts graph_bonus_map without error
# ---------------------------------------------------------------------------


def test_lexical_backend_accepts_graph_bonus_map():
    backend = LexicalPolicyRetrievalBackend()
    bonus_map = {"doc_a__dieu_1": 2.5, "doc_b__dieu_5": 1.0}
    results = backend.retrieve(
        chunks=SYNTHETIC_CHUNKS,
        query="tốt nghiệp",
        top_k=5,
        graph_bonus_map=bonus_map,
    )
    assert isinstance(results, list)


def test_lexical_backend_graph_bonus_map_none_is_accepted():
    backend = LexicalPolicyRetrievalBackend()
    results = backend.retrieve(
        chunks=SYNTHETIC_CHUNKS,
        query="tốt nghiệp",
        top_k=5,
        graph_bonus_map=None,
    )
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test: backend name attribute is accessible as class attribute
# ---------------------------------------------------------------------------


def test_lexical_backend_name_is_class_attribute():
    assert LexicalPolicyRetrievalBackend.name == "lexical_v0"
    backend = LexicalPolicyRetrievalBackend()
    assert backend.name == "lexical_v0"


# ---------------------------------------------------------------------------
# Test: retrieve returns empty list when no chunks match filter
# ---------------------------------------------------------------------------


def test_lexical_backend_returns_empty_for_non_matching_filter():
    backend = LexicalPolicyRetrievalBackend()
    results = backend.retrieve(
        chunks=SYNTHETIC_CHUNKS,
        query="tốt nghiệp",
        top_k=5,
        policy_area="nonexistent_area",
    )
    assert results == []


# ---------------------------------------------------------------------------
# Test: top_k is respected
# ---------------------------------------------------------------------------


def test_lexical_backend_top_k_limit():
    backend = LexicalPolicyRetrievalBackend()
    results = backend.retrieve(
        chunks=SYNTHETIC_CHUNKS,
        query="điều",
        top_k=2,
    )
    assert len(results) <= 2
