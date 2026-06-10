"""
tests/test_policy_retrieval_service.py

Tests for PolicyRetrievalService.
"""

import sys
from pathlib import Path
from typing import Any
import pytest

# Ensure scripts and root are in path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from policy_retrieval_service import PolicyRetrievalService


# ---------------------------------------------------------------------------
# Fake backend for injection tests
# ---------------------------------------------------------------------------


class FakeBackend:
    """Minimal fake backend that always returns the first chunk with score 99."""

    name = "fake_backend"

    def retrieve(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        top_k: int = 5,
        policy_area: str | None = None,
        action_tag: str | None = None,
        requirement_tag: str | None = None,
        risk_tag: str | None = None,
        graph_bonus_map: dict[str, float] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        return [(chunks[0], 99.0)] if chunks else []


def get_real_chunks():
    chunks_path = Path(__file__).parent.parent / "data" / "chunks" / "policy_chunks.annotated.jsonl"
    if not chunks_path.exists():
        return None
    
    import json
    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def test_service_initializes():
    dummy_chunks = [{"chunk_id": "test_chunk", "text": "test content", "policy_area": ["graduation"]}]
    service = PolicyRetrievalService(chunks=dummy_chunks)
    assert service.chunks == dummy_chunks
    assert service.nodes_file is None
    assert service.edges_file is None


def test_retrieve_for_issue_graduation():
    chunks = get_real_chunks()
    if not chunks:
        pytest.skip("Real annotated chunks not found. Skipping graduation retrieval test.")
    
    nodes_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
    edges_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"
    service = PolicyRetrievalService(chunks=chunks, nodes_file=nodes_path, edges_file=edges_path)
    
    issue = {
        "issue_type": "graduation",
        "query": "điều kiện xét tốt nghiệp",
        "policy_area": "graduation"
    }
    selected = service.retrieve_for_issue(
        issue=issue,
        question="Điều kiện xét tốt nghiệp là gì?",
        top_k=5,
        max_sources=3
    )
    assert len(selected) > 0
    # First chunk should be Điều 27
    first_chunk = selected[0][0]
    assert first_chunk["section_number"] == "27" or "dieu_27" in first_chunk["chunk_id"]


def test_retrieve_for_issue_exemption_hoso():
    chunks = get_real_chunks()
    if not chunks:
        pytest.skip("Real annotated chunks not found.")
    
    nodes_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
    edges_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"
    service = PolicyRetrievalService(chunks=chunks, nodes_file=nodes_path, edges_file=edges_path)
    
    # Question: "Miễn môn học cần hồ sơ gì?"
    issue = {
        "issue_type": "course_exemption",
        "query": "mien mon hoc can ho so gi",
        "policy_area": "course_exemption"
    }
    selected = service.retrieve_for_issue(
        issue=issue,
        question="Miễn môn học cần hồ sơ gì?",
        top_k=5,
        max_sources=3
    )
    assert len(selected) > 0
    first_chunk = selected[0][0]
    # Should be Điều 5 (Hồ sơ)
    assert first_chunk["section_number"] == "5" or "dieu_5" in first_chunk["chunk_id"]


def test_retrieve_for_issue_exemption_conditions():
    chunks = get_real_chunks()
    if not chunks:
        pytest.skip("Real annotated chunks not found.")
    
    nodes_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
    edges_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"
    service = PolicyRetrievalService(chunks=chunks, nodes_file=nodes_path, edges_file=edges_path)
    
    # Question: "Điều kiện miễn môn học là gì?"
    issue = {
        "issue_type": "course_exemption",
        "query": "dieu kien mien mon hoc la gi",
        "policy_area": "course_exemption"
    }
    selected = service.retrieve_for_issue(
        issue=issue,
        question="Điều kiện miễn môn học là gì?",
        top_k=5,
        max_sources=3
    )
    assert len(selected) > 0
    first_chunk = selected[0][0]
    # Should be Điều 4 (Điều kiện)
    assert first_chunk["section_number"] == "4" or "dieu_4" in first_chunk["chunk_id"]


def test_retrieve_for_issue_english_ielts():
    chunks = get_real_chunks()
    if not chunks:
        pytest.skip("Real annotated chunks not found.")
    
    nodes_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
    edges_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"
    service = PolicyRetrievalService(chunks=chunks, nodes_file=nodes_path, edges_file=edges_path)
    
    # Question: "IELTS 6.0 được miễn tiếng Anh không?"
    issue = {
        "issue_type": "foreign_language_requirement",
        "query": "ielts 6.0 duoc mien tieng anh khong",
        "policy_area": "foreign_language_requirement"
    }
    selected = service.retrieve_for_issue(
        issue=issue,
        question="IELTS 6.0 được miễn tiếng Anh không?",
        top_k=5,
        max_sources=3
    )
    assert len(selected) > 0
    # Should contain Điều 9 and/or Phụ lục I
    has_target = any("dieu_9" in c[0]["chunk_id"] or "phu_luc_i" in c[0]["chunk_id"] or "dieu 9" in str(c[0].get("section_number")) for c in selected)
    assert has_target


def test_retrieve_for_issues_deduplicates():
    dummy_chunks = [
        {"chunk_id": "chunk_1", "text": "chunk one", "section_number": "1", "policy_area": ["graduation"]},
        {"chunk_id": "chunk_2", "text": "chunk two", "section_number": "2", "policy_area": ["graduation"]}
    ]
    service = PolicyRetrievalService(chunks=dummy_chunks)
    issues = [
        {"issue_type": "graduation", "query": "chunk", "policy_area": "graduation"},
        {"issue_type": "graduation", "query": "chunk", "policy_area": "graduation"}
    ]
    # Because retrieve_for_issues retrieves for each issue, they might both retrieve chunk_1.
    # Deduplication should ensure it only appears once in the final list.
    selected = service.retrieve_for_issues(
        issues=issues,
        question="chunk",
        top_k=5,
        max_sources_per_issue=3,
        use_graph=False
    )
    chunk_ids = [c[0]["chunk_id"] for c in selected]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_service_works_when_graph_files_missing():
    dummy_chunks = [{"chunk_id": "chunk_1", "text": "graduation info", "section_number": "27", "policy_area": ["graduation"]}]
    service = PolicyRetrievalService(
        chunks=dummy_chunks,
        nodes_file="nonexistent_nodes.jsonl",
        edges_file="nonexistent_edges.jsonl"
    )
    issue = {"issue_type": "graduation", "query": "graduation", "policy_area": "graduation"}
    selected = service.retrieve_for_issue(
        issue=issue,
        question="graduation",
        top_k=5,
        max_sources=3,
        use_graph=True
    )
    assert len(selected) == 1
    assert selected[0][0]["chunk_id"] == "chunk_1"


def test_real_query_graduation_returns_dieu_27_exactly():
    chunks = get_real_chunks()
    if not chunks:
        pytest.skip("Real annotated chunks not found.")
    
    nodes_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
    edges_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"
    service = PolicyRetrievalService(chunks=chunks, nodes_file=nodes_path, edges_file=edges_path)
    
    issue = {
        "issue_type": "graduation",
        "query": "dieu kien xet tot nghiep la gi",
        "policy_area": "graduation"
    }
    selected = service.retrieve_for_issue(
        issue=issue,
        question="Điều kiện xét tốt nghiệp là gì?",
        top_k=5,
        max_sources=3
    )
    assert len(selected) > 0
    chunk_ids = [c[0]["chunk_id"] for c in selected]
    assert "ou_fulltime_credit_training_regulation_2016__dieu_27" in chunk_ids


# ---------------------------------------------------------------------------
# New: Backend abstraction tests
# ---------------------------------------------------------------------------


def test_service_default_backend_name_is_lexical_v0():
    """PolicyRetrievalService uses lexical_v0 backend by default."""
    dummy_chunks = [{"chunk_id": "c1", "text": "test", "policy_area": ["graduation"]}]
    service = PolicyRetrievalService(chunks=dummy_chunks)
    assert service.backend_name == "lexical_v0"


def test_service_accepts_custom_backend():
    """PolicyRetrievalService stores and exposes injected backend."""
    dummy_chunks = [{"chunk_id": "c1", "text": "test", "policy_area": ["graduation"]}]
    fake = FakeBackend()
    service = PolicyRetrievalService(chunks=dummy_chunks, backend=fake)
    assert service.backend is fake
    assert service.backend_name == "fake_backend"


def test_service_retrieve_for_issue_uses_injected_backend():
    """retrieve_for_issue delegates to the injected backend, not the default."""
    dummy_chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "doc_test",
            "text": "điều kiện tốt nghiệp",
            "section_title": "Điều 1",
            "section_number": "1",
            "chapter_title": "Chương I",
            "chunk_type": "article",
            "policy_area": ["graduation"],
            "action_tags": ["xét tốt nghiệp"],
            "requirement_tags": [],
            "procedure_tags": [],
            "risk_tags": [],
            "evidence_groups": [],
            "time_tags": [],
        }
    ]
    fake = FakeBackend()
    service = PolicyRetrievalService(chunks=dummy_chunks, backend=fake)
    issue = {"issue_type": "graduation", "query": "tốt nghiệp", "policy_area": "graduation"}
    selected = service.retrieve_for_issue(
        issue=issue,
        question="Điều kiện tốt nghiệp?",
        top_k=5,
        max_sources=3,
        use_graph=False,
    )
    # FakeBackend always returns the first chunk with score 99.0
    assert len(selected) >= 1
    assert selected[0][0]["chunk_id"] == "c1"


def test_service_fake_backend_empty_chunks():
    """FakeBackend returns empty list when chunks is empty."""
    fake = FakeBackend()
    service = PolicyRetrievalService(chunks=[], backend=fake)
    issue = {"issue_type": "graduation", "query": "tốt nghiệp", "policy_area": "graduation"}
    selected = service.retrieve_for_issue(
        issue=issue,
        question="test",
        top_k=5,
        max_sources=3,
        use_graph=False,
    )
    assert selected == []


# ---------------------------------------------------------------------------
# New: backend_name param and BM25 service integration
# ---------------------------------------------------------------------------

_BM25_TEST_CHUNK = {
    "chunk_id": "test_doc__dieu_1",
    "doc_id": "test_doc",
    "text": "Điều kiện tốt nghiệp bao gồm tích lũy đủ số tín chỉ theo chương trình đào tạo.",
    "section_title": "Điều 1. Điều kiện tốt nghiệp",
    "section_number": "1",
    "chapter_title": "Chương I",
    "chunk_type": "article",
    "policy_area": ["graduation"],
    "action_tags": ["graduation_audit"],
    "requirement_tags": ["credit_hours"],
    "procedure_tags": [],
    "risk_tags": [],
    "evidence_groups": [],
    "time_tags": [],
}


def test_service_backend_name_param_bm25():
    """PolicyRetrievalService(backend_name='bm25_like_v0').backend_name == 'bm25_like_v0'."""
    service = PolicyRetrievalService(
        chunks=[_BM25_TEST_CHUNK],
        backend_name="bm25_like_v0",
    )
    assert service.backend_name == "bm25_like_v0"


def test_service_bm25_backend_retrieves_matching_chunk():
    """Service with BM25 backend retrieves a matching synthetic chunk."""
    service = PolicyRetrievalService(
        chunks=[_BM25_TEST_CHUNK],
        backend_name="bm25_like_v0",
    )
    issue = {
        "issue_type": "graduation",
        "query": "điều kiện tốt nghiệp",
        "policy_area": "graduation",
    }
    selected = service.retrieve_for_issue(
        issue=issue,
        question="Điều kiện tốt nghiệp là gì?",
        top_k=5,
        max_sources=3,
        use_graph=False,
    )
    assert len(selected) >= 1
    assert selected[0][0]["chunk_id"] == "test_doc__dieu_1"


def test_service_explicit_backend_wins_over_backend_name():
    """Explicit backend= object takes priority over backend_name=."""
    fake = FakeBackend()
    service = PolicyRetrievalService(
        chunks=[_BM25_TEST_CHUNK],
        backend=fake,
        backend_name="bm25_like_v0",
    )
    # The fake backend is used, not BM25
    assert service.backend is fake
    assert service.backend_name == "fake_backend"
