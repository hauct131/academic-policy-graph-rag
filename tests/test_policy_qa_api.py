"""
tests/test_policy_qa_api.py

Integration and unit tests for the FastAPI Academic Policy QA API.
"""

import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.policy_qa_service import PolicyQAService

client = TestClient(app)


def test_health_check_still_works():
    response = client.get("/api/v1/graph-rag/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_empty_question_returns_validation_error():
    response = client.post("/policy/ask", json={"question": "", "top_k": 5})
    assert response.status_code == 422


def test_ask_too_high_top_k_returns_validation_error():
    response = client.post("/policy/ask", json={"question": "Điều kiện xét tốt nghiệp là gì?", "top_k": 11})
    assert response.status_code == 422


def test_ask_success_when_chunks_exist():
    chunks_path = Path("data/chunks/policy_chunks.annotated.jsonl")
    if not chunks_path.exists():
        pytest.skip("Annotated chunks file not found, skipping integration QA tests.")

    response = client.post("/policy/ask", json={
        "question": "Điều kiện xét tốt nghiệp là gì?",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "status" in data
    assert data["status"] == "ok"
    assert "metadata" in data
    
    ans = data["answer"]
    assert "Điều 27" in ans or "xét tốt nghiệp" in ans.lower()


def test_ask_temporal_warning():
    chunks_path = Path("data/chunks/policy_chunks.annotated.jsonl")
    if not chunks_path.exists():
        pytest.skip("Annotated chunks file not found, skipping integration QA tests.")

    response = client.post("/policy/ask", json={
        "question": "Học kỳ này khi nào nộp hồ sơ miễn môn?",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "warnings" in data
    
    warnings_found = any("thông báo" in w.lower() for w in data["warnings"])
    answer_has_fallback = ("chưa có thông báo" in data["answer"].lower() or "chưa thể kết luận" in data["answer"].lower())
    assert warnings_found or answer_has_fallback


def test_service_with_missing_graph_files():
    service = PolicyQAService(
        chunks_path="data/chunks/policy_chunks.annotated.jsonl",
        domain_config_path="domains/ou_academic_policy_v1/domain.json",
        document_registry_path="domains/ou_academic_policy_v1/document_registry.jsonl",
        nodes_path="nonexistent_nodes.jsonl",
        edges_path="nonexistent_edges.jsonl"
    )
    if not service.chunks_path.exists() or not service.domain_config_path.exists() or not service.document_registry_path.exists():
        pytest.skip("Required chunks/config files not found, skipping service test.")
        
    service.load_resources()
    assert service.initialized is True
    
    ans, meta, warnings = service.get_qa_response("Điều kiện xét tốt nghiệp là gì?")
    assert meta["uses_graph"] is False
