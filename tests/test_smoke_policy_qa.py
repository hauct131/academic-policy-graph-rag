"""
tests/test_smoke_policy_qa.py

Tests for the scripts/09_smoke_policy_qa.py script.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module


def test_smoke_script_imports_and_compiles():
    # 1. scripts/09_smoke_policy_qa.py imports/compiles.
    smoke = import_module("09_smoke_policy_qa")
    assert hasattr(smoke, "run_smoke_test")
    assert hasattr(smoke, "DEFAULT_QUESTIONS")


def test_smoke_script_execution_exits_zero_when_chunks_exist(capsys):
    smoke = import_module("09_smoke_policy_qa")
    
    chunks_path = Path(__file__).parent.parent / "data" / "chunks" / "policy_chunks.annotated.jsonl"
    if not chunks_path.exists():
        pytest.skip("Annotated chunks file not found, skipping smoke execution test.")

    # 2. If generated annotated chunks exist, running the script with default options exits 0.
    nodes_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
    edges_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"
    config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
    registry_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "document_registry.jsonl"

    res = smoke.run_smoke_test(
        chunks_path=chunks_path,
        nodes_path=nodes_path,
        edges_path=edges_path,
        config_path=config_path,
        registry_path=registry_path,
        top_k=5,
        full_answer=False
    )
    assert res == 0

    # 4. Ensure output contains at least one known demo question or chunk reference when generated chunks exist.
    captured = capsys.readouterr()
    assert "Điều kiện xét tốt nghiệp là gì?" in captured.out
    assert "Smoke Test Complete" in captured.out


def test_smoke_script_returns_one_when_files_missing():
    # Test helper behavior when files are missing
    smoke = import_module("09_smoke_policy_qa")
    nonexistent = Path("nonexistent_file.jsonl")
    
    res = smoke.run_smoke_test(
        chunks_path=nonexistent,
        nodes_path=nonexistent,
        edges_path=nonexistent,
        config_path=nonexistent,
        registry_path=nonexistent,
        top_k=5,
        full_answer=False
    )
    assert res == 1


def test_smoke_script_bm25_backend_exits_zero_when_chunks_exist(capsys):
    """run_smoke_test with bm25_like_v0 should exit 0 and print backend name."""
    smoke = import_module("09_smoke_policy_qa")

    chunks_path = Path(__file__).parent.parent / "data" / "chunks" / "policy_chunks.annotated.jsonl"
    if not chunks_path.exists():
        pytest.skip("Annotated chunks file not found, skipping BM25 smoke test.")

    nodes_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
    edges_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"
    config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
    registry_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "document_registry.jsonl"

    res = smoke.run_smoke_test(
        chunks_path=chunks_path,
        nodes_path=nodes_path,
        edges_path=edges_path,
        config_path=config_path,
        registry_path=registry_path,
        top_k=5,
        full_answer=False,
        retrieval_backend="bm25_like_v0",
    )
    assert res == 0

    captured = capsys.readouterr()
    assert "Retrieval backend: bm25_like_v0" in captured.out
    assert "Smoke Test Complete" in captured.out
