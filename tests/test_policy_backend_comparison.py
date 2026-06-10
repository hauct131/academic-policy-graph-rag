"""
tests/test_policy_backend_comparison.py

Tests for scripts/10_compare_retrieval_backends.py.
"""

import json
import sys
from pathlib import Path
from typing import Any
import pytest

# Ensure scripts directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_CHUNKS = [
    {
        "chunk_id": "doc_test__dieu_27",
        "doc_id": "doc_test",
        "text": "Điều kiện xét tốt nghiệp bao gồm tích lũy đủ số tín chỉ theo chương trình đào tạo.",
        "section_title": "Điều 27. Điều kiện xét tốt nghiệp và công nhận tốt nghiệp",
        "section_number": "27",
        "chapter_title": "Chương VII",
        "chunk_type": "article",
        "policy_area": ["graduation"],
        "action_tags": ["graduation_audit"],
        "requirement_tags": ["credit_hours"],
        "procedure_tags": [],
        "risk_tags": [],
        "evidence_groups": [],
        "time_tags": [],
    },
    {
        "chunk_id": "doc_test__dieu_4",
        "doc_id": "doc_test",
        "text": "Điều kiện miễn môn học gồm hoàn thành môn tương đương ở trường khác.",
        "section_title": "Điều 4. Điều kiện miễn môn",
        "section_number": "4",
        "chapter_title": "Chương II",
        "chunk_type": "article",
        "policy_area": ["course_exemption"],
        "action_tags": ["request_course_exemption"],
        "requirement_tags": [],
        "procedure_tags": [],
        "risk_tags": [],
        "evidence_groups": [],
        "time_tags": [],
    },
]

SYNTHETIC_CASES = [
    {
        "case_id": "synthetic_graduation_001",
        "question": "Điều kiện xét tốt nghiệp là gì?",
        "expected_issue_types": ["graduation"],
        "expected_first_chunk_id": "doc_test__dieu_27",
        "expected_chunk_ids_any": [],
        "expected_answer_contains_any": [],
        "negative_chunk_ids": [],
    },
    {
        "case_id": "synthetic_exemption_001",
        "question": "Điều kiện miễn môn học là gì?",
        "expected_issue_types": ["course_exemption"],
        "expected_first_chunk_id": None,
        "expected_chunk_ids_any": ["doc_test__dieu_4"],
        "expected_answer_contains_any": [],
        "negative_chunk_ids": [],
    },
]

SYNTHETIC_DOMAIN_CONFIG = {
    "domain_id": "test_domain",
    "issues": [
        {
            "issue_type": "graduation",
            "label": "Tốt nghiệp",
            "policy_area": "graduation",
            "keywords": ["tốt nghiệp", "xét tốt nghiệp"],
            "query": "điều kiện xét tốt nghiệp",
        },
        {
            "issue_type": "course_exemption",
            "label": "Miễn môn",
            "policy_area": "course_exemption",
            "keywords": ["miễn môn", "miễn học phần"],
            "query": "điều kiện miễn môn học",
        },
    ],
}


def _get_compare_module():
    return import_module("10_compare_retrieval_backends")


# ---------------------------------------------------------------------------
# Test 1: script imports and compiles
# ---------------------------------------------------------------------------


def test_compare_script_imports_and_compiles():
    mod = _get_compare_module()
    assert hasattr(mod, "load_cases")
    assert hasattr(mod, "evaluate_backend_on_case")
    assert hasattr(mod, "compare_backends")
    assert hasattr(mod, "main")
    assert hasattr(mod, "print_comparison_report")


# ---------------------------------------------------------------------------
# Test 2: lexical_v0 and bm25_like_v0 are accepted backend names
# ---------------------------------------------------------------------------


def test_lexical_v0_is_accepted_backend_name():
    from policy_retrieval_backends import get_retrieval_backend
    backend = get_retrieval_backend("lexical_v0")
    assert backend.name == "lexical_v0"


def test_bm25_like_v0_is_accepted_backend_name():
    from policy_retrieval_backends import get_retrieval_backend
    backend = get_retrieval_backend("bm25_like_v0")
    assert backend.name == "bm25_like_v0"


# ---------------------------------------------------------------------------
# Test 3: unknown backend name raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_backend_raises_value_error():
    from policy_retrieval_backends import get_retrieval_backend
    with pytest.raises(ValueError, match="Unknown retrieval backend"):
        get_retrieval_backend("super_fancy_neural_backend_v99")


# ---------------------------------------------------------------------------
# Test 4: compare_backends runs on synthetic cases (no real data needed)
# ---------------------------------------------------------------------------


def test_compare_backends_runs_on_synthetic_cases(tmp_path):
    mod = _get_compare_module()

    results = mod.compare_backends(
        cases=SYNTHETIC_CASES,
        chunks=SYNTHETIC_CHUNKS,
        domain_config=SYNTHETIC_DOMAIN_CONFIG,
        backend_names=["lexical_v0", "bm25_like_v0"],
        top_k=5,
        nodes_path=tmp_path / "nonexistent_nodes.jsonl",
        edges_path=tmp_path / "nonexistent_edges.jsonl",
        verbose=False,
    )

    assert "lexical_v0" in results
    assert "bm25_like_v0" in results
    assert len(results["lexical_v0"]) == len(SYNTHETIC_CASES)
    assert len(results["bm25_like_v0"]) == len(SYNTHETIC_CASES)

    # Every result must have required keys
    for backend_name, case_results in results.items():
        for r in case_results:
            assert "case_id" in r
            assert "passed" in r
            assert "checks" in r
            assert "selected_chunk_ids" in r
            assert "first_chunk_id" in r
            assert "backend_name" in r
            assert r["backend_name"] == backend_name


def test_evaluate_backend_on_case_returns_expected_fields(tmp_path):
    mod = _get_compare_module()

    result = mod.evaluate_backend_on_case(
        case=SYNTHETIC_CASES[0],
        chunks=SYNTHETIC_CHUNKS,
        domain_config=SYNTHETIC_DOMAIN_CONFIG,
        backend_name="lexical_v0",
        top_k=5,
        nodes_path=tmp_path / "nonexistent_nodes.jsonl",
        edges_path=tmp_path / "nonexistent_edges.jsonl",
    )

    assert result["case_id"] == "synthetic_graduation_001"
    assert result["backend_name"] == "lexical_v0"
    assert "passed" in result
    assert "first_chunk_pass" in result["checks"]
    assert "chunk_any_pass" in result["checks"]
    assert "negative_chunk_pass" in result["checks"]
    assert isinstance(result["selected_chunk_ids"], list)


def test_evaluate_backend_on_case_bm25_returns_expected_fields(tmp_path):
    mod = _get_compare_module()

    result = mod.evaluate_backend_on_case(
        case=SYNTHETIC_CASES[1],
        chunks=SYNTHETIC_CHUNKS,
        domain_config=SYNTHETIC_DOMAIN_CONFIG,
        backend_name="bm25_like_v0",
        top_k=5,
        nodes_path=tmp_path / "nonexistent_nodes.jsonl",
        edges_path=tmp_path / "nonexistent_edges.jsonl",
    )

    assert result["case_id"] == "synthetic_exemption_001"
    assert result["backend_name"] == "bm25_like_v0"
    assert isinstance(result["passed"], bool)


def test_evaluate_backend_on_case_unknown_backend_raises(tmp_path):
    mod = _get_compare_module()

    with pytest.raises(ValueError, match="Unknown retrieval backend"):
        mod.evaluate_backend_on_case(
            case=SYNTHETIC_CASES[0],
            chunks=SYNTHETIC_CHUNKS,
            domain_config=SYNTHETIC_DOMAIN_CONFIG,
            backend_name="does_not_exist",
            top_k=5,
            nodes_path=tmp_path / "nodes.jsonl",
            edges_path=tmp_path / "edges.jsonl",
        )


# ---------------------------------------------------------------------------
# Test 5: print_comparison_report does not crash (synthetic)
# ---------------------------------------------------------------------------


def test_print_comparison_report_runs_without_error(tmp_path, capsys):
    mod = _get_compare_module()

    results = mod.compare_backends(
        cases=SYNTHETIC_CASES,
        chunks=SYNTHETIC_CHUNKS,
        domain_config=SYNTHETIC_DOMAIN_CONFIG,
        backend_names=["lexical_v0", "bm25_like_v0"],
        top_k=5,
        nodes_path=tmp_path / "nodes.jsonl",
        edges_path=tmp_path / "edges.jsonl",
        verbose=True,
    )
    mod.print_comparison_report(results, verbose=True)

    captured = capsys.readouterr()
    assert "lexical_v0" in captured.out
    assert "bm25_like_v0" in captured.out
    assert "Retrieval Backend Comparison" in captured.out


# ---------------------------------------------------------------------------
# Tests using real generated chunks (skipped if missing)
# ---------------------------------------------------------------------------


def _real_paths():
    root = Path(__file__).parent.parent
    return {
        "cases": root / "data" / "eval" / "ou_policy_cases.jsonl",
        "chunks": root / "data" / "chunks" / "policy_chunks.annotated.jsonl",
        "nodes": root / "data" / "graph" / "policy_graph_nodes.jsonl",
        "edges": root / "data" / "graph" / "policy_graph_edges.jsonl",
        "config": root / "domains" / "ou_academic_policy_v1" / "domain.json",
    }


def test_comparison_script_runs_with_real_files_if_available():
    """Full comparison run with default files exits without exception."""
    paths = _real_paths()
    if not paths["chunks"].exists():
        pytest.skip("Generated chunks not found. Skipping real-data comparison test.")

    mod = _get_compare_module()
    from policy_retrieval_backends import get_retrieval_backend

    chunks = mod.read_jsonl(paths["chunks"])

    import importlib as _il
    _dom = _il.import_module("policy_domain_config")
    domain_config = _dom.load_domain_config(paths["config"])

    cases = mod.load_cases(paths["cases"])

    results = mod.compare_backends(
        cases=cases,
        chunks=chunks,
        domain_config=domain_config,
        backend_names=["lexical_v0", "bm25_like_v0"],
        top_k=5,
        nodes_path=paths["nodes"],
        edges_path=paths["edges"],
        verbose=False,
    )

    assert "lexical_v0" in results
    assert len(results["lexical_v0"]) == len(cases)


def test_grad_conditions_001_passes_lexical_with_real_chunks():
    """grad_conditions_001 must pass first_chunk check for lexical_v0."""
    paths = _real_paths()
    if not paths["chunks"].exists():
        pytest.skip("Generated chunks not found. Skipping real-data test.")

    mod = _get_compare_module()

    import importlib as _il
    _dom = _il.import_module("policy_domain_config")
    domain_config = _dom.load_domain_config(paths["config"])

    chunks = mod.read_jsonl(paths["chunks"])

    # Find the specific case
    cases = mod.load_cases(paths["cases"])
    target = next((c for c in cases if c["case_id"] == "grad_conditions_001"), None)
    assert target is not None, "grad_conditions_001 not found in eval cases"

    result = mod.evaluate_backend_on_case(
        case=target,
        chunks=chunks,
        domain_config=domain_config,
        backend_name="lexical_v0",
        top_k=5,
        nodes_path=paths["nodes"],
        edges_path=paths["edges"],
    )

    assert result["passed"], (
        f"grad_conditions_001 failed for lexical_v0. "
        f"checks={result['checks']}, "
        f"first_chunk={result['first_chunk_id']}, "
        f"selected={result['selected_chunk_ids']}"
    )
    assert result["first_chunk_id"] == "ou_fulltime_credit_training_regulation_2016__dieu_27"
