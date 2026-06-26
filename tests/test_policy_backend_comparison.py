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

import compare_retrieval_backends as _compare_mod
import policy_domain_config as _pdc_mod


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
    return _compare_mod


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
    mod.print_comparison_report(results, cases=SYNTHETIC_CASES, top_k=5, verbose=True)

    captured = capsys.readouterr()
    assert "lexical_v0" in captured.out
    assert "bm25_like_v0" in captured.out
    assert "Retrieval Backend Comparison" in captured.out


# ---------------------------------------------------------------------------
# Tests using real generated chunks (skipped if missing)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# IR metric tests (synthetic, no real data needed)
# ---------------------------------------------------------------------------


def test_relevant_ids_for_case_combines_first_and_any():
    mod = _get_compare_module()
    case = {
        "expected_first_chunk_id": "chunk_a",
        "expected_chunk_ids_any": ["chunk_b", "chunk_c"],
    }
    ids = mod._relevant_ids_for_case(case)
    assert ids == ["chunk_a", "chunk_b", "chunk_c"]


def test_relevant_ids_for_case_deduplicates():
    mod = _get_compare_module()
    case = {
        "expected_first_chunk_id": "chunk_a",
        "expected_chunk_ids_any": ["chunk_a", "chunk_b"],
    }
    ids = mod._relevant_ids_for_case(case)
    assert ids.count("chunk_a") == 1
    assert "chunk_b" in ids


def test_relevant_ids_for_case_empty_when_none_defined():
    mod = _get_compare_module()
    case = {"expected_first_chunk_id": None, "expected_chunk_ids_any": []}
    assert mod._relevant_ids_for_case(case) == []


def test_compute_ir_metrics_perfect_retrieval():
    """When the expected chunk is ranked first, all metrics should be 1.0."""
    mod = _get_compare_module()
    cases = [
        {
            "case_id": "c1",
            "expected_first_chunk_id": "chunk_a",
            "expected_chunk_ids_any": [],
        }
    ]
    results = [
        {
            "case_id": "c1",
            "selected_chunk_ids": ["chunk_a", "chunk_b"],
        }
    ]
    ir = mod.compute_ir_metrics(results, cases, k=5)
    assert ir["recall_at_k"] == 1.0
    assert ir["mrr"] == 1.0
    assert ir["ndcg_at_k"] == 1.0


def test_compute_ir_metrics_zero_when_not_retrieved():
    """When the expected chunk is absent, all metrics should be 0.0."""
    mod = _get_compare_module()
    cases = [
        {
            "case_id": "c1",
            "expected_first_chunk_id": "chunk_a",
            "expected_chunk_ids_any": [],
        }
    ]
    results = [
        {
            "case_id": "c1",
            "selected_chunk_ids": ["chunk_z"],
        }
    ]
    ir = mod.compute_ir_metrics(results, cases, k=5)
    assert ir["recall_at_k"] == 0.0
    assert ir["mrr"] == 0.0
    assert ir["ndcg_at_k"] == 0.0


def test_compute_ir_metrics_partial_mrr():
    """Relevant chunk at rank 2 → MRR = 0.5, Recall = 1."""
    mod = _get_compare_module()
    cases = [
        {
            "case_id": "c1",
            "expected_first_chunk_id": "chunk_a",
            "expected_chunk_ids_any": [],
        }
    ]
    results = [
        {
            "case_id": "c1",
            "selected_chunk_ids": ["chunk_z", "chunk_a"],
        }
    ]
    ir = mod.compute_ir_metrics(results, cases, k=5)
    assert ir["recall_at_k"] == 1.0
    assert abs(ir["mrr"] - 0.5) < 1e-9


def test_compute_ir_metrics_excludes_unannotated_cases():
    """Cases without any relevant IDs must not affect metric denominators."""
    mod = _get_compare_module()
    cases = [
        {
            "case_id": "c_annotated",
            "expected_first_chunk_id": "chunk_a",
            "expected_chunk_ids_any": [],
        },
        {
            "case_id": "c_unannotated",
            "expected_first_chunk_id": None,
            "expected_chunk_ids_any": [],
        },
    ]
    results = [
        {"case_id": "c_annotated", "selected_chunk_ids": ["chunk_a"]},
        {"case_id": "c_unannotated", "selected_chunk_ids": []},
    ]
    ir = mod.compute_ir_metrics(results, cases, k=5)
    # Denominator should be 1 (only the annotated case)
    assert ir["recall_n"] == 1
    assert ir["mrr_n"] == 1
    assert ir["recall_at_k"] == 1.0
    assert ir["mrr"] == 1.0


def test_compute_ir_metrics_recall_uses_expected_chunk_ids_any():
    """Recall@k should fire when any chunk from expected_chunk_ids_any is present."""
    mod = _get_compare_module()
    cases = [
        {
            "case_id": "c1",
            "expected_first_chunk_id": None,
            "expected_chunk_ids_any": ["chunk_b", "chunk_c"],
        }
    ]
    results = [
        {"case_id": "c1", "selected_chunk_ids": ["chunk_b"]},
    ]
    ir = mod.compute_ir_metrics(results, cases, k=5)
    assert ir["recall_at_k"] == 1.0
    # MRR denominator is 0 (no expected_first_chunk_id)
    assert ir["mrr_n"] == 0


def test_print_comparison_report_shows_ir_metrics(tmp_path, capsys):
    """IR metrics table is printed when cases are provided."""
    mod = _get_compare_module()
    results = mod.compare_backends(
        cases=SYNTHETIC_CASES,
        chunks=SYNTHETIC_CHUNKS,
        domain_config=SYNTHETIC_DOMAIN_CONFIG,
        backend_names=["lexical_v0"],
        top_k=5,
        nodes_path=tmp_path / "nodes.jsonl",
        edges_path=tmp_path / "edges.jsonl",
        verbose=False,
    )
    mod.print_comparison_report(results, cases=SYNTHETIC_CASES, top_k=5, verbose=False)
    captured = capsys.readouterr()
    assert "Recall@5" in captured.out
    assert "MRR" in captured.out
    assert "nDCG@5" in captured.out





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

    domain_config = _pdc_mod.load_domain_config(paths["config"])

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

    domain_config = _pdc_mod.load_domain_config(paths["config"])

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


# ---------------------------------------------------------------------------
# New tests for analyze-bm25-retrieval-gap
# ---------------------------------------------------------------------------

def test_case_id_filtering_helper_works():
    mod = _get_compare_module()
    cases = [
        {"case_id": "case_1", "question": "Q1"},
        {"case_id": "case_2", "question": "Q2"},
    ]
    # Filter by specific case_id
    filtered = mod.filter_cases_by_id(cases, "case_1")
    assert len(filtered) == 1
    assert filtered[0]["case_id"] == "case_1"

    # Filter with None/empty should return all
    assert len(mod.filter_cases_by_id(cases, None)) == 2
    assert len(mod.filter_cases_by_id(cases, "")) == 2


def test_diagnostic_output_renders_failed_synthetic_case(capsys):
    mod = _get_compare_module()
    synthetic_case = {
        "case_id": "synth_fail_001",
        "question": "Question text",
        "notes": "Some notes here",
    }
    synthetic_result = {
        "case_id": "synth_fail_001",
        "question": "Question text",
        "backend_name": "bm25_like_v0",
        "passed": False,
        "checks": {
            "first_chunk_pass": False,
            "chunk_any_pass": True,
            "negative_chunk_pass": True,
        },
        "selected_chunk_ids": ["chunk_b"],
        "selected_pairs": [
            ({"chunk_id": "chunk_b", "text": "This is chunk b text", "doc_id": "doc_1", "section_title": "Sec B", "policy_area": ["test"], "action_tags": []}, 0.85)
        ],
        "first_chunk_id": "chunk_b",
        "expected_first_chunk_id": "chunk_a",
        "expected_chunk_ids_any": ["chunk_b"],
    }
    raw_pairs = [
        ({"chunk_id": "chunk_a", "text": "This is chunk a text", "doc_id": "doc_1", "section_title": "Sec A", "policy_area": ["test"]}, 0.90),
        ({"chunk_id": "chunk_b", "text": "This is chunk b text", "doc_id": "doc_1", "section_title": "Sec B", "policy_area": ["test"]}, 0.85),
    ]

    mod.print_failure_diagnosis(synthetic_result, synthetic_case, raw_pairs=raw_pairs)
    captured = capsys.readouterr()
    assert "DIAGNOSIS: [bm25_like_v0] synth_fail_001" in captured.out
    assert "Question text" in captured.out
    assert "Some notes here" in captured.out
    assert "chunk_a" in captured.out
    assert "chunk_b" in captured.out


def test_unknown_case_id_returns_clear_failure_or_warning_message(capsys):
    mod = _get_compare_module()
    # Running main with non-existent case_id should return 1
    # We pass temporary paths to avoid errors from main loading chunks
    paths = _real_paths()
    if not paths["chunks"].exists():
        pytest.skip("Real chunks file needed for integration test of main()")

    # Run main with a bad case-id
    ret = mod.main([
        "--case-id", "non_existent_case_9999",
        "--cases-file", str(paths["cases"]),
        "--chunks-file", str(paths["chunks"]),
        "--nodes-file", str(paths["nodes"]),
        "--edges-file", str(paths["edges"]),
        "--domain-config", str(paths["config"]),
    ])
    assert ret == 1
    captured = capsys.readouterr()
    assert "No case found with case_id='non_existent_case_9999'" in captured.err


def test_real_case_diagnostic_run_exits_successfully():
    paths = _real_paths()
    if not paths["chunks"].exists():
        pytest.skip("Generated chunks not found. Skipping real-data diagnostic test.")

    mod = _get_compare_module()
    # Run main targeting graduation_transcript_procedure_001 with diagnostics and raw top
    ret = mod.main([
        "--case-id", "graduation_transcript_procedure_001",
        "--diagnose-failures",
        "--show-raw-top",
        "--cases-file", str(paths["cases"]),
        "--chunks-file", str(paths["chunks"]),
        "--nodes-file", str(paths["nodes"]),
        "--edges-file", str(paths["edges"]),
        "--domain-config", str(paths["config"]),
        "--backends", "lexical_v0,bm25_like_v0",
    ])
    # Should exit successfully (0) because lexical_v0 passes graduation_transcript_procedure_001 perfectly
    assert ret == 0

