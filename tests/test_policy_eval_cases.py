"""
tests/test_policy_eval_cases.py

Unit and integration tests for the evaluation runner and cases.

Run with:
    python -m pytest tests/test_policy_eval_cases.py
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module

_eval = import_module("eval_policy_cases")
load_cases = _eval.load_cases
evaluate_case = _eval.evaluate_case

_svc = import_module("policy_retrieval_service")
PolicyRetrievalService = _svc.PolicyRetrievalService


class TestPolicyEvalCases:
    _CASES_PATH = Path(__file__).parent.parent / "data" / "eval" / "ou_policy_cases.jsonl"
    _CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks" / "policy_chunks.annotated.jsonl"
    _CONFIG_PATH = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"

    def test_loads_at_least_30_cases(self):
        assert self._CASES_PATH.exists()
        cases = load_cases(self._CASES_PATH)
        assert len(cases) >= 30

    def test_every_case_has_id_and_question(self):
        cases = load_cases(self._CASES_PATH)
        for idx, case in enumerate(cases):
            assert "case_id" in case, f"Case at index {idx} has no case_id"
            assert "question" in case, f"Case at index {idx} has no question"
            assert isinstance(case["case_id"], str)
            assert isinstance(case["question"], str)

    def test_eval_case_ids_are_unique(self):
        cases = load_cases(self._CASES_PATH)
        ids = [c["case_id"] for c in cases]
        assert len(ids) == len(set(ids)), f"Duplicate case IDs found: {ids}"

    def test_synthetic_case_passes_matching_issue_and_chunk(self):
        synthetic_case = {
            "case_id": "synth_001",
            "question": "Điều kiện xét tốt nghiệp",
            "expected_issue_types": ["graduation"],
            "expected_first_chunk_id": "chunk_27",
            "expected_chunk_ids_any": [],
            "expected_answer_contains_any": ["Điều 27"],
            "negative_chunk_ids": ["chunk_unrelated"]
        }
        
        synthetic_chunks = [
            {
                "chunk_id": "chunk_27",
                "doc_id": "doc_1",
                "section_number": "27",
                "section_title": "Điều 27. Xét tốt nghiệp",
                "text": "Điều kiện xét tốt nghiệp là...",
                "policy_area": ["graduation"]
            }
        ]
        
        # Load minimal configuration
        import policy_domain_config
        config = policy_domain_config.load_domain_config(self._CONFIG_PATH)

        service = PolicyRetrievalService(
            chunks=synthetic_chunks,
            nodes_file=Path("nonexistent_nodes.jsonl"),
            edges_file=Path("nonexistent_edges.jsonl"),
        )
        res = evaluate_case(
            case=synthetic_case,
            chunks=synthetic_chunks,
            domain_config=config,
            top_k=5,
            retrieval_service=service,
        )
        assert res["passed"] is True
        assert res["checks"]["issue_types_pass"] is True
        assert res["checks"]["first_chunk_pass"] is True
        assert res["checks"]["chunk_any_pass"] is True
        assert res["checks"]["answer_contains_pass"] is True
        assert res["checks"]["negative_chunk_pass"] is True

    def test_synthetic_case_fails_mismatching_first_chunk(self):
        synthetic_case = {
            "case_id": "synth_002",
            "question": "Điều kiện xét tốt nghiệp",
            "expected_issue_types": ["graduation"],
            "expected_first_chunk_id": "expected_chunk_27",
            "expected_chunk_ids_any": [],
            "expected_answer_contains_any": [],
            "negative_chunk_ids": []
        }
        
        synthetic_chunks = [
            {
                "chunk_id": "chunk_different",
                "doc_id": "doc_1",
                "section_number": "27",
                "section_title": "Điều 27. Xét tốt nghiệp",
                "text": "Điều kiện xét tốt nghiệp là...",
                "policy_area": ["graduation"]
            }
        ]
        
        import policy_domain_config
        config = policy_domain_config.load_domain_config(self._CONFIG_PATH)

        service = PolicyRetrievalService(
            chunks=synthetic_chunks,
            nodes_file=Path("nonexistent_nodes.jsonl"),
            edges_file=Path("nonexistent_edges.jsonl"),
        )
        res = evaluate_case(
            case=synthetic_case,
            chunks=synthetic_chunks,
            domain_config=config,
            top_k=5,
            retrieval_service=service,
        )
        assert res["passed"] is False
        assert res["checks"]["issue_types_pass"] is True
        assert res["checks"]["first_chunk_pass"] is False

    def test_real_chunks_grad_conditions_passes(self):
        if not self._CHUNKS_PATH.exists():
            pytest.skip("policy_chunks.annotated.jsonl not built")
            
        cases = load_cases(self._CASES_PATH)
        target_case = None
        for c in cases:
            if c["case_id"] == "grad_conditions_001":
                target_case = c
                break
                
        assert target_case is not None
        
        import json
        real_chunks = []
        with self._CHUNKS_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    real_chunks.append(json.loads(line))
                    
        import policy_domain_config
        config = policy_domain_config.load_domain_config(self._CONFIG_PATH)

        # Retrieve paths to graph files if they exist
        nodes_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
        edges_path = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"

        service = PolicyRetrievalService(
            chunks=real_chunks,
            nodes_file=nodes_path,
            edges_file=edges_path,
        )
        res = evaluate_case(
            case=target_case,
            chunks=real_chunks,
            domain_config=config,
            top_k=5,
            retrieval_service=service,
        )

        # Check assertions for grad_conditions_001 under real chunks
        assert res["checks"]["issue_types_pass"] is True
        assert res["checks"]["first_chunk_pass"] is True
        assert res["checks"]["answer_contains_pass"] is True
        assert res["passed"] is True
