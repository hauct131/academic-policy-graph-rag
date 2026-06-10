#!/usr/bin/env python3
"""
scripts/08_eval_policy_cases.py

Deterministic evaluation script to run policy queries against evaluation cases,
verifying correctness of retrieval and generated answers.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure scripts directory is in path
sys.path.insert(0, str(Path(__file__).parent.absolute()))
import importlib

try:
    _qa = importlib.import_module("06_answer_policy_question")
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _qa = importlib.import_module("06_answer_policy_question")

infer_case_issues = _qa.infer_case_issues
answer_question = _qa.answer_question
read_jsonl = _qa.read_jsonl
load_graph_expansion = _qa.load_graph_expansion
retrieve_chunks = _qa._retriever.retrieve_chunks

try:
    _selector = importlib.import_module("07_select_policy_sources")
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _selector = importlib.import_module("07_select_policy_sources")

select_sources_for_issue = _selector.select_sources_for_issue

try:
    _domain = importlib.import_module("policy_domain_config")
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _domain = importlib.import_module("policy_domain_config")

load_domain_config = _domain.load_domain_config


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation cases from a JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cases file not found: {p}")
    cases = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def evaluate_case(
    case: dict[str, Any],
    chunks: list[dict[str, Any]],
    domain_config: dict[str, Any],
    top_k: int,
    nodes_path: Path,
    edges_path: Path
) -> dict[str, Any]:
    """Evaluate a single test case and return verification results."""
    question = case["question"]
    expected_issue_types = case.get("expected_issue_types", [])
    expected_first_chunk_id = case.get("expected_first_chunk_id")
    expected_chunk_ids_any = case.get("expected_chunk_ids_any", [])
    expected_answer_contains_any = case.get("expected_answer_contains_any", [])
    negative_chunk_ids = case.get("negative_chunk_ids", [])

    # Load graph bonus map if files exist
    graph_bonus_map = {}
    if nodes_path.exists() and edges_path.exists():
        try:
            graph_bonus_map = load_graph_expansion(question, nodes_path, edges_path)
        except Exception:
            pass

    # 1. Infer issues
    inferred = infer_case_issues(question, domain_config=domain_config)
    inferred_types = [iss["issue_type"] for iss in inferred]

    # 2. Retrieve selected sources
    selected_chunks = []
    for issue in inferred:
        p_area = issue["policy_area"]
        results = retrieve_chunks(
            chunks=chunks,
            query=issue["query"],
            top_k=top_k,
            policy_area=p_area,
            graph_bonus_map=graph_bonus_map,
        )
        selected = select_sources_for_issue(issue, results, max_sources=min(3, top_k))
        for chunk, score in selected:
            selected_chunks.append(chunk)

    selected_ids = [c["chunk_id"] for c in selected_chunks]
    first_chunk_id = selected_ids[0] if selected_ids else None

    # 3. Generate answer
    answer = answer_question(
        question=question,
        chunks=chunks,
        top_k=top_k,
        graph_bonus_map=graph_bonus_map,
        domain_config=domain_config,
    )

    # 4. Run assertions
    issue_types_pass = all(t in inferred_types for t in expected_issue_types)

    first_chunk_pass = True
    if expected_first_chunk_id:
        first_chunk_pass = (first_chunk_id == expected_first_chunk_id)

    chunk_any_pass = True
    if expected_chunk_ids_any:
        chunk_any_pass = any(cid in selected_ids for cid in expected_chunk_ids_any)

    answer_contains_pass = True
    if expected_answer_contains_any:
        answer_lower = answer.lower()
        answer_contains_pass = any(phrase.lower() in answer_lower for phrase in expected_answer_contains_any)

    negative_chunk_pass = True
    if negative_chunk_ids:
        negative_chunk_pass = not any(cid in selected_ids for cid in negative_chunk_ids)

    passed = (
        issue_types_pass
        and first_chunk_pass
        and chunk_any_pass
        and answer_contains_pass
        and negative_chunk_pass
    )

    return {
        "case_id": case["case_id"],
        "passed": passed,
        "checks": {
            "issue_types_pass": issue_types_pass,
            "first_chunk_pass": first_chunk_pass,
            "chunk_any_pass": chunk_any_pass,
            "answer_contains_pass": answer_contains_pass,
            "negative_chunk_pass": negative_chunk_pass,
        },
        "inferred_issues": inferred_types,
        "selected_chunk_ids": selected_ids,
        "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic Evaluation Runner.")
    parser.add_argument(
        "--cases-file",
        default="data/eval/ou_policy_cases.jsonl",
        help="Path to evaluation cases JSONL",
    )
    parser.add_argument(
        "--chunks-file",
        default="data/chunks/policy_chunks.annotated.jsonl",
        help="Path to annotated chunks JSONL",
    )
    parser.add_argument(
        "--nodes-file",
        default="data/graph/policy_graph_nodes.jsonl",
        help="Path to graph nodes JSONL",
    )
    parser.add_argument(
        "--edges-file",
        default="data/graph/policy_graph_edges.jsonl",
        help="Path to graph edges JSONL",
    )
    parser.add_argument(
        "--domain-config",
        default="domains/ou_academic_policy_v1/domain.json",
        help="Path to domain configuration JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K chunks to retrieve per issue",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose details for passing cases as well",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    
    cases_path = Path(args.cases_file)
    chunks_path = Path(args.chunks_file)
    nodes_path = Path(args.nodes_file)
    edges_path = Path(args.edges_file)
    config_path = Path(args.domain_config)

    if not cases_path.exists():
        print(f"[ERROR] Cases file not found: {cases_path}", file=sys.stderr)
        return 1
    if not chunks_path.exists():
        print(f"[ERROR] Chunks file not found: {chunks_path}", file=sys.stderr)
        return 1
    if not config_path.exists():
        print(f"[ERROR] Domain config not found: {config_path}", file=sys.stderr)
        return 1

    # Load data
    cases = load_cases(cases_path)
    chunks = read_jsonl(chunks_path)
    domain_config = load_domain_config(config_path)

    passed_count = 0
    results = []

    print("============================================================")
    print("Running Academic Policy RAG Evaluation")
    print("============================================================")

    for case in cases:
        res = evaluate_case(
            case=case,
            chunks=chunks,
            domain_config=domain_config,
            top_k=args.top_k,
            nodes_path=nodes_path,
            edges_path=edges_path,
        )
        results.append(res)
        if res["passed"]:
            passed_count += 1
            if args.verbose:
                print(f"[PASS] {res['case_id']}")
        else:
            print(f"[FAIL] {res['case_id']}")
            print(f"  Question: {case['question']}")
            print(f"  Checks: {res['checks']}")
            print(f"  Inferred: {res['inferred_issues']}")
            print(f"  Selected chunks: {res['selected_chunk_ids']}")
            print(f"  Answer preview: {res['answer_preview']}")
            print()

    total = len(cases)
    failed_count = total - passed_count
    pass_rate = (passed_count / total) * 100 if total > 0 else 0.0

    print("============================================================")
    print("Evaluation Summary")
    print("============================================================")
    print(f"  Total cases : {total}")
    print(f"  Passed      : {passed_count}")
    print(f"  Failed      : {failed_count}")
    print(f"  Pass Rate   : {pass_rate:.1f}%")
    print("============================================================")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
