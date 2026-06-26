#!/usr/bin/env python3
"""
scripts/08_eval_policy_cases.py

Deterministic evaluation script to run policy queries against evaluation cases,
verifying correctness of retrieval and generated answers.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure scripts directory is in path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import answer_policy_question as _qa
import policy_retrieval_service as _service
import policy_domain_config as _domain

infer_case_issues = _qa.infer_case_issues
answer_question = _qa.answer_question
read_jsonl = _qa.read_jsonl
PolicyRetrievalService = _service.PolicyRetrievalService
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
    retrieval_service: "PolicyRetrievalService",
) -> dict[str, Any]:
    """
    Evaluate a single test case and return verification results, including latency.

    Latency covers the real production pipeline: answer_question() which
    internally performs retrieval + answer generation. The service is NOT
    re-created per case; it is passed in from main() to avoid measuring
    initialization overhead.

    retrieve_for_issues() is called separately (outside the timer) only to
    obtain selected_ids needed for correctness assertions (first_chunk_pass,
    negative_chunk_pass). This is evaluation bookkeeping, not part of the
    measured production path.
    """
    question = case["question"]
    expected_issue_types = case.get("expected_issue_types", [])
    expected_first_chunk_id = case.get("expected_first_chunk_id")
    expected_chunk_ids_any = case.get("expected_chunk_ids_any", [])
    expected_answer_contains_any = case.get("expected_answer_contains_any", [])
    negative_chunk_ids = case.get("negative_chunk_ids", [])

    # 1. Infer issues (lightweight keyword pass, not part of timed pipeline)
    inferred = infer_case_issues(question, domain_config=domain_config)
    inferred_types = [iss["issue_type"] for iss in inferred]

    # --- Latency measurement: covers retrieve + answer generation ---
    # answer_question() calls retrieval_service.retrieve_for_issue() per issue
    # internally before generating the answer. This is the real production path.
    start = time.perf_counter()

    answer = answer_question(
        question=question,
        chunks=chunks,
        top_k=top_k,
        domain_config=domain_config,
        retrieval_service=retrieval_service,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    # ----------------------------------------------------------------

    # 2. Obtain selected_ids for correctness assertions.
    # retrieve_for_issues() is called once outside the timer as evaluation
    # bookkeeping. answer_question() already performed retrieval internally
    # above; this call is purely so we can inspect which chunks were ranked first.
    selected = retrieval_service.retrieve_for_issues(
        issues=inferred,
        question=question,
        top_k=top_k,
        max_sources_per_issue=min(3, top_k),
        use_graph=True,
        strict_pruning=True,
    )
    selected_ids = [chunk["chunk_id"] for chunk, _score in selected]
    first_chunk_id = selected_ids[0] if selected_ids else None

    # 3. Run assertions
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
        "elapsed_ms": elapsed_ms,
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

    # Build the retrieval service ONCE. Initializing per-case would load graph
    # files repeatedly and inflate the latency baseline with setup overhead.
    retrieval_service = PolicyRetrievalService(
        chunks=chunks,
        nodes_file=nodes_path,
        edges_file=edges_path,
    )

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
            retrieval_service=retrieval_service,
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

    # --- Compute latency statistics ---
    latencies = [r["elapsed_ms"] for r in results]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # --- Compute pass rates for baseline result (reuse per-case check results) ---
    first_chunk_pass_count = sum(
        1 for r in results if r["checks"]["first_chunk_pass"]
    )
    negative_chunk_pass_count = sum(
        1 for r in results if r["checks"]["negative_chunk_pass"]
    )
    first_chunk_pass_rate = first_chunk_pass_count / total if total > 0 else 0.0
    negative_chunk_pass_rate = negative_chunk_pass_count / total if total > 0 else 0.0

    print("============================================================")
    print("Evaluation Summary")
    print("============================================================")
    print(f"  Total cases : {total}")
    print(f"  Passed      : {passed_count}")
    print(f"  Failed      : {failed_count}")
    print(f"  Pass Rate   : {pass_rate:.1f}%")
    print("============================================================")
    print(f"Average latency: {avg_latency_ms:.2f} ms")

    # --- Save baseline result ---
    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = eval_dir / "baseline_result.json"
    baseline_result = {
        "run_at": datetime.now().isoformat(),
        "backend": "lexical_v0",
        "total_cases": total,
        "passed": passed_count,
        "first_chunk_pass_rate": first_chunk_pass_rate,
        "negative_chunk_pass_rate": negative_chunk_pass_rate,
        "avg_latency_ms": avg_latency_ms,
    }
    with baseline_path.open("w", encoding="utf-8") as fh:
        json.dump(baseline_result, fh, indent=2, ensure_ascii=False)
    print(f"Baseline result saved to: {baseline_path}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
