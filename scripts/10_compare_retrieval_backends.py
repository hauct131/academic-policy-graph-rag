#!/usr/bin/env python3
"""
scripts/10_compare_retrieval_backends.py

Deterministic retrieval backend comparison script for Academic Policy Graph RAG.

Compares the retrieval quality of multiple backends (e.g. lexical_v0, bm25_like_v0)
against the existing evaluation cases in data/eval/ou_policy_cases.jsonl.

Only retrieval-level checks are evaluated:
    - expected_first_chunk_id  (if present in case)
    - expected_chunk_ids_any   (if present in case)
    - negative_chunk_ids       (if present in case)

Answer-text evaluation is NOT performed here. Use 08_eval_policy_cases.py for that.

Usage:
    python scripts/10_compare_retrieval_backends.py
    python scripts/10_compare_retrieval_backends.py --backends lexical_v0,bm25_like_v0 --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure scripts folder is in python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import importlib

try:
    _retriever = importlib.import_module("05_retrieve_policy_chunks")
    _qa = importlib.import_module("06_answer_policy_question")
    _domain = importlib.import_module("policy_domain_config")
    _service = importlib.import_module("policy_retrieval_service")
    _backends_mod = importlib.import_module("policy_retrieval_backends")
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _retriever = importlib.import_module("05_retrieve_policy_chunks")
    _qa = importlib.import_module("06_answer_policy_question")
    _domain = importlib.import_module("policy_domain_config")
    _service = importlib.import_module("policy_retrieval_service")
    _backends_mod = importlib.import_module("policy_retrieval_backends")

read_jsonl = _retriever.read_jsonl
infer_case_issues = _qa.infer_case_issues
load_domain_config = _domain.load_domain_config
PolicyRetrievalService = _service.PolicyRetrievalService
get_retrieval_backend = _backends_mod.get_retrieval_backend



# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-case evaluation (retrieval only)
# ---------------------------------------------------------------------------


def evaluate_backend_on_case(
    case: dict[str, Any],
    chunks: list[dict[str, Any]],
    domain_config: dict[str, Any],
    backend_name: str,
    top_k: int,
    nodes_path: Path,
    edges_path: Path,
) -> dict[str, Any]:
    """
    Run retrieval for one eval case with the specified backend.

    Returns a result dict with:
        case_id, backend_name, passed, checks, selected_chunk_ids, first_chunk_id
    """
    question = case["question"]
    expected_first_chunk_id = case.get("expected_first_chunk_id")
    expected_chunk_ids_any: list[str] = case.get("expected_chunk_ids_any") or []
    negative_chunk_ids: list[str] = case.get("negative_chunk_ids") or []

    # Validate backend name early (raises ValueError for unknown names)
    get_retrieval_backend(backend_name)

    # Instantiate service with requested backend
    service = PolicyRetrievalService(
        chunks=chunks,
        nodes_file=nodes_path,
        edges_file=edges_path,
        backend_name=backend_name,
    )

    # Infer issues
    issues = infer_case_issues(question, domain_config=domain_config)

    # Retrieve
    selected_pairs = service.retrieve_for_issues(
        issues=issues,
        question=question,
        top_k=top_k,
        max_sources_per_issue=min(3, top_k),
        use_graph=True,
        strict_pruning=True,
    )
    selected_ids = [chunk["chunk_id"] for chunk, _ in selected_pairs]
    first_chunk_id = selected_ids[0] if selected_ids else None

    # --- Retrieval checks ---
    first_chunk_pass = True
    if expected_first_chunk_id:
        first_chunk_pass = (first_chunk_id == expected_first_chunk_id)

    chunk_any_pass = True
    if expected_chunk_ids_any:
        chunk_any_pass = any(cid in selected_ids for cid in expected_chunk_ids_any)

    negative_chunk_pass = True
    if negative_chunk_ids:
        negative_chunk_pass = not any(cid in selected_ids for cid in negative_chunk_ids)

    passed = first_chunk_pass and chunk_any_pass and negative_chunk_pass

    return {
        "case_id": case["case_id"],
        "backend_name": backend_name,
        "passed": passed,
        "checks": {
            "first_chunk_pass": first_chunk_pass,
            "chunk_any_pass": chunk_any_pass,
            "negative_chunk_pass": negative_chunk_pass,
        },
        "selected_chunk_ids": selected_ids,
        "first_chunk_id": first_chunk_id,
        "expected_first_chunk_id": expected_first_chunk_id,
        "expected_chunk_ids_any": expected_chunk_ids_any,
    }


# ---------------------------------------------------------------------------
# Backend comparison across all cases
# ---------------------------------------------------------------------------


def compare_backends(
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    domain_config: dict[str, Any],
    backend_names: list[str],
    top_k: int,
    nodes_path: Path,
    edges_path: Path,
    verbose: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """
    Run retrieval evaluation for all backends and all cases.

    Returns a dict mapping backend_name → list of per-case result dicts.
    """
    all_results: dict[str, list[dict[str, Any]]] = {}

    for backend_name in backend_names:
        results = []
        for case in cases:
            res = evaluate_backend_on_case(
                case=case,
                chunks=chunks,
                domain_config=domain_config,
                backend_name=backend_name,
                top_k=top_k,
                nodes_path=nodes_path,
                edges_path=edges_path,
            )
            results.append(res)
        all_results[backend_name] = results

    return all_results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def _col(text: str, width: int) -> str:
    """Left-justified column with fixed width."""
    return str(text).ljust(width)


def print_comparison_report(
    all_results: dict[str, list[dict[str, Any]]],
    verbose: bool = False,
) -> None:
    backend_names = list(all_results.keys())

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("Retrieval Backend Comparison — Summary")
    print("=" * 74)
    header = (
        _col("Backend", 18)
        + _col("Cases", 7)
        + _col("1st✓", 6)
        + _col("Any✓", 6)
        + _col("Neg✓", 6)
        + _col("Pass", 6)
        + "Rate"
    )
    print(header)
    print("-" * 74)

    summary: dict[str, dict[str, Any]] = {}
    for backend_name, results in all_results.items():
        total = len(results)
        first_ok = sum(1 for r in results if r["checks"]["first_chunk_pass"])
        any_ok = sum(1 for r in results if r["checks"]["chunk_any_pass"])
        neg_ok = sum(1 for r in results if r["checks"]["negative_chunk_pass"])
        passed = sum(1 for r in results if r["passed"])
        rate = (passed / total * 100) if total > 0 else 0.0
        summary[backend_name] = {
            "total": total,
            "first_ok": first_ok,
            "any_ok": any_ok,
            "neg_ok": neg_ok,
            "passed": passed,
            "rate": rate,
        }
        print(
            _col(backend_name, 18)
            + _col(total, 7)
            + _col(first_ok, 6)
            + _col(any_ok, 6)
            + _col(neg_ok, 6)
            + _col(passed, 6)
            + f"{rate:.1f}%"
        )
    print("=" * 74)

    # ── Per-case differences ─────────────────────────────────────────────────
    if len(backend_names) >= 2:
        print("\nPer-case differences (cases where backends disagree):")
        print("-" * 74)

        # Collect all case IDs in order
        first_backend = backend_names[0]
        case_ids = [r["case_id"] for r in all_results[first_backend]]

        diffs_found = False
        for case_id in case_ids:
            results_by_backend = {}
            for bn in backend_names:
                for r in all_results[bn]:
                    if r["case_id"] == case_id:
                        results_by_backend[bn] = r
                        break

            pass_states = [results_by_backend[bn]["passed"] for bn in backend_names if bn in results_by_backend]
            if len(set(pass_states)) > 1 or verbose:
                diffs_found = True
                print(f"\n  Case: {case_id}")
                for bn in backend_names:
                    if bn not in results_by_backend:
                        continue
                    r = results_by_backend[bn]
                    status = "PASS" if r["passed"] else "FAIL"
                    checks = r["checks"]
                    print(
                        f"    [{bn}] {status}"
                        f"  1st={checks['first_chunk_pass']}"
                        f"  any={checks['chunk_any_pass']}"
                        f"  neg={checks['negative_chunk_pass']}"
                    )
                    if verbose:
                        print(f"      First chunk : {r['first_chunk_id']}")
                        print(f"      Selected    : {r['selected_chunk_ids']}")

        if not diffs_found:
            print("  (all backends agree on all cases)")

    print("-" * 74)

    # ── Verbose per-backend full listing ─────────────────────────────────────
    if verbose:
        for backend_name, results in all_results.items():
            print(f"\nDetailed results for backend: {backend_name}")
            print("-" * 60)
            for r in results:
                status = "PASS" if r["passed"] else "FAIL"
                print(f"  [{status}] {r['case_id']}")
                if not r["passed"]:
                    checks = r["checks"]
                    print(f"         checks      : {checks}")
                    print(f"         first_chunk : {r['first_chunk_id']}")
                    print(f"         expected_1st: {r['expected_first_chunk_id']}")
                    print(f"         selected    : {r['selected_chunk_ids']}")
            print("-" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare retrieval backends on evaluation cases."
    )
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
        "--backends",
        default="lexical_v0,bm25_like_v0",
        help="Comma-separated list of backend names to compare (default: lexical_v0,bm25_like_v0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K chunks to retrieve per issue (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose case-level details",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cases_path = Path(args.cases_file)
    chunks_path = Path(args.chunks_file)
    nodes_path = Path(args.nodes_file)
    edges_path = Path(args.edges_file)
    config_path = Path(args.domain_config)
    backend_names = [b.strip() for b in args.backends.split(",") if b.strip()]

    # Validate required files
    errors = []
    if not cases_path.exists():
        errors.append(f"Cases file not found: {cases_path}")
    if not chunks_path.exists():
        errors.append(f"Chunks file not found: {chunks_path}")
    if not config_path.exists():
        errors.append(f"Domain config not found: {config_path}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        if not chunks_path.exists():
            print(
                "\nPlease run the ingestion pipeline first:\n"
                "  python scripts/01_build_policy_chunks.py\n"
                "  python scripts/02_annotate_policy_chunks.py\n"
                "  python scripts/03_build_policy_graph.py",
                file=sys.stderr,
            )
        return 1

    # Validate backend names early
    for bn in backend_names:
        try:
            get_retrieval_backend(bn)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    # Load data
    print(f"Loading evaluation cases from: {cases_path}")
    cases = load_cases(cases_path)
    print(f"Loading chunks from          : {chunks_path}")
    chunks = read_jsonl(chunks_path)
    print(f"Loading domain config from   : {config_path}")
    domain_config = load_domain_config(config_path)
    print(f"Backends to compare          : {', '.join(backend_names)}")
    print(f"Cases                        : {len(cases)}")
    print(f"Chunks                       : {len(chunks)}")

    # Run comparison
    all_results = compare_backends(
        cases=cases,
        chunks=chunks,
        domain_config=domain_config,
        backend_names=backend_names,
        top_k=args.top_k,
        nodes_path=nodes_path,
        edges_path=edges_path,
        verbose=args.verbose,
    )

    # Print report
    print_comparison_report(all_results, verbose=args.verbose)

    # Return 0 if at least one backend achieved 100% pass on retrieval checks
    any_perfect = any(
        all(r["passed"] for r in results)
        for results in all_results.values()
    )
    return 0 if any_perfect else 1


if __name__ == "__main__":
    sys.exit(main())
