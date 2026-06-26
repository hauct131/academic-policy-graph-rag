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
    python scripts/10_compare_retrieval_backends.py \\
        --backends lexical_v0,bm25_like_v0 \\
        --case-id graduation_transcript_procedure_001 \\
        --diagnose-failures \\
        --show-raw-top \\
        --verbose
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Ensure scripts folder is in python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import retrieve_policy_chunks as _retriever
import answer_policy_question as _qa
import policy_domain_config as _domain
import policy_retrieval_service as _service
import policy_retrieval_backends as _backends_mod

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


def filter_cases_by_id(
    cases: list[dict[str, Any]],
    case_id: str | None,
) -> list[dict[str, Any]]:
    """
    Return cases filtered to a single case_id when specified.

    Args:
        cases: Full list of evaluation cases.
        case_id: If provided, return only the case with this ID.
                 If None or empty, return the full list unchanged.

    Returns:
        Filtered (or original) list of cases.
    """
    if not case_id:
        return cases
    matched = [c for c in cases if c["case_id"] == case_id]
    return matched  # may be empty — caller handles that


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

    # Retrieve through service (source selection + strict pruning applied)
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
        "question": question,
        "backend_name": backend_name,
        "passed": passed,
        "checks": {
            "first_chunk_pass": first_chunk_pass,
            "chunk_any_pass": chunk_any_pass,
            "negative_chunk_pass": negative_chunk_pass,
        },
        "selected_chunk_ids": selected_ids,
        "selected_pairs": selected_pairs,  # (chunk, score) list for diagnostics
        "first_chunk_id": first_chunk_id,
        "expected_first_chunk_id": expected_first_chunk_id,
        "expected_chunk_ids_any": expected_chunk_ids_any,
    }


def get_raw_backend_results(
    case: dict[str, Any],
    chunks: list[dict[str, Any]],
    domain_config: dict[str, Any],
    backend_name: str,
    top_k: int,
    nodes_path: Path,
    edges_path: Path,
) -> list[tuple[dict[str, Any], float]]:
    """
    Retrieve raw backend scores BEFORE source selection and strict pruning.

    This exposes the backend's raw ranking, useful for diagnosing whether a
    failure is due to backend scoring, source selection, or strict pruning.

    Returns:
        List of (chunk, score) tuples from the backend, sorted by score desc.
    """
    backend = get_retrieval_backend(backend_name)
    issues = infer_case_issues(case["question"], domain_config=domain_config)

    # Build graph bonus map via a temporary service (same logic, no production change)
    service = PolicyRetrievalService(
        chunks=chunks,
        nodes_file=nodes_path,
        edges_file=edges_path,
        backend_name=backend_name,
    )
    graph_bonus_map = service.build_graph_bonus_map(case["question"])

    # Call backend directly for each issue and aggregate raw results
    all_raw: list[tuple[dict[str, Any], float]] = []
    seen_ids: set[str] = set()

    for issue in issues:
        raw = backend.retrieve(
            chunks=chunks,
            query=issue["query"],
            top_k=top_k,
            policy_area=issue.get("policy_area"),
            graph_bonus_map=graph_bonus_map,
        )
        for chunk, score in raw:
            cid = chunk.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_raw.append((chunk, score))

    # Re-sort by score descending
    all_raw.sort(key=lambda x: -x[1])
    return all_raw[:top_k]


# ---------------------------------------------------------------------------
# IR metric computation
# ---------------------------------------------------------------------------


def _relevant_ids_for_case(case: dict[str, Any]) -> list[str]:
    """
    Return the list of chunk IDs considered relevant for a case.
    Combines expected_first_chunk_id and expected_chunk_ids_any.
    Returns an empty list when no relevant chunks are defined (case excluded
    from IR metric denominators).
    """
    relevant: list[str] = []
    first = case.get("expected_first_chunk_id")
    if first:
        relevant.append(first)
    for cid in (case.get("expected_chunk_ids_any") or []):
        if cid not in relevant:
            relevant.append(cid)
    return relevant


def compute_ir_metrics(
    results: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    k: int,
) -> dict[str, float]:
    """
    Compute standard IR metrics from retrieval results.

    Metrics:
        recall_at_k  — fraction of cases (with relevant chunks defined) where
                       at least one relevant chunk appears in the top-k results.
        mrr          — Mean Reciprocal Rank over cases that have
                       expected_first_chunk_id defined (rank of that chunk).
        ndcg_at_k    — Mean nDCG@k with binary relevance, averaged over cases
                       that have at least one relevant chunk defined.

    Cases with no relevant chunks defined are excluded from each metric's
    denominator, so the metrics are never diluted by unannotated cases.

    Args:
        results: Per-case result dicts from evaluate_backend_on_case.
        cases:   Original eval case dicts (same order as results).
        k:       Cutoff for Recall@k and nDCG@k.

    Returns:
        Dict with keys: recall_at_k, mrr, ndcg_at_k.
    """
    case_by_id = {c["case_id"]: c for c in cases}

    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg_scores: list[float] = []

    for res in results:
        original_case = case_by_id.get(res["case_id"], {})
        relevant_ids = _relevant_ids_for_case(original_case)
        selected_ids: list[str] = res.get("selected_chunk_ids", [])
        top_k_ids = selected_ids[:k]

        # --- Recall@k ---
        if relevant_ids:
            hit = any(cid in top_k_ids for cid in relevant_ids)
            recall_scores.append(1.0 if hit else 0.0)

        # --- MRR (uses expected_first_chunk_id only) ---
        first_expected = original_case.get("expected_first_chunk_id")
        if first_expected:
            rr = 0.0
            for rank, cid in enumerate(selected_ids, 1):
                if cid == first_expected:
                    rr = 1.0 / rank
                    break
            mrr_scores.append(rr)

        # --- nDCG@k ---
        if relevant_ids:
            # Binary relevance: 1 if chunk in relevant set, else 0
            gains = [
                1.0 if cid in relevant_ids else 0.0
                for cid in top_k_ids
            ]
            # DCG
            dcg = sum(
                gain / math.log2(rank + 1)
                for rank, gain in enumerate(gains, 1)
            )
            # Ideal DCG: all relevant at top positions
            n_relevant_in_k = min(len(relevant_ids), k)
            idcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, n_relevant_in_k + 1)
            )
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    def _mean(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    return {
        "recall_at_k": _mean(recall_scores),
        "mrr": _mean(mrr_scores),
        "ndcg_at_k": _mean(ndcg_scores),
        "recall_n": len(recall_scores),
        "mrr_n": len(mrr_scores),
        "ndcg_n": len(ndcg_scores),
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
# Diagnostics rendering
# ---------------------------------------------------------------------------


def _chunk_preview(chunk: dict[str, Any], max_len: int = 120) -> str:
    text = chunk.get("text", "")
    return (text[:max_len] + "...") if len(text) > max_len else text


def print_failure_diagnosis(
    result: dict[str, Any],
    case: dict[str, Any],
    raw_pairs: list[tuple[dict[str, Any], float]] | None = None,
) -> None:
    """
    Print detailed diagnostics for a single failed retrieval result.

    Args:
        result: The result dict from evaluate_backend_on_case.
        case: The original eval case dict (for question, notes, etc.).
        raw_pairs: If provided, raw backend scores before source selection/pruning.
    """
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  DIAGNOSIS: [{result['backend_name']}] {result['case_id']}")
    print(sep)
    print(f"  Question              : {result['question']}")
    print(f"  Notes                 : {case.get('notes', '')}")
    print(f"  Checks                : {result['checks']}")
    print(f"  Expected first chunk  : {result['expected_first_chunk_id']}")
    print(f"  Expected any of       : {result['expected_chunk_ids_any']}")
    print(f"  Actual first chunk    : {result['first_chunk_id']}")
    print(f"  Selected IDs          : {result['selected_chunk_ids']}")

    # Selected chunks with detail
    selected_pairs: list[tuple[dict[str, Any], float]] = result.get("selected_pairs", [])
    if selected_pairs:
        print(f"\n  Selected chunks (after source selection + pruning):")
        for rank, (chunk, score) in enumerate(selected_pairs, 1):
            print(f"    [{rank}] score={score:.4f}  id={chunk.get('chunk_id')}")
            print(f"         doc_id       : {chunk.get('doc_id')}")
            print(f"         section_title: {chunk.get('section_title')}")
            print(f"         policy_area  : {chunk.get('policy_area')}")
            print(f"         action_tags  : {chunk.get('action_tags')}")
            print(f"         preview      : {_chunk_preview(chunk)}")
    else:
        print("  (no chunks selected after pruning)")

    # Raw backend results (before source selection/pruning)
    if raw_pairs is not None:
        print(f"\n  Raw backend results (before source selection + pruning), top {len(raw_pairs)}:")
        if raw_pairs:
            for rank, (chunk, score) in enumerate(raw_pairs, 1):
                cid = chunk.get("chunk_id", "")
                # Flag expected chunks
                is_expected_first = (cid == result["expected_first_chunk_id"])
                is_expected_any = cid in (result["expected_chunk_ids_any"] or [])
                marker = ""
                if is_expected_first:
                    marker = "  ← EXPECTED FIRST"
                elif is_expected_any:
                    marker = "  ← EXPECTED ANY"
                print(f"    [{rank}] score={score:.4f}  id={cid}{marker}")
                print(f"         section_title: {chunk.get('section_title')}")
                print(f"         policy_area  : {chunk.get('policy_area')}")
                print(f"         preview      : {_chunk_preview(chunk, 100)}")
        else:
            print("  (no raw results)")

    print(sep)


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def _col(text: str, width: int) -> str:
    """Left-justified column with fixed width."""
    return str(text).ljust(width)


def print_comparison_report(
    all_results: dict[str, list[dict[str, Any]]],
    cases: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    verbose: bool = False,
) -> None:
    backend_names = list(all_results.keys())

    # ── Accuracy summary table ────────────────────────────────────────────────
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

    for backend_name, results in all_results.items():
        total = len(results)
        first_ok = sum(1 for r in results if r["checks"]["first_chunk_pass"])
        any_ok = sum(1 for r in results if r["checks"]["chunk_any_pass"])
        neg_ok = sum(1 for r in results if r["checks"]["negative_chunk_pass"])
        passed = sum(1 for r in results if r["passed"])
        rate = (passed / total * 100) if total > 0 else 0.0
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

    # ── IR metrics table ─────────────────────────────────────────────────────
    if cases is not None:
        print(f"\nIR Metrics (k={top_k})")
        print("-" * 74)
        ir_header = (
            _col("Backend", 18)
            + _col(f"Recall@{top_k}", 12)
            + _col("MRR", 10)
            + _col(f"nDCG@{top_k}", 12)
        )
        print(ir_header)
        print("-" * 74)
        for backend_name, results in all_results.items():
            ir = compute_ir_metrics(results, cases, k=top_k)
            print(
                _col(backend_name, 18)
                + _col(f"{ir['recall_at_k']:.4f}", 12)
                + _col(f"{ir['mrr']:.4f}", 10)
                + _col(f"{ir['ndcg_at_k']:.4f}", 12)
            )
        print("-" * 74)
        # Note on denominator
        sample_results = next(iter(all_results.values()))
        sample_ir = compute_ir_metrics(sample_results, cases, k=top_k)
        print(
            f"  (Recall/nDCG denominator: {sample_ir['recall_n']} cases with relevant chunks defined;"
            f" MRR denominator: {sample_ir['mrr_n']} cases with expected_first_chunk_id)"
        )
        print("=" * 74)

    # ── Per-case differences ─────────────────────────────────────────────────
    if len(backend_names) >= 2:
        print("\nPer-case differences (cases where backends disagree):")
        print("-" * 74)

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

            pass_states = [
                results_by_backend[bn]["passed"]
                for bn in backend_names
                if bn in results_by_backend
            ]
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
    parser.add_argument(
        "--case-id",
        default=None,
        metavar="CASE_ID",
        help="Only evaluate this single case ID",
    )
    parser.add_argument(
        "--diagnose-failures",
        action="store_true",
        help="Print detailed diagnostics for each failing case/backend",
    )
    parser.add_argument(
        "--show-raw-top",
        action="store_true",
        help=(
            "For failed cases, also print raw backend scores before "
            "source selection/pruning (helps identify where the failure originates)"
        ),
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
                "  python scripts/build_policy_chunks.py\n"
                "  python scripts/annotate_policy_chunks.py\n"
                "  python scripts/build_policy_graph.py",
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
    all_cases = load_cases(cases_path)
    print(f"Loading chunks from          : {chunks_path}")
    chunks = read_jsonl(chunks_path)
    print(f"Loading domain config from   : {config_path}")
    domain_config = load_domain_config(config_path)

    # Apply case-id filter
    cases = filter_cases_by_id(all_cases, args.case_id)
    if args.case_id and not cases:
        print(
            f"[WARNING] No case found with case_id={args.case_id!r}. "
            f"Available IDs: {[c['case_id'] for c in all_cases]}",
            file=sys.stderr,
        )
        return 1

    print(f"Backends to compare          : {', '.join(backend_names)}")
    print(f"Cases evaluated              : {len(cases)}"
          + (f" (filtered to: {args.case_id})" if args.case_id else ""))
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

    # Print comparison report (pass cases for IR metrics)
    print_comparison_report(
        all_results,
        cases=cases,
        top_k=args.top_k,
        verbose=args.verbose,
    )

    # Diagnostics for failures
    if args.diagnose_failures:
        # Build a lookup from case_id → case dict for easy access
        case_by_id = {c["case_id"]: c for c in cases}

        any_failure_printed = False
        for backend_name, results in all_results.items():
            for result in results:
                if not result["passed"]:
                    any_failure_printed = True
                    original_case = case_by_id.get(result["case_id"], {})
                    raw_pairs = None
                    if args.show_raw_top:
                        raw_pairs = get_raw_backend_results(
                            case=original_case,
                            chunks=chunks,
                            domain_config=domain_config,
                            backend_name=backend_name,
                            top_k=args.top_k,
                            nodes_path=nodes_path,
                            edges_path=edges_path,
                        )
                    print_failure_diagnosis(
                        result=result,
                        case=original_case,
                        raw_pairs=raw_pairs,
                    )

        if not any_failure_printed:
            print("\n[OK] No failures to diagnose — all cases passed for all backends.")

    # Return 0 if at least one backend achieved 100% pass on retrieval checks
    any_perfect = any(
        all(r["passed"] for r in results)
        for results in all_results.values()
    )
    return 0 if any_perfect else 1


if __name__ == "__main__":
    sys.exit(main())
