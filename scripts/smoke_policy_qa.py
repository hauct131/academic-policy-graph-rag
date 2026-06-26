#!/usr/bin/env python3
"""
scripts/09_smoke_policy_qa.py

Smoke test runner for Academic Policy QA. Runs default demo questions
and prints compact metadata and answer previews.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure scripts folder is in python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import importlib
try:
    _qa = importlib.import_module("answer_policy_question")
    _retriever = importlib.import_module("retrieve_policy_chunks")
    _domain = importlib.import_module("policy_domain_config")
    _reg = importlib.import_module("policy_document_registry")
    _service = importlib.import_module("policy_retrieval_service")
except ImportError:
    # Alternate path fallback
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _qa = importlib.import_module("answer_policy_question")
    _retriever = importlib.import_module("retrieve_policy_chunks")
    _domain = importlib.import_module("policy_domain_config")
    _reg = importlib.import_module("policy_document_registry")
    _service = importlib.import_module("policy_retrieval_service")

read_jsonl = _retriever.read_jsonl
infer_case_issues = _qa.infer_case_issues
answer_question = _qa.answer_question
load_domain_config = _domain.load_domain_config
load_document_registry = _reg.load_document_registry
should_warn_missing_current_notice = _reg.should_warn_missing_current_notice
PolicyRetrievalService = _service.PolicyRetrievalService


DEFAULT_QUESTIONS = [
    "Điều kiện xét tốt nghiệp là gì?",
    "Miễn môn học cần hồ sơ gì?",
    "IELTS 6.0 được miễn tiếng Anh không?",
    "Học kỳ này khi nào nộp hồ sơ miễn môn?",
    "Em có IELTS 6.0, từng học trường khác có bảng điểm, muốn xin miễn môn và hỏi điều kiện xét tốt nghiệp."
]


def run_smoke_test(
    chunks_path: Path,
    nodes_path: Path,
    edges_path: Path,
    config_path: Path,
    registry_path: Path,
    top_k: int,
    full_answer: bool,
    retrieval_backend: str = "lexical_v0",
) -> int:
    # 1. Check required files
    missing = []
    if not chunks_path.exists():
        missing.append(f"Annotated chunks file not found: {chunks_path}")
    if not config_path.exists():
        missing.append(f"Domain config file not found: {config_path}")
    if not registry_path.exists():
        missing.append(f"Document registry file not found: {registry_path}")

    if missing:
        print("[ERROR] Required files are missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease run the data ingestion and processing pipeline first:")
        print("  python scripts/build_policy_chunks.py")
        print("  python scripts/annotate_policy_chunks.py")
        print("  python scripts/build_policy_graph.py")
        return 1

    # 2. Load resources
    print("Loading resources...")
    chunks = read_jsonl(chunks_path)
    domain_config = load_domain_config(config_path)
    document_registry = load_document_registry(registry_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path.name}")
    print(f"Loaded domain config: {domain_config.get('domain_id', 'unknown')}")
    print(f"Loaded document registry: {len(document_registry)} documents")

    print("\n============================================================")
    print("Running Academic Policy QA Smoke Test")
    print("============================================================\n")

    # Instantiate retrieval service
    retrieval_service = PolicyRetrievalService(
        chunks=chunks,
        nodes_file=nodes_path,
        edges_file=edges_path,
        backend_name=retrieval_backend,
    )
    print(f"Retrieval backend: {retrieval_service.backend_name}")

    for idx, question in enumerate(DEFAULT_QUESTIONS, 1):
        print(f"[{idx}] Question: \"{question}\"")

        # Expose inferred issues
        issues = infer_case_issues(question, domain_config=domain_config)
        issue_types = [iss["issue_type"] for iss in issues]
        issue_labels = [iss["label"] for iss in issues]
        print(f"    - Inferred Issues: {issue_types} ({', '.join(issue_labels)})")

        # Expose first evidence chunk details
        selected_pairs = retrieval_service.retrieve_for_issues(
            issues=issues,
            question=question,
            top_k=top_k,
            max_sources_per_issue=min(3, top_k),
            use_graph=True,
            strict_pruning=True
        )
        selected_chunks = [chunk for chunk, score in selected_pairs]

        if selected_chunks:
            c = selected_chunks[0]
            print(f"    - First Evidence : [Doc: {c.get('doc_id')}] [Sec: {c.get('section_title')}]")
        else:
            print("    - First Evidence : None")

        # Expose warnings
        has_warning = False
        warning_areas = []
        for issue in issues:
            if should_warn_missing_current_notice(question, document_registry, issue["policy_area"]):
                has_warning = True
                warning_areas.append(issue["policy_area"])
        print(f"    - Has Warn Msg   : {has_warning} (Areas: {warning_areas})" if has_warning else "    - Has Warn Msg   : False")

        # Generate answer
        answer = answer_question(
            question=question,
            chunks=chunks,
            top_k=top_k,
            domain_config=domain_config,
            document_registry=document_registry,
            retrieval_service=retrieval_service
        )

        if full_answer:
            print("\n    --- Full Answer ---")
            print(answer)
            print("    -------------------\n")
        else:
            preview = answer[:150] + "..." if len(answer) > 150 else answer
            preview_clean = preview.replace("\n", " ")
            print(f"    - Answer Preview : {preview_clean}\n")

    print("============================================================")
    print("Smoke Test Complete")
    print("============================================================")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic QA Smoke Test.")
    parser.add_argument(
        "--chunks-file",
        default="data/chunks/policy_chunks.annotated.jsonl",
        help="Path to annotated chunks JSONL"
    )
    parser.add_argument(
        "--nodes-file",
        default="data/graph/policy_graph_nodes.jsonl",
        help="Path to graph nodes JSONL"
    )
    parser.add_argument(
        "--edges-file",
        default="data/graph/policy_graph_edges.jsonl",
        help="Path to graph edges JSONL"
    )
    parser.add_argument(
        "--domain-config",
        default="domains/ou_academic_policy_v1/domain.json",
        help="Path to domain config JSON"
    )
    parser.add_argument(
        "--document-registry",
        default="domains/ou_academic_policy_v1/document_registry.jsonl",
        help="Path to document registry JSONL"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K retrieved chunks"
    )
    parser.add_argument(
        "--full-answer",
        action="store_true",
        help="Print full generated answer output"
    )
    parser.add_argument(
        "--retrieval-backend",
        default="lexical_v0",
        choices=["lexical_v0", "bm25_like_v0"],
        help="Retrieval backend to use (default: lexical_v0)"
    )
    args = parser.parse_args()

    return run_smoke_test(
        chunks_path=Path(args.chunks_file),
        nodes_path=Path(args.nodes_file),
        edges_path=Path(args.edges_file),
        config_path=Path(args.domain_config),
        registry_path=Path(args.document_registry),
        top_k=args.top_k,
        full_answer=args.full_answer,
        retrieval_backend=args.retrieval_backend,
    )


if __name__ == "__main__":
    sys.exit(main())
