#!/usr/bin/env python3
"""
scripts/05_retrieve_policy_chunks.py

Lightweight retrieval utility for Academic Policy Graph RAG.
Finds relevant policy chunks from annotated chunks using lexical scoring,
optional tag filters, and optional graph-backed tag expansion.

Usage:
    python scripts/05_retrieve_policy_chunks.py --query "điều kiện xét tốt nghiệp" --top-k 5
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


# Add parent directory to path to load core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import normalize_text, tokenize, read_jsonl

# ---------------------------------------------------------------------------
# Scoring and Retrieval Logic
# ---------------------------------------------------------------------------

def score_chunk(
    chunk: dict[str, Any],
    query_tokens: list[str],
    normalized_query_phrase: str,
    graph_bonus: float = 0.0,
) -> float:
    """Calculate lexical relevance score for a chunk against a query."""
    if not query_tokens:
        return 0.0

    # Tokenize chunk fields
    title_tokens = tokenize(chunk.get("section_title", ""))
    chapter_tokens = tokenize(chunk.get("chapter_title", ""))
    text_tokens = tokenize(chunk.get("text", ""))

    # Gather all metadata tag tokens
    meta_tokens = []
    for field in [
        "policy_area",
        "action_tags",
        "requirement_tags",
        "procedure_tags",
        "risk_tags",
        "evidence_groups",
        "time_tags",
    ]:
        for tag in chunk.get(field, []):
            meta_tokens.extend(tokenize(str(tag)))

    # Compute overlap scores with weights
    score = 0.0
    for token in query_tokens:
        # section_title match weighs more than body text
        score += title_tokens.count(token) * 2.0
        score += chapter_tokens.count(token) * 0.5
        score += text_tokens.count(token) * 1.0
        score += meta_tokens.count(token) * 1.0

    # Exact normalized phrase match bonuses
    norm_title = " ".join(title_tokens)
    norm_text = " ".join(text_tokens)

    if normalized_query_phrase:
        if normalized_query_phrase in norm_title:
            score += 5.0
        if normalized_query_phrase in norm_text:
            score += 3.0

    # Add graph expansion bonus if applicable
    score += graph_bonus

    return score


def filter_chunks(
    chunks: list[dict[str, Any]],
    policy_area: str | None = None,
    action_tag: str | None = None,
    requirement_tag: str | None = None,
    risk_tag: str | None = None,
) -> list[dict[str, Any]]:
    """Apply metadata/tag filters to the chunk list."""
    filtered = []
    for chunk in chunks:
        if policy_area and policy_area not in chunk.get("policy_area", []):
            continue
        if action_tag and action_tag not in chunk.get("action_tags", []):
            continue
        if requirement_tag and requirement_tag not in chunk.get("requirement_tags", []):
            continue
        if risk_tag and risk_tag not in chunk.get("risk_tags", []):
            continue
        filtered.append(chunk)
    return filtered


def get_section_number_key(section_num_str: str) -> tuple[int, Any]:
    """Helper to sort section numbers naturally (numeric first, then string)."""
    s = str(section_num_str).strip()
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def retrieve_chunks(
    chunks: list[dict[str, Any]],
    query: str,
    top_k: int = 5,
    policy_area: str | None = None,
    action_tag: str | None = None,
    requirement_tag: str | None = None,
    risk_tag: str | None = None,
    graph_bonus_map: dict[str, float] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """
    Filter, score, sort, and return top_k retrieved chunks.
    """
    # 1. Filter chunks
    filtered = filter_chunks(
        chunks,
        policy_area=policy_area,
        action_tag=action_tag,
        requirement_tag=requirement_tag,
        risk_tag=risk_tag,
    )

    # 2. Score chunks
    query_tokens = tokenize(query)
    normalized_phrase = " ".join(query_tokens)
    bonus_map = graph_bonus_map or {}

    scored_items = []
    for chunk in filtered:
        c_id = chunk.get("chunk_id", "")
        g_bonus = bonus_map.get(c_id, 0.0)
        score = score_chunk(chunk, query_tokens, normalized_phrase, graph_bonus=g_bonus)
        if score > 0.0:
            scored_items.append((chunk, score))

    # 3. Sort chunks
    # Sort order: score desc, doc_id asc, section_number asc (numeric first), chunk_id asc
    def sort_key(item: tuple[dict[str, Any], float]) -> tuple[float, str, tuple[int, Any], str]:
        c, s = item
        sec_num = c.get("section_number", "")
        return (
            -s,
            c.get("doc_id", ""),
            get_section_number_key(sec_num),
            c.get("chunk_id", ""),
        )

    sorted_items = sorted(scored_items, key=sort_key)
    return sorted_items[:top_k]


# ---------------------------------------------------------------------------
# Graph-backed tag expansion helper
# ---------------------------------------------------------------------------

def load_graph_expansion(
    query: str,
    nodes_file: Path,
    edges_file: Path,
) -> dict[str, float]:
    """
    Identify tag nodes that match query tokens, then return a dict mapping
    chunk_id → graph_bonus for connected chunks.
    """
    bonus_map: dict[str, float] = {}
    if not (nodes_file.exists() and edges_file.exists()):
        return bonus_map

    try:
        nodes = read_jsonl(nodes_file)
        edges = read_jsonl(edges_file)

        query_tokens = set(tokenize(query))
        norm_query_phrase = normalize_text(query)

        # Detect which tag nodes are matched by the query
        activated_tag_ids = set()
        for node in nodes:
            node_type = node.get("node_type", "")
            # Verify it's a tag node
            if node_type in (
                "PolicyArea",
                "ActionTag",
                "StudentStatusTag",
                "ProcedureTag",
                "EvidenceGroup",
                "RiskTag",
                "RequirementTag",
                "TimeTag",
            ):
                val = node.get("properties", {}).get("value", "")
                norm_val = normalize_text(val)
                if not norm_val:
                    continue
                # Match if tag value is one of query tokens or exists as substring
                if norm_val in query_tokens or norm_val in norm_query_phrase:
                    activated_tag_ids.add(node["node_id"])

        # Find chunks connected to activated tags
        # Chunk nodes point to tag nodes (CHUNK_HAS_...)
        for edge in edges:
            target = edge.get("target", "")
            source = edge.get("source", "")
            if target in activated_tag_ids and source.startswith("chunk:"):
                # Extract chunk_id from chunk:doc_id__dieu_1
                chunk_id = source.split("chunk:", 1)[1]
                bonus_map[chunk_id] = bonus_map.get(chunk_id, 0.0) + 1.5

    except Exception as e:
        print(f"[WARNING] Failed to load/parse graph expansion: {e}", file=sys.stderr)

    return bonus_map


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def format_preview(text: str, limit: int = 300) -> str:
    cleaned = text.replace("\n", " ").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def print_results(
    results: list[tuple[dict[str, Any], float]],
    show_text: bool = False,
) -> None:
    print("=" * 70)
    print(f"Retrieval Results (Total: {len(results)})")
    print("=" * 70)
    for rank, (chunk, score) in enumerate(results, 1):
        print(f"\nRank {rank} | Score: {score:.2f} | Chunk ID: {chunk.get('chunk_id')}")
        print(f"  Document   : {chunk.get('doc_id')}")
        print(f"  Chapter    : {chunk.get('chapter_title') or 'N/A'}")
        print(f"  Section    : {chunk.get('section_title') or 'N/A'}")
        print(f"  Chunk Type : {chunk.get('chunk_type')}")
        print(f"  Policy Area: {', '.join(chunk.get('policy_area', [])) or 'None'}")
        print(f"  Action Tags: {', '.join(chunk.get('action_tags', [])) or 'None'}")
        print(f"  Req Tags   : {', '.join(chunk.get('requirement_tags', [])) or 'None'}")
        print(f"  Source PDF : {chunk.get('source_pdf') or 'N/A'}")
        if show_text:
            print(f"  Text       :\n{chunk.get('text')}")
        else:
            print(f"  Preview    : {format_preview(chunk.get('text', ''))}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve relevant policy chunks using lexical scoring & filters."
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
        "--query",
        required=True,
        help="Search query text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to return (default: 5)",
    )
    parser.add_argument(
        "--policy-area",
        help="Filter by policy area",
    )
    parser.add_argument(
        "--action-tag",
        help="Filter by action tag",
    )
    parser.add_argument(
        "--requirement-tag",
        help="Filter by requirement tag",
    )
    parser.add_argument(
        "--risk-tag",
        help="Filter by risk tag",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Show full text of retrieved chunks",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    chunks_path = Path(args.chunks_file)
    nodes_path = Path(args.nodes_file)
    edges_path = Path(args.edges_file)

    if not chunks_path.exists():
        print(f"[ERROR] Chunks file not found: {chunks_path}", file=sys.stderr)
        sys.exit(1)

    # Load chunks
    chunks = read_jsonl(chunks_path)

    # Perform graph-backed tag expansion if graph exists
    graph_bonus_map = load_graph_expansion(args.query, nodes_path, edges_path)

    # Retrieve matching chunks
    results = retrieve_chunks(
        chunks=chunks,
        query=args.query,
        top_k=args.top_k,
        policy_area=args.policy_area,
        action_tag=args.action_tag,
        requirement_tag=args.requirement_tag,
        risk_tag=args.risk_tag,
        graph_bonus_map=graph_bonus_map,
    )

    print_results(results, show_text=args.show_text)


if __name__ == "__main__":
    main()
