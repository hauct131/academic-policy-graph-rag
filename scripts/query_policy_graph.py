#!/usr/bin/env python3
"""
scripts/04_query_policy_graph.py

Lightweight CLI to inspect and query the Academic Policy metadata graph
without a database — reads JSONL files built by 03_build_policy_graph.py.

Usage:
    python scripts/04_query_policy_graph.py --summary

    python scripts/04_query_policy_graph.py \\
        --find-chunks-by-tag \\
        --tag-type policy_area \\
        --tag-value graduation

    python scripts/04_query_policy_graph.py \\
        --find-chunks-by-tag \\
        --tag-type action_tag \\
        --tag-value request_course_exemption \\
        --limit 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import read_jsonl


# Supported tag types and their node_id prefixes (mirrors 03_build_policy_graph.py)
_VALID_TAG_TYPES: dict[str, str] = {
    "policy_area":          "policy_area",
    "action_tag":           "action_tag",
    "student_status_tag":   "student_status_tag",
    "procedure_tag":        "procedure_tag",
    "evidence_group":       "evidence_group",
    "risk_tag":             "risk_tag",
    "requirement_tag":      "requirement_tag",
    "time_tag":             "time_tag",
}

_TEXT_PREVIEW_CHARS = 250


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

# read_jsonl is imported from core



def index_nodes_by_id(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a dict mapping node_id → node object for fast lookup."""
    return {n["node_id"]: n for n in nodes}


# ---------------------------------------------------------------------------
# Core query
# ---------------------------------------------------------------------------

def find_chunks_by_tag(
    tag_type: str,
    tag_value: str,
    nodes_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Find Chunk nodes connected to a specific tag node via an edge.

    Returns a list of chunk node dicts (up to `limit` items).
    The target node_id is constructed as ``{tag_prefix}:{tag_value}``.
    """
    prefix = _VALID_TAG_TYPES.get(tag_type)
    if prefix is None:
        return []

    target_node_id = f"{prefix}:{tag_value}"

    # Collect chunk node_ids that have an edge pointing to the target tag node
    chunk_node_ids: list[str] = []
    seen: set[str] = set()
    for edge in edges:
        if edge.get("target") == target_node_id:
            src = edge.get("source", "")
            if src.startswith("chunk:") and src not in seen:
                chunk_node_ids.append(src)
                seen.add(src)

    # Resolve chunk nodes
    results: list[dict[str, Any]] = []
    for nid in chunk_node_ids:
        node = nodes_by_id.get(nid)
        if node and node.get("node_type") == "Chunk":
            results.append(node)
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def print_summary(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    node_counts: Counter[str] = Counter(n.get("node_type", "?") for n in nodes)
    edge_counts: Counter[str] = Counter(e.get("edge_type", "?") for e in edges)

    print("=" * 60)
    print("Graph summary")
    print("=" * 60)
    print(f"  Total nodes : {len(nodes)}")
    print(f"  Total edges : {len(edges)}")
    print()
    print("  Nodes by type:")
    for ntype, cnt in sorted(node_counts.items()):
        print(f"    {ntype}: {cnt}")
    print()
    print("  Edges by type:")
    for etype, cnt in sorted(edge_counts.items()):
        print(f"    {etype}: {cnt}")
    print("=" * 60)


def print_chunk_matches(
    chunk_nodes: list[dict[str, Any]],
    tag_type: str,
    tag_value: str,
) -> None:
    print("=" * 60)
    print(f"Chunks tagged  [{tag_type}:{tag_value}]")
    print(f"Results found : {len(chunk_nodes)}")
    print("=" * 60)
    for i, node in enumerate(chunk_nodes, 1):
        props = node.get("properties", {})
        text  = props.get("text", "")
        preview = text[:_TEXT_PREVIEW_CHARS].replace("\n", " ")
        if len(text) > _TEXT_PREVIEW_CHARS:
            preview += "…"
        print(f"\n[{i}] {node.get('node_id', '')}")
        print(f"  chunk_id      : {props.get('chunk_id', '')}")
        print(f"  doc_id        : {props.get('doc_id', '')}")
        print(f"  section_title : {props.get('section_title', '')}")
        print(f"  chapter_title : {props.get('chapter_title', '')}")
        print(f"  chunk_type    : {props.get('chunk_type', '')}")
        print(f"  source_pdf    : {props.get('source_pdf', '')}")
        print(f"  text preview  : {preview}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query the Academic Policy metadata graph (JSONL-backed)."
    )
    p.add_argument(
        "--nodes-file",
        default="data/graph/policy_graph_nodes.jsonl",
        help="Nodes JSONL file (default: data/graph/policy_graph_nodes.jsonl)",
    )
    p.add_argument(
        "--edges-file",
        default="data/graph/policy_graph_edges.jsonl",
        help="Edges JSONL file (default: data/graph/policy_graph_edges.jsonl)",
    )
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print graph summary (node/edge counts by type)",
    )
    p.add_argument(
        "--find-chunks-by-tag",
        action="store_true",
        help="Find chunks connected to a given tag node",
    )
    p.add_argument(
        "--tag-type",
        choices=list(_VALID_TAG_TYPES.keys()),
        help="Tag type (e.g. policy_area, action_tag, risk_tag, …)",
    )
    p.add_argument(
        "--tag-value",
        help="Tag value (e.g. graduation, register_course, …)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)",
    )
    return p.parse_args(argv)


def _load_graph(
    nodes_path: Path,
    edges_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load nodes and edges, exiting with a clear message if files are missing."""
    missing = [p for p in (nodes_path, edges_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
        print(
            "[HINT] Run the pipeline first:\n"
            "  python scripts/build_policy_chunks.py\n"
            "  python scripts/annotate_policy_chunks.py\n"
            "  python scripts/build_policy_graph.py",
            file=sys.stderr,
        )
        sys.exit(1)

    nodes = read_jsonl(nodes_path)
    edges = read_jsonl(edges_path)
    return nodes, edges


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    nodes_path = Path(args.nodes_file)
    edges_path = Path(args.edges_file)

    # Validate: at least one action requested
    if not args.summary and not args.find_chunks_by_tag:
        print("[ERROR] Specify --summary or --find-chunks-by-tag.", file=sys.stderr)
        sys.exit(1)

    # Validate find-chunks-by-tag arguments
    if args.find_chunks_by_tag:
        if not args.tag_type:
            print("[ERROR] --find-chunks-by-tag requires --tag-type.", file=sys.stderr)
            sys.exit(1)
        if not args.tag_value:
            print("[ERROR] --find-chunks-by-tag requires --tag-value.", file=sys.stderr)
            sys.exit(1)

    nodes, edges = _load_graph(nodes_path, edges_path)

    if args.summary:
        print_summary(nodes, edges)

    if args.find_chunks_by_tag:
        nodes_by_id = index_nodes_by_id(nodes)
        matches = find_chunks_by_tag(
            tag_type=args.tag_type,
            tag_value=args.tag_value,
            nodes_by_id=nodes_by_id,
            edges=edges,
            limit=args.limit,
        )
        print_chunk_matches(matches, args.tag_type, args.tag_value)


if __name__ == "__main__":
    main()
