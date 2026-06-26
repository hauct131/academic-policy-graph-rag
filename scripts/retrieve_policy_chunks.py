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
    bonus_seed: float = 1.5,
    bonus_hop2: float = 1.25,
    max_bonus_per_chunk: float = 4.5,
) -> dict[str, float]:
    """
    2-hop graph traversal expansion for query-driven chunk boosting.

    Algorithm
    ---------
    Phase 1 — Tag Activation (unchanged from original):
      Match query tokens against tag node values (PolicyArea, ActionTag, …).
      Find all seed chunk nodes connected to matched tags via CHUNK_HAS_* edges.

    Phase 2 — 1-hop neighbour traversal:
      Follow REFERENCES and EXTERNAL_REFERENCES edges (both directions) from
      every seed chunk node to find immediately connected chunk nodes.

    Phase 3 — 2-hop neighbour traversal:
      Repeat Phase 2 starting from hop-1 neighbours to capture hidden
      prerequisite chains up to 2 edges away.

    Bonus assignment
    ----------------
      Seeds and hop-1 nodes:  +bonus_seed (default 1.5) per activating path.
      Hop-2 only nodes:       +bonus_hop2 (default 1.25) per activating path.
      Accumulated bonus is capped at max_bonus_per_chunk (default 4.5) to
      keep combined scores within the [0, 1] normalisation band used by the
      scoring layer.

    Returns
    -------
    dict mapping chunk_id (without the "chunk:" prefix) -> total graph bonus.
    """
    bonus_map: dict[str, float] = {}
    if not (nodes_file.exists() and edges_file.exists()):
        return bonus_map

    try:
        nodes = read_jsonl(nodes_file)
        edges = read_jsonl(edges_file)

        query_tokens = set(tokenize(query))
        norm_query_phrase = normalize_text(query)

        # ── Phase 1: Tag activation ──────────────────────────────────────────
        _TAG_NODE_TYPES = {
            "PolicyArea", "ActionTag", "StudentStatusTag", "ProcedureTag",
            "EvidenceGroup", "RiskTag", "RequirementTag", "TimeTag",
        }
        activated_tag_nids: set[str] = set()
        for node in nodes:
            if node.get("node_type", "") not in _TAG_NODE_TYPES:
                continue
            val = node.get("properties", {}).get("value", "")
            norm_val = normalize_text(val)
            if not norm_val:
                continue
            if norm_val in query_tokens or norm_val in norm_query_phrase:
                activated_tag_nids.add(node["node_id"])

        # ── Build edge adjacency index for chunk↔chunk traversal ────────────
        # Index edges by source and target for O(1) neighbour lookup.
        # Only REFERENCES and EXTERNAL_REFERENCES edges carry article links.
        _TRAVERSAL_EDGE_TYPES = {"REFERENCES", "EXTERNAL_REFERENCES"}

        # forward: source_nid -> set of target_nids
        fwd: dict[str, set[str]] = {}
        # backward: target_nid -> set of source_nids
        bwd: dict[str, set[str]] = {}

        for edge in edges:
            etype = edge.get("edge_type", "")
            if etype not in _TRAVERSAL_EDGE_TYPES:
                continue
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            fwd.setdefault(src, set()).add(tgt)
            bwd.setdefault(tgt, set()).add(src)

        def _neighbours(nid: str) -> set[str]:
            """Return all chunk nodes reachable in one traversal hop."""
            return (fwd.get(nid, set()) | bwd.get(nid, set()))

        # ── Identify seed chunk nodes via tag edges ──────────────────────────
        seed_nids: set[str] = set()
        for edge in edges:
            tgt = edge.get("target", "")
            src = edge.get("source", "")
            if tgt in activated_tag_nids and src.startswith("chunk:"):
                seed_nids.add(src)

        def _add_bonus(chunk_nid: str, amount: float) -> None:
            cid = chunk_nid[len("chunk:"):]   # strip prefix
            bonus_map[cid] = min(
                bonus_map.get(cid, 0.0) + amount,
                max_bonus_per_chunk,
            )

        # ── Phase 2: 1-hop traversal from seeds ─────────────────────────────
        hop1_nids: set[str] = set()
        for seed in seed_nids:
            _add_bonus(seed, bonus_seed)
            for nbr in _neighbours(seed):
                if nbr.startswith("chunk:"):
                    hop1_nids.add(nbr)

        # Seeds already get the full bonus; hop-1 nodes that are NOT seeds
        # also receive the full bonus (they are direct article neighbours).
        new_hop1 = hop1_nids - seed_nids
        for nid in new_hop1:
            _add_bonus(nid, bonus_seed)

        # ── Phase 3: 2-hop traversal from hop-1 neighbours ──────────────────
        hop2_nids: set[str] = set()
        for h1 in hop1_nids:
            for nbr in _neighbours(h1):
                if nbr.startswith("chunk:"):
                    hop2_nids.add(nbr)

        # Only nodes that are new at hop-2 (not already seeded/hop-1) get
        # the decayed bonus to penalise inference drift.
        new_hop2 = hop2_nids - seed_nids - hop1_nids
        for nid in new_hop2:
            _add_bonus(nid, bonus_hop2)

    except Exception as e:
        print(f"[WARNING] Failed during graph expansion: {e}", file=sys.stderr)

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
