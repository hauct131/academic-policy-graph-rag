#!/usr/bin/env python3
"""
scripts/build_policy_graph.py

Build the metadata graph for the Academic Policy Graph RAG project.

Reads policy_chunks.annotated.jsonl and emits:
  data/graph/policy_graph_nodes.jsonl
  data/graph/policy_graph_edges.jsonl

Usage:
    python scripts/build_policy_graph.py [--input-file ...] [--nodes-file ...] [--edges-file ...]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import read_jsonl


# Maps annotation field -> (node_id_prefix, node_type, edge_type)
_TAG_FIELD_META: dict[str, tuple[str, str, str]] = {
    "policy_area":          ("policy_area",       "PolicyArea",       "CHUNK_HAS_POLICY_AREA"),
    "action_tags":          ("action_tag",         "ActionTag",        "CHUNK_HAS_ACTION_TAG"),
    "student_status_tags":  ("student_status_tag", "StudentStatusTag", "CHUNK_HAS_STUDENT_STATUS_TAG"),
    "procedure_tags":       ("procedure_tag",      "ProcedureTag",     "CHUNK_HAS_PROCEDURE_TAG"),
    "evidence_groups":      ("evidence_group",     "EvidenceGroup",    "CHUNK_HAS_EVIDENCE_GROUP"),
    "risk_tags":            ("risk_tag",           "RiskTag",          "CHUNK_HAS_RISK_TAG"),
    "requirement_tags":     ("requirement_tag",    "RequirementTag",   "CHUNK_HAS_REQUIREMENT_TAG"),
    "time_tags":            ("time_tag",           "TimeTag",          "CHUNK_HAS_TIME_TAG"),
}

# ---------------------------------------------------------------------------
# Cross-reference extraction: prioritized masking patterns
# ---------------------------------------------------------------------------

# Patterns ordered longest-to-shortest so deep hierarchies are captured
# before shorter sub-patterns can steal sub-strings.  Each pattern captures
# the Dieu number (or the literal "nay") in its first capture group, and
# optionally captures Khoan and Diem in subsequent groups.
#
# All patterns are compiled with re.IGNORECASE and re.DOTALL so newlines
# inside a quoted article span (e.g. "Khoản 1 Điều\n7") are absorbed.
REFERENCE_PATTERNS: list[re.Pattern] = [
    # Pattern 1: Deep hierarchy  — Điểm + Khoản + Điều
    re.compile(
        r"đi[eể]m\s+([a-z\u0111])\s*,?\s*"
        r"kho[aả]n\s+(\d+[a-z\u0111]?)\s*,?\s*"
        r"đi[eề]u\s+(\d+|n[aà]y)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Pattern 2: Mid hierarchy   — Khoản + Điều
    re.compile(
        r"kho[aả]n\s+(\d+[a-z\u0111]?)\s*,?\s*"
        r"đi[eề]u\s+(\d+|n[aà]y)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Pattern 3: Base            — Điều N  (standalone, not inside a larger match)
    re.compile(
        r"đi[eề]u\s+(\d+)",
        re.IGNORECASE | re.DOTALL,
    ),
]

_MASK_TOKEN = "\x00MASKEDREF\x00"   # ASCII NUL-based sentinel; never appears in text


def _resolve_dieu_number(raw: str, section_number: str | None) -> str | None:
    """
    Convert the raw Dieu token (e.g. '14' or 'này'/'nay') to a numeric string.
    Returns None if the self-reference anchor cannot be resolved.
    """
    norm = raw.strip().lower()
    if norm in ("n\u00e0y", "nay", "n\u00e0y"):   # 'này'
        # Resolve to the current chunk's own Dieu number
        sn = str(section_number or "").strip()
        return sn if sn and sn.isdigit() else None
    if raw.strip().isdigit():
        return raw.strip()
    return None


def extract_cross_references(
    chunk: dict[str, Any],
) -> list[tuple[str | None, str | None, str]]:
    """
    Extract structured cross-references from the chunk's text using a
    prioritized masking loop.

    Returns a list of (khoan, diem, dieu_number) tuples where dieu_number
    is always a numeric string, khoan and diem may be None.
    """
    text  = chunk.get("text", "")
    sn    = str(chunk.get("section_number", "")).strip()
    refs: list[tuple[str | None, str | None, str]] = []

    # Blank out Markdown heading lines (# ... ) and table lines (| ... |) so
    # that section titles like "## Điều 6. ..." are not matched as cross-
    # references.  We replace each such line with spaces of the same length to
    # preserve character offsets for the masking loop.
    def _blank_headings(s: str) -> str:
        lines = s.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith("|"):
                cleaned.append(" " * len(line))
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    # Work on a mutable copy for mask replacement (with headings blanked)
    working = _blank_headings(text)

    for pat in REFERENCE_PATTERNS:
        new_working = working
        for m in pat.finditer(working):
            groups = [g for g in m.groups() if g is not None]
            if len(groups) == 3:
                # Deep: diem, khoan, dieu
                diem_raw, khoan_raw, dieu_raw = groups
                dieu = _resolve_dieu_number(dieu_raw, sn)
                if dieu:
                    refs.append((khoan_raw.strip(), diem_raw.strip(), dieu))
            elif len(groups) == 2:
                # Mid: khoan, dieu
                khoan_raw, dieu_raw = groups
                dieu = _resolve_dieu_number(dieu_raw, sn)
                if dieu:
                    refs.append((khoan_raw.strip(), None, dieu))
            elif len(groups) == 1:
                # Base: dieu only
                dieu_raw = groups[0]
                dieu = _resolve_dieu_number(dieu_raw, sn)
                if dieu:
                    refs.append((None, None, dieu))
            # Mask the consumed span so shorter patterns cannot steal it
            new_working = new_working[:m.start()] + _MASK_TOKEN + new_working[m.end():]
        working = new_working

    return refs





# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

# read_jsonl is imported from core



def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_document_node(chunk: dict[str, Any]) -> dict[str, Any]:
    doc_id = chunk["doc_id"]
    return {
        "node_id":   f"document:{doc_id}",
        "node_type": "Document",
        "label":     chunk.get("title", doc_id),
        "properties": {
            "doc_id":         doc_id,
            "title":          chunk.get("title", ""),
            "decision_no":    chunk.get("decision_no", ""),
            "issued_date":    chunk.get("issued_date", ""),
            "institution":    chunk.get("institution", ""),
            "education_mode": chunk.get("education_mode", ""),
            "source_pdf":     chunk.get("source_pdf", ""),
        },
    }


def make_chunk_node(chunk: dict[str, Any]) -> dict[str, Any]:
    chunk_id = chunk["chunk_id"]
    label = chunk.get("section_title") or chunk_id
    return {
        "node_id":   f"chunk:{chunk_id}",
        "node_type": "Chunk",
        "label":     label,
        "properties": {
            "chunk_id":       chunk_id,
            "doc_id":         chunk.get("doc_id", ""),
            "title":          chunk.get("title", ""),
            "chapter_title":  chunk.get("chapter_title", ""),
            "section_title":  chunk.get("section_title", ""),
            "section_number": chunk.get("section_number", ""),
            "chunk_type":     chunk.get("chunk_type", ""),
            "source_path":    chunk.get("source_path", ""),
            "source_pdf":     chunk.get("source_pdf", ""),
            "char_count":     chunk.get("char_count", 0),
            "word_count":     chunk.get("word_count", 0),
            "text":           chunk.get("text", ""),
        },
    }


def make_tag_node(tag_prefix: str, node_type: str, tag_value: str) -> dict[str, Any]:
    return {
        "node_id":    f"{tag_prefix}:{tag_value}",
        "node_type":  node_type,
        "label":      tag_value,
        "properties": {"value": tag_value},
    }


def make_external_ref_node(target_nid: str, dieu_num: str, khoan: str | None) -> dict[str, Any]:
    """
    Create a stub ExternalReference node for a cross-document target that
    is referenced by at least 2 distinct source chunks (minimum support >= 2).
    """
    label = f"Điều {dieu_num}" + (f" Khoản {khoan}" if khoan else "")
    return {
        "node_id":    target_nid,
        "node_type":  "ExternalReference",
        "label":      label,
        "properties": {
            "referenced_dieu":  dieu_num,
            "referenced_khoan": khoan or "",
            "support":          0,   # will be updated after counting
        },
    }


# ---------------------------------------------------------------------------
# Edge factory
# ---------------------------------------------------------------------------

def make_edge(
    source: str,
    target: str,
    edge_type: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "edge_id":    f"{source}->{target}:{edge_type}",
        "source":     source,
        "target":     target,
        "edge_type":  edge_type,
        "properties": properties or {},
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    chunks: list[dict[str, Any]],
    min_external_support: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build deduplicated, deterministically-sorted node and edge lists.

    Cross-reference containment rules
    ----------------------------------
    1. Active Chunk Registry: a set of all materialized chunk node IDs is
       built before any cross-reference processing.
    2. Dangling Edge Safeguard: a REFERENCES edge whose target does not exist
       in the registry is NOT emitted as an internal edge.
    3. Minimum Support for External References: external targets intercepted
       in step 2 are counted across all source chunks. Only targets cited by
       >= min_external_support distinct chunks receive a physical
       ExternalReference node and an EXTERNAL_REFERENCES edge.
    4. Isolated External References (support < min_external_support) are
       stored as a JSON-encoded string in the source chunk node's
       `isolated_external_refs` property to preserve context without
       polluting graph topology.

    Returns (nodes, edges).
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    def add_node(node: dict[str, Any]) -> None:
        nodes.setdefault(node["node_id"], node)

    def add_edge(edge: dict[str, Any]) -> None:
        edges.setdefault(edge["edge_id"], edge)

    # ── Step 1: Build Active Chunk Registry ───────────────────────────────
    active_chunk_nids: set[str] = {f"chunk:{c['chunk_id']}" for c in chunks}

    # ── Pass 1: standard nodes / tag edges + collect cross-ref candidates ─
    # external_refs: target_nid -> set of source chunk_nids that reference it
    external_refs: dict[str, set[str]] = {}
    # isolated metadata accumulator: chunk_nid -> list of ref strings
    isolated_ref_strings: dict[str, list[str]] = {}
    # Store cross-refs per chunk for Pass 2
    chunk_crossrefs: dict[str, list[tuple[str, str | None, str | None, str]]] = {}

    for chunk in chunks:
        doc_id    = chunk["doc_id"]
        chunk_id  = chunk["chunk_id"]
        doc_nid   = f"document:{doc_id}"
        chunk_nid = f"chunk:{chunk_id}"

        add_node(make_document_node(chunk))
        add_node(make_chunk_node(chunk))
        add_edge(make_edge(doc_nid, chunk_nid, "DOCUMENT_HAS_CHUNK"))

        for field, (tag_prefix, node_type, edge_type) in _TAG_FIELD_META.items():
            for tag_value in chunk.get(field, []):
                tag_node = make_tag_node(tag_prefix, node_type, tag_value)
                add_node(tag_node)
                add_edge(make_edge(chunk_nid, tag_node["node_id"], edge_type))

        # Collect cross-references for classification
        crossrefs = []
        for khoan, diem, dieu_num in extract_cross_references(chunk):
            target_chunk_id = f"{doc_id}__dieu_{dieu_num}"
            if khoan:
                target_chunk_id += f"__khoan_{khoan}"
            target_nid = f"chunk:{target_chunk_id}"
            crossrefs.append((target_nid, khoan, diem, dieu_num))

            # ── Step 2: Dangling Edge Safeguard ───────────────────────────
            if target_nid not in active_chunk_nids:
                # External target — aggregate for minimum support counting
                external_refs.setdefault(target_nid, set()).add(chunk_nid)

        chunk_crossrefs[chunk_nid] = crossrefs

    # ── Step 3: Enforce Minimum Support >= 2 for External References ──────
    # Determine which external targets meet the threshold
    high_support_externals: set[str] = {
        target_nid
        for target_nid, sources in external_refs.items()
        if len(sources) >= min_external_support
    }

    # Materialise ExternalReference nodes for high-support targets
    for target_nid, sources in external_refs.items():
        if target_nid in high_support_externals:
            # Parse dieu/khoan back from target_nid for label construction
            # Format: chunk:{doc_id}__dieu_{N}[__khoan_{K}]
            tail = target_nid.split("__dieu_", 1)[-1] if "__dieu_" in target_nid else ""
            dieu_part, _, khoan_part = tail.partition("__khoan_")
            ext_node = make_external_ref_node(target_nid, dieu_part, khoan_part or None)
            ext_node["properties"]["support"] = len(sources)
            add_node(ext_node)

    # ── Pass 2: Emit edges now that external support is known ──────────────
    for chunk in chunks:
        chunk_nid = f"chunk:{chunk['chunk_id']}"
        isolated: list[str] = []

        for target_nid, khoan, diem, dieu_num in chunk_crossrefs.get(chunk_nid, []):
            if target_nid in active_chunk_nids:
                # Internal link — emit standard REFERENCES edge
                props: dict[str, Any] = {"referenced_dieu": dieu_num}
                if khoan:
                    props["referenced_khoan"] = khoan
                if diem:
                    props["referenced_diem"] = diem
                add_edge(make_edge(chunk_nid, target_nid, "REFERENCES", props))

            elif target_nid in high_support_externals:
                # High-support external — emit EXTERNAL_REFERENCES edge
                props = {"referenced_dieu": dieu_num}
                if khoan:
                    props["referenced_khoan"] = khoan
                if diem:
                    props["referenced_diem"] = diem
                add_edge(make_edge(chunk_nid, target_nid, "EXTERNAL_REFERENCES", props))

            else:
                # ── Step 4: Isolated external ref — store in metadata only ─
                ref_str = f"Điều {dieu_num}"
                if khoan:
                    ref_str += f" Khoản {khoan}"
                if diem:
                    ref_str += f" Điểm {diem}"
                isolated.append(ref_str)

        # Inject isolated refs into the chunk node's properties
        if isolated and chunk_nid in nodes:
            nodes[chunk_nid]["properties"]["isolated_external_refs"] = isolated

    sorted_nodes = sorted(nodes.values(), key=lambda n: n["node_id"])
    sorted_edges = sorted(edges.values(), key=lambda e: e["edge_id"])
    return sorted_nodes, sorted_edges

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    chunks_read: int,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    nodes_path: Path,
    edges_path: Path,
) -> None:
    node_counts: Counter[str] = Counter(n["node_type"] for n in nodes)
    edge_counts: Counter[str] = Counter(e["edge_type"] for e in edges)

    print("=" * 60)
    print("Policy graph build complete")
    print("=" * 60)
    print(f"  Chunks read   : {chunks_read}")
    print(f"  Nodes written : {len(nodes)}  -> {nodes_path.resolve()}")
    print(f"  Edges written : {len(edges)}  -> {edges_path.resolve()}")
    print()
    print("  Nodes by type:")
    for ntype, cnt in sorted(node_counts.items()):
        print(f"    {ntype}: {cnt}")
    print()
    print("  Edges by type:")
    for etype, cnt in sorted(edge_counts.items()):
        print(f"    {etype}: {cnt}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build policy metadata graph from annotated chunks."
    )
    p.add_argument(
        "--input-file",
        default="data/chunks/policy_chunks.annotated.jsonl",
    )
    p.add_argument(
        "--nodes-file",
        default="data/graph/policy_graph_nodes.jsonl",
    )
    p.add_argument(
        "--edges-file",
        default="data/graph/policy_graph_edges.jsonl",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input_file)
    nodes_path = Path(args.nodes_file)
    edges_path = Path(args.edges_file)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    chunks = read_jsonl(input_path)
    nodes, edges = build_graph(chunks)
    write_jsonl(nodes_path, nodes)
    write_jsonl(edges_path, edges)
    print_summary(len(chunks), nodes, edges, nodes_path, edges_path)


if __name__ == "__main__":
    main()
