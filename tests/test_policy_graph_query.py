"""
tests/test_policy_graph_query.py

Unit and integration tests for the graph query utility.

Run with:
    python -m pytest
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module

_mod = import_module("query_policy_graph")

read_jsonl          = _mod.read_jsonl
index_nodes_by_id   = _mod.index_nodes_by_id
find_chunks_by_tag  = _mod.find_chunks_by_tag
print_summary       = _mod.print_summary


# ---------------------------------------------------------------------------
# Minimal graph fixtures
# ---------------------------------------------------------------------------

def _doc_node(doc_id: str = "doc_a") -> dict:
    return {
        "node_id":    f"document:{doc_id}",
        "node_type":  "Document",
        "label":      "Test Document",
        "properties": {"doc_id": doc_id, "title": "Test", "decision_no": "",
                       "issued_date": "", "institution": "", "education_mode": "",
                       "source_pdf": "test.pdf"},
    }


def _chunk_node(
    chunk_id: str = "doc_a__dieu_1",
    doc_id: str = "doc_a",
    section_title: str = "Điều 1",
    chapter_title: str = "Chương I",
    chunk_type: str = "dieu",
) -> dict:
    return {
        "node_id":    f"chunk:{chunk_id}",
        "node_type":  "Chunk",
        "label":      section_title,
        "properties": {
            "chunk_id":       chunk_id,
            "doc_id":         doc_id,
            "title":          "Test Document",
            "chapter_title":  chapter_title,
            "section_title":  section_title,
            "section_number": "1",
            "chunk_type":     chunk_type,
            "source_path":    "data/raw/cleaned/doc_a.md",
            "source_pdf":     "test.pdf",
            "char_count":     100,
            "word_count":     20,
            "text":           "Sample text content for this chunk.",
        },
    }


def _tag_node(prefix: str, node_type: str, value: str) -> dict:
    return {
        "node_id":    f"{prefix}:{value}",
        "node_type":  node_type,
        "label":      value,
        "properties": {"value": value},
    }


def _edge(source: str, target: str, edge_type: str) -> dict:
    return {
        "edge_id":    f"{source}->{target}:{edge_type}",
        "source":     source,
        "target":     target,
        "edge_type":  edge_type,
        "properties": {},
    }


def _minimal_graph(
    *,
    policy_area: str = "graduation",
    action_tag: str | None = None,
    chunk_id: str = "doc_a__dieu_1",
    doc_id: str = "doc_a",
) -> tuple[list[dict], list[dict]]:
    """Build a small in-memory graph for testing."""
    nodes = [
        _doc_node(doc_id),
        _chunk_node(chunk_id, doc_id),
        _tag_node("policy_area", "PolicyArea", policy_area),
    ]
    edges = [
        _edge(f"document:{doc_id}", f"chunk:{chunk_id}", "DOCUMENT_HAS_CHUNK"),
        _edge(f"chunk:{chunk_id}", f"policy_area:{policy_area}", "CHUNK_HAS_POLICY_AREA"),
    ]
    if action_tag:
        nodes.append(_tag_node("action_tag", "ActionTag", action_tag))
        edges.append(_edge(f"chunk:{chunk_id}", f"action_tag:{action_tag}", "CHUNK_HAS_ACTION_TAG"))
    return nodes, edges


# ---------------------------------------------------------------------------
# 1. Reading JSONL
# ---------------------------------------------------------------------------

class TestReadJsonl:
    def test_reads_valid_jsonl(self, tmp_path):
        f = tmp_path / "test.jsonl"
        records = [{"a": 1}, {"b": 2}]
        f.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        result = read_jsonl(f)
        assert result == records

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
        result = read_jsonl(f)
        assert len(result) == 2

    def test_unicode_preserved(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"text": "Điều 1. Phạm vi"}\n', encoding="utf-8")
        result = read_jsonl(f)
        assert result[0]["text"] == "Điều 1. Phạm vi"


# ---------------------------------------------------------------------------
# 2. Indexing nodes by node_id
# ---------------------------------------------------------------------------

class TestIndexNodesById:
    def test_index_maps_node_id_to_node(self):
        nodes = [_doc_node("doc_a"), _chunk_node()]
        idx = index_nodes_by_id(nodes)
        assert "document:doc_a" in idx
        assert "chunk:doc_a__dieu_1" in idx

    def test_lookup_returns_correct_node(self):
        nodes = [_doc_node("doc_x")]
        idx = index_nodes_by_id(nodes)
        assert idx["document:doc_x"]["node_type"] == "Document"

    def test_empty_list_returns_empty_dict(self):
        assert index_nodes_by_id([]) == {}


# ---------------------------------------------------------------------------
# 3. Finding chunks by policy_area
# ---------------------------------------------------------------------------

class TestFindChunksByPolicyArea:
    def test_finds_single_chunk(self):
        nodes, edges = _minimal_graph(policy_area="graduation")
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "graduation", idx, edges, limit=10)
        assert len(results) == 1
        assert results[0]["node_type"] == "Chunk"

    def test_result_is_chunk_node(self):
        nodes, edges = _minimal_graph(policy_area="course_registration")
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "course_registration", idx, edges)
        assert all(r["node_type"] == "Chunk" for r in results)

    def test_properties_accessible(self):
        nodes, edges = _minimal_graph(policy_area="graduation", chunk_id="doc_a__dieu_5")
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "graduation", idx, edges)
        assert results[0]["properties"]["chunk_id"] == "doc_a__dieu_5"

    def test_multiple_chunks_same_tag(self):
        """Two chunks tagged with the same policy_area should both appear."""
        nodes = [
            _doc_node("doc_a"),
            _chunk_node("doc_a__dieu_1", "doc_a"),
            _chunk_node("doc_a__dieu_2", "doc_a"),
            _tag_node("policy_area", "PolicyArea", "graduation"),
        ]
        edges = [
            _edge("document:doc_a", "chunk:doc_a__dieu_1", "DOCUMENT_HAS_CHUNK"),
            _edge("document:doc_a", "chunk:doc_a__dieu_2", "DOCUMENT_HAS_CHUNK"),
            _edge("chunk:doc_a__dieu_1", "policy_area:graduation", "CHUNK_HAS_POLICY_AREA"),
            _edge("chunk:doc_a__dieu_2", "policy_area:graduation", "CHUNK_HAS_POLICY_AREA"),
        ]
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "graduation", idx, edges, limit=10)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 4. Finding chunks by action_tag
# ---------------------------------------------------------------------------

class TestFindChunksByActionTag:
    def test_finds_chunk_by_action_tag(self):
        nodes, edges = _minimal_graph(
            policy_area="course_registration",
            action_tag="register_course",
        )
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("action_tag", "register_course", idx, edges, limit=10)
        assert len(results) == 1
        assert results[0]["node_type"] == "Chunk"

    def test_action_tag_not_mixed_with_policy_area(self):
        """Querying action_tag should not return policy_area matches."""
        nodes, edges = _minimal_graph(
            policy_area="graduation",
            action_tag="graduation_audit",
        )
        idx = index_nodes_by_id(nodes)
        # Search for a policy_area tag — should not return the action_tag match
        pa_results = find_chunks_by_tag("policy_area", "graduation_audit", idx, edges)
        assert len(pa_results) == 0


# ---------------------------------------------------------------------------
# 5. Limit
# ---------------------------------------------------------------------------

class TestLimit:
    def _build_multi_chunk_graph(self, n: int, tag_value: str = "graduation") -> tuple:
        nodes: list[dict] = [_doc_node("doc_a"), _tag_node("policy_area", "PolicyArea", tag_value)]
        edges: list[dict] = []
        for i in range(1, n + 1):
            cid = f"doc_a__dieu_{i}"
            nodes.append(_chunk_node(cid, "doc_a"))
            edges.append(_edge("document:doc_a", f"chunk:{cid}", "DOCUMENT_HAS_CHUNK"))
            edges.append(_edge(f"chunk:{cid}", f"policy_area:{tag_value}", "CHUNK_HAS_POLICY_AREA"))
        return nodes, edges

    def test_limit_restricts_results(self):
        nodes, edges = self._build_multi_chunk_graph(8)
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "graduation", idx, edges, limit=3)
        assert len(results) == 3

    def test_limit_1_returns_single_result(self):
        nodes, edges = self._build_multi_chunk_graph(5)
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "graduation", idx, edges, limit=1)
        assert len(results) == 1

    def test_limit_larger_than_results_returns_all(self):
        nodes, edges = self._build_multi_chunk_graph(3)
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "graduation", idx, edges, limit=100)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# 6. Unknown tag returns empty list
# ---------------------------------------------------------------------------

class TestUnknownTag:
    def test_unknown_tag_type_returns_empty(self):
        nodes, edges = _minimal_graph()
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("nonexistent_type", "graduation", idx, edges)
        assert results == []

    def test_unknown_tag_value_returns_empty(self):
        nodes, edges = _minimal_graph(policy_area="graduation")
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "no_such_area", idx, edges)
        assert results == []

    def test_wrong_tag_type_for_existing_value_returns_empty(self):
        """graduation exists as policy_area; looking for it as action_tag should fail."""
        nodes, edges = _minimal_graph(policy_area="graduation")
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("action_tag", "graduation", idx, edges)
        assert results == []


# ---------------------------------------------------------------------------
# 7. Summary counters on small in-memory graph
# ---------------------------------------------------------------------------

class TestSummaryCounters:
    def test_summary_runs_without_error(self, capsys):
        nodes, edges = _minimal_graph(policy_area="graduation", action_tag="graduation_audit")
        print_summary(nodes, edges)
        captured = capsys.readouterr()
        assert "Total nodes" in captured.out
        assert "Total edges" in captured.out

    def test_summary_counts_correct(self, capsys):
        nodes = [
            _doc_node("doc_a"),
            _chunk_node(),
            _tag_node("policy_area", "PolicyArea", "graduation"),
        ]
        edges = [
            _edge("document:doc_a", "chunk:doc_a__dieu_1", "DOCUMENT_HAS_CHUNK"),
            _edge("chunk:doc_a__dieu_1", "policy_area:graduation", "CHUNK_HAS_POLICY_AREA"),
        ]
        print_summary(nodes, edges)
        captured = capsys.readouterr()
        assert "3" in captured.out   # total nodes
        assert "2" in captured.out   # total edges

    def test_summary_shows_node_types(self, capsys):
        nodes = [_doc_node(), _chunk_node(), _tag_node("policy_area", "PolicyArea", "graduation")]
        print_summary(nodes, [])
        captured = capsys.readouterr()
        assert "Document" in captured.out
        assert "Chunk" in captured.out
        assert "PolicyArea" in captured.out

    def test_summary_shows_edge_types(self, capsys):
        nodes, edges = _minimal_graph()
        print_summary(nodes, edges)
        captured = capsys.readouterr()
        assert "DOCUMENT_HAS_CHUNK" in captured.out
        assert "CHUNK_HAS_POLICY_AREA" in captured.out


# ---------------------------------------------------------------------------
# 8. Integration: real graph files
# ---------------------------------------------------------------------------

class TestIntegration:
    _NODES = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_nodes.jsonl"
    _EDGES = Path(__file__).parent.parent / "data" / "graph" / "policy_graph_edges.jsonl"

    def _skip_if_missing(self):
        if not self._NODES.exists() or not self._EDGES.exists():
            pytest.skip("Graph files not available — run the pipeline first")

    def test_graduation_returns_at_least_one_chunk(self):
        self._skip_if_missing()
        nodes = read_jsonl(self._NODES)
        edges = read_jsonl(self._EDGES)
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "graduation", idx, edges, limit=20)
        assert len(results) >= 1, "Expected at least one chunk tagged graduation"

    def test_results_are_chunk_nodes(self):
        self._skip_if_missing()
        nodes = read_jsonl(self._NODES)
        edges = read_jsonl(self._EDGES)
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("policy_area", "graduation", idx, edges, limit=5)
        assert all(r["node_type"] == "Chunk" for r in results)

    def test_action_tag_register_course_returns_chunks(self):
        self._skip_if_missing()
        nodes = read_jsonl(self._NODES)
        edges = read_jsonl(self._EDGES)
        idx = index_nodes_by_id(nodes)
        results = find_chunks_by_tag("action_tag", "register_course", idx, edges, limit=10)
        assert len(results) >= 1, "Expected at least one chunk tagged register_course"

    def test_index_contains_all_node_ids(self):
        self._skip_if_missing()
        nodes = read_jsonl(self._NODES)
        idx = index_nodes_by_id(nodes)
        assert len(idx) == len(nodes)
        for node in nodes:
            assert node["node_id"] in idx
