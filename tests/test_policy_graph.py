"""
tests/test_policy_graph.py

Unit and integration tests for the policy graph builder.

Run with:
    python -m pytest
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module

_mod = import_module("build_policy_graph")

make_document_node = _mod.make_document_node
make_chunk_node    = _mod.make_chunk_node
make_tag_node      = _mod.make_tag_node
make_edge          = _mod.make_edge
build_graph        = _mod.build_graph
read_jsonl         = _mod.read_jsonl
write_jsonl        = _mod.write_jsonl


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _sample_chunk(
    chunk_id: str = "doc_a__dieu_1",
    doc_id: str = "doc_a",
    section_title: str = "Điều 1. Test",
    **kwargs,
) -> dict:
    base = {
        "chunk_id":       chunk_id,
        "doc_id":         doc_id,
        "title":          "Test Document",
        "decision_no":    "001/QD",
        "issued_date":    "2023-01-01",
        "institution":    "OU",
        "education_mode": "full_time",
        "chapter_title":  "Chương I",
        "section_title":  section_title,
        "section_number": "1",
        "chunk_type":     "dieu",
        "source_path":    "data/raw/cleaned/doc_a.md",
        "source_pdf":     "data/raw/documents/doc_a.pdf",
        "char_count":     100,
        "word_count":     20,
        "text":           "Sample text content.",
        # Annotation fields
        "policy_area":         [],
        "action_tags":         [],
        "student_status_tags": [],
        "procedure_tags":      [],
        "evidence_groups":     [],
        "risk_tags":           [],
        "requirement_tags":    [],
        "time_tags":           [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. Document node creation
# ---------------------------------------------------------------------------

class TestDocumentNode:
    def test_node_id_format(self):
        chunk = _sample_chunk(doc_id="my_doc")
        node = make_document_node(chunk)
        assert node["node_id"] == "document:my_doc"

    def test_node_type(self):
        node = make_document_node(_sample_chunk())
        assert node["node_type"] == "Document"

    def test_label_is_title(self):
        chunk = _sample_chunk()
        chunk["title"] = "My Policy Title"
        node = make_document_node(chunk)
        assert node["label"] == "My Policy Title"

    def test_required_properties(self):
        node = make_document_node(_sample_chunk())
        for field in ("doc_id", "title", "decision_no", "issued_date",
                      "institution", "education_mode", "source_pdf"):
            assert field in node["properties"], f"Missing property: {field}"

    def test_properties_values(self):
        chunk = _sample_chunk(doc_id="doc_x")
        chunk["institution"] = "Test University"
        node = make_document_node(chunk)
        assert node["properties"]["doc_id"] == "doc_x"
        assert node["properties"]["institution"] == "Test University"


# ---------------------------------------------------------------------------
# 2. Chunk node creation
# ---------------------------------------------------------------------------

class TestChunkNode:
    def test_node_id_format(self):
        chunk = _sample_chunk(chunk_id="doc_a__dieu_5")
        node = make_chunk_node(chunk)
        assert node["node_id"] == "chunk:doc_a__dieu_5"

    def test_node_type(self):
        node = make_chunk_node(_sample_chunk())
        assert node["node_type"] == "Chunk"

    def test_label_uses_section_title_when_present(self):
        chunk = _sample_chunk(section_title="Điều 5. Some rule")
        node = make_chunk_node(chunk)
        assert node["label"] == "Điều 5. Some rule"

    def test_label_falls_back_to_chunk_id(self):
        chunk = _sample_chunk(chunk_id="doc_a__preamble", section_title="")
        node = make_chunk_node(chunk)
        assert node["label"] == "doc_a__preamble"

    def test_text_in_properties(self):
        chunk = _sample_chunk()
        chunk["text"] = "Important policy text."
        node = make_chunk_node(chunk)
        assert node["properties"]["text"] == "Important policy text."

    def test_required_properties(self):
        node = make_chunk_node(_sample_chunk())
        for field in ("chunk_id", "doc_id", "title", "chapter_title",
                      "section_title", "section_number", "chunk_type",
                      "source_path", "source_pdf", "char_count",
                      "word_count", "text"):
            assert field in node["properties"], f"Missing property: {field}"


# ---------------------------------------------------------------------------
# 3. Tag node creation
# ---------------------------------------------------------------------------

class TestTagNode:
    def test_policy_area_node(self):
        node = make_tag_node("policy_area", "PolicyArea", "course_registration")
        assert node["node_id"] == "policy_area:course_registration"
        assert node["node_type"] == "PolicyArea"
        assert node["label"] == "course_registration"
        assert node["properties"]["value"] == "course_registration"

    def test_action_tag_node(self):
        node = make_tag_node("action_tag", "ActionTag", "register_course")
        assert node["node_id"] == "action_tag:register_course"
        assert node["node_type"] == "ActionTag"

    def test_requirement_tag_node(self):
        node = make_tag_node("requirement_tag", "RequirementTag", "credit_completion_required")
        assert node["node_id"] == "requirement_tag:credit_completion_required"
        assert node["node_type"] == "RequirementTag"

    def test_properties_contain_value(self):
        node = make_tag_node("risk_tag", "RiskTag", "forced_dropout")
        assert node["properties"] == {"value": "forced_dropout"}


# ---------------------------------------------------------------------------
# 4. DOCUMENT_HAS_CHUNK edge creation
# ---------------------------------------------------------------------------

class TestDocumentHasChunkEdge:
    def test_edge_id_format(self):
        edge = make_edge("document:doc_a", "chunk:doc_a__dieu_9", "DOCUMENT_HAS_CHUNK")
        expected = "document:doc_a->chunk:doc_a__dieu_9:DOCUMENT_HAS_CHUNK"
        assert edge["edge_id"] == expected

    def test_edge_type(self):
        edge = make_edge("document:doc_a", "chunk:doc_a__dieu_9", "DOCUMENT_HAS_CHUNK")
        assert edge["edge_type"] == "DOCUMENT_HAS_CHUNK"

    def test_source_and_target(self):
        edge = make_edge("document:doc_a", "chunk:doc_a__dieu_9", "DOCUMENT_HAS_CHUNK")
        assert edge["source"] == "document:doc_a"
        assert edge["target"] == "chunk:doc_a__dieu_9"

    def test_empty_properties_by_default(self):
        edge = make_edge("document:doc_a", "chunk:doc_a__dieu_1", "DOCUMENT_HAS_CHUNK")
        assert edge["properties"] == {}

    def test_build_graph_produces_doc_chunk_edge(self):
        chunk = _sample_chunk(doc_id="doc_a", chunk_id="doc_a__dieu_1")
        _, edges = build_graph([chunk])
        edge_types = {e["edge_type"] for e in edges}
        assert "DOCUMENT_HAS_CHUNK" in edge_types

        doc_chunk_edges = [e for e in edges if e["edge_type"] == "DOCUMENT_HAS_CHUNK"]
        assert any(
            e["source"] == "document:doc_a" and e["target"] == "chunk:doc_a__dieu_1"
            for e in doc_chunk_edges
        )


# ---------------------------------------------------------------------------
# 5. Chunk-to-policy-area edge
# ---------------------------------------------------------------------------

class TestChunkToPolicyAreaEdge:
    def test_edge_created_for_policy_area(self):
        chunk = _sample_chunk(
            chunk_id="doc_a__dieu_1",
            policy_area=["course_registration"],
        )
        _, edges = build_graph([chunk])
        pa_edges = [e for e in edges if e["edge_type"] == "CHUNK_HAS_POLICY_AREA"]
        assert len(pa_edges) == 1
        assert pa_edges[0]["source"] == "chunk:doc_a__dieu_1"
        assert pa_edges[0]["target"] == "policy_area:course_registration"

    def test_edge_id_format_for_policy_area(self):
        chunk = _sample_chunk(policy_area=["graduation"])
        _, edges = build_graph([chunk])
        pa_edge = next(e for e in edges if e["edge_type"] == "CHUNK_HAS_POLICY_AREA")
        assert "CHUNK_HAS_POLICY_AREA" in pa_edge["edge_id"]

    def test_multiple_areas_produce_multiple_edges(self):
        chunk = _sample_chunk(policy_area=["course_registration", "graduation"])
        _, edges = build_graph([chunk])
        pa_edges = [e for e in edges if e["edge_type"] == "CHUNK_HAS_POLICY_AREA"]
        assert len(pa_edges) == 2

    def test_all_tag_edge_types_present_when_tags_set(self):
        chunk = _sample_chunk(
            policy_area=["course_registration"],
            action_tags=["register_course"],
            student_status_tags=["academic_warning"],
            procedure_tags=["application_required"],
            evidence_groups=["transcript"],
            risk_tags=["forced_dropout"],
            requirement_tags=["minimum_gpa_required"],
            time_tags=["registration_period"],
        )
        _, edges = build_graph([chunk])
        edge_types = {e["edge_type"] for e in edges}
        for expected in (
            "CHUNK_HAS_POLICY_AREA",
            "CHUNK_HAS_ACTION_TAG",
            "CHUNK_HAS_STUDENT_STATUS_TAG",
            "CHUNK_HAS_PROCEDURE_TAG",
            "CHUNK_HAS_EVIDENCE_GROUP",
            "CHUNK_HAS_RISK_TAG",
            "CHUNK_HAS_REQUIREMENT_TAG",
            "CHUNK_HAS_TIME_TAG",
        ):
            assert expected in edge_types, f"Missing edge type: {expected}"


# ---------------------------------------------------------------------------
# 6. Deduplication of repeated tag nodes
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_shared_policy_area_deduped(self):
        chunk1 = _sample_chunk("doc_a__dieu_1", "doc_a", policy_area=["graduation"])
        chunk2 = _sample_chunk("doc_a__dieu_2", "doc_a", policy_area=["graduation"])
        nodes, _ = build_graph([chunk1, chunk2])
        pa_nodes = [n for n in nodes if n["node_type"] == "PolicyArea"]
        assert len(pa_nodes) == 1

    def test_shared_document_deduped(self):
        chunk1 = _sample_chunk("doc_a__dieu_1", "doc_a")
        chunk2 = _sample_chunk("doc_a__dieu_2", "doc_a")
        nodes, _ = build_graph([chunk1, chunk2])
        doc_nodes = [n for n in nodes if n["node_type"] == "Document"]
        assert len(doc_nodes) == 1

    def test_repeated_edge_deduped(self):
        """Two chunks sharing the same tag → two edges (chunk→tag), not duplicates."""
        chunk1 = _sample_chunk("doc_a__dieu_1", "doc_a", policy_area=["graduation"])
        chunk2 = _sample_chunk("doc_a__dieu_2", "doc_a", policy_area=["graduation"])
        _, edges = build_graph([chunk1, chunk2])
        pa_edges = [e for e in edges if e["edge_type"] == "CHUNK_HAS_POLICY_AREA"]
        # Each chunk gets its own edge to the shared node
        assert len(pa_edges) == 2
        edge_ids = [e["edge_id"] for e in pa_edges]
        assert len(edge_ids) == len(set(edge_ids)), "Duplicate edge_ids found"

    def test_two_docs_two_document_nodes(self):
        chunk1 = _sample_chunk("doc_a__dieu_1", "doc_a")
        chunk2 = _sample_chunk("doc_b__dieu_1", "doc_b")
        nodes, _ = build_graph([chunk1, chunk2])
        doc_nodes = [n for n in nodes if n["node_type"] == "Document"]
        assert len(doc_nodes) == 2


# ---------------------------------------------------------------------------
# 7. Deterministic sorting
# ---------------------------------------------------------------------------

class TestDeterministicSorting:
    def test_nodes_sorted_by_node_id(self):
        chunks = [
            _sample_chunk("doc_b__dieu_1", "doc_b", policy_area=["graduation"]),
            _sample_chunk("doc_a__dieu_1", "doc_a", policy_area=["course_registration"]),
        ]
        nodes, _ = build_graph(chunks)
        ids = [n["node_id"] for n in nodes]
        assert ids == sorted(ids)

    def test_edges_sorted_by_edge_id(self):
        chunks = [
            _sample_chunk("doc_b__dieu_1", "doc_b", policy_area=["graduation"]),
            _sample_chunk("doc_a__dieu_1", "doc_a", policy_area=["course_registration"]),
        ]
        _, edges = build_graph(chunks)
        ids = [e["edge_id"] for e in edges]
        assert ids == sorted(ids)

    def test_same_input_order_gives_same_output(self):
        chunks = [
            _sample_chunk("doc_a__dieu_1", "doc_a", policy_area=["graduation"]),
            _sample_chunk("doc_a__dieu_2", "doc_a", action_tags=["register_course"]),
        ]
        nodes1, edges1 = build_graph(chunks)
        nodes2, edges2 = build_graph(chunks)
        assert [n["node_id"] for n in nodes1] == [n["node_id"] for n in nodes2]
        assert [e["edge_id"] for e in edges1] == [e["edge_id"] for e in edges2]


# ---------------------------------------------------------------------------
# 8. In-memory sample graph build
# ---------------------------------------------------------------------------

class TestInMemoryBuild:
    def test_node_and_edge_counts(self):
        chunk = _sample_chunk(
            "doc_a__dieu_1",
            "doc_a",
            policy_area=["course_registration"],
            action_tags=["register_course", "study_ahead"],
        )
        nodes, edges = build_graph([chunk])
        # Expect: 1 Document + 1 Chunk + 1 PolicyArea + 2 ActionTag = 5 nodes
        assert len(nodes) == 5
        # Expect: 1 DOCUMENT_HAS_CHUNK + 1 CHUNK_HAS_POLICY_AREA + 2 CHUNK_HAS_ACTION_TAG = 4 edges
        assert len(edges) == 4

    def test_all_node_schema_fields_present(self):
        nodes, _ = build_graph([_sample_chunk(policy_area=["graduation"])])
        for node in nodes:
            for field in ("node_id", "node_type", "label", "properties"):
                assert field in node, f"Field '{field}' missing from node {node.get('node_id')}"

    def test_all_edge_schema_fields_present(self):
        _, edges = build_graph([_sample_chunk(policy_area=["graduation"])])
        for edge in edges:
            for field in ("edge_id", "source", "target", "edge_type", "properties"):
                assert field in edge, f"Field '{field}' missing from edge {edge.get('edge_id')}"

    def test_write_read_roundtrip(self, tmp_path):
        chunks = [_sample_chunk(policy_area=["graduation"], action_tags=["graduation_audit"])]
        nodes, edges = build_graph(chunks)

        nodes_path = tmp_path / "nodes.jsonl"
        edges_path = tmp_path / "edges.jsonl"
        write_jsonl(nodes_path, nodes)
        write_jsonl(edges_path, edges)

        read_nodes = read_jsonl(nodes_path)
        read_edges = read_jsonl(edges_path)
        assert len(read_nodes) == len(nodes)
        assert len(read_edges) == len(edges)
        assert read_nodes[0]["node_id"] == nodes[0]["node_id"]


# ---------------------------------------------------------------------------
# 9. Integration: real annotated file
# ---------------------------------------------------------------------------

class TestIntegration:
    _INPUT = Path(__file__).parent.parent / "data" / "chunks" / "policy_chunks.annotated.jsonl"

    def test_produces_nonempty_nodes_and_edges(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        nodes, edges = build_graph(chunks)

        nodes_path = tmp_path / "nodes.jsonl"
        edges_path = tmp_path / "edges.jsonl"
        write_jsonl(nodes_path, nodes)
        write_jsonl(edges_path, edges)

        assert len(nodes) > 0, "Expected non-empty nodes"
        assert len(edges) > 0, "Expected non-empty edges"

    def test_document_nodes_match_unique_doc_ids(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        unique_docs = {c["doc_id"] for c in chunks}
        nodes, _ = build_graph(chunks)
        doc_nodes = [n for n in nodes if n["node_type"] == "Document"]
        assert len(doc_nodes) == len(unique_docs)

    def test_chunk_count_matches(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        nodes, _ = build_graph(chunks)
        chunk_nodes = [n for n in nodes if n["node_type"] == "Chunk"]
        assert len(chunk_nodes) == len(chunks)

    def test_nodes_sorted(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        nodes, _ = build_graph(chunks)
        ids = [n["node_id"] for n in nodes]
        assert ids == sorted(ids)

    def test_edges_sorted(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        _, edges = build_graph(chunks)
        ids = [e["edge_id"] for e in edges]
        assert ids == sorted(ids)

    def test_no_duplicate_node_ids(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        nodes, _ = build_graph(chunks)
        ids = [n["node_id"] for n in nodes]
        assert len(ids) == len(set(ids)), "Duplicate node_ids found"

    def test_no_duplicate_edge_ids(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        _, edges = build_graph(chunks)
        ids = [e["edge_id"] for e in edges]
        assert len(ids) == len(set(ids)), "Duplicate edge_ids found"
