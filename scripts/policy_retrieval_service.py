#!/usr/bin/env python3
"""
scripts/policy_retrieval_service.py

Service layer wrapping policy chunk retrieval, source selection,
and pruning with graph bonus integration.

Retrieval is delegated to a swappable PolicyRetrievalBackend.
The default backend is LexicalPolicyRetrievalBackend (lexical_v0).

Future BM25/vector/hybrid retrieval should be implemented as new
retrieval backends and injected here — not by bypassing source
selection, strict evidence pruning, citation guardrails, or temporal
notice checks.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure scripts folder is in python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import importlib

try:
    _retriever = importlib.import_module("05_retrieve_policy_chunks")
    _selector = importlib.import_module("07_select_policy_sources")
    _backends = importlib.import_module("policy_retrieval_backends")
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _retriever = importlib.import_module("05_retrieve_policy_chunks")
    _selector = importlib.import_module("07_select_policy_sources")
    _backends = importlib.import_module("policy_retrieval_backends")

load_graph_expansion = _retriever.load_graph_expansion
select_sources_for_issue = _selector.select_sources_for_issue
prune_selected_sources_for_issue = _selector.prune_selected_sources_for_issue
PolicyRetrievalBackend = _backends.PolicyRetrievalBackend
get_default_retrieval_backend = _backends.get_default_retrieval_backend
get_retrieval_backend = _backends.get_retrieval_backend


class PolicyRetrievalService:
    def __init__(
        self,
        chunks: list[dict],
        nodes_file: str | Path | None = None,
        edges_file: str | Path | None = None,
        backend: Any | None = None,
        backend_name: str | None = None,
    ):
        """
        Initialize the retrieval service.

        Args:
            chunks: Annotated policy chunks.
            nodes_file: Optional path to graph nodes JSONL.
            edges_file: Optional path to graph edges JSONL.
            backend: Optional retrieval backend instance. Takes priority over
                     backend_name. Defaults to LexicalPolicyRetrievalBackend.
            backend_name: Optional backend name string (e.g. "bm25_like_v0").
                          Ignored if backend is provided. Uses get_retrieval_backend()
                          to resolve. Defaults to LexicalPolicyRetrievalBackend.
        """
        self.chunks = chunks
        self.nodes_file = Path(nodes_file) if nodes_file else None
        self.edges_file = Path(edges_file) if edges_file else None
        if backend is not None:
            self.backend = backend
        elif backend_name is not None:
            self.backend = get_retrieval_backend(backend_name)
        else:
            self.backend = get_default_retrieval_backend()

    @property
    def backend_name(self) -> str:
        """Return the name of the active retrieval backend."""
        return self.backend.name

    def build_graph_bonus_map(self, question: str) -> dict[str, float]:
        """
        Builds a graph bonus map if graph files exist and are loaded.
        Otherwise returns an empty dictionary.
        """
        if not self.nodes_file or not self.edges_file:
            return {}
        if not self.nodes_file.exists() or not self.edges_file.exists():
            return {}
        try:
            return load_graph_expansion(question, self.nodes_file, self.edges_file)
        except Exception:
            return {}

    def retrieve_for_issue(
        self,
        issue: dict,
        question: str,
        top_k: int = 5,
        max_sources: int = 3,
        use_graph: bool = True,
        strict_pruning: bool = True,
        # Additional filters/bonus maps to match exactly 06_answer_policy_question needs
        policy_area_filter: str | None = None,
        action_tag_filter: str | None = None,
        requirement_tag_filter: str | None = None,
        risk_tag_filter: str | None = None,
        graph_bonus_map: dict[str, float] | None = None,
    ) -> list[tuple[dict, float]]:
        """
        Retrieve evidence chunks for a single issue, applying source selection
        and optional strict pruning.
        """
        g_map = graph_bonus_map
        if g_map is None:
            g_map = self.build_graph_bonus_map(question) if use_graph else {}

        p_area = policy_area_filter or issue["policy_area"]

        # 1. Retrieve via backend
        results = self.backend.retrieve(
            chunks=self.chunks,
            query=issue["query"],
            top_k=top_k,
            policy_area=p_area,
            action_tag=action_tag_filter,
            requirement_tag=requirement_tag_filter,
            risk_tag=risk_tag_filter,
            graph_bonus_map=g_map,
        )

        # 2. Source Selection
        selected = select_sources_for_issue(issue, results, max_sources=max_sources)

        # 3. Strict Pruning
        if strict_pruning:
            selected = prune_selected_sources_for_issue(issue, selected, max_sources=max_sources)

        return selected

    def retrieve_for_issues(
        self,
        issues: list[dict],
        question: str,
        top_k: int = 5,
        max_sources_per_issue: int = 3,
        use_graph: bool = True,
        strict_pruning: bool = True,
        # Overrides/Filters
        policy_area_filter: str | None = None,
        action_tag_filter: str | None = None,
        requirement_tag_filter: str | None = None,
        risk_tag_filter: str | None = None,
        graph_bonus_map: dict[str, float] | None = None,
    ) -> list[tuple[dict, float]]:
        """
        Retrieve and aggregate evidence chunks for multiple issues.
        Deduplicates final sources by chunk_id while preserving order.
        """
        all_results = []
        seen_chunk_ids = set()

        for issue in issues:
            selected = self.retrieve_for_issue(
                issue=issue,
                question=question,
                top_k=top_k,
                max_sources=max_sources_per_issue,
                use_graph=use_graph,
                strict_pruning=strict_pruning,
                policy_area_filter=policy_area_filter,
                action_tag_filter=action_tag_filter,
                requirement_tag_filter=requirement_tag_filter,
                risk_tag_filter=risk_tag_filter,
                graph_bonus_map=graph_bonus_map,
            )
            for chunk, score in selected:
                c_id = chunk.get("chunk_id")
                if c_id not in seen_chunk_ids:
                    seen_chunk_ids.add(c_id)
                    all_results.append((chunk, score))

        return all_results
