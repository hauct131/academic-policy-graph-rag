#!/usr/bin/env python3
"""
scripts/policy_retrieval_backends.py

Retrieval backend abstraction for Academic Policy Graph RAG.

Defines the PolicyRetrievalBackend Protocol and the default
LexicalPolicyRetrievalBackend that wraps the existing lexical
retrieve_chunks function from 05_retrieve_policy_chunks.py.

Future BM25/vector/hybrid retrieval backends should be implemented here
and injected into PolicyRetrievalService — never by bypassing source
selection, strict evidence pruning, citation guardrails, or temporal
notice checks.
"""

import sys
from pathlib import Path
from typing import Any, Protocol

# Ensure scripts folder is in python path for the importlib dance
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import importlib

try:
    _retriever = importlib.import_module("05_retrieve_policy_chunks")
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _retriever = importlib.import_module("05_retrieve_policy_chunks")

_retrieve_chunks = _retriever.retrieve_chunks


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PolicyRetrievalBackend(Protocol):
    """
    Protocol that all retrieval backends must satisfy.

    A backend is responsible only for scoring and ranking chunks.
    Source selection, strict pruning, citation guardrails, and temporal
    notice checks are handled at the PolicyRetrievalService layer.
    """

    name: str

    def retrieve(
        self,
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
        Retrieve and score chunks for the given query.

        Returns a list of (chunk, score) tuples sorted by descending score.
        """
        ...


# ---------------------------------------------------------------------------
# Default backend: lexical / metadata / graph-bonus scoring
# ---------------------------------------------------------------------------


class LexicalPolicyRetrievalBackend:
    """
    Default production retrieval backend.

    Delegates directly to the existing retrieve_chunks() function from
    05_retrieve_policy_chunks.py — lexical token overlap + metadata tag
    scoring + optional graph-bonus expansion. No scoring changes.
    """

    name: str = "lexical_v0"

    def retrieve(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        top_k: int = 5,
        policy_area: str | None = None,
        action_tag: str | None = None,
        requirement_tag: str | None = None,
        risk_tag: str | None = None,
        graph_bonus_map: dict[str, float] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Thin wrapper around retrieve_chunks with identical semantics."""
        return _retrieve_chunks(
            chunks=chunks,
            query=query,
            top_k=top_k,
            policy_area=policy_area,
            action_tag=action_tag,
            requirement_tag=requirement_tag,
            risk_tag=risk_tag,
            graph_bonus_map=graph_bonus_map,
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def get_default_retrieval_backend() -> LexicalPolicyRetrievalBackend:
    """Return the default production retrieval backend (lexical_v0)."""
    return LexicalPolicyRetrievalBackend()
