#!/usr/bin/env python3
"""
scripts/policy_retrieval_backends.py

Retrieval backend abstraction for Academic Policy Graph RAG.

Defines the PolicyRetrievalBackend Protocol and concrete backends:
  - LexicalPolicyRetrievalBackend (lexical_v0): default production backend,
    wraps the existing retrieve_chunks() from 05_retrieve_policy_chunks.py.
  - BM25LikePolicyRetrievalBackend (bm25_like_v0): experimental backend,
    uses a BM25-like formula with standard-library-only implementation.

Future vector/hybrid retrieval backends should be added here and injected
into PolicyRetrievalService — never by bypassing source selection, strict
evidence pruning, citation guardrails, or temporal notice checks.
"""

import math
import sys
from pathlib import Path
from typing import Any, Protocol

# Ensure scripts folder is in python path for the importlib dance
sys.path.insert(0, str(Path(__file__).parent.absolute()))

import importlib

try:
    _retriever = importlib.import_module("retrieve_policy_chunks")
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _retriever = importlib.import_module("retrieve_policy_chunks")

_retrieve_chunks = _retriever.retrieve_chunks
_filter_chunks = _retriever.filter_chunks
_normalize_text = _retriever.normalize_text
_tokenize = _retriever.tokenize
_get_section_number_key = _retriever.get_section_number_key


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
# Experimental BM25-like backend
# ---------------------------------------------------------------------------


def _build_chunk_text(chunk: dict[str, Any]) -> str:
    """
    Build the searchable text for a chunk.

    Section title is prepended twice to give it a natural boost: the
    BM25 formula will see those tokens appear more often in the document,
    raising their TF contribution without any ad-hoc weight override.
    """
    title = chunk.get("section_title", "")
    body = chunk.get("text", "")
    # Repeat title so title matches score higher than body-only matches
    return f"{title}\n{title}\n{body}"


class BM25LikePolicyRetrievalBackend:
    """
    Experimental BM25-like retrieval backend.

    Uses only the Python standard library (math module).
    Tokenization and normalization reuse the Vietnamese accent-insensitive
    helpers from 05_retrieve_policy_chunks.py for consistency.

    BM25 parameters:
        k1 = 1.5  (term-frequency saturation)
        b  = 0.75 (length normalization strength)

    Section title gets a natural boost by appearing twice in the
    searchable text field (see _build_chunk_text).

    graph_bonus_map values are added directly to the BM25 score to match
    the behaviour of LexicalPolicyRetrievalBackend.

    Tie-breaking order (same as lexical backend):
        1. score descending
        2. doc_id ascending
        3. section_number numeric if possible, then string
        4. chunk_id ascending
    """

    name: str = "bm25_like_v0"

    # BM25 hyper-parameters
    K1: float = 1.5
    B: float = 0.75

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
        """Score and return top_k chunks using BM25-like formula."""

        # 1. Filter by metadata tags (reuse existing filter logic)
        filtered = _filter_chunks(
            chunks,
            policy_area=policy_area,
            action_tag=action_tag,
            requirement_tag=requirement_tag,
            risk_tag=risk_tag,
        )
        if not filtered:
            return []

        # 2. Tokenize query
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        bonus_map = graph_bonus_map or {}

        # 3. Build token lists for each chunk (searchable text)
        tokenized_docs: list[list[str]] = [
            _tokenize(_build_chunk_text(c)) for c in filtered
        ]

        # 4. Compute document frequencies across the filtered corpus
        n_docs = len(filtered)
        df: dict[str, int] = {}
        for doc_tokens in tokenized_docs:
            for tok in set(doc_tokens):
                df[tok] = df.get(tok, 0) + 1

        # 5. Average document length (in tokens)
        avgdl = sum(len(d) for d in tokenized_docs) / n_docs if n_docs else 1.0

        # 6. Score each chunk
        k1 = self.K1
        b = self.B
        scored: list[tuple[dict[str, Any], float]] = []

        for chunk, doc_tokens in zip(filtered, tokenized_docs):
            dl = len(doc_tokens)
            score = 0.0

            for tok in query_tokens:
                tf = doc_tokens.count(tok)
                if tf == 0:
                    continue
                # IDF: add 1 smoothing to avoid division by zero and log(0)
                idf = math.log((n_docs - df.get(tok, 0) + 0.5) / (df.get(tok, 0) + 0.5) + 1)
                # BM25 TF component with length normalisation
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                score += idf * tf_norm

            # Add graph bonus
            c_id = chunk.get("chunk_id", "")
            score += bonus_map.get(c_id, 0.0)

            if score > 0.0:
                scored.append((chunk, score))

        # 7. Sort: score desc, doc_id asc, section_number asc, chunk_id asc
        def sort_key(item: tuple[dict[str, Any], float]) -> tuple[float, str, tuple, str]:
            c, s = item
            return (
                -s,
                c.get("doc_id", ""),
                _get_section_number_key(c.get("section_number", "")),
                c.get("chunk_id", ""),
            )

        scored.sort(key=sort_key)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type] = {
    "lexical_v0": LexicalPolicyRetrievalBackend,
    "bm25_like_v0": BM25LikePolicyRetrievalBackend,
}


def get_default_retrieval_backend() -> LexicalPolicyRetrievalBackend:
    """Return the default production retrieval backend (lexical_v0)."""
    return LexicalPolicyRetrievalBackend()


def get_retrieval_backend(name: str | None = None) -> Any:
    """
    Return a retrieval backend by name.

    Args:
        name: Backend name. None / empty string / "lexical_v0" → lexical_v0.
              "bm25_like_v0" → BM25LikePolicyRetrievalBackend.

    Returns:
        An instantiated retrieval backend.

    Raises:
        ValueError: If name is not a known backend.
    """
    if not name or name == "lexical_v0":
        return LexicalPolicyRetrievalBackend()
    cls = _BACKEND_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(
            f"Unknown retrieval backend: {name!r}. Known backends: {known}"
        )
    return cls()
