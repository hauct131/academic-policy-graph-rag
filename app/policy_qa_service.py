#!/usr/bin/env python3
"""
app/policy_qa_service.py

Service layer for Academic Policy QA API.
Loads chunks, domain config, and document registry, and answers questions.
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

# Add scripts directory to path to load QA pipeline modules
scripts_path = str(Path(__file__).parent.parent / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

import importlib
try:
    _qa = importlib.import_module("06_answer_policy_question")
    _retriever = importlib.import_module("05_retrieve_policy_chunks")
    _domain = importlib.import_module("policy_domain_config")
    _reg = importlib.import_module("policy_document_registry")
    _service = importlib.import_module("policy_retrieval_service")
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    _qa = importlib.import_module("06_answer_policy_question")
    _retriever = importlib.import_module("05_retrieve_policy_chunks")
    _domain = importlib.import_module("policy_domain_config")
    _reg = importlib.import_module("policy_document_registry")
    _service = importlib.import_module("policy_retrieval_service")

read_jsonl = _retriever.read_jsonl
answer_question = _qa.answer_question
load_domain_config = _domain.load_domain_config
infer_issues_from_domain = _domain.infer_issues_from_domain
load_document_registry = _reg.load_document_registry
should_warn_missing_current_notice = _reg.should_warn_missing_current_notice
PolicyRetrievalService = _service.PolicyRetrievalService


class PolicyQAService:
    def __init__(
        self,
        chunks_path: str = "data/chunks/policy_chunks.annotated.jsonl",
        domain_config_path: str = "domains/ou_academic_policy_v1/domain.json",
        document_registry_path: str = "domains/ou_academic_policy_v1/document_registry.jsonl",
        nodes_path: str = "data/graph/policy_graph_nodes.jsonl",
        edges_path: str = "data/graph/policy_graph_edges.jsonl",
    ):
        self.chunks_path = Path(chunks_path)
        self.domain_config_path = Path(domain_config_path)
        self.document_registry_path = Path(document_registry_path)
        self.nodes_path = Path(nodes_path)
        self.edges_path = Path(edges_path)

        self.chunks = []
        self.domain_config = None
        self.document_registry = None
        
        self.initialized = False
        self.missing_resources = []

    def load_resources(self):
        """Load required and optional resources, setting status."""
        self.missing_resources = []
        
        # Check required files
        if not self.chunks_path.exists():
            self.missing_resources.append("missing annotated chunks")
        if not self.domain_config_path.exists():
            self.missing_resources.append("missing domain config")
        if not self.document_registry_path.exists():
            self.missing_resources.append("missing document registry")

        if self.missing_resources:
            self.initialized = False
            return

        try:
            self.chunks = read_jsonl(self.chunks_path)
            self.domain_config = load_domain_config(self.domain_config_path)
            self.document_registry = load_document_registry(self.document_registry_path)
            self.retrieval_service = PolicyRetrievalService(
                chunks=self.chunks,
                nodes_file=self.nodes_path,
                edges_file=self.edges_path
            )
            self.initialized = True
        except Exception as e:
            self.initialized = False
            self.missing_resources.append(f"error loading resources: {str(e)}")

    def get_qa_response(
        self,
        question: str,
        top_k: int = 5,
        show_evidence_text: bool = False
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """
        Run the QA pipeline.
        Returns Tuple[answer, metadata, warnings].
        """
        if not self.initialized:
            msg = "Required data files are missing:\n"
            for res in self.missing_resources:
                msg += f"- {res}\n"
            if "missing annotated chunks" in self.missing_resources:
                msg += "\nPlease run the ingestion pipeline:\n"
                msg += "  python scripts/01_build_policy_chunks.py\n"
                msg += "  python scripts/02_annotate_policy_chunks.py\n"
            raise RuntimeError(msg)

        # Try to load graph expansion
        uses_graph = False
        if self.nodes_path.exists() and self.edges_path.exists():
            uses_graph = True

        # Call core QA answer function
        answer = answer_question(
            question=question,
            chunks=self.chunks,
            top_k=top_k,
            domain_config=self.domain_config,
            document_registry=self.document_registry,
            show_evidence_text=show_evidence_text,
            retrieval_service=self.retrieval_service,
        )

        # Detect warnings
        warnings = []
        issues = infer_issues_from_domain(question, self.domain_config)
        for issue in issues:
            if should_warn_missing_current_notice(
                question=question,
                records=self.document_registry,
                policy_area=issue["policy_area"],
            ):
                warnings.append(
                    f"chưa có thông báo học kỳ hiện tại cho {issue['policy_area']}"
                )

        metadata = {
            "domain_id": self.domain_config.get("domain_id", "ou_academic_policy_v1") if self.domain_config else "ou_academic_policy_v1",
            "chunks_loaded": len(self.chunks),
            "uses_document_registry": True,
            "uses_graph": uses_graph
        }

        return answer, metadata, warnings


def log_request_if_enabled(
    question_len: int,
    status: str,
    top_k: int,
    warnings: List[str]
):
    """Log QA request metadata if opt-in logging is enabled via environment variable."""
    if os.environ.get("POLICY_QA_ENABLE_REQUEST_LOGGING") == "1":
        try:
            log_dir = Path("data/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "policy_qa_requests.jsonl"
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "question_length": question_len,
                "status": status,
                "top_k": top_k,
                "warnings": warnings
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass
