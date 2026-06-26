#!/usr/bin/env python3
"""
scripts/policy_domain_config.py

Domain configuration loader and validation layer for OU Academic Policy RAG.
"""

import json
import unicodedata
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core import normalize_text

def load_domain_config(path: str | Path = "domains/ou_academic_policy_v1/domain.json") -> dict:
    """Load the JSON domain config from the specified path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Domain config file not found: {p}")
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate_domain_config(config: dict) -> list[str]:
    """
    Validate structure, fields, and types of the domain configuration dict.
    Returns a list of error strings. If empty, the configuration is valid.
    """
    errors = []
    if not isinstance(config, dict):
        errors.append("Config must be a dictionary.")
        return errors

    # Check top-level required fields
    required_top = [
        "domain_id",
        "domain_name",
        "version",
        "language",
        "institution",
        "fallback_scope_disclaimer",
        "current_semester_missing_notice_message",
        "certificate_keywords",
        "current_semester_keywords",
        "document_types",
        "issue_definitions"
    ]
    for key in required_top:
        if key not in config:
            errors.append(f"Missing top-level key: {key}")

    # Check lists
    for k in ["certificate_keywords", "current_semester_keywords", "document_types"]:
        if k in config and not isinstance(config[k], list):
            errors.append(f"Top-level key '{k}' must be a list of strings.")

    # Validate issue definitions
    if "issue_definitions" in config:
        defs = config["issue_definitions"]
        if not isinstance(defs, list):
            errors.append("'issue_definitions' must be a list.")
        else:
            required_def_keys = [
                "issue_type",
                "label",
                "keywords",
                "query_default",
                "answer_template"
            ]
            for idx, item in enumerate(defs):
                if not isinstance(item, dict):
                    errors.append(f"issue_definitions[{idx}] must be a dictionary.")
                    continue
                for key in required_def_keys:
                    if key not in item:
                        errors.append(f"issue_definitions[{idx}] (type: {item.get('issue_type', 'unknown')}) is missing key: {key}")
                if "keywords" in item and not isinstance(item["keywords"], list):
                    errors.append(f"issue_definitions[{idx}] keywords must be a list of strings.")
                if "query_when" in item:
                    q_when = item["query_when"]
                    if not isinstance(q_when, list):
                        errors.append(f"issue_definitions[{idx}] query_when must be a list.")
                    else:
                        for cond_idx, cond in enumerate(q_when):
                            if not isinstance(cond, dict) or "condition_keywords" not in cond or "query" not in cond:
                                errors.append(f"issue_definitions[{idx}] query_when[{cond_idx}] must be a dict with condition_keywords and query.")

    return errors


def infer_issues_from_domain(question: str, config: dict) -> list[dict]:
    """
    Infer academic policy issues from the user question based on domain config rules.
    Returns a list of dicts with:
      - issue_type
      - policy_area
      - query
      - label
    """
    norm_q = normalize_text(question)
    issues = []
    
    definitions = config.get("issue_definitions", [])
    
    # Identify non-generic issues whose keywords are in the question
    for item in definitions:
        itype = item.get("issue_type")
        if itype == "generic":
            continue
            
        kws = [normalize_text(kw) for kw in item.get("keywords", [])]
        if any(kw in norm_q for kw in kws):
            # Determine the search query
            query = item.get("query_default", "")
            
            # Check for conditional overrides
            for cond in item.get("query_when", []):
                cond_kws = [normalize_text(ckw) for ckw in cond.get("condition_keywords", [])]
                if any(ckw in norm_q for ckw in cond_kws):
                    query = cond.get("query")
                    break
            
            issues.append({
                "issue_type": itype,
                "policy_area": item.get("policy_area"),
                "query": query,
                "label": item.get("label", "")
            })
            
    # Fallback to generic if no issue detected
    if not issues:
        generic_item = None
        for item in definitions:
            if item.get("issue_type") == "generic":
                generic_item = item
                break
        
        if generic_item:
            issues.append({
                "issue_type": "generic",
                "policy_area": generic_item.get("policy_area"),
                "query": question,
                "label": generic_item.get("label", "Quy định liên quan")
            })
        else:
            issues.append({
                "issue_type": "generic",
                "policy_area": None,
                "query": question,
                "label": "Quy định liên quan"
            })
            
    return issues
