#!/usr/bin/env python3
"""
scripts/policy_document_registry.py

Temporal document registry and query selection support for OU Academic Policy RAG.
"""

import json
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import normalize_text, read_jsonl



# normalize_text is imported from core



# read_jsonl is imported from core



def load_document_registry(path: str | Path = "domains/ou_academic_policy_v1/document_registry.jsonl") -> list[dict[str, Any]]:
    """Load the document registry JSONL."""
    return read_jsonl(path)


def validate_date(date_str: Any) -> bool:
    """Check if date matches YYYY-MM-DD."""
    if date_str is None:
        return True
    if not isinstance(date_str, str):
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def policy_area_matches(record: dict[str, Any], policy_area: str | None) -> bool:
    """Check if the record's policy_area matches the given policy_area."""
    if policy_area is None:
        return True
    rec_area = record.get("policy_area")
    if rec_area is None:
        return False
    if isinstance(rec_area, list):
        return policy_area in rec_area
    return rec_area == policy_area


def validate_document_record(record: dict[str, Any]) -> list[str]:
    """Validate fields and format of a single document record."""
    errors = []
    if not isinstance(record, dict):
        errors.append("Record must be a dictionary.")
        return errors

    required_fields = ["doc_id", "title", "document_type", "status", "temporal_scope", "update_cadence"]
    for f in required_fields:
        if f not in record or not record[f]:
            errors.append(f"Missing required field: {f}")

    if "policy_area" in record and record["policy_area"] is not None:
        pa = record["policy_area"]
        if isinstance(pa, list):
            for idx, item in enumerate(pa):
                if not isinstance(item, str):
                    errors.append(f"policy_area list item at index {idx} must be a string.")
        elif not isinstance(pa, str):
            errors.append("policy_area must be a string, list of strings, or null.")

    for df in ["issued_date", "effective_from", "effective_to"]:
        if df in record and record[df] is not None:
            if not validate_date(record[df]):
                errors.append(f"Field '{df}' has invalid date format: {record[df]}. Must be YYYY-MM-DD.")
    return errors


def validate_document_registry(records: list[dict[str, Any]]) -> list[str]:
    """Validate all records in the registry and check for duplicate IDs."""
    errors = []
    if not isinstance(records, list):
        errors.append("Registry must be a list of records.")
        return errors

    doc_ids = set()
    for idx, r in enumerate(records):
        rec_errors = validate_document_record(r)
        for e in rec_errors:
            errors.append(f"Record {idx}: {e}")
        if "doc_id" in r and r["doc_id"]:
            if r["doc_id"] in doc_ids:
                errors.append(f"Duplicate doc_id found: {r['doc_id']}")
            doc_ids.add(r["doc_id"])
    return errors


def infer_time_context(question: str) -> dict[str, Any]:
    """Infer semester, academic year, and deadline intent from a question."""
    norm = normalize_text(question)
    context = {
        "current_semester": False,
        "semester": None,
        "academic_year": None,
        "has_deadline_intent": False
    }

    if "hoc ky nay" in norm or "hoc ky hien tai" in norm or "nam hoc nay" in norm or "han dang ky" in norm:
        context["current_semester"] = True

    if re.search(r"\b(hk|hoc ky)\s*1\b", norm):
        context["semester"] = 1
    elif re.search(r"\b(hk|hoc ky)\s*2\b", norm):
        context["semester"] = 2
    elif re.search(r"\b(hk|hoc ky)\s*3\b", norm):
        context["semester"] = 3

    year_match = re.search(r"nam hoc\s*(\d{4})[-/](\d{4})", norm)
    if year_match:
        context["academic_year"] = f"{year_match.group(1)}-{year_match.group(2)}"

    deadline_kws = ["deadline", "thoi han", "han chot", "khi nao", "bao gio", "lich thi", "lich hoc", "ngay thi", "ngay nop", "thoi gian"]
    if any(kw in norm for kw in deadline_kws):
        context["has_deadline_intent"] = True

    return context


def is_document_active(record: dict[str, Any], target_date: date | None = None) -> bool:
    """Check if the document is active currently or at target_date."""
    if record.get("status") != "active":
        return False
    if target_date is None:
        target_date = date.today()

    eff_from = None
    if record.get("effective_from"):
        try:
            eff_from = datetime.strptime(record["effective_from"], "%Y-%m-%d").date()
        except ValueError:
            pass

    eff_to = None
    if record.get("effective_to"):
        try:
            eff_to = datetime.strptime(record["effective_to"], "%Y-%m-%d").date()
        except ValueError:
            pass

    if eff_from and target_date < eff_from:
        return False
    if eff_to and target_date > eff_to:
        return False

    return True


def requires_current_notice(question: str) -> bool:
    """Check if query is asking for operational/schedule/deadline details."""
    time_ctx = infer_time_context(question)
    if time_ctx["current_semester"] or time_ctx["has_deadline_intent"]:
        return True
    norm = normalize_text(question)
    additional_kws = ["lich nop", "thoi khoa bieu", "lich thi", "lich hoc", "ke hoach"]
    if any(kw in norm for kw in additional_kws):
        return True
    return False


def has_current_notice(records: list[dict[str, Any]], policy_area: str | None = None, target_date: date | None = None) -> bool:
    """Check if there is an active semester or annual notice in the registry."""
    for r in records:
        if r.get("document_type") in ["semester_notice", "annual_notice"]:
            if is_document_active(r, target_date):
                if policy_area_matches(r, policy_area):
                    return True
    return False


def should_warn_missing_current_notice(
    question: str,
    records: list[dict[str, Any]],
    policy_area: str | None = None,
    target_date: date | None = None
) -> bool:
    """Check if warning about missing operational notice is necessary."""
    if not requires_current_notice(question):
        return False
    return not has_current_notice(records, policy_area=policy_area, target_date=target_date)


def select_documents_for_query(
    records: list[dict[str, Any]],
    policy_area: str | None = None,
    question: str = "",
    target_date: date | None = None
) -> list[dict[str, Any]]:
    """Prioritize and select documents matching the query requirements."""
    active_docs = [r for r in records if is_document_active(r, target_date)]
    req_notice = requires_current_notice(question)

    def doc_sort_key(r: dict[str, Any]) -> tuple[int, int]:
        area_match = 0 if policy_area_matches(r, policy_area) else 1
        dtype = r.get("document_type")
        if req_notice:
            if dtype == "semester_notice":
                type_priority = 0
            elif dtype == "annual_notice":
                type_priority = 1
            elif dtype == "regulation":
                type_priority = 2
            else:
                type_priority = 3
        else:
            if dtype == "regulation":
                type_priority = 0
            elif dtype == "semester_notice":
                type_priority = 1
            elif dtype == "annual_notice":
                type_priority = 2
            else:
                type_priority = 3
        return (area_match, type_priority)

    return sorted(active_docs, key=doc_sort_key)
