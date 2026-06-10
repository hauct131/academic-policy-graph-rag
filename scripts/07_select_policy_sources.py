#!/usr/bin/env python3
"""
scripts/07_select_policy_sources.py

Deterministic policy source selector to reduce noisy evidence after retrieval
and before answer generation.
"""

import sys
import unicodedata
from typing import Any


def normalize_text(text: str | None) -> str:
    """Lowercase and remove Vietnamese/Latin diacritics."""
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def section_number_as_int(chunk: dict) -> int | None:
    """Safely convert section_number to int if numeric."""
    s = str(chunk.get("section_number", "")).strip()
    if s.isdigit():
        return int(s)
    return None


def source_selector_priority(issue: dict, chunk: dict) -> int:
    """
    Determine priority (higher is better) for the chunk relative to the issue.
    Returns:
      - 3: Top priority (exact direct regulation match)
      - 2: Medium-high priority (closely related or supporting regulation match)
      - 1: Low-medium priority (general matching policy area)
      - 0: Default / neutral
      - Negative: Unrelated or penalized chunks
    """
    issue_type = issue.get("issue_type", "")
    query = normalize_text(issue.get("query", ""))
    
    # Extract chunk fields
    sec_num = str(chunk.get("section_number", "")).strip()
    sec_title = normalize_text(chunk.get("section_title", ""))
    policy_area = chunk.get("policy_area", [])
    
    if issue_type == "graduation":
        if "graduation" in policy_area:
            if sec_num == "27" or "dieu kien xet tot nghiep" in sec_title or "cong nhan tot nghiep" in sec_title:
                return 3
            if sec_num in {"28", "29"} or "bang tot nghiep" in sec_title or "van bang tot nghiep" in sec_title:
                return 2
            return 1
        return -1

    elif issue_type == "course_exemption":
        if "course_exemption" in policy_area:
            if "ho so" in query:
                if sec_num == "5" and "ho so" in sec_title:
                    return 3
                if sec_num == "4":
                    return 2
                return 1
            else:
                if sec_num == "4" and "dieu kien" in sec_title:
                    return 3
                if sec_num == "5":
                    return 2
                return 1
        return -1

    elif issue_type == "foreign_language_requirement":
        if "foreign_language_requirement" in policy_area:
            cert_kws = ["ielts", "toeic", "toefl", "aptis", "cambridge", "chung chi"]
            is_cert = any(kw in query for kw in cert_kws)
            if is_cert:
                if "phu luc i" in sec_title or sec_num == "9" or "xet mien ngoai ngu" in sec_title:
                    return 3
                return 1
            else:
                if sec_num in {"5", "8", "9"} or any(kw in sec_title for kw in ["chuan dau ra", "ky thi", "xet mien"]):
                    return 2
                return 1
        return -1

    elif issue_type == "course_registration":
        if "course_registration" in policy_area:
            if any(kw in sec_title for kw in ["dang ky", "khoi luong hoc tap", "hoc vuot"]):
                return 2
            return 1
        return -1

    elif issue_type == "retake_and_grade_improvement":
        if "retake_and_grade_improvement" in policy_area:
            if any(kw in sec_title for kw in ["hoc lai", "cai thien diem"]):
                return 2
            return 1
        return -1

    elif issue_type == "academic_standing":
        if "academic_standing" in policy_area:
            if any(kw in sec_title for kw in ["canh bao", "buoc thoi hoc"]):
                return 2
            return 1
        return -1

    return 0


def source_selector_bonus(issue: dict, chunk: dict) -> float:
    """
    Calculate re-ranking/filtering bonus for a chunk based on the issue type.
    We also keep this as specified by public function requirements.
    """
    issue_type = issue.get("issue_type", "")
    query = normalize_text(issue.get("query", ""))
    
    # Extract chunk fields
    sec_num = str(chunk.get("section_number", "")).strip()
    sec_title = normalize_text(chunk.get("section_title", ""))
    policy_area = chunk.get("policy_area", [])
    
    bonus = 0.0
    
    if issue_type == "graduation":
        if "graduation" in policy_area:
            bonus += 10.0
        if sec_num == "27":
            bonus += 50.0
        if "dieu kien xet tot nghiep" in sec_title or "cong nhan tot nghiep" in sec_title:
            bonus += 50.0
        if sec_num in {"28", "29"}:
            bonus += 20.0
        if "bang tot nghiep" in sec_title or "van bang tot nghiep" in sec_title:
            bonus += 20.0
        if "graduation" not in policy_area and sec_num not in {"27", "28", "29"}:
            bonus -= 30.0

    elif issue_type == "course_exemption":
        if "course_exemption" in policy_area:
            bonus += 10.0
        if "ho so" in query:
            if sec_num == "5" and "ho so" in sec_title:
                bonus += 50.0
            if sec_num == "4":
                bonus += 20.0
        else:
            if sec_num == "4" and "dieu kien" in sec_title:
                bonus += 50.0
            if sec_num == "5":
                bonus += 20.0
        if "course_exemption" not in policy_area and sec_num not in {"4", "5"}:
            bonus -= 30.0

    elif issue_type == "foreign_language_requirement":
        if "foreign_language_requirement" in policy_area:
            bonus += 10.0
        cert_kws = ["ielts", "toeic", "toefl", "aptis", "cambridge", "chung chi"]
        is_cert_query = any(ckw in query for ckw in cert_kws)
        if is_cert_query:
            if "phu luc i" in sec_title:
                bonus += 50.0
            if sec_num == "9" or "xet mien ngoai ngu" in sec_title:
                bonus += 50.0
        else:
            has_target_title = any(kw in sec_title for kw in ["chuan dau ra", "ky thi", "xet mien"])
            if sec_num in {"5", "8", "9"} or has_target_title:
                bonus += 30.0
        if "foreign_language_requirement" not in policy_area and sec_num not in {"5", "8", "9"} and "phu luc i" not in sec_title:
            bonus -= 30.0

    elif issue_type == "course_registration":
        if "course_registration" in policy_area:
            bonus += 10.0
        if any(kw in sec_title for kw in ["dang ky", "khoi luong hoc tap", "hoc vuot"]):
            bonus += 30.0
        if "course_registration" not in policy_area:
            bonus -= 30.0

    elif issue_type == "retake_and_grade_improvement":
        if "retake_and_grade_improvement" in policy_area:
            bonus += 10.0
        if any(kw in sec_title for kw in ["hoc lai", "cai thien diem"]):
            bonus += 30.0
        if "retake_and_grade_improvement" not in policy_area:
            bonus -= 30.0

    elif issue_type == "academic_standing":
        if "academic_standing" in policy_area:
            bonus += 10.0
        if any(kw in sec_title for kw in ["canh bao", "buoc thoi hoc"]):
            bonus += 30.0
        if "academic_standing" not in policy_area:
            bonus -= 30.0

    return bonus


def select_sources_for_issue(
    issue: dict,
    results: list[tuple[dict, float]],
    max_sources: int = 3
) -> list[tuple[dict, float]]:
    """
    Deduplicate, score with selector bonus and priority, sort deterministically, and filter/limit.
    Falls back to original top results if no useful candidate remains.
    """
    if not results:
        return []

    # 1. Deduplicate by chunk_id
    seen = set()
    deduped_results = []
    for chunk, score in results:
        c_id = chunk.get("chunk_id")
        if c_id not in seen:
            seen.add(c_id)
            deduped_results.append((chunk, score))

    # 2. Score and prioritize candidates
    scored_candidates = []
    for chunk, score in deduped_results:
        priority = source_selector_priority(issue, chunk)
        bonus = source_selector_bonus(issue, chunk)
        final_score = score + bonus
        scored_candidates.append((chunk, final_score, priority))

    # 3. Filter for "useful" chunks (priority >= 0)
    useful_candidates = [(c, f_s, p) for c, f_s, p in scored_candidates if p >= 0 and f_s > 0]

    def get_fallback_sort_key(item: tuple[dict, float]) -> tuple[float, str, tuple[int, Any], str]:
        c, s = item
        sec_num = str(c.get("section_number", "")).strip()
        is_numeric = 0 if sec_num.isdigit() else 1
        num_val = int(sec_num) if sec_num.isdigit() else sec_num
        return (-s, c.get("doc_id", ""), (is_numeric, num_val), c.get("chunk_id", ""))

    # 4. Fallback if no useful candidate exists
    if not useful_candidates:
        sorted_orig = sorted(deduped_results, key=get_fallback_sort_key)
        return sorted_orig[:max_sources]

    # Sort useful candidates deterministically by priority, then score, etc.
    def get_useful_sort_key(item: tuple[dict, float, int]) -> tuple[int, float, str, tuple[int, Any], str]:
        c, s, p = item
        sec_num = str(c.get("section_number", "")).strip()
        is_numeric = 0 if sec_num.isdigit() else 1
        num_val = int(sec_num) if sec_num.isdigit() else sec_num
        return (-p, -s, c.get("doc_id", ""), (is_numeric, num_val), c.get("chunk_id", ""))

    sorted_useful = sorted(useful_candidates, key=get_useful_sort_key)
    return [(c, f_s) for c, f_s, p in sorted_useful][:max_sources]
