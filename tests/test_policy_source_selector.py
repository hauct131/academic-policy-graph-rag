"""
tests/test_policy_source_selector.py

Unit tests for the policy source selector.

Run with:
    python -m pytest tests/test_policy_source_selector.py
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module

_selector = import_module("07_select_policy_sources")
select_sources_for_issue = _selector.select_sources_for_issue
prune_selected_sources_for_issue = _selector.prune_selected_sources_for_issue


def _make_chunk(text: str, section_number: str = "1", section_title: str = "", **kwargs) -> dict:
    """Build a minimal chunk dict for testing selection."""
    base = {
        "chunk_id": "test_chunk",
        "doc_id": "test_doc",
        "section_number": section_number,
        "section_title": section_title,
        "text": text,
        "policy_area": [],
    }
    base.update(kwargs)
    return base


class TestPolicySourceSelector:
    def test_course_exemption_hoso_selects_dieu_5_first(self):
        # Hồ sơ query should prioritize Điều 5 (hồ sơ) over Điều 4 (điều kiện)
        issue = {
            "issue_type": "course_exemption",
            "query": "ho so xin mien giam mon hoc"
        }
        
        chunk_4 = _make_chunk("Điều kiện được xét miễn giảm môn học", section_number="4", section_title="Điều 4. Điều kiện được xét miễn, giảm môn học", policy_area=["course_exemption"], chunk_id="chunk_4")
        chunk_5 = _make_chunk("Hồ sơ xin miễn giảm môn học gồm đơn và bảng điểm", section_number="5", section_title="Điều 5. Hồ sơ xin miễn, giảm môn học", policy_area=["course_exemption"], chunk_id="chunk_5")
        
        # Original retrieval order has chunk_4 first
        results = [
            (chunk_4, 10.0),
            (chunk_5, 8.0)
        ]
        
        selected = select_sources_for_issue(issue, results, max_sources=3)
        assert len(selected) == 2
        # Chunk 5 must rank first
        assert selected[0][0]["chunk_id"] == "chunk_5"
        assert selected[1][0]["chunk_id"] == "chunk_4"

    def test_course_exemption_conditions_selects_dieu_4_first(self):
        # Conditions query should prioritize Điều 4 (điều kiện) over Điều 5 (hồ sơ)
        issue = {
            "issue_type": "course_exemption",
            "query": "dieu kien mien mon hoc"
        }
        
        chunk_4 = _make_chunk("Điều kiện được xét miễn giảm môn học", section_number="4", section_title="Điều 4. Điều kiện được xét miễn, giảm môn học", policy_area=["course_exemption"], chunk_id="chunk_4")
        chunk_5 = _make_chunk("Hồ sơ xin miễn giảm môn học gồm đơn và bảng điểm", section_number="5", section_title="Điều 5. Hồ sơ xin miễn, giảm môn học", policy_area=["course_exemption"], chunk_id="chunk_5")
        
        # Original retrieval order has chunk_5 first
        results = [
            (chunk_5, 10.0),
            (chunk_4, 8.0)
        ]
        
        selected = select_sources_for_issue(issue, results, max_sources=3)
        assert len(selected) == 2
        # Chunk 4 must rank first
        assert selected[0][0]["chunk_id"] == "chunk_4"
        assert selected[1][0]["chunk_id"] == "chunk_5"

    def test_graduation_selects_dieu_27_first(self):
        # Graduation query should prioritize Điều 27 over other graduation-related sections
        issue = {
            "issue_type": "graduation",
            "query": "dieu kien xet tot nghiep"
        }
        
        chunk_27 = _make_chunk("Sinh viên đủ điều kiện xét tốt nghiệp", section_number="27", section_title="Điều 27. Điều kiện xét tốt nghiệp và công nhận tốt nghiệp", policy_area=["graduation"], chunk_id="chunk_27")
        chunk_28 = _make_chunk("Cấp bằng tốt nghiệp", section_number="28", section_title="Điều 28. Cấp bằng tốt nghiệp", policy_area=["graduation"], chunk_id="chunk_28")
        chunk_29 = _make_chunk("Thủ tục quản lý cấp văn bằng", section_number="29", section_title="Điều 29. Thủ tục quản lý và cấp văn bằng tốt nghiệp", policy_area=["graduation"], chunk_id="chunk_29")
        
        # Original retrieval puts 28 or 29 first
        results = [
            (chunk_29, 10.0),
            (chunk_28, 9.0),
            (chunk_27, 5.0)
        ]
        
        selected = select_sources_for_issue(issue, results, max_sources=3)
        assert len(selected) == 3
        # Chunk 27 must rank first
        assert selected[0][0]["chunk_id"] == "chunk_27"

    def test_foreign_language_ielts_selects_phu_luc_i_and_dieu_9_first(self):
        # IELTS query should prioritize Phụ lục I and/or Điều 9 over general English chunks
        issue = {
            "issue_type": "foreign_language_requirement",
            "query": "xet mien tieng anh ielts"
        }
        
        chunk_gen = _make_chunk("Cấp độ tiếng Anh sinh viên học", section_number="5", section_title="Điều 5. Tiếng Anh và chuẩn đầu ra", policy_area=["foreign_language_requirement"], chunk_id="chunk_gen")
        chunk_9 = _make_chunk("Xét miễn ngoại ngữ không chuyên", section_number="9", section_title="Điều 9. Xét miễn ngoại ngữ", policy_area=["foreign_language_requirement"], chunk_id="chunk_9")
        chunk_app = _make_chunk("Danh mục các chứng chỉ tiếng Anh được xét miễn", section_number="Phụ lục I", section_title="Phụ lục I. Danh mục các chứng chỉ", policy_area=["foreign_language_requirement"], chunk_id="chunk_app")
        
        results = [
            (chunk_gen, 15.0),
            (chunk_9, 10.0),
            (chunk_app, 5.0)
        ]
        
        selected = select_sources_for_issue(issue, results, max_sources=3)
        # Should prioritize chunk_app and chunk_9 over chunk_gen
        assert selected[0][0]["chunk_id"] in ("chunk_app", "chunk_9")
        assert selected[1][0]["chunk_id"] in ("chunk_app", "chunk_9")
        assert selected[2][0]["chunk_id"] == "chunk_gen"

    def test_deduplicates_repeated_chunk_id(self):
        issue = {
            "issue_type": "graduation",
            "query": "dieu kien xet tot nghiep"
        }
        chunk = _make_chunk("Sinh viên đủ điều kiện xét tốt nghiệp", section_number="27", section_title="Điều 27. Xét tốt nghiệp", policy_area=["graduation"], chunk_id="chunk_27")
        results = [
            (chunk, 10.0),
            (chunk, 8.0)
        ]
        selected = select_sources_for_issue(issue, results, max_sources=3)
        assert len(selected) == 1

    def test_respects_max_sources(self):
        issue = {
            "issue_type": "graduation",
            "query": "dieu kien xet tot nghiep"
        }
        chunks = [
            (_make_chunk(f"text {i}", section_number="27", chunk_id=f"c_{i}", policy_area=["graduation"]), 10.0 - i)
            for i in range(5)
        ]
        selected = select_sources_for_issue(issue, chunks, max_sources=2)
        assert len(selected) == 2

    def test_fallback_returns_original_top_results(self):
        # If no rule-specific candidate exists (e.g. all get penalized, but still need to return fallback)
        issue = {
            "issue_type": "graduation",
            "query": "dieu kien xet tot nghiep"
        }
        # Chunks not related to graduation, hence getting penalized
        chunk_unrelated_1 = _make_chunk("Học cùng lúc hai chương trình", section_number="15", policy_area=["course_registration"], chunk_id="chunk_15")
        chunk_unrelated_2 = _make_chunk("Nghỉ học tạm thời", section_number="12", policy_area=["leave_and_withdrawal"], chunk_id="chunk_12")
        
        results = [
            (chunk_unrelated_1, 10.0),
            (chunk_unrelated_2, 8.0)
        ]
        # In our source selector, non-graduation chunks get -30 bonus.
        # Both end up with final_score <= 0 or bonus < 0, so no "useful" chunks remain.
        # Should fallback to original top results (sorted)
        selected = select_sources_for_issue(issue, results, max_sources=2)
        assert len(selected) == 2
        # Deterministic order sorting by original score desc, then doc_id, etc.
        assert selected[0][0]["chunk_id"] == "chunk_15"
        assert selected[1][0]["chunk_id"] == "chunk_12"

    def test_pruning_course_exemption_conditions(self):
        # Question: "Điều kiện miễn môn học là gì?" -> issue_type "course_exemption", query does NOT contain "ho so"
        # Keeps Điều 4 (priority 3), and should remove unrelated Điều 10 / Điều 9 (penalized/not matched)
        issue = {
            "issue_type": "course_exemption",
            "query": "dieu kien mien mon hoc"
        }
        chunk_4 = _make_chunk("Điều kiện xét miễn môn học", section_number="4", section_title="Điều 4. Điều kiện xét miễn", policy_area=["course_exemption"], chunk_id="chunk_4")
        chunk_10 = _make_chunk("Điều chỉnh khối lượng", section_number="10", section_title="Điều 10. Điều chỉnh", policy_area=["course_registration"], chunk_id="chunk_10")
        chunk_9 = _make_chunk("Xét miễn ngoại ngữ", section_number="9", section_title="Điều 9. Xét miễn ngoại ngữ", policy_area=["foreign_language_requirement"], chunk_id="chunk_9")

        selected = [
            (chunk_4, 15.0),
            (chunk_10, 8.0),
            (chunk_9, 7.0)
        ]

        pruned = prune_selected_sources_for_issue(issue, selected, max_sources=3)
        assert len(pruned) == 1
        assert pruned[0][0]["chunk_id"] == "chunk_4"

    def test_pruning_course_exemption_hoso(self):
        # Question: "Miễn môn học cần hồ sơ gì?" -> issue_type "course_exemption", query contains "ho so"
        # Keeps Điều 5 (priority 3) and Điều 4 (priority 2), removes unrelated chunks (priority -1)
        issue = {
            "issue_type": "course_exemption",
            "query": "ho so mien mon hoc"
        }
        chunk_4 = _make_chunk("Điều kiện xét miễn môn học", section_number="4", section_title="Điều 4. Điều kiện xét miễn", policy_area=["course_exemption"], chunk_id="chunk_4")
        chunk_5 = _make_chunk("Hồ sơ xét miễn môn học", section_number="5", section_title="Điều 5. Hồ sơ xét miễn", policy_area=["course_exemption"], chunk_id="chunk_5")
        chunk_10 = _make_chunk("Điều chỉnh khối lượng", section_number="10", section_title="Điều 10. Điều chỉnh", policy_area=["course_registration"], chunk_id="chunk_10")

        selected = [
            (chunk_5, 18.0),
            (chunk_4, 12.0),
            (chunk_10, 8.0)
        ]

        pruned = prune_selected_sources_for_issue(issue, selected, max_sources=3)
        assert len(pruned) == 2
        assert pruned[0][0]["chunk_id"] == "chunk_5"
        assert pruned[1][0]["chunk_id"] == "chunk_4"

    def test_pruning_ielts_query(self):
        # Question: "IELTS 6.0..." -> cert query. Keeps Điều 9 and Phụ lục I (priority 3), removes generic/unrelated chunks (priority < 3)
        issue = {
            "issue_type": "foreign_language_requirement",
            "query": "ielts 6.0 duoc mien tieng anh khong"
        }
        chunk_9 = _make_chunk("Xét miễn ngoại ngữ", section_number="9", section_title="Điều 9. Xét miễn ngoại ngữ", policy_area=["foreign_language_requirement"], chunk_id="chunk_9")
        chunk_app = _make_chunk("Danh mục chứng chỉ", section_number="Phụ lục I", section_title="Phụ lục I. Chứng chỉ tiếng Anh", policy_area=["foreign_language_requirement"], chunk_id="chunk_app")
        chunk_gen = _make_chunk("Tiếng Anh và chuẩn đầu ra", section_number="5", section_title="Điều 5. Chuẩn đầu ra", policy_area=["foreign_language_requirement"], chunk_id="chunk_gen")

        selected = [
            (chunk_9, 15.0),
            (chunk_app, 14.0),
            (chunk_gen, 10.0)
        ]

        pruned = prune_selected_sources_for_issue(issue, selected, max_sources=3)
        assert len(pruned) == 2
        assert {p[0]["chunk_id"] for p in pruned} == {"chunk_9", "chunk_app"}

    def test_pruning_graduation(self):
        # Question: "Điều kiện xét tốt nghiệp..." -> keeps Điều 27 (priority 3) and optionally 28/29 (priority 2), removes unrelated chunks (priority < 2)
        issue = {
            "issue_type": "graduation",
            "query": "dieu kien xet tot nghiep"
        }
        chunk_27 = _make_chunk("Điều kiện xét tốt nghiệp", section_number="27", policy_area=["graduation"], chunk_id="chunk_27")
        chunk_28 = _make_chunk("Cấp bằng tốt nghiệp", section_number="28", policy_area=["graduation"], chunk_id="chunk_28")
        chunk_unrelated = _make_chunk("Cảnh báo học tập", section_number="13", policy_area=["academic_standing"], chunk_id="chunk_13")

        selected = [
            (chunk_27, 20.0),
            (chunk_28, 15.0),
            (chunk_unrelated, 8.0)
        ]

        pruned = prune_selected_sources_for_issue(issue, selected, max_sources=3)
        assert len(pruned) == 2
        assert pruned[0][0]["chunk_id"] == "chunk_27"
        assert pruned[1][0]["chunk_id"] == "chunk_28"

    def test_pruning_fallback_keeps_first_selected_source(self):
        # If all sources are pruned, fallback keeps the first selected source
        issue = {
            "issue_type": "graduation",
            "query": "dieu kien xet tot nghiep"
        }
        chunk_unrelated = _make_chunk("Cảnh báo học tập", section_number="13", policy_area=["academic_standing"], chunk_id="chunk_13")
        selected = [
            (chunk_unrelated, 12.0)
        ]

        pruned = prune_selected_sources_for_issue(issue, selected, max_sources=3)
        assert len(pruned) == 1
        assert pruned[0][0]["chunk_id"] == "chunk_13"
