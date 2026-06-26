"""
tests/test_policy_answer.py

Unit and integration tests for the rule-based policy QA answer utility.

Run with:
    python -m pytest tests/test_policy_answer.py
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import answer_policy_question as _mod

infer_case_issues   = _mod.infer_case_issues
answer_question     = _mod.answer_question


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, section_title: str = "", **kwargs) -> dict:
    """Build a minimal chunk dict for QA testing."""
    base = {
        "chunk_id": "test_chunk",
        "doc_id": "test_doc",
        "section_number": "1",
        "section_title": section_title,
        "chapter_title": "",
        "text": text,
        "policy_area": [],
        "action_tags": [],
        "requirement_tags": [],
        "procedure_tags": [],
        "risk_tags": [],
        "evidence_groups": [],
        "time_tags": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. Inference rules
# ---------------------------------------------------------------------------

class TestInference:
    def test_graduation_inference(self):
        issues = infer_case_issues("Điều kiện xét tốt nghiệp là gì?")
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "graduation"
        assert issues[0]["policy_area"] == "graduation"

    def test_course_exemption_inference(self):
        issues = infer_case_issues("Miễn môn học cần hồ sơ gì?")
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "course_exemption"
        assert issues[0]["policy_area"] == "course_exemption"
        assert "ho so" in issues[0]["query"]

    def test_foreign_language_inference(self):
        issues = infer_case_issues("IELTS 6.0 được miễn tiếng Anh không?")
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "foreign_language_requirement"
        assert issues[0]["policy_area"] == "foreign_language_requirement"
        assert "ielts" in issues[0]["query"]

    def test_long_case_multiple_issues(self):
        issues = infer_case_issues(
            "Em có IELTS 6.0, từng học trường khác có bảng điểm, muốn xin miễn môn và hỏi điều kiện xét tốt nghiệp."
        )
        # Should detect:
        # - foreign language (IELTS)
        # - course exemption (bảng điểm / miễn môn)
        # - graduation (xét tốt nghiệp)
        assert len(issues) >= 3
        types = {iss["issue_type"] for iss in issues}
        assert "graduation" in types
        assert "course_exemption" in types
        assert "foreign_language_requirement" in types


# ---------------------------------------------------------------------------
# 2. No-answer behavior
# ---------------------------------------------------------------------------

class TestNoAnswer:
    def test_no_matches_returns_no_answer_message(self):
        chunks = [
            _make_chunk("Không liên quan gì", section_title="Điều 1")
        ]
        # Query will result in 0 score for all chunks
        ans = answer_question("Học bổng khuyến khích học tập thế nào?", chunks)
        assert ans == "Chưa tìm thấy quy định phù hợp trong dữ liệu hiện có."


# ---------------------------------------------------------------------------
# 3. Evidence formatting
# ---------------------------------------------------------------------------

class TestEvidenceFormatting:
    def test_formatted_evidence_structure(self):
        chunks = [
            _make_chunk(
                text="Sinh viên hoàn thành các thủ tục xét tốt nghiệp.",
                section_title="Điều 27. Xét tốt nghiệp",
                chunk_id="dieu_27_chunk",
                doc_id="test_doc_abc",
                source_pdf="my_document.pdf",
                policy_area=["graduation"]
            )
        ]
        ans = answer_question("Điều kiện xét tốt nghiệp là gì?", chunks)
        
        # Verify markdown content elements
        assert "# Căn cứ chi tiết" in ans
        assert "dieu_27_chunk" in ans
        assert "test_doc_abc" in ans
        assert "my_document.pdf" in ans
        assert "Điều 27. Xét tốt nghiệp" in ans
        # Should print score
        assert "score" in ans

    def test_show_evidence_text_true_shows_full_text(self):
        long_text = "Dòng 1.\nDòng 2.\nDòng 3.\n" + ("A" * 400)
        chunks = [
            _make_chunk(
                text=long_text,
                section_title="Điều 5. Miễn giảm môn học",
                chunk_id="dieu_5_chunk",
                policy_area=["course_exemption"]
            )
        ]
        
        # Test default show_evidence_text=False (truncates/previews)
        ans_default = answer_question("Miễn môn học cần hồ sơ gì?", chunks, show_evidence_text=False)
        assert "..." in ans_default
        assert "Dòng 1. Dòng 2. Dòng 3." in ans_default

        # Test show_evidence_text=True (full text, preserves newlines)
        ans_full = answer_question("Miễn môn học cần hồ sơ gì?", chunks, show_evidence_text=True)
        assert "..." not in ans_full
        assert long_text in ans_full

    def test_ielts_cautious_wording(self):
        chunks = [
            _make_chunk(
                text="Chứng chỉ tiếng Anh quốc tế IELTS được dùng để xét miễn học phần tiếng Anh.",
                section_title="Điều 9. Chuẩn ngoại ngữ",
                chunk_id="dieu_9_chunk",
                policy_area=["foreign_language_requirement"]
            )
        ]
        ans = answer_question("IELTS 6.0 được miễn tiếng Anh không?", chunks)
        # Should contain cautious phrase
        assert "đối chiếu" in ans.lower() or "chưa nên kết luận chắc chắn" in ans.lower()

    def test_course_exemption_conditions_refers_to_dieu_4(self):
        chunks = [
            _make_chunk(
                text="Xét miễn môn học cho sinh viên theo quy chế đào tạo.",
                section_title="Điều 4. Miễn giảm môn học",
                chunk_id="dieu_4_chunk",
                policy_area=["course_exemption"]
            )
        ]
        ans = answer_question("Điều kiện miễn môn học là gì?", chunks)
        # Assert the answer contains "Điều 4"
        assert "Điều 4" in ans
        # Assert it does not suggest ONLY Điều 5 is the rule for conditions
        # (It shouldn't have the Điều 5 sole statement)
        assert "quy định cụ thể tại Điều 5" not in ans


# ---------------------------------------------------------------------------
# 4. Integration Tests against real data/chunks/policy_chunks.annotated.jsonl
# ---------------------------------------------------------------------------

class TestIntegrationQA:
    _INPUT = Path(__file__).parent.parent / "data" / "chunks" / "policy_chunks.annotated.jsonl"

    def _load_real_chunks(self):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.annotated.jsonl is not built")
        records = []
        import json
        with self._INPUT.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def test_integration_graduation_query(self):
        chunks = self._load_real_chunks()
        ans = answer_question("Điều kiện xét tốt nghiệp là gì?", chunks)
        
        # Should formulate an answer containing "Điều 27" or "xét tốt nghiệp"
        assert "Điều 27" in ans or "xét tốt nghiệp" in ans.lower()
        # Should have citations
        assert "Căn cứ chính" in ans
        assert "# Căn cứ chi tiết" in ans
        
        # Check that Điều 27 is first in the list of Căn cứ chính
        lines = ans.splitlines()
        cc_start = -1
        for idx, line in enumerate(lines):
            if "Căn cứ chính:" in line:
                cc_start = idx
                break
        assert cc_start != -1
        # The first evidence line after "Căn cứ chính:" should contain "Điều 27"
        assert "Điều 27" in lines[cc_start + 1]

    def test_integration_course_exemption_query(self):
        chunks = self._load_real_chunks()
        ans = answer_question("Miễn môn học cần hồ sơ gì?", chunks)
        
        # Should return evidence and mention "hồ sơ" or "miễn"
        assert "hồ sơ" in ans.lower() or "miễn" in ans.lower()
        assert "Căn cứ chính" in ans
        
        # Check that Điều 5 is first in the list of Căn cứ chính
        lines = ans.splitlines()
        cc_start = -1
        for idx, line in enumerate(lines):
            if "Căn cứ chính:" in line:
                cc_start = idx
                break
        assert cc_start != -1
        assert "Điều 5" in lines[cc_start + 1]

    def test_integration_course_exemption_conditions_query(self):
        chunks = self._load_real_chunks()
        ans = answer_question("Điều kiện miễn môn học là gì?", chunks)
        
        assert "miễn" in ans.lower()
        assert "Căn cứ chính" in ans
        
        # Check that Điều 4 is first in the list of Căn cứ chính
        lines = ans.splitlines()
        cc_start = -1
        for idx, line in enumerate(lines):
            if "Căn cứ chính:" in line:
                cc_start = idx
                break
        assert cc_start != -1
        assert "Điều 4" in lines[cc_start + 1]

    def test_integration_ielts_query(self):
        chunks = self._load_real_chunks()
        ans = answer_question("IELTS 6.0 được miễn tiếng Anh không?", chunks)
        
        # Should find relevant language chunks/evidence
        assert "Căn cứ chính" in ans
        assert "# Căn cứ chi tiết" in ans
        assert len(ans) > 100
        # Should include Điều 9 and/or Phụ lục I
        assert "Điều 9" in ans or "Phụ lục I" in ans

    def test_integration_course_exemption_no_noisy_evidence(self):
        chunks = self._load_real_chunks()
        ans = answer_question("Điều kiện miễn môn học là gì?", chunks)
        
        # Should NOT contain noisy/unrelated chunks
        assert "ou_fulltime_credit_training_regulation_2016__dieu_10" not in ans
        assert "ou_non_major_foreign_language_regulation_2023__dieu_9" not in ans

    def test_integration_with_domain_config_parameter(self):
        chunks = self._load_real_chunks()
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        
        import policy_domain_config
        config = policy_domain_config.load_domain_config(config_path)
        
        ans = answer_question("Miễn môn học cần hồ sơ gì?", chunks, domain_config=config)
        assert "Căn cứ chính" in ans
        
        lines = ans.splitlines()
        cc_start = -1
        for idx, line in enumerate(lines):
            if "Căn cứ chính:" in line:
                cc_start = idx
                break
        assert cc_start != -1
        # "Miễn môn học cần hồ sơ gì?" should include Điều 5 first
        assert "Điều 5" in lines[cc_start + 1]

    def test_integration_with_document_registry_parameter(self):
        chunks = self._load_real_chunks()
        registry_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "document_registry.jsonl"
        
        import policy_document_registry
        registry = policy_document_registry.load_document_registry(registry_path)
        
        ans = answer_question("Học kỳ này khi nào nộp hồ sơ miễn môn?", chunks, document_registry=registry)
        assert "Căn cứ chính" in ans
        
        assert ("chưa có thông báo học kỳ hiện tại" in ans.lower() 
                or "chưa thể kết luận thời hạn cụ thể" in ans.lower())
