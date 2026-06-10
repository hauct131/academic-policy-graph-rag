"""
tests/test_policy_annotation.py

Unit and integration tests for the rule-based annotation pipeline.

Run with:
    python -m pytest
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test (name starts with a digit, use importlib)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module

_mod = import_module("02_annotate_policy_chunks")

normalize_text = _mod.normalize_text
add_unique = _mod.add_unique
annotate_chunk = _mod.annotate_chunk
annotate_chunks = _mod.annotate_chunks
read_jsonl = _mod.read_jsonl
write_jsonl = _mod.write_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, section_title: str = "") -> dict:
    """Build a minimal chunk dict for annotation testing."""
    return {
        "chunk_id": "test__dieu_0",
        "doc_id": "test_doc",
        "title": "Test",
        "decision_no": "001",
        "issued_date": "2023-01-01",
        "institution": "OU",
        "education_mode": "full_time",
        "chapter_title": "",
        "section_title": section_title,
        "section_number": "",
        "chunk_type": "dieu",
        "source_path": "test.md",
        "source_pdf": "test.pdf",
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def _annotated(text: str, section_title: str = "") -> dict:
    return annotate_chunk(_make_chunk(text, section_title))


# ---------------------------------------------------------------------------
# 1. normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_removes_vietnamese_accents(self):
        assert normalize_text("Điều") == "dieu"

    def test_lowercases_output(self):
        assert normalize_text("HELLO") == "hello"

    def test_d_stroke_handled(self):
        assert normalize_text("đại học") == "dai hoc"

    def test_mixed_vietnamese(self):
        result = normalize_text("Phụ lục I. Danh mục chứng chỉ")
        assert result == result.lower()
        assert all(c.isascii() or c == " " or c == "." for c in result)

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_plain_ascii_unchanged_modulo_case(self):
        assert normalize_text("hello world 123") == "hello world 123"


# ---------------------------------------------------------------------------
# 2. Course registration
# ---------------------------------------------------------------------------

class TestCourseRegistration:
    def test_dang_ky_mon_hoc(self):
        a = _annotated("Sinh viên đăng ký môn học theo hướng dẫn của Phòng QLĐT.")
        assert "course_registration" in a["policy_area"]
        assert "register_course" in a["action_tags"]

    def test_khoi_luong_hoc_tap(self):
        a = _annotated("Khối lượng học tập tối thiểu mỗi học kỳ là 14 tín chỉ.")
        assert "course_registration" in a["policy_area"]

    def test_hoc_vuot_adds_study_ahead(self):
        a = _annotated(
            "Sinh viên muốn đăng ký học vượt phải được sự chấp thuận của cố vấn học tập."
        )
        assert "study_ahead" in a["action_tags"]

    def test_hoc_vuot_with_co_van_hoc_tap(self):
        a = _annotated(
            "Sinh viên muốn đăng ký học vượt phải được sự chấp thuận của cố vấn học tập."
        )
        assert "advisor_approval_required" in a["procedure_tags"]

    def test_late_registration_tag(self):
        a = _annotated("Đăng ký muộn được thực hiện trong 2 tuần đầu của học kỳ.")
        assert "late_registration_period" in a["time_tags"]

    def test_original_text_unchanged(self):
        original = "Sinh viên đăng ký môn học theo hướng dẫn."
        chunk = _make_chunk(original)
        annotated = annotate_chunk(chunk)
        assert annotated["text"] == original


# ---------------------------------------------------------------------------
# 3. Retake / grade improvement
# ---------------------------------------------------------------------------

class TestRetakeAndGradeImprovement:
    def test_hoc_lai(self):
        a = _annotated("Sinh viên phải đăng ký học lại môn học không đạt.")
        assert "retake_and_grade_improvement" in a["policy_area"]
        assert "retake_course" in a["action_tags"]

    def test_cai_thien_diem(self):
        a = _annotated(
            "Sinh viên có thể đăng ký học lại để cải thiện điểm. "
            "Điểm thi cao nhất sẽ được chọn."
        )
        assert "retake_and_grade_improvement" in a["policy_area"]
        assert "improve_grade" in a["action_tags"]
        assert "passed_course_grade_improvement" in a["requirement_tags"]

    def test_failed_required_course_tag(self):
        a = _annotated(
            "Nếu môn học bắt buộc không đạt, sinh viên bắt buộc phải đăng ký học lại."
        )
        assert "failed_required_course" in a["requirement_tags"]


# ---------------------------------------------------------------------------
# 4. Academic warning / dismissal
# ---------------------------------------------------------------------------

class TestAcademicStanding:
    def test_canh_bao_ket_qua(self):
        a = _annotated("Cảnh báo kết quả học tập được thực hiện theo từng học kỳ.")
        assert "academic_standing" in a["policy_area"]
        assert "academic_warning" in a["action_tags"]
        assert "academic_warning" in a["student_status_tags"]

    def test_buoc_thoi_hoc(self):
        a = _annotated(
            "Sinh viên bị buộc thôi học nếu đã hết thời gian đào tạo tối đa."
        )
        assert "academic_standing" in a["policy_area"]
        assert "dismissal" in a["action_tags"]
        assert "forced_dropout" in a["risk_tags"]

    def test_buoc_nghi_hoc_tam_thoi(self):
        a = _annotated("Sinh viên bị buộc nghỉ học tạm thời do vi phạm kỷ luật.")
        assert "temporary_suspension" in a["action_tags"]
        assert "forced_temporary_leave" in a["risk_tags"]

    def test_tu_y_bo_hoc(self):
        a = _annotated("Trường hợp tự ý bỏ học không lý do một học kỳ.")
        assert "unauthorized_absence" in a["risk_tags"]

    def test_khong_hoan_thanh_hoc_phi(self):
        a = _annotated("Không hoàn thành nghĩa vụ học phí theo quy định.")
        assert "tuition_debt" in a["risk_tags"]

    def test_hoc_luc_kem(self):
        a = _annotated("Sinh viên xếp hạng học lực kém chỉ được đăng ký tối đa 14 tín chỉ.")
        assert "poor_academic_performance" in a["student_status_tags"]


# ---------------------------------------------------------------------------
# 5. Graduation
# ---------------------------------------------------------------------------

class TestGraduation:
    def test_xet_tot_nghiep(self):
        a = _annotated("Hội đồng xét tốt nghiệp xem xét các điều kiện tốt nghiệp.")
        assert "graduation" in a["policy_area"]
        assert "graduation_audit" in a["action_tags"]

    def test_bang_tot_nghiep(self):
        a = _annotated("Bằng tốt nghiệp được cấp sau khi công nhận tốt nghiệp.")
        assert "graduation" in a["policy_area"]
        assert "degree_conferral" in a["action_tags"]

    def test_quoc_phong_requirement(self):
        a = _annotated(
            "Sinh viên phải có chứng chỉ giáo dục quốc phòng để được xét tốt nghiệp."
        )
        assert "national_defense_certificate_required" in a["requirement_tags"]

    def test_giao_duc_the_chat(self):
        a = _annotated(
            "Hoàn thành môn học giáo dục thể chất là điều kiện tốt nghiệp."
        )
        assert "physical_education_completion_required" in a["requirement_tags"]

    def test_graduation_application_procedure(self):
        a = _annotated(
            "Sinh viên nộp đơn đề nghị xét tốt nghiệp tại Phòng QLĐT."
        )
        assert "graduation_application_required" in a["procedure_tags"]


# ---------------------------------------------------------------------------
# 6. Course exemption / credit transfer
# ---------------------------------------------------------------------------

class TestCourseExemption:
    def test_mien_giam_mon_hoc(self):
        a = _annotated("Quy định về miễn môn học và giảm môn học cho sinh viên.")
        assert "course_exemption" in a["policy_area"]
        assert "request_course_exemption" in a["action_tags"]
        assert "request_course_reduction" in a["action_tags"]

    def test_xet_mien_giam(self):
        a = _annotated("Phòng QLĐT tiến hành xét miễn giảm môn học mỗi học kỳ.")
        assert "course_exemption" in a["policy_area"]

    def test_credit_transfer(self):
        a = _annotated("Sinh viên xin chuyển điểm các môn học đã học ngoài trường.")
        assert "credit_transfer" in a["action_tags"]

    def test_50_percent_limit(self):
        a = _annotated(
            "Tổng số tín chỉ được xét miễn không vượt quá 50% tổng số tín chỉ tối thiểu."
        )
        assert "credit_limit_50_percent" in a["requirement_tags"]

    def test_bang_diem_evidence(self):
        a = _annotated("Sinh viên nộp bảng điểm xin miễn môn học tại Phòng QLĐT.")
        assert "transcript" in a["evidence_groups"]
        assert "transcript_required" in a["procedure_tags"]

    def test_de_cuong_evidence(self):
        a = _annotated(
            "Sinh viên cung cấp đề cương môn học nếu có yêu cầu khi xét miễn."
        )
        assert "course_syllabus" in a["evidence_groups"]

    def test_certificate_evidence(self):
        a = _annotated("Sinh viên nộp chứng chỉ hợp lệ để xét miễn môn học.")
        assert "certificate" in a["evidence_groups"]


# ---------------------------------------------------------------------------
# 7. Foreign language / English
# ---------------------------------------------------------------------------

class TestForeignLanguageRequirement:
    def test_ngoai_ngu_khong_chuyen(self):
        a = _annotated("Quy định đào tạo ngoại ngữ không chuyên thuộc chương trình đào tạo.")
        assert "foreign_language_requirement" in a["policy_area"]

    def test_ielts_in_text(self):
        a = _annotated("Sinh viên có IELTS từ 6.0 trở lên được xét miễn tiếng Anh nâng cao.")
        assert "foreign_language_requirement" in a["policy_area"]
        assert "certificate_requirement" in a["requirement_tags"]
        assert "language_certificate" in a["evidence_groups"]

    def test_toeic_in_text(self):
        a = _annotated("Chứng chỉ TOEIC từ 650 điểm trở lên.")
        assert "foreign_language_requirement" in a["policy_area"]
        assert "language_certificate" in a["evidence_groups"]

    def test_tieng_anh_placement(self):
        a = _annotated(
            "Trường tổ chức kiểm tra trình độ tiếng Anh đầu vào để xếp lớp."
        )
        assert "english_placement_test" in a["action_tags"]
        assert "english_entry_requirement" in a["requirement_tags"]

    def test_exit_exam(self):
        a = _annotated(
            "Sinh viên dự kỳ thi đánh giá năng lực sử dụng tiếng Anh đầu ra."
        )
        assert "english_exit_exam" in a["action_tags"]
        assert "english_exit_requirement" in a["requirement_tags"]

    def test_english_exemption_action(self):
        a = _annotated(
            "Trường xét miễn tiếng Anh căn cứ vào lộ trình học của sinh viên."
        )
        assert "request_english_exemption" in a["action_tags"]


# ---------------------------------------------------------------------------
# 8. Second foreign language
# ---------------------------------------------------------------------------

class TestSecondForeignLanguage:
    def test_ngoai_ngu_hai(self):
        a = _annotated("Ngoại ngữ hai dành cho sinh viên ngành ngôn ngữ.")
        assert "second_foreign_language_requirement" in a["policy_area"]
        assert "request_second_language_exemption" in a["action_tags"]

    def test_jlpt_certificate(self):
        a = _annotated("Chứng chỉ năng lực tiếng Nhật JLPT cấp độ N4.")
        assert "second_foreign_language_requirement" in a["policy_area"]
        assert "language_certificate" in a["evidence_groups"]
        assert "second_language_certificate_requirement" in a["requirement_tags"]

    def test_topik_certificate(self):
        a = _annotated("Chứng chỉ TOPIK II cấp độ 3 cho tiếng Hàn.")
        assert "second_foreign_language_requirement" in a["policy_area"]

    def test_hsk_certificate(self):
        a = _annotated("HSK cấp độ 3 cho tiếng Trung.")
        assert "second_foreign_language_requirement" in a["policy_area"]

    def test_goethe_certificate(self):
        a = _annotated("Goethe-Zertifikat B1 cho tiếng Đức.")
        assert "second_foreign_language_requirement" in a["policy_area"]

    def test_delf_certificate(self):
        a = _annotated("DELF trình độ B1 cho tiếng Pháp.")
        assert "second_foreign_language_requirement" in a["policy_area"]


# ---------------------------------------------------------------------------
# 9. Assessment and grading
# ---------------------------------------------------------------------------

class TestAssessmentAndGrading:
    def test_thi_cuoi_ky(self):
        a = _annotated("Nhà trường tổ chức kỳ thi cuối kỳ để đánh giá sinh viên.")
        assert "assessment_and_grading" in a["policy_area"]
        assert "take_final_exam" in a["action_tags"]

    def test_phuc_tra_khieu_nai_diem(self):
        a = _annotated(
            "Sinh viên có quyền phúc tra và khiếu nại điểm theo quy định."
        )
        assert "appeal_grade" in a["action_tags"]
        assert "appeal_application_required" in a["procedure_tags"]

    def test_diem_trung_binh_chung(self):
        a = _annotated("Điểm trung bình chung tích lũy được tính theo thang điểm 4.")
        assert "calculate_gpa" in a["action_tags"]

    def test_vang_thi_zero_score(self):
        a = _annotated(
            "Sinh viên vắng thi không có lý do phải nhận điểm 0 môn học."
        )
        assert "zero_score" in a["risk_tags"]
        assert "exam_absence" in a["risk_tags"]


# ---------------------------------------------------------------------------
# 10. Integration: run on real data/chunks/policy_chunks.jsonl
# ---------------------------------------------------------------------------

class TestIntegrationAnnotation:
    _INPUT = Path(__file__).parent.parent / "data" / "chunks" / "policy_chunks.jsonl"

    def test_same_row_count(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.jsonl not available")

        output = tmp_path / "annotated.jsonl"
        chunks = read_jsonl(self._INPUT)
        annotated = annotate_chunks(chunks)
        write_jsonl(output, annotated)

        in_lines = [l for l in self._INPUT.read_text(encoding="utf-8").splitlines() if l.strip()]
        out_lines = [l for l in output.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(in_lines) == len(out_lines), (
            f"Row count mismatch: {len(in_lines)} in vs {len(out_lines)} out"
        )

    def test_annotation_fields_present(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        annotated = annotate_chunks(chunks)

        required_fields = {
            "policy_area", "action_tags", "student_status_tags",
            "procedure_tags", "evidence_groups", "risk_tags",
            "requirement_tags", "time_tags",
        }
        for i, chunk in enumerate(annotated):
            missing = required_fields - set(chunk.keys())
            assert not missing, f"Chunk {i} missing fields: {missing}"

    def test_annotation_fields_are_lists(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        annotated = annotate_chunks(chunks)

        list_fields = [
            "policy_area", "action_tags", "student_status_tags",
            "procedure_tags", "evidence_groups", "risk_tags",
            "requirement_tags", "time_tags",
        ]
        for chunk in annotated:
            for field in list_fields:
                assert isinstance(chunk[field], list), (
                    f"Field '{field}' should be a list in chunk {chunk.get('chunk_id')}"
                )

    def test_original_text_preserved(self, tmp_path):
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        annotated = annotate_chunks(chunks)

        for orig, ann in zip(chunks, annotated):
            assert orig["text"] == ann["text"], (
                f"Text modified for chunk {orig.get('chunk_id')}"
            )

    def test_at_least_one_area_tagged(self, tmp_path):
        """At least one chunk per doc should have a non-empty policy_area."""
        if not self._INPUT.exists():
            pytest.skip("policy_chunks.jsonl not available")

        chunks = read_jsonl(self._INPUT)
        annotated = annotate_chunks(chunks)
        tagged = [c for c in annotated if c["policy_area"]]
        assert len(tagged) > 0, "No chunks received any policy_area tag"
