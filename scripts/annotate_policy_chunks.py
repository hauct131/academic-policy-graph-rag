#!/usr/bin/env python3
"""
scripts/02_annotate_policy_chunks.py

Rule-based metadata annotation for Academic Policy Graph RAG chunks.

Reads policy_chunks.jsonl, applies deterministic keyword-matching rules to
each chunk, and writes policy_chunks.annotated.jsonl with 8 new annotation
fields added to every chunk object.

Usage:
    python scripts/02_annotate_policy_chunks.py [--input-file ...] [--output-file ...]
"""

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import normalize_text, read_jsonl



# ---------------------------------------------------------------------------
# Text normalisation (accent removal + lowercase, Vietnamese-safe)
# ---------------------------------------------------------------------------

# normalize_text is imported from core



# ---------------------------------------------------------------------------
# Helper: add a tag only once (list stays ordered, no duplicates)
# ---------------------------------------------------------------------------

def add_unique(lst: list[str], *items: str) -> None:
    """Append each item to lst if not already present."""
    for item in items:
        if item not in lst:
            lst.append(item)


# ---------------------------------------------------------------------------
# Per-chunk annotation logic
# ---------------------------------------------------------------------------

def annotate_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Apply all rule groups to a single chunk.

    Returns a new dict with the 8 annotation fields added.
    The original fields are preserved unchanged.
    """
    # Build normalised search corpus from section_title + text
    section_title_norm = normalize_text(chunk.get("section_title", ""))
    text_norm = normalize_text(chunk.get("text", ""))
    corpus = section_title_norm + " " + text_norm

    # Annotation containers
    policy_area: list[str] = []
    action_tags: list[str] = []
    student_status_tags: list[str] = []
    procedure_tags: list[str] = []
    evidence_groups: list[str] = []
    risk_tags: list[str] = []
    requirement_tags: list[str] = []
    time_tags: list[str] = []

    def has(*phrases: str) -> bool:
        """Return True if any of the normalised phrases appears in corpus."""
        return any(normalize_text(p) in corpus for p in phrases)

    # -----------------------------------------------------------------------
    # A. Course registration
    #
    # NOTE: "khối lượng học tập" không còn là trigger độc lập nữa (trước ở
    # đây) vì cụm này quá chung chung — xuất hiện cả trong định nghĩa "tín
    # chỉ" ở Điều 2 (Giải thích từ ngữ), không liên quan tới thủ tục đăng ký.
    # Vẫn giữ tín hiệu này nhưng chỉ tính khi đi kèm cụm chỉ rõ giới hạn
    # khối lượng theo học kỳ/năm học ("mỗi học kỳ", "mỗi năm học") — đúng
    # ngữ cảnh quy định mức đăng ký tối thiểu/tối đa, không phải câu định
    # nghĩa chung.
    # -----------------------------------------------------------------------
    if has(
        "đăng ký môn học",
        "đăng ký học phần",
        "đăng ký muộn",
        "điều chỉnh khối lượng",
        "học vượt",
    ) or (has("khối lượng học tập") and has("mỗi học kỳ", "mỗi năm học")):
        add_unique(policy_area, "course_registration")
        add_unique(action_tags, "register_course")

        if has("học vượt"):
            add_unique(action_tags, "study_ahead")

        if has("cố vấn học tập"):
            add_unique(procedure_tags, "advisor_approval_required")

        if has("đăng ký muộn"):
            add_unique(time_tags, "late_registration_period")

        if has("thời hạn", "ngoài thời hạn"):
            add_unique(time_tags, "registration_period")
            add_unique(risk_tags, "missed_registration_deadline")

    # -----------------------------------------------------------------------
    # B. Retake / grade improvement
    # -----------------------------------------------------------------------
    if has("học lại", "cải thiện điểm", "điểm thi cao nhất"):
        add_unique(policy_area, "retake_and_grade_improvement")
        add_unique(action_tags, "retake_course")

        if has("cải thiện điểm", "điểm thi cao nhất"):
            add_unique(action_tags, "improve_grade")
            add_unique(requirement_tags, "passed_course_grade_improvement")

        if has("học lại") and has("bắt buộc", "không đạt"):
            add_unique(requirement_tags, "failed_required_course")

    # -----------------------------------------------------------------------
    # C. Academic warning / dismissal
    # -----------------------------------------------------------------------
    if has(
        "cảnh báo kết quả học tập",
        "học lực kém",
        "buộc thôi học",
        "buộc nghỉ học tạm thời",
        "tự ý bỏ học",
        "không hoàn thành nghĩa vụ học phí",
    ):
        add_unique(policy_area, "academic_standing")
        add_unique(student_status_tags, "academic_warning", "poor_academic_performance")

        if has("cảnh báo kết quả học tập", "học lực kém"):
            add_unique(action_tags, "academic_warning")

        if has("buộc nghỉ học tạm thời"):
            add_unique(action_tags, "temporary_suspension")
            add_unique(risk_tags, "forced_temporary_leave")

        if has("buộc thôi học"):
            add_unique(action_tags, "dismissal")
            add_unique(risk_tags, "forced_dropout")

        if has("tự ý bỏ học"):
            add_unique(risk_tags, "unauthorized_absence")

        if has("không hoàn thành nghĩa vụ học phí"):
            add_unique(risk_tags, "tuition_debt")

    # -----------------------------------------------------------------------
    # D. Temporary leave / withdrawal
    # -----------------------------------------------------------------------
    if has(
        "nghỉ học tạm thời",
        "bảo lưu kết quả",
        "xin thôi học",
        "tạm dừng học tập",
    ):
        add_unique(policy_area, "leave_and_withdrawal")

        if has("nghỉ học tạm thời", "tạm dừng học tập"):
            add_unique(action_tags, "temporary_leave")

        if has("xin thôi học"):
            add_unique(action_tags, "withdrawal")

        if has("bảo lưu kết quả"):
            add_unique(action_tags, "reserve_study_results")

        add_unique(procedure_tags, "application_required", "decision_required")

        if has("giấy xác nhận", "cơ quan y tế", "bệnh viện"):
            add_unique(evidence_groups, "medical_certificate")

        if has("lực lượng vũ trang", "quân sự", "điều động"):
            add_unique(evidence_groups, "military_order")

    # -----------------------------------------------------------------------
    # E. Graduation
    # Trigger ONLY on genuine graduation context — not on bare GDTC/GDQP mentions
    # that appear in course-exemption or other policy chunks.
    # Additionally, Chuơng VIII (Sections 27-29) of the fulltime training regulation
    # are always graduation sections by definition.
    # -----------------------------------------------------------------------
    section_number = str(chunk.get("section_number", "")).strip()
    doc_id_val     = str(chunk.get("doc_id", "")).strip()

    is_fulltime_graduation_section = (
        doc_id_val == "ou_fulltime_credit_training_regulation_2016"
        and section_number in {"27", "28", "29"}
    )

    _is_graduation = has(
        "xét tốt nghiệp",
        "công nhận tốt nghiệp",
        "điều kiện tốt nghiệp",
        "hội đồng xét tốt nghiệp",
    ) or is_fulltime_graduation_section
    if _is_graduation:
        add_unique(policy_area, "graduation")
        add_unique(action_tags, "graduation_audit")

        if has("bằng tốt nghiệp", "công nhận tốt nghiệp"):
            add_unique(action_tags, "degree_conferral")

        if has("điểm trung bình chung tích lũy", "2,00", "2.00"):
            add_unique(requirement_tags, "minimum_gpa_required")

        if has("tích lũy đủ số", "khối lượng", "số tín chỉ"):
            add_unique(requirement_tags, "credit_completion_required")

        # Requirement tags for GDQP/GDTC only when already in graduation context
        if has("chứng chỉ giáo dục quốc phòng", "quốc phòng"):
            add_unique(requirement_tags, "national_defense_certificate_required")

        if has("giáo dục thể chất"):
            add_unique(requirement_tags, "physical_education_completion_required")

        if has("đơn", "đề nghị"):
            add_unique(procedure_tags, "graduation_application_required")

    # -----------------------------------------------------------------------
    # F. Course exemption / credit transfer
    #
    # NOTE: "bảo lưu kết quả" đã bị loại khỏi trigger list (trước ở đây) vì
    # cụm này đã thuộc rule D (leave_and_withdrawal) — "bảo lưu kết quả học
    # tập" (giữ kết quả khi tạm nghỉ học) là khái niệm khác hoàn toàn với
    # "miễn/giảm môn học" (course_exemption). Việc dùng chung trigger phrase
    # khiến các Điều nói về bảo lưu kết quả (vd. Điều 28, 31) bị gán nhầm
    # thêm course_exemption dù không liên quan.
    # -----------------------------------------------------------------------
    if has(
        "miễn môn học",
        "giảm môn học",
        "xét miễn",
        "xét miễn giảm",
        "chuyển điểm",
        "tín chỉ được xét miễn",
    ):
        add_unique(policy_area, "course_exemption")

        if has("miễn môn học", "xét miễn"):
            add_unique(action_tags, "request_course_exemption")

        if has("giảm môn học", "xét miễn giảm"):
            add_unique(action_tags, "request_course_reduction")

        if has("chuyển điểm"):
            add_unique(action_tags, "credit_transfer")

        if has("5,0", "5.0", "điểm đạt"):
            add_unique(requirement_tags, "minimum_grade_required")

        if has("số tín chỉ", "tín chỉ tương đương", "tín chỉ lớn hơn hoặc bằng"):
            add_unique(requirement_tags, "credit_equivalence_required")

        if has("50%", "không vượt quá 50"):
            add_unique(requirement_tags, "credit_limit_50_percent")

        add_unique(procedure_tags, "application_required")

        if has("bảng điểm"):
            add_unique(procedure_tags, "transcript_required")
            add_unique(evidence_groups, "transcript")

        if has("đề cương môn học"):
            add_unique(procedure_tags, "syllabus_required")
            add_unique(evidence_groups, "course_syllabus")

        if has("chứng chỉ"):
            add_unique(evidence_groups, "certificate")

    # -----------------------------------------------------------------------
    # G. Foreign language / English (non-major)
    #
    # NOTE: "chuẩn đầu vào"/"chuẩn đầu ra" đã bị loại khỏi trigger list ở
    # đây (trước dùng để gán policy_area) vì đây là thuật ngữ giáo dục đại
    # học CHUNG, áp dụng cho MỌI chương trình đào tạo (không riêng ngoại
    # ngữ) — Điều 2 (Giải thích từ ngữ) định nghĩa "Chuẩn đầu ra" như một
    # khái niệm chung nên bị gán nhầm policy_area="foreign_language_requirement".
    # Các anchor còn lại trong danh sách dưới đều đủ đặc hiệu (nhắc thẳng
    # "tiếng anh"/ngoại ngữ) nên không cần "chuẩn đầu vào"/"chuẩn đầu ra"
    # làm điều kiện kích hoạt độc lập. Bare "đầu vào"/"đầu ra" ở các
    # sub-tag bên dưới vẫn giữ nguyên vì chỉ chạy SAU KHI outer trigger đã
    # xác nhận ngữ cảnh ngoại ngữ.
    # -----------------------------------------------------------------------
    if has(
        "ngoại ngữ không chuyên",
        "tiếng anh",
        "tiếng anh dự bị",
        "kỳ thi đánh giá năng lực",
        "chứng chỉ tiếng anh",
        "ielts",
        "toeic",
        "toefl",
        "aptis",
        "cambridge",
    ):
        add_unique(policy_area, "foreign_language_requirement")

        if has("kiểm tra trình độ", "xếp lớp", "đầu vào"):
            add_unique(action_tags, "english_placement_test")

        if has("kỳ thi đánh giá năng lực", "thi đánh giá"):
            add_unique(action_tags, "english_exit_exam")

        if has("xét miễn", "miễn tiếng anh"):
            add_unique(action_tags, "request_english_exemption")

        # english_entry_requirement: match both canonical form and bare "đầu vào"
        if has("chuẩn đầu vào", "đầu vào"):
            add_unique(requirement_tags, "english_entry_requirement")

        # english_exit_requirement: match both canonical form and bare "đầu ra"
        if has("chuẩn đầu ra", "đầu ra"):
            add_unique(requirement_tags, "english_exit_requirement")

        if has("chứng chỉ", "ielts", "toeic", "toefl", "aptis", "cambridge"):
            add_unique(requirement_tags, "certificate_requirement")
            add_unique(evidence_groups, "language_certificate")

        if has("kế hoạch đào tạo", "lần/năm", "3 lần"):
            add_unique(time_tags, "annual_training_plan")
            add_unique(time_tags, "exam_schedule")

    # -----------------------------------------------------------------------
    # H. Second foreign language
    # -----------------------------------------------------------------------
    if has(
        "ngoại ngữ hai",
        "tiếng pháp",
        "tiếng nhật",
        "tiếng hàn",
        "tiếng trung",
        "tiếng nga",
        "tiếng đức",
        "tiếng tây ban nha",
        "delf",
        "jlpt",
        "nat-test",
        "topik",
        "hsk",
        "tocfl",
        "trki",
        "goethe",
    ):
        add_unique(policy_area, "second_foreign_language_requirement")
        add_unique(action_tags, "request_second_language_exemption")
        add_unique(requirement_tags, "second_language_certificate_requirement")
        add_unique(evidence_groups, "language_certificate")

    # -----------------------------------------------------------------------
    # I. Exams / grades
    # -----------------------------------------------------------------------
    if has(
        "thi cuối kỳ",
        "kỳ thi",
        "phúc tra",
        "khiếu nại điểm",
        "thang điểm",
        "điểm trung bình chung",
        "điểm trung bình chung tích lũy",
        "bảng điểm",
        "vắng thi",
        "điểm 0",
    ):
        add_unique(policy_area, "assessment_and_grading")

        if has("thi cuối kỳ", "kỳ thi cuối"):
            add_unique(action_tags, "take_final_exam")

        if has("phúc tra", "khiếu nại điểm"):
            add_unique(action_tags, "appeal_grade")
            add_unique(procedure_tags, "appeal_application_required")

        if has("điểm trung bình chung"):
            add_unique(action_tags, "calculate_gpa")

        if has("điểm 0", "vắng thi"):
            add_unique(risk_tags, "zero_score")
            add_unique(risk_tags, "exam_absence")

    # -----------------------------------------------------------------------
    # Assemble annotated chunk (copy + add fields)
    # -----------------------------------------------------------------------
    annotated = dict(chunk)
    annotated["policy_area"] = policy_area
    annotated["action_tags"] = action_tags
    annotated["student_status_tags"] = student_status_tags
    annotated["procedure_tags"] = procedure_tags
    annotated["evidence_groups"] = evidence_groups
    annotated["risk_tags"] = risk_tags
    annotated["requirement_tags"] = requirement_tags
    annotated["time_tags"] = time_tags
    return annotated


def annotate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate a list of chunks. Returns a new list; originals are not mutated."""
    return [annotate_chunk(c) for c in chunks]


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

# read_jsonl is imported from core



def write_jsonl(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    chunks_in: int,
    chunks_out: int,
    output_path: Path,
    annotated: list[dict[str, Any]],
) -> None:
    area_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()

    for chunk in annotated:
        for area in chunk.get("policy_area", []):
            area_counter[area] += 1
        for tag in chunk.get("action_tags", []):
            tag_counter[tag] += 1

    print("=" * 60)
    print("Policy annotation complete")
    print("=" * 60)
    print(f"  Chunks read    : {chunks_in}")
    print(f"  Chunks written : {chunks_out}")
    print(f"  Output         : {output_path.resolve()}")
    print()
    print("  Counts by policy_area:")
    for area, count in sorted(area_counter.items()):
        print(f"    {area}: {count}")
    print()
    print("  Counts by action_tags:")
    for tag, count in sorted(tag_counter.items()):
        print(f"    {tag}: {count}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate policy chunks with rule-based metadata tags."
    )
    parser.add_argument(
        "--input-file",
        default="data/chunks/policy_chunks.jsonl",
        help="Input JSONL file (default: data/chunks/policy_chunks.jsonl)",
    )
    parser.add_argument(
        "--output-file",
        default="data/chunks/policy_chunks.annotated.jsonl",
        help="Output JSONL file (default: data/chunks/policy_chunks.annotated.jsonl)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    chunks = read_jsonl(input_path)
    annotated = annotate_chunks(chunks)
    write_jsonl(output_path, annotated)
    print_summary(len(chunks), len(annotated), output_path, annotated)


if __name__ == "__main__":
    main()