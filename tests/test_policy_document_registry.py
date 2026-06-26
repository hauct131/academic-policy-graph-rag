"""
tests/test_policy_document_registry.py

Unit tests for the policy document registry loader, validation, temporal context inference,
active checks, query selection, and missing notices warning rules.

Run with:
    python -m pytest tests/test_policy_document_registry.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import policy_document_registry as _reg
load_document_registry = _reg.load_document_registry
validate_document_registry = _reg.validate_document_registry
infer_time_context = _reg.infer_time_context
is_document_active = _reg.is_document_active
requires_current_notice = _reg.requires_current_notice
has_current_notice = _reg.has_current_notice
should_warn_missing_current_notice = _reg.should_warn_missing_current_notice
select_documents_for_query = _reg.select_documents_for_query
policy_area_matches = _reg.policy_area_matches


class TestPolicyDocumentRegistry:
    _REGISTRY_PATH = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "document_registry.jsonl"

    def test_loads_registry(self):
        assert self._REGISTRY_PATH.exists()
        records = load_document_registry(self._REGISTRY_PATH)
        assert isinstance(records, list)
        assert len(records) >= 3

    def test_registry_validation_returns_no_errors(self):
        records = load_document_registry(self._REGISTRY_PATH)
        errors = validate_document_registry(records)
        assert len(errors) == 0, f"Registry validation errors: {errors}"

    def test_all_three_current_doc_ids_exist(self):
        records = load_document_registry(self._REGISTRY_PATH)
        doc_ids = {r["doc_id"] for r in records}
        expected_ids = {
            "ou_fulltime_credit_training_regulation_2016",
            "ou_course_exemption_regulation_2023",
            "ou_non_major_foreign_language_regulation_2023"
        }
        assert expected_ids.issubset(doc_ids)

    def test_registry_has_no_placeholder_decision_nos(self):
        records = load_document_registry(self._REGISTRY_PATH)
        placeholders = {"123/QĐ-ĐHM", "456/QĐ-ĐHM", "789/QĐ-ĐHM"}
        for r in records:
            dno = r.get("decision_no")
            assert dno not in placeholders, f"Record {r['doc_id']} contains placeholder decision_no: {dno}"

    def test_policy_area_matching_with_list(self):
        # Record with list of policy areas
        record_list = {"policy_area": ["course_exemption", "graduation"]}
        assert policy_area_matches(record_list, "course_exemption") is True
        assert policy_area_matches(record_list, "graduation") is True
        assert policy_area_matches(record_list, "academic_standing") is False
        assert policy_area_matches(record_list, None) is True

        # Record with string policy area
        record_str = {"policy_area": "course_exemption"}
        assert policy_area_matches(record_str, "course_exemption") is True
        assert policy_area_matches(record_str, "graduation") is False

        # Record with null policy area
        record_null = {"policy_area": None}
        assert policy_area_matches(record_null, "course_exemption") is False

    def test_stable_regulation_active(self):
        records = load_document_registry(self._REGISTRY_PATH)
        target_doc = next(r for r in records if r["doc_id"] == "ou_course_exemption_regulation_2023")
        
        # effective_to is null, status is active
        assert target_doc["effective_to"] is None
        assert target_doc["status"] == "active"
        
        # Should be active generally and at a target date
        assert is_document_active(target_doc) is True
        assert is_document_active(target_doc, target_date=date(2026, 6, 10)) is True

    def test_infer_time_context_current_semester_and_deadline(self):
        ctx = infer_time_context("Học kỳ này khi nào nộp hồ sơ miễn môn?")
        assert ctx["current_semester"] is True
        assert ctx["has_deadline_intent"] is True
        assert ctx["semester"] is None
        assert ctx["academic_year"] is None

    def test_infer_time_context_semester_and_academic_year(self):
        ctx = infer_time_context("học kỳ 1 năm học 2025-2026")
        assert ctx["current_semester"] is False
        assert ctx["semester"] == 1
        assert ctx["academic_year"] == "2025-2026"

    def test_requires_current_notice(self):
        assert requires_current_notice("Học kỳ này khi nào nộp hồ sơ miễn môn?") is True
        assert requires_current_notice("Điều kiện để được xét tốt nghiệp là gì?") is False

    def test_has_current_notice_false_with_current_registry(self):
        records = load_document_registry(self._REGISTRY_PATH)
        assert has_current_notice(records, "course_exemption") is False

    def test_should_warn_missing_current_notice_true(self):
        records = load_document_registry(self._REGISTRY_PATH)
        q = "Học kỳ này khi nào nộp hồ sơ miễn môn?"
        assert should_warn_missing_current_notice(q, records, "course_exemption") is True

    def test_select_documents_for_query_prioritizes_policy_area(self):
        records = load_document_registry(self._REGISTRY_PATH)
        selected = select_documents_for_query(
            records,
            policy_area="course_exemption",
            question="Điều kiện miễn môn học là gì?"
        )
        assert len(selected) >= 3
        # First document should be the one matching policy_area 'course_exemption'
        assert selected[0]["doc_id"] == "ou_course_exemption_regulation_2023"

    def test_synthetic_semester_notice_detection(self):
        # Base records
        records = load_document_registry(self._REGISTRY_PATH)
        
        # Add synthetic semester notice with policy_area as list
        synthetic_notice = {
            "doc_id": "ou_course_exemption_notice_hk1_2025",
            "title": "Thông báo nộp hồ sơ miễn môn HK1 năm học 2025-2026",
            "document_type": "semester_notice",
            "policy_area": ["course_exemption"],
            "decision_no": "999/TB-ĐHM",
            "issued_date": "2025-08-01",
            "effective_from": "2025-08-01",
            "effective_to": "2026-01-31",
            "academic_year": "2025-2026",
            "semester": 1,
            "status": "active",
            "temporal_scope": "semester",
            "update_cadence": "semester",
            "source_pdf": "ou_course_exemption_notice_hk1_2025.pdf",
            "source_path": "data/raw/ou_course_exemption_notice_hk1_2025.pdf",
            "notes": "Synthetic notice for test."
        }
        
        extended_records = records + [synthetic_notice]
        
        # At target date 2025-10-10, this synthetic notice is active
        target_d = date(2025, 10, 10)
        assert has_current_notice(extended_records, "course_exemption", target_date=target_d) is True
        
        # For a temporal question, select_documents_for_query should put the semester notice first
        selected = select_documents_for_query(
            extended_records,
            policy_area="course_exemption",
            question="Học kỳ này khi nào nộp hồ sơ miễn môn?",
            target_date=target_d
        )
        assert selected[0]["doc_id"] == "ou_course_exemption_notice_hk1_2025"
