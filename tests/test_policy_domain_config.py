"""
tests/test_policy_domain_config.py

Unit tests for the OU Academic Policy domain configuration layer.

Run with:
    python -m pytest tests/test_policy_domain_config.py
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from importlib import import_module

_domain = import_module("policy_domain_config")
load_domain_config = _domain.load_domain_config
validate_domain_config = _domain.validate_domain_config
infer_issues_from_domain = _domain.infer_issues_from_domain


class TestPolicyDomainConfig:
    def test_loads_domain_json(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        assert config_path.exists()
        config = load_domain_config(config_path)
        assert isinstance(config, dict)
        assert config["domain_id"] == "ou_academic_policy_v1"

    def test_validate_domain_config_returns_no_errors(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        config = load_domain_config(config_path)
        errors = validate_domain_config(config)
        assert len(errors) == 0, f"Validation errors found: {errors}"

    def test_graduation_issue_inferred(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        config = load_domain_config(config_path)
        issues = infer_issues_from_domain("Điều kiện xét tốt nghiệp là gì?", config)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "graduation"
        assert issues[0]["query"] == "dieu kien xet tot nghiep"

    def test_course_exemption_hoso_query(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        config = load_domain_config(config_path)
        issues = infer_issues_from_domain("Miễn môn học cần hồ sơ gì?", config)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "course_exemption"
        assert issues[0]["query"] == "ho so xin mien giam mon hoc"

    def test_course_exemption_conditions_query(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        config = load_domain_config(config_path)
        issues = infer_issues_from_domain("Điều kiện miễn môn học là gì?", config)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "course_exemption"
        assert issues[0]["query"] == "dieu kien xet mien giam mon hoc"

    def test_foreign_language_ielts_query(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        config = load_domain_config(config_path)
        issues = infer_issues_from_domain("IELTS 6.0 có được miễn học phần tiếng Anh không?", config)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "foreign_language_requirement"
        assert "ielts" in issues[0]["query"]

    def test_long_case_returns_three_issues(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        config = load_domain_config(config_path)
        # Query with IELTS (foreign_language_requirement), bảng điểm (course_exemption), tốt nghiệp (graduation)
        case_text = "Em đã thi đạt IELTS 6.0 và có bảng điểm môn học ở trường cũ. Cho em hỏi làm sao để xin miễn môn học và điều kiện để xét tốt nghiệp sau này?"
        issues = infer_issues_from_domain(case_text, config)
        assert len(issues) >= 3
        issue_types = {iss["issue_type"] for iss in issues}
        assert "foreign_language_requirement" in issue_types
        assert "course_exemption" in issue_types
        assert "graduation" in issue_types

    def test_unknown_question_returns_generic_issue(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        config = load_domain_config(config_path)
        issues = infer_issues_from_domain("Đăng ký ký túc xá thế nào?", config)
        assert len(issues) == 1
        assert issues[0]["issue_type"] == "generic"

    def test_current_semester_keywords_include_time_terms(self):
        config_path = Path(__file__).parent.parent / "domains" / "ou_academic_policy_v1" / "domain.json"
        config = load_domain_config(config_path)
        kws = config["current_semester_keywords"]
        assert "deadline" in kws
        assert "thoi han" in kws or "thời hạn" in kws or any("thoi han" in kw for kw in kws)
