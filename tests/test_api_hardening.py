"""
tests/test_api_hardening.py

Unit tests for API hardening features:
- Question length limitations (min_length, max_length)
- Rate limiting middleware (per-IP request limits)
"""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app, _request_log, RATE_LIMIT_PER_MINUTE

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_request_log():
    """Clear request log before and after each test to prevent test interference."""
    _request_log.clear()
    yield
    _request_log.clear()


def test_question_too_long():
    # Max length is 2000 characters. 2001 characters should trigger a 422 Validation Error.
    long_question = "a" * 2001
    response = client.post("/policy/ask", json={
        "question": long_question,
        "top_k": 5
    })
    assert response.status_code == 422
    assert "question" in response.text or "msg" in response.text


def test_question_empty_or_too_short():
    # Min length is 1 character. Empty question should trigger 422.
    response = client.post("/policy/ask", json={
        "question": "",
        "top_k": 5
    })
    assert response.status_code == 422


@patch("app.main.qa_service")
def test_rate_limit_exceeded(mock_qa_service):
    # Setup mock to avoid loading real files or calling external services
    mock_qa_service.initialized = True
    mock_qa_service.get_qa_response.return_value = ("Mocked Answer", {}, [])

    # Send RATE_LIMIT_PER_MINUTE requests
    for i in range(RATE_LIMIT_PER_MINUTE):
        response = client.post("/policy/ask", json={
            "question": f"Valid question {i}",
            "top_k": 5
        })
        assert response.status_code == 200

    # The (RATE_LIMIT_PER_MINUTE + 1)-th request must be rate limited (429)
    response = client.post("/policy/ask", json={
        "question": "One too many questions",
        "top_k": 5
    })
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "60"
    assert response.json() == {"detail": "Too many requests. Please try again later."}
