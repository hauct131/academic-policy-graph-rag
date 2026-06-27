"""
tests/test_annotate_new_chunks_offline.py

Unit tests for scripts/annotate_new_chunks_offline.py under Hybrid Architecture:
- validate_tags_against_domain() utility testing
- Fallback to rule-based annotations on tag validation errors (tag lạ)
- Model rotation on 429 followed by LLM success
- Fallback to rule-based annotations on client exception (network error, timeout)
- Proper metadata tracking for model_used and extracted_by
"""

import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.annotate_new_chunks_offline import (
    annotate_chunk_core,
    validate_tags_against_domain,
)


def test_validate_tags_against_domain():
    domain_config = {
        "action_tags": ["tag1", "tag2"],
        "risk_tags": ["tag3"],
        "procedure_tags": ["tag4"]
    }
    # All valid tags preserved
    assert validate_tags_against_domain(["tag1", "tag3"], domain_config) == ["tag1", "tag3"]
    # Invalid tags filtered out
    assert validate_tags_against_domain(["tag1", "tag_invalid", "tag4"], domain_config) == ["tag1", "tag4"]
    # Empty tags list
    assert validate_tags_against_domain([], domain_config) == []


@pytest.mark.anyio
async def test_fallback_on_tag_la():
    # Setup mock client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    # LLM returns a response containing a forbidden tag ("tag_a_forbidden")
    content_json = {
        "action_tags": ["tag_a_allowed", "tag_a_forbidden"],
        "risk_tags": ["tag_r_allowed"],
        "procedure_tags": ["tag_p_allowed"]
    }
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content_json)
                }
            }
        ]
    }
    mock_client.post.return_value = mock_response

    chunk = {"chunk_id": "chunk_1", "text": "Quy chế đăng ký môn học"}
    action_allowed = ["tag_a_allowed"]
    risk_allowed = ["tag_r_allowed"]
    proc_allowed = ["tag_p_allowed"]
    semaphore = asyncio.Semaphore(1)
    global_state = {"active_idx": 0}
    models_pool = ["model_1", "model_2"]

    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        idx=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        semaphore=semaphore,
        api_key="dummy_key",
        models_pool=models_pool,
        global_state=global_state,
        current_cycle=1
    )

    # Must succeed (pipeline doesn't crash) but fall back to rule-based because of "tag_a_forbidden"
    assert success is True
    assert annotated["extracted_by"] == "rule_based_fallback"
    assert annotated["model_used"] is None
    # "Quy chế đăng ký môn học" rule-based matching yields "course_registration"
    assert "course_registration" in annotated["policy_area"]
    assert "register_course" in annotated["action_tags"]


@pytest.mark.anyio
@patch("asyncio.sleep", return_value=None)
async def test_rate_limit_rotation(mock_sleep):
    mock_client = AsyncMock()
    
    # First response is 429, second response is 200 (success)
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429
    
    mock_resp_success = MagicMock()
    mock_resp_success.status_code = 200
    mock_resp_success.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"action_tags": ["tag_a"], "risk_tags": [], "procedure_tags": []}'
                }
            }
        ]
    }
    
    mock_client.post.side_effect = [mock_resp_429, mock_resp_success]
    
    chunk = {"chunk_id": "chunk_2", "text": "Quy chế học vụ"}
    action_allowed = ["tag_a"]
    risk_allowed = []
    proc_allowed = []
    semaphore = asyncio.Semaphore(1)
    global_state = {"active_idx": 0}
    models_pool = ["model_1", "model_2", "model_3"]
    
    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        idx=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        semaphore=semaphore,
        api_key="dummy_key",
        models_pool=models_pool,
        global_state=global_state,
        current_cycle=1
    )
    
    assert success is True
    # The first attempt used model_1 (idx 0), got 429, rotated active_idx to (0+1)%3 = 1.
    # The second attempt (attempt=1) used index (1+1)%3 = 2 (model_3) and succeeded.
    # Since model_3 succeeded, model_used should be model_3, and active_idx becomes 2.
    assert annotated["model_used"] == "model_3"
    assert annotated["extracted_by"] == "llm_first_pass"
    assert global_state["active_idx"] == 2
    mock_sleep.assert_called()


@pytest.mark.anyio
@patch("asyncio.sleep", return_value=None)
async def test_fallback_on_network_error(mock_sleep):
    mock_client = AsyncMock()
    # Mock OpenRouter connection failure
    mock_client.post.side_effect = httpx.ConnectError(
        "Connection failed",
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    )
    
    chunk = {"chunk_id": "chunk_3", "text": "Đăng ký môn học và xét tốt nghiệp"}
    action_allowed = ["register_course", "graduation_audit"]
    risk_allowed = []
    proc_allowed = []
    semaphore = asyncio.Semaphore(1)
    global_state = {"active_idx": 0}
    models_pool = ["model_1", "model_2"]
    
    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        idx=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        semaphore=semaphore,
        api_key="dummy_key",
        models_pool=models_pool,
        global_state=global_state,
        current_cycle=1
    )
    
    # Must succeed by falling back to rule-based labeling immediately
    assert success is True
    assert annotated["extracted_by"] == "rule_based_fallback"
    assert annotated["model_used"] is None
    # Tag results match rule-based mapping
    assert "course_registration" in annotated["policy_area"]
    assert "graduation" in annotated["policy_area"]
    assert "register_course" in annotated["action_tags"]
    assert "graduation_audit" in annotated["action_tags"]


@pytest.mark.anyio
async def test_model_used_tracking():
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"action_tags": [], "risk_tags": [], "procedure_tags": []}'
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp
    
    chunk = {"chunk_id": "chunk_4", "text": "Quy chế"}
    action_allowed = []
    risk_allowed = []
    proc_allowed = []
    semaphore = asyncio.Semaphore(1)
    
    # Start with active_idx = 1 (model_2)
    global_state = {"active_idx": 1}
    models_pool = ["model_1", "model_2", "model_3"]
    
    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        idx=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        semaphore=semaphore,
        api_key="dummy_key",
        models_pool=models_pool,
        global_state=global_state,
        current_cycle=2
    )
    
    assert success is True
    assert annotated["model_used"] == "model_2"
    assert annotated["extracted_by"] == "llm_first_pass"
    assert annotated["pipeline_cycle_attempts"] == 2
