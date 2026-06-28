"""
tests/test_annotate_new_chunks_offline.py

Unit tests for scripts/annotate_new_chunks_offline.py (v2.6):
- validate_and_filter_tags() utility testing
- Tag filtering behavior (bad tags filtered, good tags kept)
- Model rotation on 429 followed by LLM success
- Model error handling (network error/timeout)
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
    validate_and_filter_tags,
    PerModelRateLimiter,
    AuthError,
)


def test_validate_and_filter_tags():
    action_allowed = ["tag1", "tag2"]
    risk_allowed = ["tag3"]
    proc_allowed = ["tag4"]

    # All valid tags preserved
    fa, fr, fp = validate_and_filter_tags(
        ["tag1"], ["tag3"], ["tag4"],
        action_allowed, risk_allowed, proc_allowed,
        "chunk_1", "model_1"
    )
    assert fa == ["tag1"]
    assert fr == ["tag3"]
    assert fp == ["tag4"]

    # Invalid tags filtered out
    fa, fr, fp = validate_and_filter_tags(
        ["tag1", "tag_invalid"], ["tag3", "tag_invalid"], ["tag4", "tag_invalid"],
        action_allowed, risk_allowed, proc_allowed,
        "chunk_1", "model_1"
    )
    assert fa == ["tag1"]
    assert fr == ["tag3"]
    assert fp == ["tag4"]


@pytest.mark.anyio
async def test_tag_filtering_behavior():
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
    
    free_models = ["model_1", "model_2"]
    rate_limiter = PerModelRateLimiter(free_models, global_interval=0.01)
    paid_chunks_counter = [0]

    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        chunk_position=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        semaphore=semaphore,
        rate_limiter=rate_limiter,
        api_key="dummy_key",
        free_models=free_models,
        paid_fallback_models=[],
        use_paid_fallback=False,
        current_cycle=1,
        paid_chunks_counter=paid_chunks_counter,
    )

    assert success is True
    assert annotated["extracted_by"] == "llm_annotated"
    assert annotated["model_used"] == "model_1"
    assert "tag_a_allowed" in annotated["action_tags"]
    assert "tag_a_forbidden" not in annotated["action_tags"]


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
    free_models = ["model_1", "model_2"]
    rate_limiter = PerModelRateLimiter(free_models, global_interval=0.01)
    paid_chunks_counter = [0]
    
    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        chunk_position=0,  # starts at model_1
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        semaphore=semaphore,
        rate_limiter=rate_limiter,
        api_key="dummy_key",
        free_models=free_models,
        paid_fallback_models=[],
        use_paid_fallback=False,
        current_cycle=1,
        paid_chunks_counter=paid_chunks_counter,
    )
    
    assert success is True
    assert annotated["model_used"] == "model_2"
    assert annotated["extracted_by"] == "llm_annotated"
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
    free_models = ["model_1", "model_2"]
    rate_limiter = PerModelRateLimiter(free_models, global_interval=0.01)
    paid_chunks_counter = [0]
    
    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        chunk_position=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        semaphore=semaphore,
        rate_limiter=rate_limiter,
        api_key="dummy_key",
        free_models=free_models,
        paid_fallback_models=[],
        use_paid_fallback=False,
        current_cycle=1,
        paid_chunks_counter=paid_chunks_counter,
    )
    
    # All LLM attempts fail, returns False with original chunk
    assert success is False
    assert annotated == chunk


@pytest.mark.anyio
async def test_auth_error_propagation():
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_client.post.return_value = mock_resp
    
    chunk = {"chunk_id": "chunk_4", "text": "Quy chế"}
    action_allowed = []
    risk_allowed = []
    proc_allowed = []
    semaphore = asyncio.Semaphore(1)
    
    free_models = ["model_1"]
    rate_limiter = PerModelRateLimiter(free_models, global_interval=0.01)
    paid_chunks_counter = [0]
    
    with pytest.raises(AuthError):
        await annotate_chunk_core(
            client=mock_client,
            chunk=chunk,
            chunk_position=0,
            action_allowed=action_allowed,
            risk_allowed=risk_allowed,
            proc_allowed=proc_allowed,
            semaphore=semaphore,
            rate_limiter=rate_limiter,
            api_key="dummy_key",
            free_models=free_models,
            paid_fallback_models=[],
            use_paid_fallback=False,
            current_cycle=2,
            paid_chunks_counter=paid_chunks_counter,
        )
