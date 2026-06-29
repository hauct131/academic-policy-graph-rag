"""
tests/test_annotate_new_chunks_offline.py

Unit tests for scripts/annotate_new_chunks_offline.py (v2.8 — Free-only, 6-field)
"""

import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from collections import Counter
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
    """Test tag filtering with 6 fields (v2.8)"""
    action_allowed = ["tag1", "tag2"]
    risk_allowed = ["tag3"]
    proc_allowed = ["tag4"]
    evidence_allowed = ["tag5"]
    req_allowed = ["tag6"]
    time_allowed = ["tag7"]

    # All valid tags preserved
    fa, fr, fp, fe, frq, ft = validate_and_filter_tags(
        ["tag1"], ["tag3"], ["tag4"], ["tag5"], ["tag6"], ["tag7"],
        action_allowed, risk_allowed, proc_allowed,
        evidence_allowed, req_allowed, time_allowed,
        "chunk_1", "model_1"
    )
    assert fa == ["tag1"]
    assert fr == ["tag3"]
    assert fp == ["tag4"]
    assert fe == ["tag5"]
    assert frq == ["tag6"]
    assert ft == ["tag7"]

    # Invalid tags filtered out
    fa, fr, fp, fe, frq, ft = validate_and_filter_tags(
        ["tag1", "invalid"], ["tag3", "invalid"], ["tag4", "invalid"],
        ["tag5", "invalid"], ["tag6", "invalid"], ["tag7", "invalid"],
        action_allowed, risk_allowed, proc_allowed,
        evidence_allowed, req_allowed, time_allowed,
        "chunk_1", "model_1"
    )
    assert fa == ["tag1"]
    assert fr == ["tag3"]
    assert fp == ["tag4"]
    assert fe == ["tag5"]
    assert frq == ["tag6"]
    assert ft == ["tag7"]


@pytest.mark.anyio
async def test_tag_filtering_behavior():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    content_json = {
        "action_tags": ["tag_a_allowed", "tag_a_forbidden"],
        "risk_tags": ["tag_r_allowed"],
        "procedure_tags": ["tag_p_allowed"],
        "evidence_groups": ["tag_e_allowed"],
        "requirement_tags": ["tag_req_allowed"],
        "time_tags": ["tag_t_allowed"]
    }
    mock_response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(content_json)}}]
    }
    mock_client.post.return_value = mock_response

    chunk = {"chunk_id": "chunk_1", "text": "Quy chế đăng ký môn học"}
    action_allowed = ["tag_a_allowed"]
    risk_allowed = ["tag_r_allowed"]
    proc_allowed = ["tag_p_allowed"]
    evidence_allowed = ["tag_e_allowed"]
    req_allowed = ["tag_req_allowed"]
    time_allowed = ["tag_t_allowed"]

    semaphore = asyncio.Semaphore(1)
    free_models = ["model_1", "model_2"]
    rate_limiter = PerModelRateLimiter(free_models, global_interval=0.01)

    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        chunk_position=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        evidence_allowed=evidence_allowed,
        req_allowed=req_allowed,
        time_allowed=time_allowed,
        semaphore=semaphore,
        rate_limiter=rate_limiter,
        api_key="dummy_key",
        free_models=free_models,
        current_cycle=1,
        model_usage_counter=Counter()
    )

    assert success is True
    assert annotated["extracted_by"] == "llm_annotated"
    assert "tag_a_allowed" in annotated["action_tags"]
    assert "tag_a_forbidden" not in annotated["action_tags"]


@pytest.mark.anyio
@patch("asyncio.sleep", return_value=None)
async def test_rate_limit_rotation(mock_sleep):
    mock_client = AsyncMock()
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429

    mock_resp_success = MagicMock()
    mock_resp_success.status_code = 200
    mock_resp_success.json.return_value = {
        "choices": [{"message": {"content": '{"action_tags": ["tag_a"], "risk_tags": [], "procedure_tags": [], "evidence_groups": [], "requirement_tags": [], "time_tags": []}'}}]
    }

    mock_client.post.side_effect = [mock_resp_429, mock_resp_success]

    chunk = {"chunk_id": "chunk_2", "text": "Quy chế học vụ"}
    action_allowed = ["tag_a"]
    risk_allowed = []
    proc_allowed = []
    evidence_allowed = []
    req_allowed = []
    time_allowed = []

    semaphore = asyncio.Semaphore(1)
    free_models = ["model_1", "model_2"]
    rate_limiter = PerModelRateLimiter(free_models, global_interval=0.01)

    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        chunk_position=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        evidence_allowed=evidence_allowed,
        req_allowed=req_allowed,
        time_allowed=time_allowed,
        semaphore=semaphore,
        rate_limiter=rate_limiter,
        api_key="dummy_key",
        free_models=free_models,
        current_cycle=1,
        model_usage_counter=Counter()
    )

    assert success is True
    assert annotated["model_used"] == "model_2"


@pytest.mark.anyio
@patch("asyncio.sleep", return_value=None)
async def test_fallback_on_network_error(mock_sleep):
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectError("Connection failed")

    chunk = {"chunk_id": "chunk_3", "text": "Đăng ký môn học"}
    action_allowed = ["register_course"]
    risk_allowed = []
    proc_allowed = []
    evidence_allowed = []
    req_allowed = []
    time_allowed = []

    semaphore = asyncio.Semaphore(1)
    free_models = ["model_1", "model_2"]
    rate_limiter = PerModelRateLimiter(free_models, global_interval=0.01)

    success, annotated = await annotate_chunk_core(
        client=mock_client,
        chunk=chunk,
        chunk_position=0,
        action_allowed=action_allowed,
        risk_allowed=risk_allowed,
        proc_allowed=proc_allowed,
        evidence_allowed=evidence_allowed,
        req_allowed=req_allowed,
        time_allowed=time_allowed,
        semaphore=semaphore,
        rate_limiter=rate_limiter,
        api_key="dummy_key",
        free_models=free_models,
        current_cycle=1,
        model_usage_counter=Counter()
    )

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
    evidence_allowed = []
    req_allowed = []
    time_allowed = []

    semaphore = asyncio.Semaphore(1)
    free_models = ["model_1"]
    rate_limiter = PerModelRateLimiter(free_models, global_interval=0.01)

    with pytest.raises(AuthError):
        await annotate_chunk_core(
            client=mock_client,
            chunk=chunk,
            chunk_position=0,
            action_allowed=action_allowed,
            risk_allowed=risk_allowed,
            proc_allowed=proc_allowed,
            evidence_allowed=evidence_allowed,
            req_allowed=req_allowed,
            time_allowed=time_allowed,
            semaphore=semaphore,
            rate_limiter=rate_limiter,
            api_key="dummy_key",
            free_models=free_models,
            current_cycle=1,
            model_usage_counter=Counter()
        )