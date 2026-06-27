"""
tests/test_annotate_new_chunks_offline.py

Unit tests for scripts/annotate_new_chunks_offline.py:
- Tag filtering logic (allowed list checking)
- Model rotation on rate limit (429)
- Handling all retries failure
- Recording the correct model name used for successful annotations
"""

import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.annotate_new_chunks_offline import annotate_chunk_core


@pytest.mark.anyio
async def test_allow_list_filtering():
    # Setup mock client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    # Returns some tags, including not allowed ones
    content_json = {
        "action_tags": ["tag_a_allowed", "tag_a_forbidden"],
        "risk_tags": ["tag_r_allowed"],
        "procedure_tags": ["tag_p_forbidden"]
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

    chunk = {"chunk_id": "chunk_1", "text": "Quy chế học vụ"}
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

    assert success is True
    # The forbidden tags must be filtered out
    assert annotated["action_tags"] == ["tag_a_allowed"]
    assert annotated["risk_tags"] == ["tag_r_allowed"]
    assert annotated["procedure_tags"] == []
    assert annotated["model_used"] == "model_1"


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
    assert global_state["active_idx"] == 2
    mock_sleep.assert_called()


@pytest.mark.anyio
@patch("asyncio.sleep", return_value=None)
async def test_all_attempts_fail(mock_sleep):
    mock_client = AsyncMock()
    mock_client.post.side_effect = Exception("API connection failure")
    
    chunk = {"chunk_id": "chunk_3", "text": "Quy chế học vụ"}
    action_allowed = []
    risk_allowed = []
    proc_allowed = []
    semaphore = asyncio.Semaphore(1)
    global_state = {"active_idx": 0}
    models_pool = ["model_1", "model_2"]
    
    success, result_chunk = await annotate_chunk_core(
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
    
    assert success is False
    assert result_chunk == chunk


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
    assert annotated["pipeline_cycle_attempts"] == 2
