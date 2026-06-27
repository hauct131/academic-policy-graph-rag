"""
tests/test_generate_chunk_embeddings.py

Unit tests and integration tests for scripts/generate_chunk_embeddings.py:
- Mocking get_nvidia_nemotron_embedding for success, rate limits, and server errors
- check_embedding_consistency unit tests
- Pipeline integration test for mismatched embedding lengths triggering sys.exit(1)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_chunk_embeddings import (
    get_nvidia_nemotron_embedding,
    check_embedding_consistency,
    process_embeddings_pipeline,
)


@pytest.mark.anyio
async def test_get_nvidia_nemotron_embedding_success():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {
                "embedding": [0.1, 0.2, 0.3]
            }
        ]
    }
    mock_client.post.return_value = mock_response

    embedding = await get_nvidia_nemotron_embedding(mock_client, "some text", "fake_key")
    assert embedding == [0.1, 0.2, 0.3]


@pytest.mark.anyio
async def test_get_nvidia_nemotron_embedding_rate_limit():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.request = httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings")
    mock_client.post.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await get_nvidia_nemotron_embedding(mock_client, "some text", "fake_key")
    assert "429 Rate Limit" in str(exc_info.value)


@pytest.mark.anyio
async def test_get_nvidia_nemotron_embedding_server_error():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Internal Server Error",
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/embeddings"),
        response=mock_response
    )
    mock_client.post.return_value = mock_response

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await get_nvidia_nemotron_embedding(mock_client, "some text", "fake_key")
    assert "500 Internal Server Error" in str(exc_info.value)


def test_check_embedding_consistency_all_valid():
    chunks = [
        {"chunk_id": "c1", "embedding": [0.1, 0.2]},
        {"chunk_id": "c2", "embedding": [0.3, 0.4]},
    ]
    failed_chunks, embedding_lengths = check_embedding_consistency(chunks)
    assert failed_chunks == []
    assert embedding_lengths == {2}


def test_check_embedding_consistency_with_empty():
    chunks = [
        {"chunk_id": "c1", "embedding": [0.1, 0.2]},
        {"chunk_id": "c2", "embedding": []},
        {"chunk_id": "c3"},
    ]
    failed_chunks, embedding_lengths = check_embedding_consistency(chunks)
    assert failed_chunks == ["c2", "c3"]
    assert embedding_lengths == {2}


def test_check_embedding_consistency_mismatched_lengths():
    chunks = [
        {"chunk_id": "c1", "embedding": [0.1, 0.2]},
        {"chunk_id": "c2", "embedding": [0.3, 0.4, 0.5]},
    ]
    failed_chunks, embedding_lengths = check_embedding_consistency(chunks)
    assert failed_chunks == []
    assert embedding_lengths == {2, 3}


@pytest.mark.anyio
@patch("scripts.generate_chunk_embeddings.get_nvidia_nemotron_embedding")
@patch.dict(os.environ, {"OPENROUTER_API_KEY": "dummy_key"})
async def test_process_embeddings_pipeline_mismatched_exit(mock_get_embedding, tmp_path):
    # Prepare input chunks file
    input_file = tmp_path / "test_input.jsonl"
    output_file = tmp_path / "test_output.jsonl"
    
    chunks = [
        {"chunk_id": "chunk_1", "text": "Text 1"},
        {"chunk_id": "chunk_2", "text": "Text 2"},
    ]
    
    with open(input_file, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")
            
    # Mock embedding returns different dimensions
    mock_get_embedding.side_effect = [
        [0.1, 0.2],
        [0.3, 0.4, 0.5]
    ]
    
    args = argparse.Namespace(
        input_file=str(input_file),
        output_file=str(output_file),
        force_cloud=True
    )
    
    with pytest.raises(SystemExit) as exc_info:
        await process_embeddings_pipeline(args)
        
    assert exc_info.value.code == 1
