"""
tests/test_spot_check_annotations.py

Unit and integration tests for scripts/spot_check_annotations.py
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.spot_check_annotations import (
    sample_policy_chunks,
    main,
)


def test_sample_policy_chunks_deterministic():
    # Generate 10 dummy chunks
    chunks = [{"chunk_id": f"c_{i}", "text": f"text_{i}"} for i in range(10)]
    
    # Call sample multiple times with same seed
    sample_1 = sample_policy_chunks(chunks, sample_ratio=0.3, seed=42)
    sample_2 = sample_policy_chunks(chunks, sample_ratio=0.3, seed=42)
    
    assert sample_1 == sample_2
    assert len(sample_1) == 3


def test_sample_policy_chunks_edge_case():
    # 3 chunks -> sample_ratio 0.1 should still return 1 chunk
    chunks = [{"chunk_id": f"c_{i}", "text": f"text_{i}"} for i in range(3)]
    sample = sample_policy_chunks(chunks, sample_ratio=0.1, seed=42)
    assert len(sample) == 1
    assert sample[0] in chunks


def test_sample_policy_chunks_ratio_60():
    # 60 chunks -> sample_ratio 0.1 should return exactly 6 chunks
    chunks = [{"chunk_id": f"c_{i}", "text": f"text_{i}"} for i in range(60)]
    sample = sample_policy_chunks(chunks, sample_ratio=0.1, seed=42)
    assert len(sample) == 6


def test_sample_policy_chunks_empty():
    assert sample_policy_chunks([], sample_ratio=0.1) == []


def test_spot_check_integration(tmp_path):
    # Setup temporary files
    input_file = tmp_path / "policy_chunks.llm_reviewed.jsonl"
    output_file = tmp_path / "spot_check_report.json"

    # Write dummy annotated chunks to input_file
    dummy_chunks = []
    for i in range(25):
        dummy_chunks.append({
            "chunk_id": f"chunk_{i}",
            "text": f"Nội dung quy chế đào tạo phần {i}. " * 10,
            "action_tags": ["register_course"] if i % 2 == 0 else [],
            "policy_area": ["course_registration"] if i % 2 == 0 else [],
            "extracted_by": "llm_first_pass" if i % 2 == 0 else "rule_based_fallback"
        })

    with input_file.open("w", encoding="utf-8") as fh:
        for chunk in dummy_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Run the main script via mocks of CLI args
    test_args = [
        "spot_check_annotations.py",
        "--input-file", str(input_file),
        "--output-file", str(output_file),
        "--ratio", "0.1",
        "--seed", "42"
    ]
    with patch("sys.argv", test_args):
        main()

    # Check output file exists
    assert output_file.exists()

    # Read output and verify structure
    with output_file.open("r", encoding="utf-8") as fh:
        report = json.load(fh)

    # 25 chunks * 0.1 = 2.5 -> int(2.5) = 2
    assert len(report) == 2

    for record in report:
        assert "chunk_id" in record
        assert "text_preview" in record
        assert len(record["text_preview"]) <= 150
        assert "action_tags" in record
        assert "policy_area" in record
        assert "extracted_by" in record
        assert record["extracted_by"] in ["llm_first_pass", "rule_based_fallback"]
        assert record["human_verified"] is None
        assert record["notes"] == ""
