#!/usr/bin/env python3
"""
scripts/spot_check_annotations.py

Spot check utility for annotation quality control.
Randomly samples 10% of chunks with deterministic seed.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path


def sample_policy_chunks(chunks: list[dict], sample_ratio: float = 0.1, seed: int = 42) -> list[dict]:
    """Lấy mẫu ngẫu nhiên deterministic. Luôn lấy ít nhất 1 chunk."""
    if not chunks:
        return []

    sample_size = max(1, int(len(chunks) * sample_ratio))
    sample_size = min(sample_size, len(chunks))

    r = random.Random(seed)
    return r.sample(chunks, sample_size)


def create_text_preview(text: str, max_length: int = 150) -> str:
    if not text:
        return ""
    if len(text) <= max_length:
        return text.strip()
    
    # Cắt tại dấu cách
    preview = text[:max_length].rsplit(' ', 1)[0]
    return preview.strip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot check random sample of annotated policy chunks")
    parser.add_argument("--input-file", default="data/chunks/policy_chunks.llm_reviewed.jsonl")
    parser.add_argument("--output-file", default="data/graph/spot_check_report.json")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Read chunks
    chunks = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line_str = line.strip()
            if line_str:
                chunks.append(json.loads(line_str))

    print(f"Loaded {len(chunks)} chunks from {input_path}")

    sampled = sample_policy_chunks(chunks, sample_ratio=args.ratio, seed=args.seed)
    print(f"Sampled {len(sampled)} chunks (ratio={args.ratio}, seed={args.seed})")

    # Build report
    report = []
    llm_count = rule_count = 0

    for chunk in sampled:
        extracted_by = chunk.get("extracted_by", "rule_based_fallback")
        if extracted_by == "llm_annotated":
            llm_count += 1
        else:
            rule_count += 1

        report.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "text_preview": create_text_preview(chunk.get("text", ""), 150),
            "action_tags": chunk.get("action_tags", []),
            "policy_area": chunk.get("policy_area", []),
            "extracted_by": extracted_by,
            "human_verified": None,
            "notes": "",
            "model_used": chunk.get("model_used")
        })

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n=== SPOT CHECK REPORT ===")
    print(f"Total sampled : {len(report)}")
    print(f"LLM annotated : {llm_count}")
    print(f"Rule-based    : {rule_count}")
    print(f"Report saved  : {output_path}")
    print(f"Generated at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()