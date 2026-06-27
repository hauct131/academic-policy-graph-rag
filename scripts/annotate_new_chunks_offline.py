#!/usr/bin/env python3
"""
scripts/annotate_new_chunks_offline.py

Batch Ingestion Tier: Self-Healing High-Performance Concurrent Annotation Pipeline (v2.1).
Natively tracks failed chunks, applies strict defensive JSON validation, and automatically 
re-queues failed items into subsequent retry cycles. Fixed argparse and list mapping bugs.
"""

import argparse
import asyncio
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotate_policy_chunks import annotate_chunk


def load_taxonomy(domain_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Load taxonomy arrays from domain.json."""
    if not domain_path.exists():
        print(f"[ERROR] domain.json not found at {domain_path}", file=sys.stderr)
        sys.exit(1)

    with domain_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("action_tags", []), data.get("risk_tags", []), data.get("procedure_tags", [])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file into a list of dictionaries."""
    if not path.exists():
        return []
    chunks = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line_str = line.strip()
            if line_str:
                chunks.append(json.loads(line_str))
    return chunks


def write_jsonl(path: Path, chunks: list[dict[str, Any]]) -> None:
    """Write list of dictionaries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def validate_tags_against_domain(tags: list[str], domain_config: dict) -> list[str]:
    """
    Validate and filter a list of tags against the allowed tags in domain_config.
    Allowed tags are the union of action_tags, risk_tags, and procedure_tags.
    """
    allowed = set()
    if isinstance(domain_config, dict):
        allowed.update(domain_config.get("action_tags", []))
        allowed.update(domain_config.get("risk_tags", []))
        allowed.update(domain_config.get("procedure_tags", []))
    return [t for t in tags if t in allowed]


async def annotate_chunk_core(
    client: httpx.AsyncClient,
    chunk: dict[str, Any],
    idx: int,
    action_allowed: list[str],
    risk_allowed: list[str],
    proc_allowed: list[str],
    semaphore: asyncio.Semaphore,
    api_key: str,
    models_pool: list[str],
    global_state: dict[str, int],
    current_cycle: int,
) -> Tuple[bool, dict[str, Any]]:
    """
    Xử lý lõi của một chunk với cơ chế phòng vệ dữ liệu nghiêm ngặt.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/hauct131/academic-policy-graph-rag",
        "X-Title": "Academic Policy Graph RAG Pipeline v2.1"
    }

    system_prompt = (
        "Bạn là một chuyên gia phân tích Quy chế Đào tạo Đại học chuyên sâu.\n"
        "Nhiệm vụ của bạn là đọc kỹ nội dung phân đoạn quy chế được cung cấp và gán các nhãn phân loại một cách chính xác, khách quan.\n\n"
        f"Danh mục Action Tags cho phép: {json.dumps(action_allowed, ensure_ascii=False)}\n"
        f"Danh mục Risk Tags cho phép: {json.dumps(risk_allowed, ensure_ascii=False)}\n"
        f"Danh mục Procedure Tags cho phép: {json.dumps(proc_allowed, ensure_ascii=False)}\n\n"
        "Quy tắc nghiêm ngặt:\n"
        "1. BẮT BUỘC chỉ sử dụng các từ khóa có trong danh mục cho phép ở trên. Tuyệt đối không tự ý phát minh tag mới.\n"
        "2. Nếu phân đoạn không chứa nội dung phù hợp với danh mục nào, hãy trả về mảng rỗng [] cho trường đó.\n"
        "3. Bạn PHẢI trả về một đối tượng JSON cấu trúc thuần túy, không có văn bản giải thích bao quanh với dạng:\n"
        "{\n"
        '  "action_tags": ["tag_hop_le_1"],\n'
        '  "risk_tags": [],\n'
        '  "procedure_tags": ["tag_hop_le_2"]\n'
        "}"
    )

    retries_per_chunk = 3
    
    async with semaphore:
        for attempt in range(retries_per_chunk):
            current_idx = (global_state["active_idx"] + attempt) % len(models_pool)
            model_name = models_pool[current_idx]
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Nội dung phân đoạn cần gán nhãn:\n\"\"\"\n{chunk.get('text', '')}\n\"\"\""}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }

            try:
                timeout_val = 30.0 + (attempt * 10.0)
                response = await client.post(url, headers=headers, json=payload, timeout=timeout_val)
                
                if response.status_code == 429:
                    global_state["active_idx"] = (current_idx + 1) % len(models_pool)
                    await asyncio.sleep(2.0 + attempt)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                result = json.loads(content)
                action_res = result.get("action_tags") if isinstance(result.get("action_tags"), list) else []
                risk_res = result.get("risk_tags") if isinstance(result.get("risk_tags"), list) else []
                proc_res = result.get("procedure_tags") if isinstance(result.get("procedure_tags"), list) else []

                # Validate tags bằng allow-list
                all_llm_tags = action_res + risk_res + proc_res
                domain_config = {
                    "action_tags": action_allowed,
                    "risk_tags": risk_allowed,
                    "procedure_tags": proc_allowed
                }
                validated_tags = validate_tags_against_domain(all_llm_tags, domain_config)
                if len(validated_tags) < len(all_llm_tags):
                    raise ValueError("Phát hiện tag lạ không nằm trong allow-list")

                annotated = dict(chunk)
                annotated["action_tags"] = action_res
                annotated["risk_tags"] = risk_res
                annotated["procedure_tags"] = proc_res

                annotated["annotated_at"] = datetime.now().isoformat()
                annotated["pipeline_cycle_attempts"] = current_cycle
                annotated["model_used"] = model_name
                annotated["extracted_by"] = "llm_first_pass"

                for field in ["policy_area", "student_status_tags", "evidence_groups", "requirement_tags", "time_tags"]:
                    if field not in annotated: annotated[field] = []

                global_state["active_idx"] = current_idx
                return True, annotated

            except Exception as e:
                print(
                    f"   [WARN] Chunk {chunk.get('chunk_id')} | model={model_name} | "
                    f"thất bại ({type(e).__name__}: {e}). Fallback ngay lập tức sang rule-based.",
                    file=sys.stderr
                )
                global_state["active_idx"] = (current_idx + 1) % len(models_pool)
                
                # Fallback sang hàm rule-based
                annotated = annotate_chunk(chunk)
                annotated["annotated_at"] = datetime.now().isoformat()
                annotated["pipeline_cycle_attempts"] = current_cycle
                annotated["model_used"] = None
                annotated["extracted_by"] = "rule_based_fallback"

                for field in ["policy_area", "student_status_tags", "evidence_groups", "requirement_tags", "time_tags"]:
                    if field not in annotated: annotated[field] = []

                return True, annotated

        # Trường hợp tất cả các lần thử đều thất bại (ví dụ đều 429)
        annotated = annotate_chunk(chunk)
        annotated["annotated_at"] = datetime.now().isoformat()
        annotated["pipeline_cycle_attempts"] = current_cycle
        annotated["model_used"] = None
        annotated["extracted_by"] = "rule_based_fallback"

        for field in ["policy_area", "student_status_tags", "evidence_groups", "requirement_tags", "time_tags"]:
            if field not in annotated: annotated[field] = []

        return True, annotated


async def async_main(args: argparse.Namespace) -> None:
    action_allowed, risk_allowed, proc_allowed = load_taxonomy(Path(args.domain_file))
    original_chunks = read_jsonl(Path(args.input_file))
    
    if not original_chunks:
        print(f"[WARN] Không tìm thấy dữ liệu thô trong file {args.input_file}", file=sys.stderr)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] Biến môi trường OPENROUTER_API_KEY không tồn tại.", file=sys.stderr)
        sys.exit(1)

    models_pool = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b:free",
        "nvidia/nemotron-nano-9b-v2:free"
    ]

    print(f"Loaded {len(original_chunks)} chunks từ {args.input_file}")
    print(f"Khởi động Pipeline tự phục hồi v2.1. Concurrency: {args.concurrency}")

    global_state = {"active_idx": 0}
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_keepalive_connections=15, max_connections=30)
    
    final_results_map = {}
    working_queue = list(original_chunks)
    
    cycle = 1

    async with httpx.AsyncClient(limits=limits) as client:
        while working_queue and cycle <= args.max_cycles:
            print(f"\n--- [VÒNG QUÉT LỖI {cycle}/{args.max_cycles}] Đang xử lý {len(working_queue)} chunks ---")
            
            tasks = [
                annotate_chunk_core(
                    client=client,
                    chunk=chunk,
                    idx=idx,
                    action_allowed=action_allowed,
                    risk_allowed=risk_allowed,
                    proc_allowed=proc_allowed,
                    semaphore=semaphore,
                    api_key=api_key,
                    models_pool=models_pool,
                    global_state=global_state,
                    current_cycle=cycle,
                )
                for idx, chunk in enumerate(working_queue)
            ]
            
            cycle_results = await asyncio.gather(*tasks)
            working_queue = []
            
            for success, data in cycle_results:
                if success:
                    final_results_map[data["chunk_id"]] = data
                else:
                    working_queue.append(data)
            
            if working_queue:
                print(f"[FAIL] Có {len(working_queue)} chunk tạm thời thất bại ở vòng này.")
                for fc in working_queue:
                    print(f"   -> Đánh dấu lỗi cần chạy lại: {fc.get('chunk_id')}")
                
                if cycle < args.max_cycles:
                    print(f"[COOLDOWN] Ép tiến trình nghỉ {args.cooldown} giây xả hàng chờ API...")
                    await asyncio.sleep(float(args.cooldown))
            
            cycle += 1

    if working_queue:
        print(f"\n[CRITICAL] Còn {len(working_queue)} chunk thất bại hoàn toàn sau {args.max_cycles} vòng quét. Fallback sang rule-based.")
        for chunk in working_queue:
            annotated = annotate_chunk(chunk)
            annotated["annotated_at"] = datetime.now().isoformat()
            annotated["pipeline_cycle_attempts"] = args.max_cycles
            annotated["model_used"] = None
            annotated["extracted_by"] = "rule_based_fallback"
            for field in ["policy_area", "student_status_tags", "evidence_groups", "requirement_tags", "time_tags"]:
                if field not in annotated: annotated[field] = []
            final_results_map[chunk["chunk_id"]] = annotated

    # Ánh xạ kết quả chính xác theo đúng thứ tự file gốc ban đầu
    ordered_output = [final_results_map[c["chunk_id"]] for c in original_chunks]

    write_jsonl(Path(args.output_file), ordered_output)
    print(f"\n[XONG TRIỆT ĐỂ] Gán nhãn bọc thép hoàn tất {len(ordered_output)} chunks.")
    print(f"Dữ liệu xuất xưởng an toàn tại: {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="High-Performance Concurrent Stateful Model-Rotation Pipeline")
    parser.add_argument("--input-file", default="data/chunks/policy_chunks.jsonl")
    parser.add_argument("--output-file", default="data/chunks/policy_chunks.llm_reviewed.jsonl")
    parser.add_argument("--domain-file", default="domains/ou_academic_policy_v1/domain.json")
    parser.add_argument("--concurrency", type=int, default=10, help="Số lượng tác vụ chạy song song (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=4, help="Số vòng quét lỗi tối đa (default: 4)")
    parser.add_argument("--cooldown", type=int, default=10, help="Thời gian làm mát giây giữa vòng lặp lỗi (default: 10)")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free")

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()