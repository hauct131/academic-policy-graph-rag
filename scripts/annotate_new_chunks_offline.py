#!/usr/bin/env python3
"""
scripts/annotate_new_chunks_offline.py

Batch Ingestion Tier: Self-Healing Annotation Pipeline (v2.6 — Tiered Model Strategy).

THAY ĐỔI SO VỚI v2.5:
  [STRATEGY] Hai tầng model rõ ràng:
    Tầng 1 (FREE_MODELS): 5 models free, thử song song theo chunk_position
    Tầng 2 (PAID_FALLBACK): 1 model paid rẻ nhất, chỉ dùng khi tầng 1 cạn quota

    Chi phí paid fallback: 28 chunks × 1.5K tokens × $0.10/1M ≈ $0.004
    Với $10 credit → paid fallback chạy thoải mái.

  [FIX] Mở rộng free models pool từ 3 lên 5:
    Thêm gpt-oss-20b và nemotron-nano-9b-v2 (đã thấy trong activity dashboard).
    Mỗi model nhận ~12 chunks thay vì 20 → ít khả năng cạn quota hơn.

  [FIX] Per-model rate limiter tự điều chỉnh theo pool size:
    5 models × 3.0s global = 15s per model = ~4 RPM/model
    Thấp hơn đáng kể so với bất kỳ provider limit nào.

  [NEW] --no-paid-fallback flag để chạy free-only khi cần.
  [NEW] Log tổng chi phí ước tính khi dùng paid fallback.
"""

import argparse
import asyncio
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotate_policy_chunks import annotate_chunk


class AuthError(Exception):
    """Lỗi xác thực API key — không thể khắc phục bằng cách đổi model."""
    pass


# ---------------------------------------------------------------------------
# Model pools — hai tầng rõ ràng
# ---------------------------------------------------------------------------

# Tầng 1: Free models. Thêm gpt-oss-20b và nemotron chưa bị cạn quota.
FREE_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",  # ~80B MoE, multilingual
    "openai/gpt-oss-120b:free",               # 120B MoE
    "meta-llama/llama-3.3-70b-instruct:free", # 70B, stable
    "openai/gpt-oss-20b:free",                # 20B, nhẹ hơn → quota riêng
    "nvidia/nemotron-nano-9b-v2:free",        # 9B, quota riêng hoàn toàn
]

# Tầng 2: Paid fallback. Chỉ dùng khi tất cả free models 429.
# meta-llama/llama-3.1-8b-instruct: $0.02-0.05/1M tokens — rẻ nhất đáng tin cậy.
PAID_FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",  # paid, ~$0.02/1M, không rate limit
]

# Ước tính chi phí: 1 chunk ~ 1500 tokens input + 100 tokens output = 1600 tokens
# $0.02/1M × 1600 tokens = $0.000032/chunk → 28 chunks = $0.0009 ≈ $0.001
PAID_COST_PER_1M_TOKENS = 0.02
TOKENS_PER_CHUNK_EST = 1600


# ---------------------------------------------------------------------------
# PerModelRateLimiter
# ---------------------------------------------------------------------------

class _SingleRateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call_time: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_call_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_time = time.monotonic()


class PerModelRateLimiter:
    """
    Mỗi model có rate limiter riêng.
    per_model_interval = global_interval × n_free_models
    → tổng throughput ≈ 20 RPM, mỗi model ≈ 4 RPM (với 5 models, 3.0s global).

    Paid models KHÔNG bị rate limit → không cần limiter riêng.
    """
    def __init__(self, free_models: list[str], global_interval: float):
        n = len(free_models)
        per_model_interval = global_interval * n
        self._limiters: dict[str, _SingleRateLimiter] = {
            model: _SingleRateLimiter(per_model_interval)
            for model in free_models
        }
        self._global_interval = global_interval
        self._paid_limiter = _SingleRateLimiter(global_interval)

    async def acquire(self, model: str) -> None:
        limiter = self._limiters.get(model)
        if limiter:
            await limiter.acquire()
        else:
            # Paid model: dùng global interval để không spam
            await self._paid_limiter.acquire()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_taxonomy(domain_path: Path) -> tuple[list[str], list[str], list[str]]:
    if not domain_path.exists():
        print(f"[ERROR] domain.json không tìm thấy: {domain_path}", file=sys.stderr)
        sys.exit(1)
    with domain_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return (
        data.get("action_tags", []),
        data.get("risk_tags", []),
        data.get("procedure_tags", []),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    chunks = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                chunks.append(json.loads(s))
    return chunks


def write_jsonl(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# JSON parsing defensive
# ---------------------------------------------------------------------------

def extract_json_from_response(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        content = "\n".join(lines[1:end]).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    s, e = content.find("{"), content.rfind("}")
    if s != -1 and e > s:
        return json.loads(content[s : e + 1])
    raise json.JSONDecodeError("Không tìm thấy JSON", content, 0)


# ---------------------------------------------------------------------------
# Tag validation
# ---------------------------------------------------------------------------

def validate_and_filter_tags(
    action_res, risk_res, proc_res,
    action_allowed, risk_allowed, proc_allowed,
    chunk_id, model_name,
):
    fa = [t for t in action_res if t in set(action_allowed)]
    fr = [t for t in risk_res if t in set(risk_allowed)]
    fp = [t for t in proc_res if t in set(proc_allowed)]
    hallucinated = set(action_res + risk_res + proc_res) - set(fa + fr + fp)
    if hallucinated:
        print(
            f"   [HALLUCINATED] {chunk_id} | {model_name} | "
            f"Tags bị loại: {sorted(hallucinated)}",
            file=sys.stderr,
        )
    return fa, fr, fp


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def make_rule_based_fallback(chunk: dict, current_cycle: int) -> dict:
    annotated = annotate_chunk(chunk)
    annotated.update({
        "annotated_at": datetime.now().isoformat(),
        "pipeline_cycle_attempts": current_cycle,
        "model_used": None,
        "extracted_by": "rule_based_fallback",
    })
    for f in ["policy_area", "student_status_tags", "evidence_groups",
              "requirement_tags", "time_tags"]:
        annotated.setdefault(f, [])
    return annotated


# ---------------------------------------------------------------------------
# Single model attempt (dùng chung cho free và paid)
# ---------------------------------------------------------------------------

async def _try_model(
    client: httpx.AsyncClient,
    model_name: str,
    payload: dict,
    headers: dict,
    semaphore: asyncio.Semaphore,
    rate_limiter: PerModelRateLimiter,
    attempt: int,
    chunk_id: str,
) -> tuple[str, Any]:
    """
    Thử gọi 1 model. Trả về (status, data):
      ("success", response_json)
      ("429", None)
      ("400", None)
      ("5xx", None)
      ("error", None)
    """
    try:
        async with semaphore:
            await rate_limiter.acquire(model_name)
            timeout = 35.0 + attempt * 5.0
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={**payload, "model": model_name},
                timeout=timeout,
            )
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        print(f"   [NET_ERR] {chunk_id} | {model_name} | {type(e).__name__}", file=sys.stderr)
        return "error", None
    except Exception as e:
        print(f"   [ERR] {chunk_id} | {model_name} | {e}", file=sys.stderr)
        return "error", None

    if response.status_code == 429:
        return "429", None
    if response.status_code in (401, 403):
        return "auth_error", None
    if response.status_code == 400:
        return "400", None
    if response.status_code >= 500:
        return "5xx", None

    try:
        response.raise_for_status()
        data = response.json()
        if "choices" not in data or not data["choices"]:
            body = str(data)[:200]
            print(
                f"   [NO_CHOICES] {chunk_id} | {model_name} | Body: {body}",
                file=sys.stderr,
            )
            return "error", None
        return "success", data
    except Exception as e:
        print(f"   [HTTP_ERR] {chunk_id} | {model_name} | {e}", file=sys.stderr)
        return "error", None


# ---------------------------------------------------------------------------
# Core annotation coroutine (v2.6 — tiered)
# ---------------------------------------------------------------------------

async def annotate_chunk_core(
    client: httpx.AsyncClient,
    chunk: dict[str, Any],
    chunk_position: int,
    action_allowed: list[str],
    risk_allowed: list[str],
    proc_allowed: list[str],
    semaphore: asyncio.Semaphore,
    rate_limiter: PerModelRateLimiter,
    api_key: str,
    free_models: list[str],
    paid_fallback_models: list[str],
    use_paid_fallback: bool,
    current_cycle: int,
    paid_chunks_counter: list,  # [count] mutable counter để track chi phí
) -> Tuple[bool, dict[str, Any]]:
    """
    TẦNG 1 — Thử free_models theo chunk_position % n_free:
      chunk_pos=0 → bắt đầu từ free_models[0]
      chunk_pos=1 → bắt đầu từ free_models[1]
      ...
      Mỗi model bị 429 → backoff ngoài semaphore → thử tiếp.

    TẦNG 2 — Nếu tất cả free models fail → thử paid_fallback_models:
      Paid models không rate limit → thường thành công ngay.
      Chi phí ước tính được log ra.

    Nếu cả paid cũng fail → return (False, chunk) để retry cycle sau.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/hauct131/academic-policy-graph-rag",
        "X-Title": "Academic Policy Graph RAG Pipeline v2.6",
    }

    system_prompt = (
        "Bạn là chuyên gia phân tích quy chế đào tạo đại học Việt Nam.\n\n"
        "NHIỆM VỤ: Phân tích đoạn văn bản quy chế và gán nhãn phân loại.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. CHỈ trả về một JSON object duy nhất — không có text nào khác, "
        "không có ```json, không có giải thích.\n"
        "2. Dòng đầu tiên PHẢI là dấu { và dòng cuối PHẢI là dấu }.\n"
        "3. Chỉ dùng tag có trong danh mục cho phép bên dưới.\n"
        "4. Trả về mảng rỗng [] nếu không có tag phù hợp.\n\n"
        f"ACTION TAGS: {json.dumps(action_allowed, ensure_ascii=False)}\n"
        f"RISK TAGS: {json.dumps(risk_allowed, ensure_ascii=False)}\n"
        f"PROCEDURE TAGS: {json.dumps(proc_allowed, ensure_ascii=False)}\n\n"
        'FORMAT: {"action_tags": ["tag1"], "risk_tags": [], "procedure_tags": ["tag2"]}'
    )

    base_payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Phân tích và gán nhãn:\n\"\"\"\n{chunk.get('text', '')}\n\"\"\"\n\n"
                    "CHỈ trả về JSON object."
                ),
            },
        ],
        "temperature": 0.1,
        "max_tokens": 768,
    }

    chunk_id = chunk.get("chunk_id", "unknown")
    n_free = len(free_models)
    start_idx = chunk_position % n_free

    # -----------------------------------------------------------------------
    # TẦNG 1: Free models
    # -----------------------------------------------------------------------
    for attempt in range(n_free):
        current_idx = (start_idx + attempt) % n_free
        model_name = free_models[current_idx]

        status, data = await _try_model(
            client, model_name, base_payload, headers,
            semaphore, rate_limiter, attempt, chunk_id,
        )

        if status == "auth_error":
            raise AuthError(f"{model_name} trả về 401/403 — kiểm tra OPENROUTER_API_KEY")

        if status == "429":
            backoff = min(10.0 * (2 ** attempt), 60.0)
            print(
                f"   [429/FREE] {model_name} | chunk_pos={chunk_position} | "
                f"Backoff {backoff:.0f}s...",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff)
            continue

        if status in ("400", "5xx", "error"):
            if status == "5xx":
                await asyncio.sleep(2.0)
            continue

        if status == "success":
            result = _parse_and_validate(
                data, chunk, action_allowed, risk_allowed, proc_allowed,
                chunk_id, model_name, current_cycle,
            )
            if result:
                return True, result
            # Parse error → thử model tiếp
            continue

    # -----------------------------------------------------------------------
    # TẦNG 2: Paid fallback (chỉ khi tất cả free models đều 429/fail)
    # -----------------------------------------------------------------------
    if use_paid_fallback and paid_fallback_models:
        for attempt, model_name in enumerate(paid_fallback_models):
            print(
                f"   [PAID] {chunk_id} | Thử paid fallback: {model_name}...",
                file=sys.stderr,
            )
            status, data = await _try_model(
                client, model_name, base_payload, headers,
                semaphore, rate_limiter, attempt, chunk_id,
            )

            if status == "auth_error":
                raise AuthError(f"{model_name} trả về 401/403 — kiểm tra OPENROUTER_API_KEY")

            if status == "success":
                result = _parse_and_validate(
                    data, chunk, action_allowed, risk_allowed, proc_allowed,
                    chunk_id, model_name, current_cycle,
                )
                if result:
                    result["extracted_by"] = "llm_annotated_paid"
                    paid_chunks_counter[0] += 1
                    return True, result

            if status == "429":
                await asyncio.sleep(5.0)
                continue

    # Tất cả fail → retry cycle sau
    print(
        f"   [FAIL] {chunk_id} | Tất cả models thất bại ở cycle {current_cycle}.",
        file=sys.stderr,
    )
    return False, chunk


def _parse_and_validate(
    data, chunk, action_allowed, risk_allowed, proc_allowed,
    chunk_id, model_name, current_cycle,
) -> dict | None:
    """Parse response JSON và validate tags. Trả None nếu parse fail."""
    try:
        raw = data["choices"][0]["message"]["content"]
        if not raw or not isinstance(raw, str):
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            raise ValueError(
                f"content rỗng/null (finish_reason={finish_reason}) — model "
                f"reasoning có thể đã hết max_tokens trước khi sinh nội dung"
            )
        result = extract_json_from_response(raw)

        action_res = result.get("action_tags") if isinstance(result.get("action_tags"), list) else []
        risk_res = result.get("risk_tags") if isinstance(result.get("risk_tags"), list) else []
        proc_res = result.get("procedure_tags") if isinstance(result.get("procedure_tags"), list) else []

        action_res, risk_res, proc_res = validate_and_filter_tags(
            action_res, risk_res, proc_res,
            action_allowed, risk_allowed, proc_allowed,
            chunk_id, model_name,
        )

        annotated = dict(chunk)
        annotated.update({
            "action_tags": action_res,
            "risk_tags": risk_res,
            "procedure_tags": proc_res,
            "annotated_at": datetime.now().isoformat(),
            "pipeline_cycle_attempts": current_cycle,
            "model_used": model_name,
            "extracted_by": "llm_annotated",
        })
        for f in ["policy_area", "student_status_tags", "evidence_groups",
                  "requirement_tags", "time_tags"]:
            annotated.setdefault(f, [])
        return annotated

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(
            f"   [PARSE_ERR] {chunk_id} | {model_name} | {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    action_allowed, risk_allowed, proc_allowed = load_taxonomy(Path(args.domain_file))
    original_chunks = read_jsonl(Path(args.input_file))

    if not original_chunks:
        print(f"[WARN] Không có dữ liệu: {args.input_file}", file=sys.stderr)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY chưa set.", file=sys.stderr)
        sys.exit(1)

    use_paid = not args.no_paid_fallback
    n = len(original_chunks)
    n_free = len(FREE_MODELS)
    per_model_interval = args.request_interval * n_free
    est_paid_cost = n * TOKENS_PER_CHUNK_EST / 1_000_000 * PAID_COST_PER_1M_TOKENS

    print(f"Loaded {n} chunks từ {args.input_file}")
    print(f"Pipeline v2.6 | Concurrency: {args.concurrency}")
    print(f"Tầng 1 (Free): {n_free} models | interval: {per_model_interval:.0f}s/model (~{60/per_model_interval:.1f} RPM/model)")
    print(f"Tầng 2 (Paid): {'BẬT' if use_paid else 'TẮT'} | {PAID_FALLBACK_MODELS}")
    if use_paid:
        print(f"  Chi phí paid fallback tối đa (nếu tất cả chunks dùng paid): ~${est_paid_cost:.4f}")
    print(f"Models free: {FREE_MODELS}")

    semaphore = asyncio.Semaphore(args.concurrency)
    rate_limiter = PerModelRateLimiter(FREE_MODELS, global_interval=args.request_interval)
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)

    paid_chunks_counter = [0]  # mutable để track từ coroutines
    final_results_map: dict[str, dict] = {}
    working_queue = list(original_chunks)
    cycle = 1

    try:
        async with httpx.AsyncClient(limits=limits) as client:
            while working_queue and cycle <= args.max_cycles:
                print(f"\n--- [VÒNG {cycle}/{args.max_cycles}] {len(working_queue)} chunks ---")

                tasks = [
                    annotate_chunk_core(
                        client=client,
                        chunk=chunk,
                        chunk_position=idx,
                        action_allowed=action_allowed,
                        risk_allowed=risk_allowed,
                        proc_allowed=proc_allowed,
                        semaphore=semaphore,
                        rate_limiter=rate_limiter,
                        api_key=api_key,
                        free_models=FREE_MODELS,
                        paid_fallback_models=PAID_FALLBACK_MODELS,
                        use_paid_fallback=use_paid,
                        current_cycle=cycle,
                        paid_chunks_counter=paid_chunks_counter,
                    )
                    for idx, chunk in enumerate(working_queue)
                ]

                current_batch = working_queue
                results = await asyncio.gather(*tasks, return_exceptions=True)
                working_queue = []

                llm_free = llm_paid = retry_needed = 0
                for item, original_chunk in zip(results, current_batch):
                    if isinstance(item, AuthError):
                        raise item
                    if isinstance(item, BaseException):
                        print(
                            f"   [TASK_CRASH] {original_chunk.get('chunk_id', 'unknown')} | "
                            f"{type(item).__name__}: {item}",
                            file=sys.stderr
                        )
                        working_queue.append(original_chunk)
                        retry_needed += 1
                        continue
                    success, data = item
                    if success:
                        final_results_map[data["chunk_id"]] = data
                        if data.get("extracted_by") == "llm_annotated_paid":
                            llm_paid += 1
                        else:
                            llm_free += 1
                    else:
                        working_queue.append(data)
                        retry_needed += 1

                print(
                    f"[VÒNG {cycle}] Free: {llm_free} | Paid: {llm_paid} | "
                    f"Cần retry: {retry_needed}"
                )

                if working_queue and cycle < args.max_cycles:
                    print(f"[COOLDOWN] Sleep {args.cooldown}s...")
                    await asyncio.sleep(float(args.cooldown))

                cycle += 1
    except AuthError as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        sys.exit(1)

    # Final fallback: rule-based cho chunks vẫn thất bại
    if working_queue:
        print(f"\n[FALLBACK] {len(working_queue)} chunks → rule-based.")
        for chunk in working_queue:
            cid = chunk.get("chunk_id", "unknown")
            final_results_map[cid] = make_rule_based_fallback(chunk, args.max_cycles)

    ordered_output = []
    for c in original_chunks:
        cid = c.get("chunk_id", "unknown")
        result = final_results_map.get(cid) or make_rule_based_fallback(c, args.max_cycles)
        ordered_output.append(result)

    write_jsonl(Path(args.output_file), ordered_output)

    total_llm_free = sum(1 for c in ordered_output if c.get("extracted_by") == "llm_annotated")
    total_llm_paid = sum(1 for c in ordered_output if c.get("extracted_by") == "llm_annotated_paid")
    total_rule = sum(1 for c in ordered_output if c.get("extracted_by") == "rule_based_fallback")
    actual_paid_cost = paid_chunks_counter[0] * TOKENS_PER_CHUNK_EST / 1_000_000 * PAID_COST_PER_1M_TOKENS

    print(f"\n[HOÀN TẤT] {len(ordered_output)} chunks.")
    print(f"  LLM free annotated : {total_llm_free} ({total_llm_free/len(ordered_output)*100:.1f}%)")
    print(f"  LLM paid annotated : {total_llm_paid} ({total_llm_paid/len(ordered_output)*100:.1f}%)")
    print(f"  Rule-based fallback: {total_rule} ({total_rule/len(ordered_output)*100:.1f}%)")
    if total_llm_paid > 0:
        print(f"  Chi phí paid ước tính: ~${actual_paid_cost:.5f}")
    print(f"  Output: {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotation Pipeline v2.6 — Tiered Free+Paid Strategy"
    )
    parser.add_argument("--input-file", default="data/chunks/policy_chunks.jsonl")
    parser.add_argument("--output-file", default="data/chunks/policy_chunks.llm_reviewed.jsonl")
    parser.add_argument("--domain-file", default="domains/ou_academic_policy_v1/domain.json")
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Số coroutines song song. Với 5 free models: concurrency=5 tối ưu.",
    )
    parser.add_argument(
        "--request-interval", type=float, default=3.0,
        help="Global interval (s). Per-model = interval × n_free_models.",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=2,
        help="Số cycles retry. Với paid fallback bật: 2 là đủ.",
    )
    parser.add_argument(
        "--cooldown", type=int, default=60,
        help="Sleep giữa cycles (s). Tăng lên 60s để free models hồi phục quota.",
    )
    parser.add_argument(
        "--no-paid-fallback", action="store_true", default=False,
        help="Tắt paid fallback (mặc định: paid fallback BẬT khi free models cạn quota)"
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()