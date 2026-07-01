#!/usr/bin/env python3
"""
scripts/annotate_new_chunks_offline.py

Batch Ingestion Tier: Self-Healing Annotation Pipeline (v2.8 — Schema Fix +
Catalog-Rotation Resilience).

THAY ĐỔI SO VỚI v2.7.1:

  [FIX QUAN TRỌNG] Bỏ "student_status_tags" khỏi schema LLM.
    v2.7.1 vẫn yêu cầu LLM trả về student_status_tags và GHI ĐÈ giá trị
    rule-based đã có sẵn trong chunk đầu vào. Điều này VI PHẠM nguyên tắc đã
    ghi rõ trong docstring của scripts/annotate_policy_chunks.py:
      "policy_area: CHỈ rule-based xác định (deterministic, auditable)
       student_status_tags: rule-based phủ toàn bộ taxonomy"
    Verify thực tế trên dữ liệu đã chạy: so với rule-based hiện tại,
    39/60 chunk có student_status_tags bị LLM ghi đè bằng MỘT TỪ VỰNG KHÁC
    (vd: domain.json dùng "on_academic_warning" còn rule-based dùng
    "academic_warning"; "on_temporary_leave" vs "on_leave_student";
    "student_with_disability" vs "student_with_special_needs"...).
    Hậu quả: chunk thành công qua LLM và chunk rơi về rule_based_fallback
    (khi hết cycles) sẽ dùng 2 từ vựng khác nhau cho CÙNG MỘT field
    → metadata không đồng nhất, phá khả năng filter/audit theo
    student_status_tags trên toàn corpus.
    → v2.8: KHÔNG hỏi LLM field này nữa. annotated["student_status_tags"]
    luôn được inherit (giữ nguyên) từ chunk đầu vào — vốn đã được set bởi
    scripts/annotate_policy_chunks.py ở bước trước đó, giống cách
    policy_area đã được xử lý đúng từ v2.6/v2.7.
    Tác dụng phụ tích cực: giảm ~15-20% input tokens/request (ít hơn 1 danh
    mục tag phải liệt kê trong system prompt) và giảm hallucination risk.

  [NEW] Thêm "openrouter/free" (auto-router free model của OpenRouter) làm
    model cuối trong FREE_MODELS pool.
    Lý do: catalog free model của OpenRouter đổi liên tục (theo tuần) — các
    model named cụ thể (gpt-oss-120b:free, gpt-oss-20b:free,
    nemotron-nano-9b-v2:free) có thể bị gỡ hoặc đổi tên mà không báo trước.
    "openrouter/free" tự động route sang free model nào đang khả dụng tại
    thời điểm gọi → tăng độ bền (resilience) của tầng free mà KHÔNG cần bật
    paid fallback, đúng yêu cầu "LLM chỉ dùng các bản free".
    Vẫn giữ nguyên cơ chế round-robin theo chunk_position: mỗi chunk thử lần
    lượt qua TẤT CẢ model trong pool (kể cả khi không phải model bắt đầu),
    nên openrouter/free vẫn được dùng như lớp dự phòng cuối cho mọi chunk,
    không chỉ 1/4 số chunk.

  [KEPT] Toàn bộ fix của v2.7.1 giữ nguyên không đổi:
    - max_tokens=2048 (cho reasoning models gpt-oss-20b / nemotron-nano đủ
      budget thinking tokens trước khi sinh JSON).
    - Pool đã loại 2 model luôn 429 (qwen3-next-80b, llama-3.3-70b).
    - Per-model rate limiter, AuthError fail-fast, model_usage_counter để
      phát hiện model "âm thầm" không nhận request nào (silent 429).

KIẾN TRÚC PIPELINE (2 bước, tách biệt rõ ràng):
  Bước 1 — OFFLINE (rule-based, không gọi API, miễn phí, xác định):
      python scripts/annotate_policy_chunks.py \
          --input-file data/chunks/policy_chunks.jsonl \
          --output-file data/chunks/policy_chunks.annotated.jsonl
      → quyết định policy_area + student_status_tags (KHÔNG đổi ở bước 2).

  Bước 2 — LLM (free models, enhance 6 field còn lại):
      python scripts/annotate_new_chunks_offline.py \
          --input-file data/chunks/policy_chunks.annotated.jsonl \
          --output-file data/chunks/policy_chunks_llm_reviewed.jsonl
      → action_tags / risk_tags / procedure_tags / evidence_groups /
        requirement_tags / time_tags được LLM tinh chỉnh lại bằng vocabulary
        chi tiết hơn trong domain.json; policy_area + student_status_tags
        được INHERIT (copy nguyên) từ Bước 1 cho mọi chunk, kể cả chunk rơi
        về rule_based_fallback.

  ⚠️ LƯU Ý CHO BƯỚC BUILD FILE "FINAL" (vd. policy_chunks_final.jsonl dùng để
  tạo embedding/đưa vào vector store): script đó phải đọc từ output của
  Bước 2 (đã LLM-enhance), KHÔNG đọc từ một bản annotated cũ hơn. Khi review
  3 file đã chạy, phát hiện policy_chunks_final.jsonl hiện tại có action_tags/
  evidence_groups/requirement_tags... thô hơn hẳn so với
  policy_chunks_llm_reviewed.jsonl (165/60 field bị lệch) — cho thấy bước
  build final đang trỏ nhầm nguồn. Cần kiểm tra lại script build final đó.
"""

import argparse
import asyncio
from datetime import datetime
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotate_policy_chunks import annotate_chunk


class AuthError(Exception):
    """Lỗi xác thực API key — không thể khắc phục bằng cách đổi model."""
    pass


# ---------------------------------------------------------------------------
# Free model pool (duy nhất — không có paid fallback)
#
# LOẠI KHỎI POOL (luôn 429 trong thực tế, đã verify ở v2.7.1):
#   - qwen/qwen3-next-80b-a3b-instruct:free  → 429 toàn bộ 2 cycles
#   - meta-llama/llama-3.3-70b-instruct:free → 429 toàn bộ 2 cycles
#
# GIỮ LẠI (có traffic thực tế ở v2.7.1):
#   - openai/gpt-oss-120b:free        → 21 chunks cycle 1
#   - openai/gpt-oss-20b:free         → 20 chunks cycle 1
#   - nvidia/nemotron-nano-9b-v2:free → 8 chunks cycle 1
#
# THÊM MỚI (v2.8 — lớp đệm chống catalog rotation):
#   - openrouter/free  → auto-router, tự chọn free model đang khả dụng.
#     Vẫn 100% free, không tốn credit. Đây KHÔNG phải paid fallback.
# ---------------------------------------------------------------------------

FREE_MODELS = [
    "openai/gpt-oss-120b:free",        # 120B MoE, chất lượng cao nhất
    "openai/gpt-oss-20b:free"           #  20B, quota riêng, throughput cao
    
]


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
    Mỗi free model có rate limiter riêng.
    per_model_interval = global_interval × n_free_models
    Với 4 models × 3.0s global = 12s/model (~5 RPM/model).
    """
    def __init__(self, free_models: list[str], global_interval: float):
        n = len(free_models)
        per_model_interval = global_interval * n
        self._limiters: dict[str, _SingleRateLimiter] = {
            model: _SingleRateLimiter(per_model_interval)
            for model in free_models
        }

    async def acquire(self, model: str) -> None:
        limiter = self._limiters.get(model)
        if limiter:
            await limiter.acquire()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_taxonomy(domain_path: Path) -> tuple[
    list[str], list[str], list[str],
    list[str], list[str], list[str], list[str],
]:
    """
    Đọc domain.json, trả về 7 lists (giữ đủ 7 để tương thích domain.json
    hiện tại — student_allowed được đọc nhưng KHÔNG dùng để hỏi LLM, xem
    ghi chú v2.8 ở đầu file):
      action_tags, risk_tags, procedure_tags,
      student_status_tags, evidence_groups, requirement_tags, time_tags
    """
    if not domain_path.exists():
        print(f"[ERROR] domain.json không tìm thấy: {domain_path}", file=sys.stderr)
        sys.exit(1)

    with domain_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    required_keys = [
        "action_tags", "risk_tags", "procedure_tags",
        "evidence_groups", "requirement_tags", "time_tags",
    ]
    for key in required_keys:
        if key not in data or not data[key]:
            print(
                f"[WARN] domain.json thiếu hoặc rỗng key '{key}' "
                f"→ LLM sẽ không extract field này.",
                file=sys.stderr,
            )

    return (
        data.get("action_tags", []),
        data.get("risk_tags", []),
        data.get("procedure_tags", []),
        data.get("student_status_tags", []),  # đọc nhưng không gửi cho LLM
        data.get("evidence_groups", []),
        data.get("requirement_tags", []),
        data.get("time_tags", []),
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
    raise json.JSONDecodeError("Không tìm thấy JSON hợp lệ", content, 0)


# ---------------------------------------------------------------------------
# Tag validation — 6 fields (đã bỏ student_status_tags, xem v2.8 ở đầu file)
# ---------------------------------------------------------------------------

def validate_and_filter_tags(
    action_res: list, risk_res: list, proc_res: list,
    evidence_res: list, req_res: list, time_res: list,
    action_allowed: list, risk_allowed: list, proc_allowed: list,
    evidence_allowed: list, req_allowed: list, time_allowed: list,
    chunk_id: str, model_name: str,
) -> tuple[list, list, list, list, list, list]:
    """Lọc bỏ hallucinated tags (không có trong danh mục cho phép)."""
    allowed_sets = {
        "action": set(action_allowed),
        "risk": set(risk_allowed),
        "procedure": set(proc_allowed),
        "evidence": set(evidence_allowed),
        "requirement": set(req_allowed),
        "time": set(time_allowed),
    }
    raw_map = {
        "action": action_res,
        "risk": risk_res,
        "procedure": proc_res,
        "evidence": evidence_res,
        "requirement": req_res,
        "time": time_res,
    }

    filtered = {k: [t for t in v if t in allowed_sets[k]] for k, v in raw_map.items()}

    all_raw = set(t for v in raw_map.values() for t in v)
    all_filtered = set(t for v in filtered.values() for t in v)
    hallucinated = all_raw - all_filtered
    if hallucinated:
        print(
            f"   [HALLUCINATED] {chunk_id} | {model_name} | "
            f"Tags bị loại: {sorted(hallucinated)}",
            file=sys.stderr,
        )

    return (
        filtered["action"], filtered["risk"], filtered["procedure"],
        filtered["evidence"], filtered["requirement"], filtered["time"],
    )


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
    for f in [
        "policy_area", "student_status_tags", "evidence_groups",
        "requirement_tags", "time_tags",
    ]:
        annotated.setdefault(f, [])
    return annotated


# ---------------------------------------------------------------------------
# Single model attempt
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
      ("auth_error", None)
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
        print(
            f"   [NET_ERR] {chunk_id} | {model_name} | {type(e).__name__}",
            file=sys.stderr,
        )
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
            print(
                f"   [NO_CHOICES] {chunk_id} | {model_name} | Body: {str(data)[:200]}",
                file=sys.stderr,
            )
            return "error", None
        return "success", data
    except Exception as e:
        print(f"   [HTTP_ERR] {chunk_id} | {model_name} | {e}", file=sys.stderr)
        return "error", None


# ---------------------------------------------------------------------------
# Parse + validate response — 6 fields
# ---------------------------------------------------------------------------

def _parse_and_validate(
    data: dict,
    chunk: dict,
    action_allowed: list, risk_allowed: list, proc_allowed: list,
    evidence_allowed: list, req_allowed: list, time_allowed: list,
    chunk_id: str,
    model_name: str,
    current_cycle: int,
) -> dict | None:
    """Parse JSON response và validate 6 tag field. Trả None nếu fail.

    policy_area và student_status_tags KHÔNG được parse từ response —
    luôn giữ nguyên giá trị rule-based đã có trong `chunk` đầu vào
    (xem nguyên tắc thiết kế ghi ở đầu file, mục [FIX QUAN TRỌNG] v2.8).
    """
    try:
        raw = data["choices"][0]["message"]["content"]
        if not raw or not isinstance(raw, str):
            finish_reason = data["choices"][0].get("finish_reason", "unknown")
            raise ValueError(
                f"content rỗng/null (finish_reason={finish_reason}) — "
                f"model reasoning có thể đã hết max_tokens"
            )

        result = extract_json_from_response(raw)

        def _get_list(key: str) -> list:
            val = result.get(key)
            return val if isinstance(val, list) else []

        (
            action_tags, risk_tags, proc_tags,
            evidence_groups, req_tags, time_tags,
        ) = validate_and_filter_tags(
            _get_list("action_tags"),
            _get_list("risk_tags"),
            _get_list("procedure_tags"),
            _get_list("evidence_groups"),
            _get_list("requirement_tags"),
            _get_list("time_tags"),
            action_allowed, risk_allowed, proc_allowed,
            evidence_allowed, req_allowed, time_allowed,
            chunk_id, model_name,
        )

        annotated = dict(chunk)
        annotated.update({
            "action_tags": action_tags,
            "risk_tags": risk_tags,
            "procedure_tags": proc_tags,
            "evidence_groups": evidence_groups,
            "requirement_tags": req_tags,
            "time_tags": time_tags,
            "annotated_at": datetime.now().isoformat(),
            "pipeline_cycle_attempts": current_cycle,
            "model_used": model_name,
            "extracted_by": "llm_annotated",
        })
        # policy_area + student_status_tags: KHÔNG ghi đè — inherit nguyên
        # giá trị rule-based đã có sẵn trong chunk đầu vào.
        annotated.setdefault("policy_area", [])
        annotated.setdefault("student_status_tags", [])
        return annotated

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(
            f"   [PARSE_ERR] {chunk_id} | {model_name} | {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Core annotation coroutine (v2.8 — free-only, 6-field prompt + rotation guard)
# ---------------------------------------------------------------------------

async def annotate_chunk_core(
    client: httpx.AsyncClient,
    chunk: dict[str, Any],
    chunk_position: int,       # position GỐC — không đổi qua các cycles
    action_allowed: list[str],
    risk_allowed: list[str],
    proc_allowed: list[str],
    evidence_allowed: list[str],
    req_allowed: list[str],
    time_allowed: list[str],
    semaphore: asyncio.Semaphore,
    rate_limiter: PerModelRateLimiter,
    api_key: str,
    free_models: list[str],
    current_cycle: int,
    model_usage_counter: Counter,
) -> Tuple[bool, dict[str, Any]]:
    """
    Thử lần lượt free_models theo chunk_position % n_free.
    chunk_position là position gốc → không thay đổi khi retry cycle sau.
    Nếu tất cả models fail → return (False, chunk) để retry.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/hauct131/academic-policy-graph-rag",
        "X-Title": "Academic Policy Graph RAG Pipeline v2.8",
    }

    system_prompt = (
        "Bạn là chuyên gia phân tích quy chế đào tạo đại học Việt Nam.\n\n"
        "NHIỆM VỤ: Phân tích đoạn văn bản quy chế và gán nhãn theo 6 danh mục.\n"
        "(Lưu ý: policy_area và student_status_tags ĐÃ được xác định bằng "
        "rule-based ở bước trước — bạn KHÔNG cần và KHÔNG được gán 2 field "
        "này.)\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. CHỈ trả về một JSON object duy nhất — không có text nào khác, "
        "không có ```json, không có giải thích.\n"
        "2. Dòng đầu tiên PHẢI là { và dòng cuối PHẢI là }.\n"
        "3. Chỉ dùng tag có trong danh mục cho phép bên dưới.\n"
        "4. Trả về mảng rỗng [] nếu không có tag phù hợp cho danh mục đó.\n"
        "5. Phải có đủ 6 keys trong JSON — không được bỏ key nào.\n\n"
        "6. Chỉ gán tag cho những nội dung được điều khoản NÀY trực tiếp quy định, không gán theo chủ đề của toàn bộ văn bản.\n"
        "7. Các điều khoản về phạm vi áp dụng, giải thích từ ngữ, hoặc trách nhiệm đơn vị → trả về action_tags: [], policy_area: [] (mảng rỗng).\n"
        f"ACTION TAGS (hành động sinh viên cần thực hiện):\n"
        f"  {json.dumps(action_allowed, ensure_ascii=False)}\n\n"
        f"RISK TAGS (rủi ro/hậu quả nếu vi phạm):\n"
        f"  {json.dumps(risk_allowed, ensure_ascii=False)}\n\n"
        f"PROCEDURE TAGS (thủ tục/yêu cầu quy trình):\n"
        f"  {json.dumps(proc_allowed, ensure_ascii=False)}\n\n"
        f"EVIDENCE GROUPS (loại hồ sơ/giấy tờ cần nộp):\n"
        f"  {json.dumps(evidence_allowed, ensure_ascii=False)}\n\n"
        f"REQUIREMENT TAGS (điều kiện tiên quyết phải đáp ứng):\n"
        f"  {json.dumps(req_allowed, ensure_ascii=False)}\n\n"
        f"TIME TAGS (mốc thời gian/deadline):\n"
        f"  {json.dumps(time_allowed, ensure_ascii=False)}\n\n"
        "FORMAT BẮT BUỘC (phải có đủ 6 keys):\n"
        '{"action_tags": [], "risk_tags": [], "procedure_tags": [], '
        '"evidence_groups": [], "requirement_tags": [], "time_tags": []}'
    )

    base_payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Phân tích và gán nhãn đoạn quy chế sau:\n"
                    f"\"\"\"\n{chunk.get('text', '')}\n\"\"\"\n\n"
                    "CHỈ trả về JSON object với đủ 6 keys."
                ),
            },
        ],
        "temperature": 0.1,
        # 2048 để reasoning models (gpt-oss-20b, nemotron) có đủ tokens cho
        # thinking chain trước khi sinh JSON output.
        "max_tokens": 2048,
    }

    chunk_id = chunk.get("chunk_id", "unknown")
    n_free = len(free_models)
    start_idx = chunk_position % n_free  # position gốc → phân tán đều, ổn định

    for attempt in range(n_free):
        current_idx = (start_idx + attempt) % n_free
        model_name = free_models[current_idx]

        status, data = await _try_model(
            client, model_name, base_payload, headers,
            semaphore, rate_limiter, attempt, chunk_id,
        )

        if status == "auth_error":
            raise AuthError(
                f"{model_name} trả về 401/403 — kiểm tra OPENROUTER_API_KEY"
            )

        if status == "429":
            backoff = min(10.0 * (2 ** attempt), 60.0)
            print(
                f"   [429] {model_name} | chunk_pos={chunk_position} | "
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
            model_usage_counter[model_name] += 1
            result = _parse_and_validate(
                data, chunk,
                action_allowed, risk_allowed, proc_allowed,
                evidence_allowed, req_allowed, time_allowed,
                chunk_id, model_name, current_cycle,
            )
            if result:
                return True, result
            # Parse error → thử model tiếp theo
            continue

    # Tất cả free models đều fail
    print(
        f"   [FAIL] {chunk_id} | pos={chunk_position} | "
        f"Tất cả {n_free} free models thất bại ở cycle {current_cycle}.",
        file=sys.stderr,
    )
    return False, chunk


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> None:
    (
        action_allowed, risk_allowed, proc_allowed,
        _student_allowed_unused, evidence_allowed, req_allowed, time_allowed,
    ) = load_taxonomy(Path(args.domain_file))

    original_chunks = read_jsonl(Path(args.input_file))

    if not original_chunks:
        print(f"[WARN] Không có dữ liệu: {args.input_file}", file=sys.stderr)
        return

    # Cảnh báo sớm nếu input chưa qua bước rule-based (thiếu policy_area).
    missing_rule_pass = sum(
        1 for c in original_chunks if "policy_area" not in c
    )
    if missing_rule_pass:
        print(
            f"[WARN] {missing_rule_pass}/{len(original_chunks)} chunk KHÔNG "
            f"có field 'policy_area' — input có vẻ CHƯA chạy qua "
            f"scripts/annotate_policy_chunks.py. policy_area và "
            f"student_status_tags của các chunk này sẽ là [] (rỗng) vì "
            f"v2.8 không hỏi LLM 2 field này nữa. Khuyến nghị: chạy rule-based "
            f"pass trước rồi mới chạy script này.",
            file=sys.stderr,
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY chưa set.", file=sys.stderr)
        sys.exit(1)

    n = len(original_chunks)
    n_free = len(FREE_MODELS)
    per_model_interval = args.request_interval * n_free

    print(f"Loaded {n} chunks từ {args.input_file}")
    print(f"Pipeline v2.8 (free-only) | Concurrency: {args.concurrency}")
    print(
        f"Models ({n_free}): {per_model_interval:.0f}s/model "
        f"(~{60/per_model_interval:.1f} RPM/model)"
    )
    for i, m in enumerate(FREE_MODELS):
        print(f"  [{i}] {m}")
    print(
        "Tag schema (6 field, LLM enhance): action, risk, procedure, "
        "evidence_groups, requirement, time"
    )
    print(
        "Tag schema (2 field, rule-based inherit, KHÔNG đổi): policy_area, "
        "student_status_tags"
    )
    print(
        f"Domain: {args.domain_file} | "
        f"action={len(action_allowed)}, risk={len(risk_allowed)}, "
        f"procedure={len(proc_allowed)}, "
        f"evidence={len(evidence_allowed)}, requirement={len(req_allowed)}, "
        f"time={len(time_allowed)}"
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    rate_limiter = PerModelRateLimiter(FREE_MODELS, global_interval=args.request_interval)
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)

    working_queue: list[tuple[int, dict]] = [
        (idx, chunk) for idx, chunk in enumerate(original_chunks)
    ]

    final_results_map: dict[str, dict] = {}
    cycle = 1

    try:
        async with httpx.AsyncClient(limits=limits) as client:
            while working_queue and cycle <= args.max_cycles:
                print(
                    f"\n--- [VÒNG {cycle}/{args.max_cycles}] "
                    f"{len(working_queue)} chunks cần xử lý ---"
                )

                model_usage_counter: Counter = Counter()

                tasks = [
                    annotate_chunk_core(
                        client=client,
                        chunk=chunk,
                        chunk_position=orig_pos,
                        action_allowed=action_allowed,
                        risk_allowed=risk_allowed,
                        proc_allowed=proc_allowed,
                        evidence_allowed=evidence_allowed,
                        req_allowed=req_allowed,
                        time_allowed=time_allowed,
                        semaphore=semaphore,
                        rate_limiter=rate_limiter,
                        api_key=api_key,
                        free_models=FREE_MODELS,
                        current_cycle=cycle,
                        model_usage_counter=model_usage_counter,
                    )
                    for orig_pos, chunk in working_queue
                ]

                current_batch = working_queue
                results = await asyncio.gather(*tasks, return_exceptions=True)
                working_queue = []

                llm_ok = retry_needed = 0
                for item, (orig_pos, original_chunk) in zip(results, current_batch):
                    if isinstance(item, AuthError):
                        raise item
                    if isinstance(item, BaseException):
                        print(
                            f"   [TASK_CRASH] {original_chunk.get('chunk_id', 'unknown')} | "
                            f"{type(item).__name__}: {item}",
                            file=sys.stderr,
                        )
                        working_queue.append((orig_pos, original_chunk))
                        retry_needed += 1
                        continue

                    success, data = item
                    if success:
                        final_results_map[data["chunk_id"]] = data
                        llm_ok += 1
                    else:
                        working_queue.append((orig_pos, data))
                        retry_needed += 1

                print(f"[VÒNG {cycle}] LLM OK: {llm_ok} | Retry: {retry_needed}")
                print(f"  Model usage: {dict(model_usage_counter)}")
                silent = [m for m in FREE_MODELS if model_usage_counter[m] == 0]
                if silent:
                    print(
                        f"  [WARN] Models không được dùng lần nào (có thể đang 429 "
                        f"hoặc đã bị gỡ khỏi catalog): {silent}",
                        file=sys.stderr,
                    )

                if working_queue and cycle < args.max_cycles:
                    print(f"[COOLDOWN] Sleep {args.cooldown}s để free models hồi phục quota...")
                    await asyncio.sleep(float(args.cooldown))

                cycle += 1

    except AuthError as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        sys.exit(1)

    if working_queue:
        n_fallback = len(working_queue)
        print(f"\n[FALLBACK] {n_fallback} chunks → rule-based (hết cycles).")
        for _, chunk in working_queue:
            cid = chunk.get("chunk_id", "unknown")
            final_results_map[cid] = make_rule_based_fallback(chunk, args.max_cycles)

    ordered_output = []
    for c in original_chunks:
        cid = c.get("chunk_id", "unknown")
        result = final_results_map.get(cid) or make_rule_based_fallback(c, args.max_cycles)
        ordered_output.append(result)

    write_jsonl(Path(args.output_file), ordered_output)

    total_llm = sum(1 for c in ordered_output if c.get("extracted_by") == "llm_annotated")
    total_rule = sum(1 for c in ordered_output if c.get("extracted_by") == "rule_based_fallback")
    model_dist = Counter(
        c.get("model_used") for c in ordered_output if c.get("model_used")
    )

    print(f"\n{'='*60}")
    print(f"[HOÀN TẤT] {len(ordered_output)} chunks → {args.output_file}")
    print(f"  LLM annotated   : {total_llm} ({total_llm/len(ordered_output)*100:.1f}%)")
    print(f"  Rule-based      : {total_rule} ({total_rule/len(ordered_output)*100:.1f}%)")
    print(f"  Model distribution:")
    for model, count in model_dist.most_common():
        print(f"    {model}: {count}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotation Pipeline v2.8 — Free-Only, 6-Field LLM Schema "
                     "(policy_area + student_status_tags luôn inherit rule-based)"
    )
    parser.add_argument(
        "--input-file",
        default="data/chunks/policy_chunks.annotated.jsonl",
        help="Input JSONL — PHẢI là output của scripts/annotate_policy_chunks.py "
             "(đã có policy_area + student_status_tags).",
    )
    parser.add_argument(
        "--output-file",
        default="data/chunks/policy_chunks_llm_reviewed.jsonl",
        help="Output JSONL với annotations đã LLM-enhance.",
    )
    parser.add_argument(
        "--domain-file",
        default="domains/ou_academic_policy_v1/domain.json",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Số coroutines song song. Với 4 free models: 4 là tối ưu.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=3.0,
        help="Global interval (s). Per-model = interval × n_models = 12s (~5 RPM/model).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=3,
        help="Số cycles retry tối đa.",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=60,
        help="Sleep giữa các cycles (s). 60s để free models hồi phục quota.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()