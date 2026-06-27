#!/usr/bin/env python3
"""
scripts/generate_chunk_embeddings.py

Batch Ingestion Tier: Hybrid Vector Embedding Generator (v1.1).
Leverages NVIDIA Llama Nemotron Embed VL 1B V2 (FREE) via OpenRouter 
to deliver production-grade dense vectors without burning wallet balance.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, List

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.annotate_new_chunks_offline import read_jsonl, write_jsonl

HAS_LOCAL_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_LOCAL_TRANSFORMERS = True
except ImportError:
    pass


async def get_nvidia_nemotron_embedding(client: httpx.AsyncClient, text: str, api_key: str) -> List[float]:
    """Gọi API OpenRouter sử dụng mô hình Embedding miễn phí của NVIDIA"""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/hauct131/academic-policy-graph-rag",
        "X-Title": "Academic Policy Graph RAG Embedding Tier"
    }
    payload = {
        # SỬ DỤNG ENDPOINT ĐÚNG CHUẨN NVIDIA EMBED FREE
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "input": text
    }
    
    try:
        response = await client.post(url, headers=headers, json=payload, timeout=40.0)
        
        if response.status_code == 429:
            raise httpx.HTTPStatusError("429 Rate Limit", request=response.request, response=response)
            
        response.raise_for_status()
        data = response.json()
        
        # OpenRouter cấu trúc dữ liệu trả về theo chuẩn OpenAI: data[0].embedding
        return data["data"][0]["embedding"]
    except Exception as e:
        raise e


async def process_embeddings_pipeline(args: argparse.Namespace) -> None:
    chunks = read_jsonl(Path(args.input_file))
    if not chunks:
        print(f"[ERROR] Không có dữ liệu trong file đầu vào: {args.input_file}", file=sys.stderr)
        return

    print(f"Loaded {len(chunks)} chunks từ file staging.")
    
    if HAS_LOCAL_TRANSFORMERS and not args.force_cloud:
        print("[ENGINE] Kích hoạt Engine Offline bằng mô hình BAAI/bge-m3...")
        model = SentenceTransformer("BAAI/bge-m3")
        for idx, chunk in enumerate(chunks):
            print(f"-> [{idx + 1}/{len(chunks)}] Sinh vector (Local): {chunk['chunk_id']}...")
            embedding = model.encode(chunk.get("text", ""), normalize_embeddings=True)
            chunk["embedding"] = embedding.tolist()
            
    else:
        print("[ENGINE] Kích hoạt Cloud Engine: [NVIDIA: Llama Nemotron Embed VL 1B V2 (free)]")
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            print("[ERROR] OPENROUTER_API_KEY trống. Hãy export biến môi trường.", file=sys.stderr)
            sys.exit(1)
            
        # Kiểm soát 4 luồng đồng thời để Slides past tường lửa OpenRouter thông suốt
        semaphore = asyncio.Semaphore(4)
        
        async def worker(chunk_data: dict, idx_info: int, http_client: httpx.AsyncClient):
            async with semaphore:
                print(f"-> [{idx_info + 1}/{len(chunks)}] Đang nạp Vector qua NVIDIA Embed: {chunk_data['chunk_id']}...")
                retries = 4
                for attempt in range(retries):
                    try:
                        vector = await get_nvidia_nemotron_embedding(http_client, chunk_data.get("text", ""), api_key)
                        chunk_data["embedding"] = vector
                        return
                    except Exception as e:
                        wait_time = 4.0 * (attempt + 1)
                        if attempt == retries - 1:
                            print(f"   [FAIL] Chunk {chunk_data['chunk_id']} thất bại hoàn toàn: {e}", file=sys.stderr)
                            chunk_data["embedding"] = []
                        else:
                            print(f"   [WARN] Cổng nghẽn ở {chunk_data['chunk_id']}. Thử lại sau {wait_time}s...")
                            await asyncio.sleep(wait_time)

        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(limits=limits) as client:
            tasks = [worker(chunk, i, client) for i, chunk in enumerate(chunks)]
            await asyncio.gather(*tasks)

    # ── Validation: không cho pipeline báo "xong" nếu dữ liệu không sạch ────
    failed_chunks = [c["chunk_id"] for c in chunks if not c.get("embedding")]
    embedding_lengths = {len(c["embedding"]) for c in chunks if c.get("embedding")}

    if len(embedding_lengths) > 1:
        print(
            f"\n[LỖI] Phát hiện embedding có nhiều chiều dài khác nhau: "
            f"{sorted(embedding_lengths)}. Dữ liệu không đồng nhất, dừng pipeline.",
            file=sys.stderr
        )
        sys.exit(1)

    if failed_chunks:
        print(
            f"\n[CẢNH BÁO] {len(failed_chunks)}/{len(chunks)} chunk có embedding "
            f"rỗng (thất bại hoàn toàn sau retry):",
            file=sys.stderr
        )
        for cid in failed_chunks:
            print(f"   - {cid}", file=sys.stderr)
        failures_path = Path(args.output_file).with_name(
            Path(args.output_file).stem + ".embedding_failures.json"
        )
        failures_path.write_text(
            json.dumps(failed_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"   Danh sách chunk lỗi đã lưu tại: {failures_path}", file=sys.stderr)

    # Xuất file kết quả sản phẩm tối cao của Phase 3
    write_jsonl(Path(args.output_file), chunks)
    if failed_chunks:
        print(
            f"\n[HOÀN TẤT CÓ CẢNH BÁO] {len(chunks) - len(failed_chunks)}/{len(chunks)} "
            f"chunk có vector hợp lệ. Xem {failures_path.name} để retry các chunk lỗi."
        )
    else:
        print(f"\n[XONG TOÀN DIỆN PHASE 3] Toàn bộ dữ liệu sạch đã được số hóa vector!")
    print(f"File đích chuẩn bị nạp Graph: {args.output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA Powered Vector Embedding Generator")
    parser.add_argument("--input-file", default="data/chunks/policy_chunks.llm_reviewed.jsonl")
    parser.add_argument("--output-file", default="data/chunks/policy_chunks.final.jsonl")
    parser.add_argument("--force-cloud", action="store_true", help="Ép buộc chạy online qua mô hình của NVIDIA")

    args = parser.parse_args()
    asyncio.run(process_embeddings_pipeline(args))


if __name__ == "__main__":
    main()