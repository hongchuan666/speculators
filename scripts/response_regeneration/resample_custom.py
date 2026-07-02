#!/usr/bin/env python3
"""
Resample responses from a custom local JSONL dataset via vLLM Chat API.

Usage:
  python scripts/response_regeneration/resample_custom.py \
    --input /path/to/data.jsonl \
    --endpoint http://localhost:8000/v1/chat/completions \
    --output /path/to/resampled.jsonl \
    --limit 20
"""
import argparse
import asyncio
import json
import os
import time
import aiohttp
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Resample via vLLM from custom JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default=None, help="Model name (auto-detect if empty)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=32768, help="max_tokens")
    parser.add_argument("--resume", action="store_true", help="Skip existing in output")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode (Qwen3 enable_thinking=false)")
    parser.add_argument("--id-key", default=None, help="Field name for unique identifier (default: auto-detect source_index > id)")
    return parser.parse_args()


async def detect_model(endpoint: str) -> str:
    models_url = endpoint.replace("/v1/chat/completions", "/v1/models")
    async with aiohttp.ClientSession() as s, s.get(models_url) as resp:
        data = await resp.json()
        return data["data"][0]["id"]


def _get_id(d: dict, id_key: str | None = None):
    """Extract unique identifier from a record, supporting arbitrary key names."""
    if id_key:
        return d.get(id_key)
    return d.get("source_index") or d.get("id") or d.get("source_id")


def _get_messages(d: dict) -> list[dict]:
    """Extract message list from record, normalizing role/content keys."""
    for field in ("conversations", "messages", "chat"):
        msgs = d.get(field, [])
        if isinstance(msgs, list) and msgs:
            normalized = []
            for m in msgs:
                role = m.get("role") or m.get("from") or ""
                content = m.get("content") or m.get("value") or ""
                normalized.append({"role": role, "content": content})
            return normalized
    return []


def _normalize_roles(msgs: list[dict]) -> list[dict]:
    """标准化 role 名并校验数据合法性。"""
    out = []
    for m in msgs:
        r = m["role"].lower()
        if r in ("human",):
            r = "user"
        elif r in ("gpt", "assistant"):
            r = "assistant"
        out.append({"role": r, "content": m["content"]})
    return out


def _find_assistant_positions(d: dict) -> tuple[list[dict], list[int]]:
    """解析对话，返回 (标准化消息, assistant位置列表)。

    以assistant开头视为脏数据，返回空列表。
    """
    msgs = _normalize_roles(_get_messages(d))
    if not msgs:
        raise ValueError(f"Cannot find messages in record keys={list(d.keys())}")

    if msgs[0]["role"] == "assistant":
        raise ValueError("Dirty data: conversation starts with assistant")

    positions = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    return msgs, positions


def load_seen(path: str, id_key: str | None = None) -> set:
    if not os.path.isfile(path):
        return set()
    seen = set()
    with open(path) as f:
        for line in f:
            try:
                seen.add(_get_id(json.loads(line), id_key))
            except json.JSONDecodeError:
                continue
    return seen


async def worker(sem, session, queue, args, out_fh, progress, stats):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        msgs, assistant_positions = item["msgs"], item["assistant_positions"]
        total_cost = 0.0
        finish_reason = None
        sample_error = None

        for idx in assistant_positions:
            # 取到当前 assistant 位置（不包括它）作为上下文
            context = msgs[:idx]
            payload = {
                "model": args.model,
                "messages": context,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            if args.no_think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            step_start = time.time()
            try:
                async with sem, session.post(args.endpoint, json=payload) as resp:
                    data = await resp.json()
                choice = data["choices"][0]
                new_content = choice["message"]["content"]
                # 替换当前 assistant 消息
                msgs[idx]["content"] = new_content
                total_cost += time.time() - step_start
                if finish_reason is None:
                    finish_reason = choice.get("finish_reason")
            except Exception as e:
                sample_error = repr(e)
                break

        if sample_error:
            output = {
                "source_index": item["source_index"],
                "source_id": item["source_id"],
                "conversations": item["original_msgs"],
                "metadata": {"error": sample_error},
            }
            stats["errors"] += 1
        else:
            output = {
                "source_index": item["source_index"],
                "source_id": item["source_id"],
                "conversations": msgs,
                "metadata": {
                    "model": args.model,
                    "latency_s": round(total_cost, 3),
                    "num_resampled": len(assistant_positions),
                    "finish_reason": finish_reason,
                },
            }
            stats["ok"] += 1

        out_fh.write(json.dumps(output, ensure_ascii=False) + "\n")
        out_fh.flush()
        progress.set_postfix(ok=stats["ok"], errors=stats["errors"], refresh=False)
        progress.update(1)
        queue.task_done()


async def main():
    args = parse_args()
    if args.model is None:
        args.model = await detect_model(args.endpoint)
    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    id_key = args.id_key
    seen = load_seen(args.output, id_key) if args.resume else set()

    print(f"ID key: {id_key or 'auto (source_index > id > source_id)'}")

    with open(args.input) as f:
        raw = f.read().strip()
        # 尝试 JSON（list 或 object），回退到 JSONL
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            samples = parsed
        elif isinstance(parsed, dict):
            samples = [parsed]
        else:
            # JSONL: 每行一个 JSON
            samples = [json.loads(line) for line in raw.split("\n") if line.strip()]

    if args.limit:
        samples = samples[:args.limit]
    print(f"Samples: {len(samples)} (resume skip: {len(seen)})")

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=None)
    out_fh = open(args.output, "a", encoding="utf-8")

    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=len(samples), desc="Resampling", unit="sample", dynamic_ncols=True) as progress:
            stats = {"ok": 0, "errors": 0}
            workers = [asyncio.create_task(worker(sem, session, queue, args, out_fh, progress, stats))
                       for _ in range(args.concurrency)]

            for s in samples:
                s_id_val = _get_id(s, id_key)
                if s_id_val in seen:
                    continue
                try:
                    msgs, positions = _find_assistant_positions(s)
                except ValueError as e:
                    stats["errors"] += 1
                    progress.set_postfix(ok=stats["ok"], errors=stats["errors"], refresh=False)
                    progress.update(1)
                    # 将脏数据写入输出标记为 error
                    error_out = {
                        "source_index": s_id_val,
                        "source_id": str(s_id_val),
                        "conversations": _get_messages(s),
                        "metadata": {"error": str(e), "skipped": True},
                    }
                    out_fh.write(json.dumps(error_out, ensure_ascii=False) + "\n")
                    out_fh.flush()
                    continue
                if not positions:
                    # 没有 assistant 消息，跳过
                    stats["errors"] += 1
                    progress.set_postfix(ok=stats["ok"], errors=stats["errors"], refresh=False)
                    progress.update(1)
                    continue
                await queue.put({
                    "source_index": s_id_val,
                    "source_id": str(s_id_val),
                    "msgs": msgs,
                    "assistant_positions": positions,
                    "original_msgs": _get_messages(s),
                })

            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)

    out_fh.close()

    skipped = len(seen) if args.resume else 0
    print(f"Done. OK={stats['ok']}, Errors={stats['errors']}, Skipped={skipped}")


if __name__ == "__main__":
    asyncio.run(main())
