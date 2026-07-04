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
    parser.add_argument("--endpoint", action="append", default=[], help="vLLM API endpoint (可指定多个，如 --endpoint http://...1 --endpoint http://...2)")
    parser.add_argument("--model", default=None, help="Model name (auto-detect if empty, 多 endpoint 时忽略)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=32768, help="max_tokens")
    parser.add_argument("--resume", action="store_true", help="Skip existing in output")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode (Qwen3 enable_thinking=false)")
    parser.add_argument("--id-key", default=None, help="Field name for unique identifier (default: auto-detect source_index > id)")
    parser.add_argument("--debug", action="store_true", help="Print debug info for resume-skip matching")
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

    # 跳过开头的 assistant 消息（如 [gpt, user, gpt] → [user, gpt]）
    trimmed = 0
    while msgs and msgs[0]["role"] == "assistant":
        msgs.pop(0)
        trimmed += 1

    positions = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    return msgs, positions, trimmed


def load_seen(path: str, id_key: str | None = None) -> tuple[set, dict[str, int]]:
    """返回 (seen_ids, reason_counts) — 记录输出文件中每条数据的状态。"""
    if not os.path.isfile(path):
        return set(), {}
    seen = set()
    reasons: dict[str, int] = {}
    no_id_lines = 0
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                sid = _get_id(d, id_key)
                if sid is None:
                    no_id_lines += 1
                    continue
                seen.add(sid)
                meta = d.get("metadata", {})
                if meta.get("error"):
                    reasons["error"] = reasons.get("error", 0) + 1
                elif meta.get("skipped"):
                    reasons["skipped_raw"] = reasons.get("skipped_raw", 0) + 1
                elif meta.get("finish_reason"):
                    reasons["ok"] = reasons.get("ok", 0) + 1
                else:
                    reasons["unknown"] = reasons.get("unknown", 0) + 1
            except json.JSONDecodeError:
                reasons["invalid_json"] = reasons.get("invalid_json", 0) + 1
                continue
    # 补 0 确保全分类可见
    for k in ("ok", "error", "skipped_raw", "unknown", "invalid_json", "no_id"):
        reasons.setdefault(k, 0)
    if no_id_lines:
        reasons["no_id"] = no_id_lines
        # 输出第一行没有 ID 的样本 key 作为提示
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if _get_id(d, id_key) is None:
                        print(f"  ⚠️  输出文件存在无 ID 的行, keys={list(d.keys())}")
                        print(f"     id_key={id_key or 'auto(source_index > id > source_id)'}")
                        break
                except json.JSONDecodeError:
                    continue
    return seen, reasons


async def worker(sem, session, queue, args, out_fh, progress, stats, endpoints, models):
    import random
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        msgs, assistant_positions = item["msgs"], item["assistant_positions"]
        total_cost = 0.0
        finish_reason = None
        sample_error = None
        used_model = ""

        for idx in assistant_positions:
            # 随机选一个 endpoint
            ep = random.choice(endpoints)
            used_model = models[ep]
            context = msgs[:idx]
            payload = {
                "model": used_model,
                "messages": context,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            if args.no_think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            step_start = time.time()
            try:
                async with sem, session.post(ep, json=payload) as resp:
                    data = await resp.json()
                choice = data["choices"][0]
                new_content = choice["message"]["content"]
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
                    "model": used_model,
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

    # 解析 endpoints
    endpoints = args.endpoint or ["http://127.0.0.1:8000/v1/chat/completions"]
    if args.model:
        models = {ep: args.model for ep in endpoints}
    else:
        models = {ep: await detect_model(ep) for ep in endpoints}
    for ep, model in models.items():
        print(f"  [{model}] {ep}")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    id_key = args.id_key
    seen = set()
    resume_reasons: dict[str, int] = {}
    if args.resume:
        seen, resume_reasons = load_seen(args.output, id_key)
        if resume_reasons:
            parts = " + ".join(f"{k}={v}" for k, v in sorted(resume_reasons.items()))
            print(f"  Resume skip breakdown: {parts}")
    stats = {"ok": 0, "errors": 0}

    with open(args.input) as f:
        raw = f.read().strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            samples = parsed
        elif isinstance(parsed, dict):
            samples = [parsed]
        else:
            samples = [json.loads(line) for line in raw.split("\n") if line.strip()]

    if args.limit:
        samples = samples[:args.limit]
    # 预过滤 resume-skipped 样本
    active = []
    for s in samples:
        sid = _get_id(s, id_key)
        if sid is None:
            if args.debug:
                print(f"  [debug] 输入样本无 ID: keys={list(s.keys())}")
        if sid in seen:
            continue
        active.append(s)
    skipped_resume = len(samples) - len(active)
    samples = active

    if args.debug:
        seen_sample = list(seen)[:5] if seen else []
        active_sample = [_get_id(s, id_key) for s in samples[:5]]
        print(f"  [debug] seen 前5: {seen_sample}")
        print(f"  [debug] active ID 前5: {active_sample}")
        print(f"  [debug] None in seen: {None in seen}")
        print(f"  [debug] seen size: {len(seen)}")

    print(f"Samples: {len(samples)} active + {skipped_resume} resume-skipped = {len(samples) + skipped_resume} total")
    print(f"Endpoints: {len(endpoints)} ({', '.join(models.values())})")

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=None)
    out_fh = open(args.output, "a", encoding="utf-8")

    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm(total=len(samples), desc="Resampling", unit="sample", dynamic_ncols=True) as progress:
            workers = [
                asyncio.create_task(worker(sem, session, queue, args, out_fh, progress, stats, endpoints, models))
                for _ in range(args.concurrency)
            ]

            for s in samples:
                s_id_val = _get_id(s, id_key)
                try:
                    msgs, positions, trimmed = _find_assistant_positions(s)
                except ValueError:
                    stats["errors"] += 1
                    progress.set_postfix(ok=stats["ok"], errors=stats["errors"], refresh=False)
                    progress.update(1)
                    continue
                if trimmed:
                    stats["trimmed"] = stats.get("trimmed", 0) + 1
                if not positions:
                    # 裁剪后没有 assistant 消息（如全为 assistant 的对话）
                    stats["empty_after_trim"] = stats.get("empty_after_trim", 0) + 1
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

    total_proc = stats["ok"] + stats["errors"] + stats.get("empty_after_trim", 0)
    if total_proc != len(samples):
        print(f"  ⚠️  Mismatch: processed {total_proc}/{len(samples)} (ok={stats['ok']} + error={stats['errors']} + empty={stats.get('empty_after_trim',0)})")
    extras = []
    if stats.get("trimmed"):
        extras.append(f"trimmed={stats['trimmed']}")
    if stats.get("empty_after_trim"):
        extras.append(f"empty_after_trim={stats['empty_after_trim']}")
    if skipped_resume and resume_reasons:
        parts = " + ".join(f"{k}={v}" for k, v in sorted(resume_reasons.items()))
        extras.append(f"resume_skip({parts})")
    elif skipped_resume:
        extras.append(f"resume_skip={skipped_resume}")
    extra_str = f" ({', '.join(extras)})" if extras else ""
    print(f"Done. OK={stats['ok']} + Error={stats['errors']} = {total_proc}/{len(samples)}{extra_str}")


if __name__ == "__main__":
    asyncio.run(main())
