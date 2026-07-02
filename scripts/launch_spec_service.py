#!/usr/bin/env python3
"""Launch vLLM server with speculative decoding (Eagle3).

Usage:
    python scripts/launch_spec_service.py <verifier> --draft <checkpoint> [options]

Example:
    python scripts/launch_spec_service.py /path/to/model \\
        --draft /path/to/checkpoint \\
        --port 18000 --gpus 4,5 --spec-tokens 3 --quantization ascend
"""
import argparse
import json
import os
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Launch vLLM spec decode server")
    parser.add_argument("model", type=str, help="Verifier (main) model path")
    parser.add_argument("--draft", type=str, required=True, help="Draft model checkpoint path")
    parser.add_argument("--port", type=int, default=18000, help="Server port (default: 18000)")
    parser.add_argument("--gpus", type=str, default=None, help="GPUs to use, e.g. '4,5'")
    parser.add_argument("--spec-tokens", type=int, default=3, help="Number of speculative tokens")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max model length")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization (e.g. ascend)")
    parser.add_argument("--gpu-memory", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    return parser.parse_args()


def main():
    args = parse_args()

    spec_config = {
        "model": args.draft,
        "method": "eagle3",
        "num_speculative_tokens": args.spec_tokens,
    }

    cmd = [
        "vllm", "serve",
        args.model,
        "--speculative_config", json.dumps(spec_config),
        "--tensor-parallel-size", str(args.tp),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory),
        "--max-num-seqs", "256",
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]

    env = os.environ.copy()
    if args.gpus:
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.gpus

    print("=" * 50)
    print("Speculative Decoding Server")
    print("=" * 50)
    print(f"  Verifier:    {args.model}")
    print(f"  Draft:       {args.draft}")
    print(f"  Port:        {args.port}")
    print(f"  GPUs:        {args.gpus or 'all'}")
    print(f"  Spec tokens: {args.spec_tokens}")
    print(f"  Command:     {' '.join(cmd)}")
    print("=" * 50)

    if args.dry_run:
        return

    proc = subprocess.Popen(cmd, env=env)
    print(f"Server PID: {proc.pid}")

    # Wait for health
    for i in range(120):
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=2)
            if resp.status == 200:
                print(f"Server ready after {i * 2 + 1}s")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("Server did not become ready within timeout")
        sys.exit(1)

    proc.wait()


if __name__ == "__main__":
    main()
