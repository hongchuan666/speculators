#!/usr/bin/env python3
"""Launch vLLM inference service for data resampling / general usage.

Usage:
    python scripts/launch_infer_service.py <model> [options]

Examples:
    # Float 模型
    python scripts/launch_infer_service.py /home/hongchuan/model/Qwen3-8B \\
        --port 8000 --gpus 0,1

    # W8A8 量化模型
    python scripts/launch_infer_service.py /home/hongchuan/model/Qwen3-8B-w8a8-iter \\
        --port 8000 --gpus 0,1 --quantization ascend
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description="Launch vLLM inference service")
    parser.add_argument("model", type=str, help="Model path")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--gpus", type=str, default=None, help="GPUs to use, e.g. '0,1'")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size")
    parser.add_argument("--max-model-len", type=int, default=8192, help="Max model length")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization (e.g. ascend)")
    parser.add_argument("--gpu-memory", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-num-seqs", type=int, default=1024, help="Max num sequences")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = [
        "vllm", "serve",
        args.model,
        "--tensor-parallel-size", str(args.tp),
        "--data-parallel-size", str(args.dp),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory),
        "--max-num-seqs", str(args.max_num_seqs),
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]

    env = os.environ.copy()
    if args.gpus:
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.gpus

    print("=" * 50)
    print("vLLM Inference Service")
    print("=" * 50)
    print(f"  Model:    {args.model}")
    print(f"  Port:     {args.port}")
    print(f"  GPUs:     {args.gpus or 'all'}")
    print(f"  TP:       {args.tp}  DP: {args.dp}")
    print("=" * 50)

    if args.dry_run:
        print("Command:", " ".join(cmd))
        return

    proc = subprocess.Popen(cmd, env=env)
    print(f"PID: {proc.pid}")

    for i in range(120):
        try:
            resp = urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=2)
            if resp.status == 200:
                print(f"Ready after {i * 2 + 1}s")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("Server did not become ready")
        sys.exit(1)

    proc.wait()


if __name__ == "__main__":
    main()
