#!/bin/bash
# 重采样：启动推理服务 → 重新采集模型输出数据
#
# 用法:
#   bash 01_resample.sh all                                      # 启动服务 → 重采样 → 停止服务
#   bash 01_resample.sh service                                  # 仅启动推理服务
#   bash 01_resample.sh resample                                 # 仅重采样（需服务已运行）
#
# 参数:
#   bash 01_resample.sh all --model /path/to/model --gpus 0,1 --port 8000
#   bash 01_resample.sh resample --input data.jsonl --limit 1000
#
# 说明:
#   重采样从指定 vLLM API 服务读取 prompt，重新生成 response，
#   输出为 speculators 训练可用的 JSONL。支持 --resume 断点续采。

set -euo pipefail

# ============================================================
# 配置参数
# ============================================================

# 推理服务
MODEL="${MODEL:-/path/to/model}"
SERVING_GPUS="${SERVING_GPUS:-0,1}"
SERVING_TP="${SERVING_TP:-2}"
SERVING_QUANT="${SERVING_QUANT:-}"
SERVING_PORT="${SERVING_PORT:-8000}"
SERVING_MAX_LEN="${SERVING_MAX_LEN:-8192}"

# 重采样
# 端点列表（支持多个 --endpoint）
ENDPOINTS=()
INPUT_DATA="${INPUT_DATA:-/path/to/input.jsonl}"
OUTPUT_DATA="${OUTPUT_DATA:-/path/to/resampled.jsonl}"
LIMIT="${LIMIT:-}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
NO_THINK="${NO_THINK:-true}"
RESUME="${RESUME:-true}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# speculators 仓库路径
SPECULATORS_DIR="${SPECULATORS_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"

# ============================================================
# 参数解析
# ============================================================

SUBCOMMAND="${1:-resample}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --gpus) SERVING_GPUS="$2"; shift 2 ;;
        --port) SERVING_PORT="$2"; shift 2 ;;
        --quantization) SERVING_QUANT="$2"; shift 2 ;;
        --endpoint) ENDPOINTS+=("$2"); shift 2 ;;
        --input) INPUT_DATA="$2"; shift 2 ;;
        --output) OUTPUT_DATA="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --no-think) NO_THINK="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --help)
            echo "用法: bash $0 <子命令> [选项]"
            echo ""
            echo "子命令:"
            echo "  all        启动服务 → 重采样 → 停止服务"
            echo "  service    仅启动推理服务"
            echo "  resample   仅重采样（默认）"
            echo ""
            echo "服务参数:"
            echo "  --model PATH      模型路径"
            echo "  --gpus STR        GPU 分配, 如 '0,1'"
            echo "  --port N          服务端口 (默认: 8000)"
            echo "  --quantization STR 量化方式 (如 ascend)"
            echo ""
            echo "重采样参数:"
            echo "  --input PATH      输入 JSONL/JSON 路径"
            echo "  --output PATH     输出 JSONL 路径"
            echo "  --limit N         采样上限"
            echo "  --concurrency N   并发数 (默认: 32)"
            echo "  --max-tokens N    最大生成长度 (默认: 4096)"
            echo "  --resume true|false 断点续采 (默认: true)"
            echo "  --no-think true|false 关闭思考 (默认: true)"
            exit 0 ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

cd "${SPECULATORS_DIR}"

# ============================================================
# 启动推理服务
# ============================================================

start_service() {
    local port="${SERVING_PORT}"
    echo "=== 启动推理服务: ${MODEL} (端口 ${port}, GPU ${SERVING_GPUS}) ==="

    local cmd="ASCEND_RT_VISIBLE_DEVICES=${SERVING_GPUS} vllm serve ${MODEL}"
    cmd+=" --tensor-parallel-size ${SERVING_TP}"
    cmd+=" --dtype bfloat16 --max-model-len ${SERVING_MAX_LEN}"
    cmd+=" --port ${port} --gpu-memory-utilization 0.9 --max-num-seqs 256"
    [ -n "${SERVING_QUANT}" ] && cmd+=" --quantization ${SERVING_QUANT}"

    eval "${cmd} > /tmp/vllm_resample.log 2>&1 &"
    VLLM_PID=$!

    echo "等待服务就绪 (PID=${VLLM_PID})..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "服务就绪 (${i}s)"
            return 0
        fi
        sleep 2
    done
    echo "服务启动超时"
    exit 1
}

stop_service() {
    echo "=== 停止推理服务 ==="
    [ -n "${VLLM_PID:-}" ] && kill "${VLLM_PID}" 2>/dev/null || true
    ps aux | grep -E "VLLM::" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
}

# ============================================================
# 重采样
# ============================================================

run_resample() {
    local args=""
    args+=" --input ${INPUT_DATA} --output ${OUTPUT_DATA}"
    # 默认 endpoint
    if [ ${#ENDPOINTS[@]} -eq 0 ]; then
        ENDPOINTS=("http://localhost:${SERVING_PORT}/v1/chat/completions")
    fi
    for ep in "${ENDPOINTS[@]}"; do
        args+=" --endpoint ${ep}"
    done
    args+=" --concurrency ${CONCURRENCY}"
    args+=" --max-tokens ${MAX_TOKENS} --temperature ${TEMPERATURE}"

    [ -n "${LIMIT}" ] && args+=" --limit ${LIMIT}"
    [ "${NO_THINK}" == "true" ] && args+=" --no-think"
    [ "${RESUME}" == "true" ] && args+=" --resume"

    echo "=========================================="
    echo "  重采样配置"
    echo "------------------------------------------"
    echo "  端点:     ${ENDPOINTS[*]}"
    echo "  输入:     ${INPUT_DATA}"
    echo "  输出:     ${OUTPUT_DATA}"
    echo "  并发:     ${CONCURRENCY}"
    echo "  max_tokens: ${MAX_TOKENS}"
    echo "  限制:     ${LIMIT:-无}"
    echo "  续采:     ${RESUME}"
    echo "  no_think: ${NO_THINK}"
    echo "=========================================="

    python scripts/response_regeneration/resample_custom.py ${args}
}

# ============================================================
# 主入口
# ============================================================

case "${SUBCOMMAND}" in
    all)
        trap stop_service EXIT
        start_service
        run_resample
        ;;
    service)
        start_service
        echo "服务运行中 (PID=${VLLM_PID})。停止: kill ${VLLM_PID}"
        wait "${VLLM_PID}" 2>/dev/null || true
        ;;
    resample)
        run_resample
        ;;
    *)
        echo "未知子命令: ${SUBCOMMAND} (可用: all, service, resample)"
        exit 1
        ;;
esac
