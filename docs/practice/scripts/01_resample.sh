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

# 推理服务实例列表
# --instance model:gpus:port[:quant]  完整指定（模型路径可不同）
# --service  gpus:port                 共享 --model 和 --quantization
INSTANCES=()
SERVICES=()

# 模型和量化（给 --service 提供默认值）
_MODEL="${_MODEL:-/home/hongchuan/model/Qwen3-8B}"
_QUANT="${_QUANT:-}"
SERVING_TP="${SERVING_TP:-2}"
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
        --instance) INSTANCES+=("$2"); shift 2 ;;
        --endpoint) ENDPOINTS+=("$2"); shift 2 ;;
        # 多服务实例（共享 --model 路径）, 格式: gpus:port[:quant]
        --service) SERVICES+=("$2"); shift 2 ;;
        --model) _MODEL="$2"; shift 2 ;;
        --quantization) _QUANT="$2"; shift 2 ;;
        # 旧单实例参数（兼容）
        --gpus) _GPUS="$2"; shift 2 ;;
        --port) _PORT="$2"; shift 2 ;;
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
            echo "  --model PATH      模型路径（多个服务共享）"
            echo "  --quantization STR 量化方式 (如 ascend, 所有 --service 共用)"
            echo "  --service STR     服务实例 (可多个), 格式: gpus:port"
            echo "                    如: --model /m --quantization ascend --service 0,1:8000 --service 2,3:8001"
            echo "  --instance STR    完整指定 (模型可不同), 格式: model:gpus:port[:quant]"
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

VLLM_PIDS=()

parse_instance() {
    local spec="$1"
    IFS=':' read -r model gpus port quant <<< "${spec}:"
    echo "${model}" "${gpus}" "${port}" "${quant}"
}

start_service() {
    # 将 --service 合并到 INSTANCES（共享 --model 和 --quantization）
    local model="${_MODEL:-/path/to/model}"
    for svc in "${SERVICES[@]}"; do
        # 若 --service 没指定 quant，但 --quantization 有值，则补上
        if [[ "${svc}" != *:*:* ]] && [ -n "${_QUANT:-}" ]; then
            svc="${svc}:${_QUANT}"
        fi
        INSTANCES+=("${model}:${svc}")
    done
    # 合并传统单实例参数
    if [ ${#INSTANCES[@]} -eq 0 ] && [ -n "${_GPUS:-}" ]; then
        INSTANCES+=("${model}:${_GPUS}:${_PORT:-8000}:${_QUANT:-}")
    fi
    if [ ${#INSTANCES[@]} -eq 0 ]; then
        INSTANCES+=("${model}:0,1:8000:")
    fi

    local log_file="/tmp/vllm_resample.log"
    > "${log_file}"

    for instance in "${INSTANCES[@]}"; do
        read -r model gpus port quant <<< "$(parse_instance "${instance}")"
        echo "=== 启动推理服务: ${model} (端口 ${port}, GPU ${gpus}) ==="

        local cmd="ASCEND_RT_VISIBLE_DEVICES=${gpus} vllm serve ${model}"
        cmd+=" --tensor-parallel-size ${SERVING_TP}"
        cmd+=" --dtype bfloat16 --max-model-len ${SERVING_MAX_LEN}"
        cmd+=" --port ${port} --gpu-memory-utilization 0.9 --max-num-seqs 256"
        [ -n "${quant}" ] && cmd+=" --quantization ${quant}"

        eval "${cmd} >> ${log_file} 2>&1 &"
        VLLM_PIDS+=($!)
    done

    # 等待所有服务就绪
    for instance in "${INSTANCES[@]}"; do
        read -r model gpus port quant <<< "$(parse_instance "${instance}")"
        echo "等待服务 ${port} 就绪..."
        for i in $(seq 1 60); do
            if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
                echo "  端口 ${port} 就绪 (${i}s)"
                break
            fi
            sleep 2
        done
    done
    echo "全部服务就绪"
}

stop_service() {
    echo "=== 停止推理服务 ==="
    for pid in "${VLLM_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    sleep 1
    ps aux | grep -E "VLLM::" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
}

# ============================================================
# 重采样
# ============================================================

run_resample() {
    local args=""
    args+=" --input ${INPUT_DATA} --output ${OUTPUT_DATA}"
    # 默认 endpoint: 从 --instance 推断
    if [ ${#ENDPOINTS[@]} -eq 0 ] && [ ${#INSTANCES[@]} -gt 0 ]; then
        for instance in "${INSTANCES[@]}"; do
            read -r _ _ port _ <<< "$(parse_instance "${instance}")"
            ENDPOINTS+=("http://localhost:${port}/v1/chat/completions")
        done
    fi
    # 兜底默认
    if [ ${#ENDPOINTS[@]} -eq 0 ]; then
        ENDPOINTS=("http://localhost:8000/v1/chat/completions")
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
