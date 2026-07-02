#!/bin/bash
# 重采样：基于某个特定的模型推理服务，重新采集模型输出数据
#
# 用法:
#   bash 01_resample.sh                                          # 使用默认配置
#   bash 01_resample.sh --endpoint http://localhost:8000/v1/chat/completions
#   bash 01_resample.sh --limit 1000 --concurrency 64
#
# 说明:
#   从指定的 vLLM API 服务读取 prompt，重新生成 response，
#   输出格式为 speculators 训练可用的 JSONL。
#   支持 --resume 断点续采。

set -euo pipefail

# ============================================================
# 配置参数
# ============================================================

# vLLM API 端点
ENDPOINT="${ENDPOINT:-http://localhost:8000/v1/chat/completions}"

# 输入数据（原始 prompt JSONL）
INPUT_DATA="${INPUT_DATA:-/path/to/input.jsonl}"

# 输出数据（重采样结果）
OUTPUT_DATA="${OUTPUT_DATA:-/path/to/resampled.jsonl}"

# 限制采样数（留空则采全部）
LIMIT="${LIMIT:-}"

# 并发数
CONCURRENCY="${CONCURRENCY:-32}"

# 最大生成长度
MAX_TOKENS="${MAX_TOKENS:-4096}"

# 关闭思考模式（Qwen3 专用，默认关闭）
NO_THINK="${NO_THINK:-true}"

# 断点续采（跳过输出中已有 source_index）
RESUME="${RESUME:-true}"

# 采样温度
TEMPERATURE="${TEMPERATURE:-0.0}"

# speculators 仓库路径
SPECULATORS_DIR="${SPECULATORS_DIR:-/path/to/speculators}"

# ============================================================
# 参数解析
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --endpoint) ENDPOINT="$2"; shift 2 ;;
        --input) INPUT_DATA="$2"; shift 2 ;;
        --output) OUTPUT_DATA="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --no-think) NO_THINK="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --help)
            echo "用法: $0 [选项]"
            echo "  --endpoint URL     vLLM API 端点"
            echo "  --input PATH       输入 JSONL 路径"
            echo "  --output PATH      输出 JSONL 路径"
            echo "  --limit N          采样上限"
            echo "  --concurrency N    并发数 (默认: 32)"
            echo "  --max-tokens N     最大生成长度 (默认: 4096)"
            echo "  --no-think true|false  关闭思考模式 (默认: true)"
            echo "  --resume true|false    断点续采 (默认: true)"
            exit 0 ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

# ============================================================
# 运行重采样
# ============================================================

ARGS="--input ${INPUT_DATA} --output ${OUTPUT_DATA} --endpoint ${ENDPOINT} --concurrency ${CONCURRENCY} --max-tokens ${MAX_TOKENS} --temperature ${TEMPERATURE}"

if [ -n "${LIMIT}" ]; then
    ARGS+=" --limit ${LIMIT}"
fi

if [ "${NO_THINK}" == "true" ]; then
    ARGS+=" --no-think"
fi

if [ "${RESUME}" == "true" ]; then
    ARGS+=" --resume"
fi

echo "=========================================="
echo "  重采样配置"
echo "------------------------------------------"
echo "  端点:     ${ENDPOINT}"
echo "  输入:     ${INPUT_DATA}"
echo "  输出:     ${OUTPUT_DATA}"
echo "  并发:     ${CONCURRENCY}"
echo "  max_tokens: ${MAX_TOKENS}"
echo "  限制:     ${LIMIT:-无}"
echo "  续采:     ${RESUME}"
echo "  no_think: ${NO_THINK}"
echo "=========================================="

cd "${SPECULATORS_DIR}"
python scripts/response_regeneration/resample_custom.py ${ARGS}
