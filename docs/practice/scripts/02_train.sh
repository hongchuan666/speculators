#!/bin/bash
# 训练草稿模型：启动服务化（vLLM hidden states 服务）与训练，可一键拉起
#
# 用法:
#   bash 02_train.sh all                    # 全流程: 准备数据 → 启动 vLLM → 训练
#   bash 02_train.sh prepare                # 仅准备数据
#   bash 02_train.sh vllm                   # 仅启动 vLLM hidden states 服务
#   bash 02_train.sh train                  # 仅训练（需 vLLM 已运行）
#   bash 02_train.sh train --epochs 2       # 训练 2 个 epoch（支持从 checkpoint 恢复）
#
# 说明:
#   - 训练脚本会自动检测 checkpoint 并恢复
#   - re-run 同配置会自动 resume

set -euo pipefail

# ============================================================
# 配置参数
# ============================================================

# 验证器（主模型）路径
VERIFIER="${VERIFIER:-/path/to/model}"

# 训练数据（原始 JSONL，用于 prepare_data）
TRAIN_DATA="${TRAIN_DATA:-/path/to/train.jsonl}"

# 训练数据输出目录（prepare_data 产出）
DATA_DIR="${DATA_DIR:-./eagle3_arrow_exp}"

# hidden states 共享路径
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-/dev/shm/hs_exp}"

# vLLM 服务端口（hidden states 提取）
VLLM_PORT="${VLLM_PORT:-12000}"

# vLLM 使用的 GPU
VLLM_GPUS="${VLLM_GPUS:-4,5}"

# 训练使用的 GPU
TRAIN_GPUS="${TRAIN_GPUS:-6,7}"

# 训练 GPU 数
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"

# 草稿词表大小
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"

# 最大采样数
MAX_SAMPLES="${MAX_SAMPLES:-258240}"

# 序列长度
SEQ_LENGTH="${SEQ_LENGTH:-2048}"

# Epochs（训练总 epoch 数）
EPOCHS="${EPOCHS:-5}"

# 学习率
LR="${LR:-1e-4}"

# speculators 仓库路径
SPECULATORS_DIR="${SPECULATORS_DIR:-.}"

# ============================================================
# 参数解析
# ============================================================

SUBCOMMAND="${1:-all}"
shift || true

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --verifier) VERIFIER="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --help) echo "子命令: all, prepare, vllm, train"; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

cd "${SPECULATORS_DIR}"

# ============================================================
# Step 1: 准备数据
# ============================================================

prepare_data() {
    echo "=== Step 1: 准备数据 ==="
    python scripts/prepare_data.py \
        --model "${VERIFIER}" \
        --data "${TRAIN_DATA}" \
        --output "${DATA_DIR}" \
        --max-samples "${MAX_SAMPLES}" \
        --seq-length "${SEQ_LENGTH}"
}

# ============================================================
# Step 2: 启动 vLLM hidden states 服务
# ============================================================

start_vllm() {
    echo "=== Step 2: 启动 vLLM hidden states 服务 (端口 ${VLLM_PORT}) ==="
    mkdir -p "${HIDDEN_STATES_PATH}"
    ASCEND_RT_VISIBLE_DEVICES="${VLLM_GPUS}" python scripts/launch_vllm.py "${VERIFIER}" \
        --hidden-states-path "${HIDDEN_STATES_PATH}" \
        -- --data-parallel-size 2 --port "${VLLM_PORT}" &
    VLLM_PID=$!

    echo "等待 vLLM 就绪..."
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "vLLM 就绪 (PID=${VLLM_PID})"
}

cleanup_vllm() {
    echo "停止 vLLM..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}

# ============================================================
# Step 3: 训练
# ============================================================

run_train() {
    echo "=== Step 3: 训练 (epochs=${EPOCHS}) ==="
    ASCEND_RT_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun \
        --standalone --nproc_per_node "${NUM_TRAIN_GPUS}" \
        scripts/train.py \
        --verifier-name-or-path "${VERIFIER}" \
        --data-path "${DATA_DIR}" \
        --hidden-states-path "${HIDDEN_STATES_PATH}" \
        --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
        --save-path "${DATA_DIR}/checkpoints" \
        --draft-vocab-size "${DRAFT_VOCAB_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --total-seq-len "${SEQ_LENGTH}" \
        --on-missing generate \
        --on-generate delete
}

# ============================================================
# 主入口
# ============================================================

case "${SUBCOMMAND}" in
    prepare)
        prepare_data
        ;;
    vllm)
        start_vllm
        echo "vLLM 运行中 (PID=${VLLM_PID})。停止: kill ${VLLM_PID}"
        wait "${VLLM_PID}"
        ;;
    train)
        run_train
        ;;
    all)
        trap cleanup_vllm EXIT
        prepare_data
        start_vllm
        run_train
        ;;
    *)
        echo "未知子命令: ${SUBCOMMAND}"
        echo "可用: all, prepare, vllm, train"
        exit 1
        ;;
esac
