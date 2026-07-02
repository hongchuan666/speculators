#!/bin/bash
# 测试草稿模型：拉起带草稿模型的 vLLM 推理服务 + ais_bench 测评
#
# 用法:
#   bash 03_test_spec.sh                                          # 使用默认配置
#   bash 03_test_spec.sh --verifier /path/to/model --draft /path/to/checkpoint
#   bash 03_test_spec.sh --limit 100 --think                       # 开思考模式，100条
#   bash 03_test_spec.sh --limit 200 --offset 200                  # 取第 201-400 条
#
# 说明:
#   - 启动 vLLM speculative decoding 服务（TP=2）
#   - 自动裁剪 GSM8K 测试集到 N 条
#   - 运行 ais_bench 测评并采集 Prometheus 接受率指标
#   - 测试完成后自动清理 vLLM 服务

set -euo pipefail

# ============================================================
# 配置参数
# ============================================================

# 验证器（主模型）路径
VERIFIER="${VERIFIER:-/path/to/model}"

# 草稿模型 checkpoint 路径
DRAFT="${DRAFT:-/path/to/checkpoint}"

# vLLM 服务端口
VLLM_PORT="${VLLM_PORT:-18000}"

# 使用的 GPU（TP=2，占用 2 卡）
SERVING_GPUS="${SERVING_GPUS:-4,5}"

# 投机 token 数
SPEC_TOKENS="${SPEC_TOKENS:-3}"

# 最大模型长度
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# 量化类型（ascend / 留空，W8A8 模型需要 --quantization ascend）
QUANTIZATION="${QUANTIZATION:-}"

# GSM8K 数据集路径
GSM8K_DATA="${GSM8K_DATA:-/path/to/gsm8k/test.jsonl}"

# 测试数据裁剪数量
LIMIT="${LIMIT:-100}"

# 测试数据偏移（跳过前 N 条）
OFFSET="${OFFSET:-0}"

# 开启思考模式（留空为关闭，--think 开启）
THINK_MODE="${THINK_MODE:-false}"

# ais_bench 模型配置名
AIS_MODEL_CONF="${AIS_MODEL_CONF:-vllm_api_spec}"

# ais_bench 工作目录
AIS_WORK_DIR="${AIS_WORK_DIR:-/tmp/ais_spec_test}"

# ais_bench 虚拟环境
AIS_VENV="${AIS_VENV:-/path/to/ais_benchmark/.venv}"

# ============================================================
# 参数解析
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verifier) VERIFIER="$2"; shift 2 ;;
        --draft) DRAFT="$2"; shift 2 ;;
        --port) VLLM_PORT="$2"; shift 2 ;;
        --gpus) SERVING_GPUS="$2"; shift 2 ;;
        --spec-tokens) SPEC_TOKENS="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --offset) OFFSET="$2"; shift 2 ;;
        --think) THINK_MODE=true; shift ;;
        --gsm8k) GSM8K_DATA="$2"; shift 2 ;;
        --help)
            echo "用法: $0 [选项]"
            echo "  --verifier PATH   主模型路径"
            echo "  --draft PATH      草稿模型 checkpoint 路径"
            echo "  --port N          vLLM 服务端口 (默认: 18000)"
            echo "  --gpus STR        服务 GPU (默认: 4,5)"
            echo "  --spec-tokens N   投机 token 数 (默认: 3)"
            echo "  --limit N         测试条数 (默认: 100)"
            echo "  --offset N        数据偏移 (默认: 0)"
            echo "  --think           开启思考模式"
            echo "  --gsm8k PATH      GSM8K 数据集路径"
            exit 0 ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

# ============================================================
# 1. 裁剪 GSM8K 测试数据
# ============================================================

SUBSET_DIR="/tmp/gsm8k_${LIMIT}"
if [ "${OFFSET}" -gt 0 ]; then
    SUBSET_DIR="/tmp/gsm8k_${LIMIT}_offset${OFFSET}"
fi

echo "=== 裁剪测试数据: ${LIMIT} 条, 偏移 ${OFFSET} ==="
mkdir -p "${SUBSET_DIR}"
sed -n "$((OFFSET + 1)),$((OFFSET + LIMIT))p" "${GSM8K_DATA}" > "${SUBSET_DIR}/test.jsonl"
echo "生成: ${SUBSET_DIR}/test.jsonl ($(wc -l < "${SUBSET_DIR}/test.jsonl") 条)"

# 生成 ais_bench 数据集配置（如未预置）
DATASET_CONF="/tmp/gsm8k_spec_test_config.py"
cat > "${DATASET_CONF}" << PYEOF
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')
gsm8k_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='Answer the following question.The last line of the response should follow this format: "answer:\$ANSWER" (without quotes), where ANSWER is a number. Let'"'"'s think step by step.\\n\\nQuestion: {question}')])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, stopping_criteria=['Question']))
gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))
gsm8k_datasets = [dict(abbr='gsm8k_spec', type=GSM8KDataset, path='${SUBSET_DIR}',
    reader_cfg=gsm8k_reader_cfg, infer_cfg=gsm8k_infer_cfg, eval_cfg=gsm8k_eval_cfg)]
PYEOF

# ============================================================
# 2. 启动 vLLM speculative decoding 服务
# ============================================================

echo "=== 启动 vLLM speculative decoding 服务 ==="

# 构建 speculative config JSON
SPEC_CONFIG=$(cat <<JSON
{"model": "${DRAFT}", "method": "eagle3", "num_speculative_tokens": ${SPEC_TOKENS}}
JSON
)

# 构建 vLLM 启动命令
VLLM_CMD="ASCEND_RT_VISIBLE_DEVICES=${SERVING_GPUS} vllm serve ${VERIFIER} \
    --speculative_config '${SPEC_CONFIG}' \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len ${MAX_MODEL_LEN} \
    --port ${VLLM_PORT} \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256"

# W8A8 模型需添加量化参数
if [ -n "${QUANTIZATION}" ]; then
    VLLM_CMD+=" --quantization ${QUANTIZATION}"
fi

# 开思考模式时，不传 chat_template_kwargs
if [ "${THINK_MODE}" == "false" ]; then
    AIS_MODE="no-think"
else
    AIS_MODE="think"
fi

# 后台启动
eval "${VLLM_CMD} > /tmp/vllm_spec_test.log 2>&1 &"
VLLM_PID=$!

# 等待服务就绪
echo "等待 vLLM 就绪..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM 就绪 (${i}s)"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "vLLM 进程异常退出！"
        tail -20 /tmp/vllm_spec_test.log
        exit 1
    fi
    sleep 5
done

# 清理函数
cleanup() {
    echo ""
    echo "=== 清理 vLLM 服务 ==="
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 2
    ps aux | grep -E "VLLM::" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    echo "清理完成"
}
trap cleanup EXIT

# ============================================================
# 3. 运行 ais_bench 测评
# ============================================================

echo ""
echo "=== 运行 ais_bench 测评 (${AIS_MODE}, ${LIMIT} 条) ==="

# 生成 aiSbench 模型配置
MODEL_CONF="/tmp/vllm_api_spec_test.py"
if [ "${THINK_MODE}" == "false" ]; then
    GEN_KWARGS="dict(temperature=0.0, top_k=1, top_p=1.0, seed=None, repetition_penalty=1.0, chat_template_kwargs=dict(enable_thinking=False))"
else
    GEN_KWARGS="dict(temperature=0.0, top_k=1, top_p=1.0, seed=None, repetition_penalty=1.0)"
fi

cat > "${MODEL_CONF}" << PYEOF
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content
models = [dict(attr="service", type=VLLMCustomAPIChat,
    abbr='spec-test', model="${VERIFIER}", host_port=${VLLM_PORT},
    max_out_len=4096, batch_size=32,
    generation_kwargs=${GEN_KWARGS},
    pred_postprocessor=dict(type=extract_non_reasoning_content))]
PYEOF

# 运行
source "${AIS_VENV}/bin/activate"
ais_bench \
    --models "${MODEL_CONF}" \
    --datasets "${DATASET_CONF}" \
    --summarizer example \
    -w "${AIS_WORK_DIR}" 2>&1 | tee /tmp/ais_spec_test.log
deactivate

# ============================================================
# 4. 采集接受率指标
# ============================================================

echo ""
echo "=== 接受率指标 ==="
curl -s "http://localhost:${VLLM_PORT}/metrics" 2>/dev/null \
    | grep "vllm:spec_decode" | grep -v "_created\|# HELP\|# TYPE" \
    || echo "(指标未采集到)"

# ============================================================
# 5. 输出汇总
# ============================================================

echo ""
echo "========================================"
echo "  测评完成"
echo "----------------------------------------"
echo "  主模型: ${VERIFIER}"
echo "  草稿:   ${DRAFT}"
echo "  模式:   ${AIS_MODE}"
echo "  条数:   ${LIMIT}"
echo "  GSM8K:  $(grep -oP 'accuracy\s+gen\s+\K[\d.]+' /tmp/ais_spec_test.log 2>/dev/null || echo '查看日志')"
echo "========================================"
