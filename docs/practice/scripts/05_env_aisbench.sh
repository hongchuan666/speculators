#!/bin/bash
# ais_bench + GSM8K 测评环境一键搭建（基于 conda）
#
# 用法:
#   bash 05_env_aisbench.sh                                          # 一键全流程
#   bash 05_env_aisbench.sh --env-name my-ais --python 3.10          # 自定义环境名和 Python 版本
#   bash 05_env_aisbench.sh --gsm8k-only                             # 仅下载 GSM8K
#
# 说明:
#   - 自动检测 conda，没有则安装 Miniconda
#   - 创建独立 Python 环境，安装 ais_bench
#   - 下载 GSM8K 数据集并裁剪子集

set -eEuo pipefail

trap 'echo "[ERROR] 脚本在第 $LINENO 行出错（exit code: $?）。请检查上面的错误信息。"; exit 1' ERR

# ============================================================
# 配置参数
# ============================================================

ENV_NAME="${ENV_NAME:-aisbench}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MINICONDA_DIR="${MINICONDA_DIR:-${HOME}/miniconda3}"
# ais_bench 路径，clone 到 ~/ais_benchmark
_DEFAULT_AIS="${HOME}/ais_benchmark"
AIS_BENCH_DIR="${AIS_BENCH_DIR:-${_DEFAULT_AIS}}"
AIS_BENCH_REPO="${AIS_BENCH_REPO:-https://github.com/AISBench/benchmark.git}"
GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-${AIS_BENCH_DIR}/ais_bench/datasets/gsm8k}"
# GSM8K 数据集（HuggingFace Parquet 格式，需用 Python 转 JSONL）
GSM8K_TRAIN_URL="${GSM8K_TRAIN_URL:-https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet}"
GSM8K_TEST_URL="${GSM8K_TEST_URL:-https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh}"
# pip 镜像源（国内加速，取消注释即可）
# PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# PyTorch CPU 版下载源（国内可换镜像加速）
#   - 上海交大: https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
#   - 阿里云:   https://mirrors.aliyun.com/pytorch-wheels/cpu/
TORCH_CPU_INDEX="${TORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"

# 代理脚本路径（pip install 时需代理可配置）
# PROXY_SCRIPT="${PROXY_SCRIPT:-/path/to/proxy.sh}"

# ============================================================
# 参数解析
# ============================================================

GSM8K_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="$2"; shift 2 ;;
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        --ais-bench-dir) AIS_BENCH_DIR="$2"; shift 2 ;;
        --gsm8k-only) GSM8K_ONLY=true; shift ;;
        --help)
            echo "用法: bash $0 [选项]"
            echo "  --env-name NAME      conda 环境名 (默认: aisbench)"
            echo "  --python VERSION     Python 版本 (默认: 3.10)"
            echo "  --ais-bench-dir PATH ais_bench 路径 (默认: ./ais_benchmark)"
            echo "  --gsm8k-only         仅下载 GSM8K 数据集"
            exit 0 ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

# ============================================================
# 0. 确保 conda 可用
# ============================================================

ensure_conda() {
    # 尝试多个路径找到 conda
    local conda_cmd
    for cmd in conda "${MINICONDA_DIR}/bin/conda" "${HOME}/miniconda3/bin/conda" "/opt/miniconda3/bin/conda"; do
        if command -v "$cmd" &> /dev/null; then
            conda_cmd="$cmd"
            break
        fi
    done

    if [ -n "${conda_cmd:-}" ]; then
        # 确保 conda 命令在当前 shell 可用
        local conda_dir
        conda_dir=$(dirname "$(dirname "$(which "${conda_cmd}")")")
        if [ -f "${conda_dir}/etc/profile.d/conda.sh" ]; then
            source "${conda_dir}/etc/profile.d/conda.sh"
        fi
        echo "conda 已就绪: $("${conda_cmd}" --version)"
        return
    fi
    echo "conda 未安装，正在安装 Miniconda..."
    local mc_tmp="${HOME}/.miniconda_install.sh"
    rm -f "${mc_tmp}"
    wget -q --show-progress "${MINICONDA_URL}" -O "${mc_tmp}"
    chmod +x "${mc_tmp}"
    bash "${mc_tmp}" -b -u -p "${MINICONDA_DIR}"
    rm -f "${mc_tmp}"
    "${MINICONDA_DIR}/bin/conda" init
    source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
    echo "Miniconda 已安装到 ${MINICONDA_DIR}"
}

conda_activate() {
    if [ -f "${MINICONDA_DIR}/etc/profile.d/conda.sh" ]; then
        source "${MINICONDA_DIR}/etc/profile.d/conda.sh"
    fi
    conda activate "$1"
}

# ============================================================
# 1. 克隆 ais_bench
# ============================================================

clone_repo() {
    if [ -d "${AIS_BENCH_DIR}" ]; then
        echo "ais_bench 目录已存在: ${AIS_BENCH_DIR}"
    else
        echo "=== 克隆 ais_bench ==="
        git clone "${AIS_BENCH_REPO}" "${AIS_BENCH_DIR}"
    fi
}

# ============================================================
# 2. 创建 conda 环境 + 安装 ais_bench
# ============================================================

install_aisbench() {
    echo "=== 创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION}) ==="

    # 检查环境是否存在（用文件避免 conda shell 函数与 pipefail 冲突）
    conda env list > /tmp/_conda_envs.txt 2>&1 || true
    if grep -q "^${ENV_NAME}\s" /tmp/_conda_envs.txt 2>/dev/null; then
        echo "conda 环境 ${ENV_NAME} 已存在，跳过创建"
    else
        # 接受 conda ToS（新版本要求）
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
        conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    fi

    conda_activate "${ENV_NAME}"

    # 加载代理（若配置）
    if [ -n "${PROXY_SCRIPT:-}" ] && [ -f "${PROXY_SCRIPT}" ]; then
        echo "加载代理: ${PROXY_SCRIPT}"
        source "${PROXY_SCRIPT}"
    fi

    if [ -n "${PIP_INDEX_URL:-}" ]; then
        pip config set global.index-url "${PIP_INDEX_URL}"
    fi

    echo "=== 安装 ais_bench ==="
    cd "${AIS_BENCH_DIR}"
    # 先装 CPU 版 torch（可配置镜像源避免拉 NVIDIA CUDA 库）
    pip install torch --index-url "${TORCH_CPU_INDEX}"
    # 再装 ais_bench 及其余依赖
    pip install -e .
    echo "ais_bench 安装完成"
    echo "ais_bench 安装完成"
}

# ============================================================
# 3. 下载 GSM8K
# ============================================================

download_gsm8k() {
    mkdir -p "${GSM8K_DATA_DIR}"
    echo "=== 下载 GSM8K 数据集 ==="

    # GSM8K 上游已改为 Parquet 格式，需用 Python 读取后转成 JSONL
    if [ ! -f "${GSM8K_DATA_DIR}/train.jsonl" ]; then
        echo "  下载 train.parquet 并转换为 JSONL..."
        python3 -c "
import pandas as pd
df = pd.read_parquet('${GSM8K_TRAIN_URL}')
df.to_json('${GSM8K_DATA_DIR}/train.jsonl', orient='records', lines=True, force_ascii=False)
" && echo "  train: $(wc -l < "${GSM8K_DATA_DIR}/train.jsonl") 条"
    fi

    if [ ! -f "${GSM8K_DATA_DIR}/test.jsonl" ]; then
        echo "  下载 test.parquet 并转换为 JSONL..."
        python3 -c "
import pandas as pd
df = pd.read_parquet('${GSM8K_TEST_URL}')
df.to_json('${GSM8K_DATA_DIR}/test.jsonl', orient='records', lines=True, force_ascii=False)
" && echo "  test:  $(wc -l < "${GSM8K_DATA_DIR}/test.jsonl") 条"
    fi
}

# ============================================================
# 4. 裁剪子集
# ============================================================

create_subsets() {
    local src="${GSM8K_DATA_DIR}/test.jsonl"
    echo "=== 生成裁剪子集 ==="
    for n in 100 200 500; do
        local dst="${HOME}/gsm8k_${n}"
        mkdir -p "${dst}"
        head -n "${n}" "${src}" > "${dst}/test.jsonl"
        cp "${dst}/test.jsonl" "${dst}/train.jsonl" 2>/dev/null || true
        echo "  ${dst}/ (${n} 条)"
    done
}

# ============================================================
# 主入口
# ============================================================

if $GSM8K_ONLY; then
    download_gsm8k
    create_subsets
    echo "GSM8K 数据就绪"
    exit 0
fi

ensure_conda
clone_repo
install_aisbench
download_gsm8k
create_subsets

echo ""
echo "========================================"
echo "  环境搭建完成"
echo "========================================"
echo ""
echo "激活环境:  conda activate ${ENV_NAME}"
echo "快速验证:  ais_bench --models vllm_api --datasets gsm8k --summarizer example"
echo ""
echo "数据子集:"
echo "  ${HOME}/gsm8k_100/   (100 条)"
echo "  ${HOME}/gsm8k_200/   (200 条)"
echo "  ${HOME}/gsm8k_500/   (500 条)"
