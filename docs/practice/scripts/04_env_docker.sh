#!/bin/bash
# Docker 容器环境搭建
#
# 两种方式:
#   方式1（默认）: 基于已有镜像直接 docker run
#   方式2:        使用 Dockerfile 构建自定义镜像
#
# 用法:
#   bash 04_env_docker.sh                   # 方式1，默认配置
#   bash 04_env_docker.sh --build           # 方式2，使用 Dockerfile 构建
#   bash 04_env_docker.sh --name my-container --tag v0.20.2rc1
#
# 前置条件:
#   - 已安装 docker
#   - NPU 驱动已安装 (/dev/davinci* 设备存在)

set -euo pipefail

# ============================================================
# 配置参数
# ============================================================

# 容器名称
CONTAINER_NAME="${CONTAINER_NAME:-deepspec-sy}"

# vLLM-Ascend 基础镜像
VLLM_IMAGE="${VLLM_IMAGE:-quay.io/ascend/vllm-ascend}"
VLLM_TAG="${VLLM_TAG:-v0.20.2rc1}"

# 工作区挂载（宿主机:容器）
WORKSPACE_MOUNTS=(
    "/home:/home"
)

# ============================================================
# Dockerfile 方式（备用）
# ============================================================
DOCKERFILE_CONTENT=$(cat <<'DOCKERFILE'
# 使用 vLLM-Ascend 基础镜像
ARG BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.20.2rc1
FROM $BASE_IMAGE

# 安装额外系统工具
RUN apt-get update -y && apt-get install -y \
    curl vim git net-tools \
    && rm -rf /var/lib/apt/lists/*

# pip 镜像源（若有需要可取消注释）
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 克隆 speculators 仓库（若有需要）
# RUN git clone https://github.com/your-org/speculators.git /workspace/speculators
# RUN cd /workspace/speculators && pip install -e .

CMD ["bash"]
DOCKERFILE
)

# ============================================================
# 辅助函数
# ============================================================

build_docker_devices() {
    local npu_devices=(
        /dev/davinci0 /dev/davinci1 /dev/davinci2 /dev/davinci3
        /dev/davinci4 /dev/davinci5 /dev/davinci6 /dev/davinci7
        /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
    )
    for dev in "${npu_devices[@]}"; do
        if [ -e "$dev" ]; then
            echo -n " --device ${dev}"
        fi
    done
}

build_mounts() {
    local mounts=(
        "${WORKSPACE_MOUNTS[@]}"
        /etc/hccn.conf:/etc/hccn.conf
        /usr/local/Ascend/driver:/usr/local/Ascend/driver
        /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons
        /usr/local/sbin:/usr/local/sbin
        /var/log/npu:/usr/slog
    )
    for m in "${mounts[@]}"; do
        local src="${m%%:*}"
        if [ -e "$src" ] || [ -d "$src" ]; then
            echo -n " -v ${m}"
        fi
    done
}

# ============================================================
# 方式1: 直接 docker run
# ============================================================

run_direct() {
    local full_image="${VLLM_IMAGE}:${VLLM_TAG}"

    echo "=== Pulling image: ${full_image} ==="
    docker pull "${full_image}"

    # 检查容器是否已存在
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "容器 ${CONTAINER_NAME} 已存在，删除重建..."
        docker rm -f "${CONTAINER_NAME}" > /dev/null
    fi

    local cmd="docker run -d --rm --name ${CONTAINER_NAME} --privileged"
    cmd+="$(build_docker_devices)"
    cmd+="$(build_mounts)"
    cmd+=" -e SOC_VERSION=ascend910b1"
    cmd+=" -e TASK_QUEUE_ENABLE=1"
    cmd+=" -e OMP_NUM_THREADS=1"
    cmd+=" -e DEBIAN_FRONTEND=noninteractive"
    cmd+=" ${full_image} bash -c 'apt-get update -y && apt-get install -y curl vim && exec bash'"

    echo "=== Creating container: ${CONTAINER_NAME} ==="
    eval "${cmd}"
    echo "=== Done ==="
    echo "进入容器: docker exec -it ${CONTAINER_NAME} bash"
}

# ============================================================
# 方式2: 使用 Dockerfile 构建
# ============================================================

run_build() {
    local build_dir
    build_dir=$(mktemp -d)

    echo "${DOCKERFILE_CONTENT}" > "${build_dir}/Dockerfile"

    local custom_image="${CONTAINER_NAME}:custom"

    echo "=== Building image: ${custom_image} ==="
    docker build \
        --build-arg BASE_IMAGE="${VLLM_IMAGE}:${VLLM_TAG}" \
        -t "${custom_image}" \
        "${build_dir}"

    rm -rf "${build_dir}"

    # 创建容器
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker rm -f "${CONTAINER_NAME}" > /dev/null
    fi

    local cmd="docker run -d --rm --name ${CONTAINER_NAME} --privileged"
    cmd+="$(build_docker_devices)"
    cmd+="$(build_mounts)"
    cmd+=" -e SOC_VERSION=ascend910b1"
    cmd+=" -e TASK_QUEUE_ENABLE=1"
    cmd+=" -e OMP_NUM_THREADS=1"
    cmd+=" ${custom_image}"

    echo "=== Creating container: ${CONTAINER_NAME} ==="
    eval "${cmd}"
    echo "=== Done ==="
    echo "进入容器: docker exec -it ${CONTAINER_NAME} bash"
}

# ============================================================
# 主入口
# ============================================================

BUILD_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD_MODE=true; shift ;;
        --name) CONTAINER_NAME="$2"; shift 2 ;;
        --tag) VLLM_TAG="$2"; shift 2 ;;
        --image) VLLM_IMAGE="$2"; shift 2 ;;
        --help)
            echo "用法: $0 [选项]"
            echo "  --name NAME    容器名称 (默认: deepspec-sy)"
            echo "  --tag TAG      vLLM 镜像标签 (默认: v0.20.2rc1)"
            echo "  --image IMAGE  vLLM 镜像名 (默认: quay.io/ascend/vllm-ascend)"
            echo "  --build        使用 Dockerfile 构建自定义镜像"
            exit 0 ;;
        *) echo "未知选项: $1"; exit 1 ;;
    esac
done

if $BUILD_MODE; then
    run_build
else
    run_direct
fi
