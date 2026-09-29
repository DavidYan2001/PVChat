#!/usr/bin/env bash
set -euo pipefail

# Prepare the pinned Qwen3.5-9B training stack (Python 3.11, CUDA 12.6, H100/H200 class GPU).
# Creates a conda env, installs Torch/TorchCodec, checks out the pinned Transformers commit
# with the TorchCodec patch, builds the fast Qwen3.5 kernels and downloads the base weights.
# API keys are intentionally not accepted as command-line arguments or stored here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

TRANSFORMERS_COMMIT="63f32a8782cb70da3365acab16f2b67947737985"
TRANSFORMERS_DIR="${PROJECT_ROOT}/transformers-qwen35"
TRANSFORMERS_PATCH="${SCRIPT_DIR}/patches/transformers_qwen35_torchcodec_compat.patch"
MODEL_REPO="Qwen/Qwen3.5-9B"
MODEL_DIR="${QWEN_MODEL_PATH:-${PROJECT_ROOT}/Qwen3.5-9B}"

QWEN_ENV_NAME="${QWEN_ENV_NAME:-pvchat-qwen35-h200}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_EXE}" ]]; then
  echo "[Error] conda was not found. Load Miniconda/Anaconda first." >&2
  exit 1
fi
CONDA_BASE="$(${CONDA_EXE} info --base)"
QWEN_ENV_DIR="${QWEN_ENV_DIR:-${CONDA_BASE}/envs/${QWEN_ENV_NAME}}"
if [[ ! -x "${QWEN_ENV_DIR}/bin/python" ]]; then
  "${CONDA_EXE}" create -y -p "${QWEN_ENV_DIR}" "python=${PYTHON_VERSION}" pip
fi
PYTHON="${QWEN_ENV_DIR}/bin/python"
if ! "${PYTHON}" -c "import sys; assert sys.version_info[:2] == tuple(map(int, '${PYTHON_VERSION}'.split('.')))"; then
  echo "[Error] ${QWEN_ENV_DIR} is not Python ${PYTHON_VERSION}. Use a new QWEN_ENV_NAME or QWEN_ENV_DIR." >&2
  exit 1
fi

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCHCODEC_VERSION="${TORCHCODEC_VERSION:-0.5.0}"
CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.6.2.post1}"
FLA_VERSION="${FLA_VERSION:-0.5.1}"

"${PYTHON}" -m pip install --upgrade pip setuptools wheel
"${PYTHON}" -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
"${PYTHON}" -m pip install \
  "torchcodec==${TORCHCODEC_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
"${PYTHON}" -m pip install -r "${PROJECT_ROOT}/requirements_qwen35.txt"

if [[ ! -d "${TRANSFORMERS_DIR}/.git" ]]; then
  git clone https://github.com/huggingface/transformers.git "${TRANSFORMERS_DIR}"
fi
current_commit="$(git -C "${TRANSFORMERS_DIR}" rev-parse HEAD)"
if [[ "${current_commit}" != "${TRANSFORMERS_COMMIT}" ]]; then
  if [[ -n "$(git -C "${TRANSFORMERS_DIR}" status --porcelain)" ]]; then
    echo "[Error] ${TRANSFORMERS_DIR} has local changes and is not at the pinned commit." >&2
    exit 1
  fi
  git -C "${TRANSFORMERS_DIR}" fetch origin "${TRANSFORMERS_COMMIT}"
  git -C "${TRANSFORMERS_DIR}" checkout --detach "${TRANSFORMERS_COMMIT}"
fi

if git -C "${TRANSFORMERS_DIR}" apply --reverse --check "${TRANSFORMERS_PATCH}" >/dev/null 2>&1; then
  echo "[Setup] TorchCodec compatibility patch is already applied."
elif git -C "${TRANSFORMERS_DIR}" apply --check "${TRANSFORMERS_PATCH}"; then
  git -C "${TRANSFORMERS_DIR}" apply "${TRANSFORMERS_PATCH}"
else
  echo "[Error] Transformers patch does not match ${TRANSFORMERS_COMMIT}." >&2
  exit 1
fi
"${PYTHON}" -m pip install -e "${TRANSFORMERS_DIR}"

INSTALL_FAST_KERNELS="${INSTALL_FAST_KERNELS:-1}"
if [[ "${INSTALL_FAST_KERNELS}" == "1" ]]; then
  if ! "${PYTHON}" -c "import causal_conv1d; from importlib.metadata import version; assert version('causal-conv1d') == '${CAUSAL_CONV1D_VERSION}'" >/dev/null 2>&1; then
    echo "[Setup] Building causal-conv1d ${CAUSAL_CONV1D_VERSION} against the local glibc..."
    "${PYTHON}" -m pip uninstall -y causal-conv1d

    "${CONDA_EXE}" install -y -p "${QWEN_ENV_DIR}" \
      "nvidia/label/cuda-12.6.3::cuda-nvcc" \
      "gcc_linux-64=11" \
      "gxx_linux-64=11"

    # Activate build-tool hooks so nvcc and the Conda GCC toolchain are used.
    set +u
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${QWEN_ENV_DIR}"
    set -u
    export CUDA_HOME="${CUDA_HOME:-${QWEN_ENV_DIR}}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    export MAX_JOBS="${MAX_JOBS:-8}"

    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
      "${PYTHON}" -m pip install \
      --no-build-isolation \
      --no-cache-dir \
      --no-binary=causal-conv1d \
      "causal-conv1d==${CAUSAL_CONV1D_VERSION}"
  fi
  "${PYTHON}" -c "import causal_conv1d; from importlib.metadata import version; assert version('causal-conv1d') == '${CAUSAL_CONV1D_VERSION}'; print('[Setup] causal-conv1d', version('causal-conv1d'), 'OK')"

  echo "[Setup] Verifying flash-linear-attention API compatibility..."
  if ! "${PYTHON}" -c "from importlib.metadata import version; assert version('fla-core') == '${FLA_VERSION}'; assert version('flash-linear-attention') == '${FLA_VERSION}'; from fla.modules import FusedRMSNormGated; from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule" >/dev/null 2>&1; then
    "${PYTHON}" -m pip uninstall -y fla-core flash-linear-attention
    "${PYTHON}" -m pip install --no-deps \
      "fla-core==${FLA_VERSION}" \
      "flash-linear-attention==${FLA_VERSION}"
  fi
  "${PYTHON}" -c "from importlib.metadata import version; from fla.modules import FusedRMSNormGated; from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule; print('[Setup] FLA', version('fla-core'), 'OK')"

  PYTHONPATH="${TRANSFORMERS_DIR}/src:${SCRIPT_DIR}" "${PYTHON}" -c \
    "from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available; assert is_fast_path_available; print('[Setup] Qwen3.5 fast path: enabled')"
fi

DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
  "${QWEN_ENV_DIR}/bin/hf" download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[Warning] ffmpeg is not on PATH. Load an FFmpeg module before training." >&2
fi

PYTHONPATH="${TRANSFORMERS_DIR}/src:${SCRIPT_DIR}" "${PYTHON}" -c \
  'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__); print("cuda", torch.cuda.is_available())'

echo "[Done] environment=${QWEN_ENV_DIR}"
echo "[Done] model=${MODEL_DIR}"
echo "[Next] Download the PVChat-R1 data bundle and run the Stage 1/2/3 commands from README.md"
