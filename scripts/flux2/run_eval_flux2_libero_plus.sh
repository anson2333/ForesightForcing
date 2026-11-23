#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

SUITE="libero_plus"
CONFIG_NAME="sim_libero"
FLUX2_VARIANT="${FLUX2_VARIANT:-4b}" # 4b | 9b
TASK="${TASK:-libero_plus}"

imagewam_require_env FLUX2_SRC
imagewam_require_env FLUX2_AE_MODEL_PATH
imagewam_require_env FLUX2_MODEL_PATH
imagewam_ckpt_from_exp
imagewam_require_env CKPT_PATH
imagewam_require_env DATASET_STATS_PATH

FLUX2_QWEN3_MODEL_SPEC="${FLUX2_QWEN3_MODEL_SPEC:-Qwen/Qwen3-4B}"
QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-}"
export PYTHONPATH="${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export WORKER_PYTHONPATH="${PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
LIBERO_WORKER_ENV_SOURCE="${LIBERO_WORKER_ENV_SOURCE:-}"
export LIBERO_WORKER_ENV_SOURCE
imagewam_prepare_eval_ckpt

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluate_results/libero_plus_no_tmux/${TASK}_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

NO_TMUX_ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --config-name "${CONFIG_NAME}"
  --task-choice "${TASK}"
  --ckpt "${CKPT_PATH}"
  --num-trials "${NUM_TRIALS:-1}"
  --gpu-ids "${GPU_IDS}"
  --max-tasks-per-gpu "${MAX_TASKS_PER_GPU:-2}"
  --chunk-size "${CHUNK_SIZE:-20}"
  --chunk-timeout-sec "${CHUNK_TIMEOUT_SEC:-3600}"
  --single-timeout-sec "${SINGLE_TIMEOUT_SEC:-900}"
  --max-retries "${MAX_RETRIES:-1}"
  --launch-stagger-sec "${LAUNCH_STAGGER_SEC:-0}"
)
if [ -n "${TASK_FILE:-}" ]; then
  NO_TMUX_ARGS+=(--task-file "${TASK_FILE}")
else
  NO_TMUX_ARGS+=(--task-suite-names "${TASK_SUITE_NAMES:-libero_10,libero_goal,libero_spatial,libero_object}")
  NO_TMUX_ARGS+=(--task-sample-ratio "${TASK_SAMPLE_RATIO:-1.0}")
fi
if [ "${RESUME:-false}" = "true" ]; then
  NO_TMUX_ARGS+=(--resume)
fi

OVERRIDES=(
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}"
  "model.flux2_src_path=${FLUX2_SRC}"
  "model.flux2_model_path=${FLUX2_MODEL_PATH}"
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}"
  "model.variant=klein-base-${FLUX2_VARIANT}"
  "model=imagewam_flux2_klein_${FLUX2_VARIANT}_base"
  "model.qwen3_model_spec=${FLUX2_QWEN3_MODEL_SPEC}"
  "model.load_text_encoder=true"
  "model.pack_proprio_after_text=true"
  "EVALUATION.action_horizon=${ACTION_HORIZON:-16}"
  "EVALUATION.replan_steps=${REPLAN_STEPS:-12}"
  "model.proprio_dim=${PROPRIO_DIM:-8}"
  "model.qwen_context_len=${QWEN_CONTEXT_LEN:-128}"
  "data.train.qwen_text_cache_format=qwen3_flux2"
  "model.flux2_lora_config.enabled=${FLUX2_LORA_ENABLED:-false}"
)
if [ -n "${QWEN_CACHE_DIR}" ]; then
  OVERRIDES+=("data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}")
fi

imagewam_print_config SUITE TASK CKPT_PATH DATASET_STATS_PATH FLUX2_SRC FLUX2_MODEL_PATH FLUX2_AE_MODEL_PATH OUTPUT_DIR GPU_IDS
imagewam_run imagewam_python experiments/libero/run_libero_no_tmux.py "${NO_TMUX_ARGS[@]}" -- "${OVERRIDES[@]}" "$@"
