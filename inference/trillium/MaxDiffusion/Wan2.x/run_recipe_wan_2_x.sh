#!/bin/bash
set -e

# --- Environment Setup ---
# This script requires uv and a Python virtual environment with xpk installed.
# If you haven't set up uv and the environment, please refer to the README.md.
WORKLOAD_TYPE="$1"

if [ -z "$WORKLOAD_TYPE" ]; then
    echo "Error: No input provided."
    echo "Usage: $0 {wan2-1-t2v|wan2-1-i2v|wan2-2-t2v|wan2-2-i2v}"
    exit 1
fi

# Activate the virtual environment
if [ -f "${UV_VENV_PATH}/bin/activate" ]; then
    source "${UV_VENV_PATH}/bin/activate"
else
    echo "Error: Virtual environment not found at ${UV_VENV_PATH}. Check README.md."
    exit 1
fi

# # Check if xpk is installed in the venv
if ! pip show xpk &> /dev/null; then
    echo "xpk not found in the virtual environment. Please install it by running:"
    echo "pip install xpk==1.3.0"
    exit 1
fi

# --- End Environment Setup ---

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---

# Environmental Variables
export WORKLOAD_IMAGE="${WORKLOAD_IMAGE}"
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-${MODEL_NAME}")-$(date +%Y%m%d-%H%M)"
export BASE_YAML_CONFIG_WAN_2_1_T2V="src/maxdiffusion/configs/base_wan_14b.yml"
export BASE_YAML_CONFIG_WAN_2_2_T2V="src/maxdiffusion/configs/base_wan_27b.yml"
export BASE_YAML_CONFIG_WAN_2_1_I2V="src/maxdiffusion/configs/base_wan_i2v_14b.yml"
export BASE_YAML_CONFIG_WAN_2_2_I2V="src/maxdiffusion/configs/base_wan_i2v_27b.yml"
export SCRIPT_PATH="src/maxdiffusion/generate_wan.py"
export COMMAND_PREFIX="pip install . && export HF_HUB_CACHE=/dev/shm && export HF_HUB_ENABLE_HF_TRANSFER=1"

# XLA Flags
XLA_FLAGS="'\"'\"' \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_enable_async_all_reduce=true'\"'\"'"

# MaxDiffusion Workload Overrides
COMMON_MAXDIFFUSION_ARGS="\
attention='\"'\"'flash'\"'\"' \
num_frames=81 \
width=1280 \
height=720 \
jax_cache_dir='\"'\"'${BASE_OUTPUT_DIR}/jax_cache/'\"'\"' \
skip_jax_distributed_system=False \
per_device_batch_size=0.25 \
ici_context_parallelism=4 \
allow_split_physical_axes=True \
flow_shift=5.0 \
enable_profiler=True \
run_name='\"'\"'${WORKLOAD_NAME}'\"'\"' \
output_dir='\"'\"'${BASE_OUTPUT_DIR}/'\"'\"' \
flash_min_seq_length=0 \
flash_block_sizes='\"'\"'{\"block_kv\":2048,\"block_kv_compute\":1024,\"block_kv_dkv\":2048,\"block_kv_dkv_compute\":1024,\"block_q\":3024,\"block_q_dkv\":3024,\"use_fused_bwd_kernel\":true}'\"'\"' \
base_output_directory='\"'\"'${BASE_OUTPUT_DIR}'\"'\"'"

# ==============================================================================
# Workload Specific Arguments
# ==============================================================================
case "$TPU_TYPE" in
    "v6e-8")
        SPECIFIC_ARGS="ici_data_parallelism=2"
        ;;
    "v6e-16")
        SPECIFIC_ARGS="ici_data_parallelism=4"
        ;;
    *)
        echo "Error: Invalid TPU_TYPE."
        echo "Only supports v6e-8 and v6e-16"
        exit 1
        ;;
esac
COMMON_MAXDIFFUSION_ARGS="${COMMON_MAXDIFFUSION_ARGS} ${SPECIFIC_ARGS}"

case "$WORKLOAD_TYPE" in
    "wan2-1-t2v")
        echo "Starting the Wan2.1-T2V..."
        SPECIFIC_ARGS="\
        model_name='\"'\"'wan2.1'\"'\"' \
        prompt='\"'\"'a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.'\"'\"' \
        guidance_scale=5.0 \
        num_inference_steps=50"
        BASE_YAML_CONFIG=${BASE_YAML_CONFIG_WAN_2_1_T2V}
        ;;
    "wan2-1-i2v")
        echo "Starting the Wan2.1-I2V..."
        SPECIFIC_ARGS="\
        model_name='\"'\"'wan2.1'\"'\"' \
        pretrained_model_name_or_path='\"'\"'Wan-AI/Wan2.1-I2V-14B-720P-Diffusers'\"'\"' \
        num_inference_steps=50"
        BASE_YAML_CONFIG=${BASE_YAML_CONFIG_WAN_2_1_I2V}
        ;;
    "wan2-2-t2v")
        echo "Starting the Wan2.2-T2V..."
        SPECIFIC_ARGS="\
        model_name='\"'\"'wan2.2'\"'\"' \
        prompt='\"'\"'a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.'\"'\"' \
        guidance_scale_low=3.0 \
        guidance_scale_high=4.0 \
        boundary_ratio=0.875 \
        num_inference_steps=40 \
        remat_policy='\"'\"'FULL'\"'\"'"
        BASE_YAML_CONFIG=${BASE_YAML_CONFIG_WAN_2_2_T2V}
        ;;
    "wan2-2-i2v")
        echo "Starting the Wan2.2-I2V..."
        SPECIFIC_ARGS="\
        model_name='\"'\"'wan2.2'\"'\"' \
        prompt="'\"'\"'a japanese pop star young woman with black hair is singing with a smile. She is inside a studio with dim lighting and musical instruments.'\"'\"'" \
        guidance_scale_low=3.0 \
        guidance_scale_high=4.0 \
        num_inference_steps=40 \
        remat_policy='\"'\"'FULL'\"'\"'"
        BASE_YAML_CONFIG=${BASE_YAML_CONFIG_WAN_2_2_I2V}
        ;;
    *)
        echo "Error: Invalid input."
        echo "Please run as: ./run_recipe_wan_2_x.sh {Wan2.1-T2V|Wan2.1-I2V|Wan2.2-T2V|Wan2.2-I2V}"
        exit 1
        ;;
esac

MAXDIFFUSION_ARGS="${COMMON_MAXDIFFUSION_ARGS} ${SPECIFIC_ARGS}"

echo "Script Path: ${SCRIPT_PATH}"
echo "MaxDiffusion Args: ${MAXDIFFUSION_ARGS}"
echo "Deploying workload via xpk..."

# ==============================================================================
# Execute Command
# ==============================================================================

cmd="xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=medium \
  --max-restarts=0 \
  --tpu-type=$TPU_TYPE \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
   \
  --workload="${WORKLOAD_NAME}" \
  --command='set -e && \
export ARTIFACT_DIR=${BASE_OUTPUT_DIR} && \
export LIBTPU_INIT_ARGS=${XLA_FLAGS} && \
${COMMAND_PREFIX} && export HF_TOKEN=${HF_TOKEN} && \
  python ${SCRIPT_PATH}  \
  ${BASE_YAML_CONFIG} \
  ${MAXDIFFUSION_ARGS} \
  run_name='\"'\"'${WORKLOAD_NAME}'\"'\"''"

eval ${cmd}
