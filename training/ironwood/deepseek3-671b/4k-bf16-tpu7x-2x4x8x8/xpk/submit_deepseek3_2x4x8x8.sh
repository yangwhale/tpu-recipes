#!/bin/bash

# =============================================================================
# DeepSeek3-671B BF16 Multi-Slice DCN 训练任务提交脚本
# TPU: 2 × tpu7x-4x8x8 (2 slices × 256 chips = 512 chips total)
# Precision: bf16
#
# Multi-Slice DCN 方案 vs 单 slice 8x8x8 方案：
#   - 8x8x8 单 slice: ici_data_parallelism=2，FSDP 效率下降 45%
#   - 2×4x8x8 multi-slice: 每个 slice 独立 FSDP（满血效率），
#     slice 间仅通过 DCN 同步梯度（通信量小，可 overlap）
#
# 并行布局（每个 slice 内）:
#   ici: pipeline=1 × fsdp_transpose=2 × fsdp=256 = 512 devices
#   dcn: data=2 × pipeline=1 (跨 2 个 slice)
#
# 预期性能：
#   - per-chip 效率接近单 slice 4x8x8 (~605 TFLOP/s/chip)
#   - 总吞吐约为 4x8x8 的 2 倍 (~1.2M Tokens/s)
#   - 对比 8x8x8 单 slice 的 330 TFLOP/s/chip，预期提升 ~83%
#
# 资源需求：
#   - 2 × 64 hosts = 128 hosts (与 8x8x8 相同)
#   - 2 个 node pool，每个 64 hosts，使用 tpu7x-4x8x8 placement policy
# =============================================================================

set -e

export PROJECT_ID="cloud-tpu-multipod-dev"
export CLUSTER_NAME="chrisya-v7x-training"
export ZONE="us-central1-ai1a"
export BASE_OUTPUT_DIR="gs://chrisya-v7x-us-central1"
export WORKLOAD_IMAGE="gcr.io/cloud-tpu-multipod-dev/chrisya-maxtext-runner"
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-ds3-bf16-2x4x8x8")-$(date +%Y%m%d-%H%M)"

echo "=========================================="
echo "Submitting DeepSeek3-671B BF16 Multi-Slice DCN Training"
echo "=========================================="
echo "Workload Name: ${WORKLOAD_NAME}"
echo "Docker Image:  ${WORKLOAD_IMAGE}"
echo "TPU Type:      2 × tpu7x-4x8x8 (2 slices × 256 chips = 512 chips)"
echo "Precision:     bf16"
echo "Output Dir:    ${BASE_OUTPUT_DIR}"
echo ""

# XLA Flags (与单 slice 4x8x8 完全相同)
XLA_FLAGS=" \
  --xla_tpu_scoped_vmem_limit_kib=65536 \
  --xla_tpu_bf16_emission_mode=NATIVE_EMISSION \
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_use_tc_device_shape_on_sc=True \
  --xla_sc_disable_megacore_partitioning=True \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_enable_async_all_gather=true \
  --xla_tpu_prefer_async_allgather_to_allreduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true \
  --xla_tpu_use_single_sparse_core_for_all_gather_offload=true \
  --xla_tpu_enable_concurrent_sparse_core_offloading=true \
  --xla_tpu_aggressive_opt_barrier_removal=true \
  --xla_tpu_enable_offloading_gather_to_sparsecore=true \
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1 \
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 \
  --xla_tpu_enable_sparse_core_collective_aggregator=true \
  --xla_tpu_enable_latency_hiding_layer_scheduler=true \
  --xla_tpu_scheduler_percent_shared_memory_limit=150 \
  --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true \
  --xla_tpu_pcie_bandwidth_multiplier=0.03 \
  --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false \
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true \
  --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true \
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false \
  --xla_tpu_enable_scheduler_memory_pressure_tracking=true \
  --xla_latency_hiding_scheduler_rerun=2 "

# MaxText 参数 (基于 4x8x8 单 slice，变更 DCN 并行)
# 关键变更：
#   dcn_data_parallelism=2  (跨 2 个 slice 做数据并行)
#   dcn_pipeline_parallelism=1 (不用流水线并行)
#   slice 内 ICI 布局与单 slice 4x8x8 完全一致
MAXTEXT_ARGS="\
model_name=deepseek3-671b \
per_device_batch_size=5.0 \
max_target_length=4096 \
dcn_pipeline_parallelism=1 \
dcn_data_parallelism=2 \
ici_pipeline_parallelism=1 \
ici_fsdp_transpose_parallelism=2 \
ici_fsdp_parallelism=-1 \
allow_split_physical_axes=True \
use_iota_embed=True \
remat_policy=custom \
decoder_layer_input=offload \
opt_type=adamw \
mu_dtype=bfloat16 \
grad_dtype=bfloat16 \
use_random_routing=True \
megablox=True \
sparse_matmul=True \
use_custom_sort_vjp=True \
fsdp_shard_on_exp=False \
use_2d_fsdp_sharding=True \
sa_use_fused_bwd_kernel=True \
sa_block_q=2048 \
sa_block_kv=2048 \
sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 \
sa_block_kv_dkv_compute=2048 \
sa_block_kv_dq=2048 \
sa_block_q_dq=2048 \
attention=flash \
use_tokamax_splash=True \
use_max_logit_estimate=-1 \
cost_estimate_flops_fwd=5000000000000 \
cost_estimate_flops_bwd=5000000000000 \
float32_weight_sum=False \
use_tokamax_gmm=True \
tokenizer_path=assets/tokenizer.mistral-v3 \
dataset_type=synthetic \
dataset_path=gs://max-datasets-rogue \
enable_checkpointing=False \
steps=30 \
profiler=xplane \
profiler_steps=3 \
skip_first_n_steps_for_profiler=5 \
base_output_directory=${BASE_OUTPUT_DIR} \
run_name=${WORKLOAD_NAME}"

xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --device-type=tpu7x-4x8x8 \
  --num-slices=2 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -m MaxText.train MaxText/configs/base.yml ${MAXTEXT_ARGS}"

echo ""
echo "=========================================="
echo "Multi-Slice Workload submitted!"
echo "=========================================="
echo ""
echo "Monitor:"
echo "  xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}"
echo ""
echo "View logs (2 slices, each with 64 pods):"
echo "  kubectl get pods | grep ${WORKLOAD_NAME}"
echo "  kubectl logs -f <POD_NAME>  # check slice-job-0 (worker 0 of slice 0)"
