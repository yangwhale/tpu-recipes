#!/bin/bash

# =============================================================================
# DeepSeek3-671B BF16 训练任务提交脚本
# TPU: tpu7x-8x8x8 (512 chips / 1024 devices / 128 hosts)
# Precision: bf16
#
# 基于 4x8x8 (256 chips) 配置扩展而来，主要变化：
#   - device-type: tpu7x-4x8x8 → tpu7x-8x8x8
#   - 预期 GBS: 8192 (= 1024 devices × 8 per_device_batch_size)
#   - 新增 ici_data_parallelism=2 (ICI 数据并行)
#     原因: 模型 tensor dim 0 = 512，fsdp × fsdp_transpose 不能超过 512
#     4x8x8: fsdp=256 × fsdp_transpose=2 = 512 ✓
#     8x8x8 无 ici_data: fsdp=512 × fsdp_transpose=2 = 1024 > 512 ✗
#     8x8x8 加 ici_data=2: fsdp=256 × fsdp_transpose=2 = 512 ✓
#     ici_fsdp 自动计算: 1024 / (2 × 1 × 2) = 256
#   - XLA_FLAGS 与 4x8x8 相同
#
# 资源需求：
#   - 512 chips = 128 hosts × 4 chips/host
#   - Placement policy: tpu7x-1024-8x8x8-placement-policy (已存在)
#   - 需要 Kueue ResourceFlavor: 1xtpu7x-1024
#   - 需要 xpk configmap 条目: tpu7x-1024: "128"
# =============================================================================

set -e

export PROJECT_ID="cloud-tpu-multipod-dev"
export CLUSTER_NAME="chrisya-v7x-training"
export ZONE="us-central1-ai1a"
export BASE_OUTPUT_DIR="gs://chrisya-v7x-us-central1"
export WORKLOAD_IMAGE="gcr.io/cloud-tpu-multipod-dev/chrisya-maxtext-runner"
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-ds3-bf16-8x8x8")-$(date +%Y%m%d-%H%M)"

echo "=========================================="
echo "Submitting DeepSeek3-671B BF16 Training (8x8x8)"
echo "=========================================="
echo "Workload Name: ${WORKLOAD_NAME}"
echo "Docker Image:  ${WORKLOAD_IMAGE}"
echo "TPU Type:      tpu7x-8x8x8 (512 chips / 1024 devices / 128 hosts)"
echo "Precision:     bf16"
echo "Output Dir:    ${BASE_OUTPUT_DIR}"
echo ""

# XLA Flags (与 4x8x8 bf16 相同)
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
  --xla_tpu_enable_3d_reduce_scatter_decomposer=false "

# MaxText 参数 (基于 4x8x8 bf16，增加 ici_data_parallelism=2)
# 并行布局: data=2 × pipeline=1 × fsdp=256 × fsdp_transpose=2 = 1024 devices
MAXTEXT_ARGS="\
model_name=deepseek3-671b \
per_device_batch_size=8.0 \
max_target_length=4096 \
dcn_pipeline_parallelism=1 \
dcn_data_parallelism=-1 \
ici_data_parallelism=2 \
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
  --device-type=tpu7x-8x8x8 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="set -e && export ENABLE_PATHWAYS_PERSISTENCE='1' && \
export LIBTPU_INIT_ARGS='${XLA_FLAGS}' && \
export JAX_PLATFORMS='tpu,cpu' && export ENABLE_PJRT_COMPATIBILITY='true' && \
python3 -m MaxText.train MaxText/configs/base.yml ${MAXTEXT_ARGS}"

echo ""
echo "=========================================="
echo "Workload submitted!"
echo "=========================================="
echo ""
echo "Monitor:"
echo "  xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}"
echo ""
echo "View logs:"
echo "  kubectl get pods | grep ${WORKLOAD_NAME}"
echo "  kubectl logs -f <POD_NAME>"
