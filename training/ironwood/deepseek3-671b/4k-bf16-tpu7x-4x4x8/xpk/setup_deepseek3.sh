#!/bin/bash

# =============================================================================
# DeepSeek3-671B - 集群 TPU Node Pool 设置
# =============================================================================
# 在现有 GKE 集群上添加 tpu7x-4x4x8 (128 chips) node pool
# 复用已有的 Docker 镜像（MaxText 框架通用）
# =============================================================================

set -e

export PROJECT_ID="cloud-tpu-multipod-dev"
export CLUSTER_NAME="chrisya-v7x-training"
export ZONE="us-central1-c"
export RESERVATION_NAME="ghostfish-n5yz4l5ckudco"

echo "=========================================="
echo "Adding tpu7x-4x4x8 node pool to cluster"
echo "=========================================="
echo "Cluster: ${CLUSTER_NAME}"
echo "TPU Type: tpu7x-4x4x8 (128 chips, 32 hosts)"
echo ""

# 检查是否已有 tpu7x-4x4x8 node pool
if gcloud container node-pools list \
    --cluster=${CLUSTER_NAME} --zone=us-central1 \
    --project=${PROJECT_ID} \
    --format="value(name)" 2>/dev/null | grep -q "np-"; then
    echo "TPU node pool already exists:"
    gcloud container node-pools list \
        --cluster=${CLUSTER_NAME} --zone=us-central1 \
        --project=${PROJECT_ID} \
        --format="table(name,config.machineType)"
    echo ""
    echo "If you need a different TPU type, delete the existing pool first:"
    echo "  gcloud container node-pools delete <POOL_NAME> --cluster=${CLUSTER_NAME} --zone=us-central1 --project=${PROJECT_ID}"
    exit 0
fi

# xpk cluster adapt 在 v0.16.1 有 bug (memory_limit 属性缺失)
# 改用 xpk cluster create 的 --update 行为（对已存在的集群会添加 node pool）
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-4x4x8 \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}

echo "=========================================="
echo "Node pool setup complete!"
echo "=========================================="
echo ""
echo "Next: ./submit_deepseek3.sh"
