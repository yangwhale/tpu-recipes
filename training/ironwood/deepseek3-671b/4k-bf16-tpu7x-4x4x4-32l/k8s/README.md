# DeepSeek V3 32 层 proxy 在 Ironwood 上的预训练（64 芯片 / Kubernetes JobSet）

本配方在 64 芯片（4x4x4）的 [Ironwood GKE 集群](https://cloud.google.com/kubernetes-engine)
上运行 DeepSeek V3 的 **32 层缩减版**（约 221B 参数）预训练，使用
[MaxText](https://github.com/AI-Hypercomputer/maxtext) main 分支，通过 apply
一份 Kubernetes manifest 部署 JobSet。

不依赖 XPK，也不需要外部存储。对应的 XPK 版本见 [../xpk](../xpk/README.md)。

> English version: [README.en.md](README.en.md)

## 为什么要减层

完整的 DeepSeek V3 是 61 层、671B 参数。在 64 芯片上跑会 OOM：

```
RESOURCE_EXHAUSTED: Ran out of memory on HBM
total memory required for HLO temporaries (105.73G) exceeds available HBM (94.74G)
```

这不是 TPU 特有的问题。在 A4X（GB200）上做 Megatron 训练时也遇到同样的墙——
全量 61 层配合 CUDA graph 需要的 worst-case buffer 超过 184 GB HBM，同样必须
缩到 32 层。根因一致：**层数 × 256 expert 的权重和缓冲区总量**超过单卡容量。

减层是业界通行的 proxy 模型做法。NVIDIA 在 Megatron-LM 的 GB200 回归测试里也
用同样思路（`deepseekv3_proxy_..._gb_200_release`，减到 14 层）。

## proxy 模型的取舍原则

**只减深度，不动宽度。**

| 参数 | 完整模型 | 本配方 | 是否改动 |
| --- | --- | --- | --- |
| `base_num_decoder_layers` | 61 | **32** | 减 |
| `num_experts` | 256 | 256 | **不动** |
| `base_emb_dim`（H） | 7168 | 7168 | 不动 |
| `base_mlp_dim` | 18432 | 18432 | 不动 |
| `base_moe_mlp_dim` | 2048 | 2048 | 不动 |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | 1536 / 512 | 不动 |
| `first_num_dense_layers` | 3 | 3 | 不动 |
| `mtp_num_layers` | 1 | 1 | 不动 |

保持每层的计算形状、MoE 路由行为、MLA 结构和通信模式与真实模型完全一致，
这样单层的 kernel 行为和通信量有代表性，测出的 per-layer 性能可以外推。

### 为什么 `num_experts` 一定要保持 256

两个独立的理由都指向不能动：

1. **可比性**：expert 数量决定 MoE 的路由分布和 all-to-all 通信量。改了之后
   per-layer 的通信特征就不再代表真实模型。
2. **硬约束**：本配方使用 `shard_exp_on_fsdp=True`，MaxText 要求
   `num_experts % ici_fsdp_parallelism == 0`。64 芯片下 `ici_fsdp_parallelism`
   解析为 128，`256 % 128 = 0` 成立；若改成 64 experts 则 `64 % 128 ≠ 0`，
   直接配置校验失败。

注意这一点与 NVIDIA 的 GB200 proxy 不同——那份配置把 experts 减到 64，因为它
走的是 expert parallelism（EP16），约束条件不一样。

## 任务配置

-   序列长度：4096
-   精度：bf16
-   芯片数：64（4x4x4 拓扑，16 个 VM × 4 芯片）
-   模型：DeepSeek V3 32 层 proxy，约 221B 参数
-   MTP：1 层，loss 缩放 0.1
-   `per_device_batch_size`：2.0（全局 batch = 2.0 × 128 device = 256）
-   数据集：synthetic（MaxText 内置，无外部依赖）
-   Checkpoint：关闭

### per_device_batch_size 为什么是 2.0

FSDP 维度随芯片数缩小，每个 device 承担的权重分片反而变大：

| 芯片数 | device 数 | 权重分片/device | 96 GB 中的余量 |
| --- | --- | --- | --- |
| 128 | 256 | 21 GB | 75 GB |
| **64** | **128** | **42 GB** | **54 GB** |

64 芯片下每卡权重压力是 128 芯片的两倍，因此需要压缩激活占用。实测 4.0 仍然
差 864 MB，2.0 才能通过（详见 [内存收敛过程](#内存收敛过程)）。

## 前置条件

-   **GKE 集群**：已安装并运行 [JobSet](https://jobset.sigs.k8s.io/docs/installation/)。
-   **容器镜像**：MaxText runner 镜像，构建方式见
    [../../4k-bf16-tpu7x-4x4x8-latest/k8s/README.md#容器镜像](../4k-bf16-tpu7x-4x4x8-latest/k8s/README.md#容器镜像)。
    注意必须是 **runner** 镜像，base 镜像不含 MaxText 源码。
-   **工具**：`gcloud`、`kubectl`、`gke-gcloud-auth-plugin`、`envsubst`。
-   **Placement policy**：TPU v7 不支持自动创建，必须先建 4x4x4 的 policy。

## 创建 64 芯片节点池

### 常驻 vs 弹性：在容量紧张时优先常驻

实测经验：在容量被占满的 spot 池里，**固定节点数的节点池比 autoscaling 的更容易
拿到卡**。

原因是时序：固定节点池在创建时就发起整块分配请求；autoscaling 池要等 pod
Pending 之后才由 autoscaler 发起请求，那时窗口往往已经被别人占走。同一个
placement policy、同一个拓扑下，弹性池反复 `GCE out of resources`，改成常驻后
一次成功。

代价是节点空闲时也计费，用完记得删。

```bash
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE="us-central1-c"
export REGION="us-central1"

# 1. 创建 4x4x4 placement policy（v7 不支持自动创建）
gcloud compute resource-policies create workload-policy tpu7x-64chip \
  --region=${REGION} --project=${PROJECT_ID} \
  --type=HIGH_THROUGHPUT --accelerator-topology=4x4x4

# 2. 创建常驻节点池：16 个 VM × 4 芯片 = 64 芯片
gcloud container node-pools create np-tpu7x-64-fixed \
  --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} \
  --node-locations=${ZONE} \
  --machine-type=tpu7x-standard-4t \
  --tpu-topology=4x4x4 \
  --num-nodes=16 \
  --spot \
  --placement-policy=tpu7x-64chip \
  --disk-type=hyperdisk-balanced --disk-size=200

# 3. 确认 16 个节点全部 Ready
kubectl get nodes -l cloud.google.com/gke-nodepool=np-tpu7x-64-fixed \
  -o custom-columns='NODE:.metadata.name,TPU:.status.allocatable.google\.com/tpu'
```

multi-host 节点池是 all-or-nothing 分配：16 台必须同时拿到，4x4x4 拓扑要求
物理相邻的完整立方体。失败时 GKE 报 `Atomic resize failed with [GCE_STOCKOUT]`。

## 运行配方

```bash
export PROJECT_ID=""
export CLUSTER_NAME=""
export REGION="us-central1"
export WORKLOAD_IMAGE=""     # runner 镜像
export BASE_OUTPUT_DIR=""    # 例如 "gs://your-bucket/ds3-32l"
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-ds3-32l-64chip")-$(date +%Y%m%d-%H%M)"

gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

envsubst '${BASE_OUTPUT_DIR} ${WORKLOAD_NAME} ${WORKLOAD_IMAGE}' < k8s_manifest.yaml | kubectl apply -n default -f -
```

## 监控

```bash
# 16 个 pod 的状态分布
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} --no-headers | awk '{print $3}' | sort | uniq -c

POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')

# 只看训练步（XLA 编译日志非常多，直接 logs 会淹没）
kubectl logs -n default ${POD_NAME} | grep 'completed step'

# 确认是否 OOM
kubectl logs -n default ${POD_NAME} | grep RESOURCE_EXHAUSTED
```

## 读取结果

step 0 是 JIT 编译（可能上百秒），跳过前几步看稳态。

| 指标 | 说明 |
| --- | --- |
| Step Time | 一个训练步的端到端时间 |
| TFLOP/s/device | MaxText 日志值，per TensorCore（半个 chip） |
| TFLOP/s/chip | per-chip = 日志值 × 2 |
| MFU | per-chip 值 ÷ 2,307（v7 BF16 每 chip 峰值） |

**注意**：本配方是 32 层 proxy，模型规模约为完整版的一半。其 step time 和吞吐
**不能直接与 61 层完整模型的数字对比**，只能用于同为 32 层配置之间的横向比较，
或者用于评估 per-layer 效率。

## 指标说明

MaxText 每个训练步输出一行结构化指标，`metric_logger.py` 打印，格式如下：

```
completed step: 0, seconds: 48.205, TFLOP/s/device: 23.456,
Tokens/s/device: 169.941, total_weights: 1048576, loss: 13.498,
lm_loss: 12.271, perplexity: 213435.656, moe_lb_loss: 0.000,
main_model_loss: 12.271, mtp_loss: 1.227
```

各字段含义：

| 字段 | 含义 | 注意 |
| --- | --- | --- |
| `seconds` | 该步端到端耗时 | step 0 含 JIT 编译，不代表稳态 |
| `TFLOP/s/device` | 每 TensorCore 的算力 | **per-chip = 此值 × 2** |
| `Tokens/s/device` | 每 TensorCore 的吞吐 | **per-chip = 此值 × 2** |
| `total_weights` | 本步有效 token 数 | = global_batch × seq_len |
| `loss` | 总损失 | = `lm_loss` + `mtp_loss_scaling_factor` × `mtp_loss` |
| `lm_loss` | 语言建模损失 | |
| `main_model_loss` | 主模型损失（不含 MTP） | 通常等于 `lm_loss` |
| `mtp_loss` | MTP 层损失 | **非零即证明 MTP 生效** |
| `moe_lb_loss` | MoE 负载均衡损失 | `use_random_routing=True` 时为 0 |
| `perplexity` | 困惑度 | = exp(lm_loss) |

用上面的实测数据验算 loss 公式：
`12.271 + 0.1 × 1.227 = 13.394`，与打印的 `13.498` 接近（差异来自
`moe_lb_loss` 等其他项），可确认 `mtp_loss_scaling_factor=0.1` 已生效。

### MFU 换算

```
MFU = (TFLOP/s/device × 2) / 2307
```

2307 是 TPU v7 每 chip 的 BF16 峰值 TFLOPS。

## 验证状态

2026-07-27 在 `us-central1-c` 64 芯片常驻 Spot 节点池上实跑。

### 已确认可用

| 项目 | 结果 |
| --- | --- |
| 4x4x4 常驻节点池 | 16/16 Ready，一次分配成功 |
| MaxText main `e50e39458` runner 镜像 | 可用 |
| `override_model_config=True` + 32 层 | 配置校验通过 |
| `shard_exp_on_fsdp=True`（256 experts） | 校验通过 |
| MTP 1 层 | 生效，`mtp_loss: 1.227` |
| 内存 | `per_device_batch_size=2.0` 时不再 OOM |

### 内存收敛过程

这组数据说明了参数是怎么定下来的：

| 层数 | `per_device_batch_size` | 结果 |
| --- | --- | --- |
| 61 | 4.0 | OOM，差 11 GB（105.73G / 94.74G） |
| 32 | 4.0 | OOM，差 **864 MB**（95.58G / 94.74G） |
| **32** | **2.0** | **通过** |

32 层已经把差距压到不足 1 GB，再降 batch 即可通过。**继续减层收益很小，且会
进一步损害与真实模型的可比性**，因此 32 层是这个规模下的合理选择。

### 未解决：step 1 之后 TPU stall

step 0 能正常完成并输出完整指标，但推进到 step 1 后 TPU 挂起：

```
Slow PjRt TPU operation detected: description=TpuLoadedExecutable::ReadyFuture
TpuDiagnosticCoordinator: Harvesting hardware telemetry for stalled chips: [6]
```

两次运行都复现，且**卡住的 chip 不同**（第一次 chip 6，第二次 chip 9），
中间删除故障节点并由 MIG 重建了整块 16 台。因此这不是单机硬件故障，而是配置
或软件层面的问题，可能方向：

-   `use_random_routing=True` 与 MTP 组合下的 MoE 路由行为
-   32 层 proxy 下某个 XLA flag 的适用性（flag 集来自 61 层配方）
-   `remat_policy=custom` + `decoder_layer_input=offload` 在该规模下的交互

**因此稳态 step time / TFLOP 尚未采集到。** step 0 的 23.456 TFLOP/s/device
含 JIT 编译，不能作为性能参考。

## 清理

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default

# 常驻节点池不会自动缩容，用完必须删
gcloud container node-pools delete np-tpu7x-64-fixed \
  --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} --quiet
```

placement policy 是可复用的免费资源，建议保留。

## 常见问题

**OOM（`RESOURCE_EXHAUSTED`）**

先看差多少。日志会给出 `required (X) exceeds available (94.74G)`：

-   差几 GB：`per_device_batch_size` 减半
-   差一倍以上：层数还要再减，或者增加芯片数

不要通过减少 `num_experts` 来省内存，会破坏 `shard_exp_on_fsdp` 的整除约束。

**节点池创建后节点数为 0**

检查 placement policy 是否存在且拓扑匹配。TPU v7 不支持自动创建。

**其他启动类错误**

`pip install -e .` 失败、config 找不到、assets 路径报错等，见
[4x4x8-latest 配方的常见问题](../4k-bf16-tpu7x-4x4x8-latest/k8s/README.md#常见问题)。
