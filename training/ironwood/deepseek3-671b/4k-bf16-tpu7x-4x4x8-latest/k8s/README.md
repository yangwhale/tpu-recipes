# 在 Ironwood GKE 集群上用 Kubernetes JobSet 预训练 deepseek3-671b（128 芯片 / 最新 MaxText）

本配方说明如何在 [Ironwood GKE 集群](https://cloud.google.com/kubernetes-engine)
上运行 deepseek3-671b 的
[MaxText](https://github.com/AI-Hypercomputer/maxtext) 预训练任务：通过 apply
一份 Kubernetes manifest 部署 JobSet 资源。

所有内容都以原生 Kubernetes 对象表达，**不依赖 XPK**。对应的 XPK 版本见
[../xpk](../xpk/README.md)。

> English version: [README.en.md](README.en.md)

## 这个配方解决什么问题

上游的 `4k-bf16-tpu7x-4x4x8` 配方固定在重构前的 MaxText API
（`MaxText.train` 入口、`fsdp_shard_on_exp` 参数名），因此拿不到 2026-03 之后
合入的 DeepSeek V3 MoE 优化。本配方保持完全相同的 128 芯片并行配置，但改用
当前的 MaxText main 分支运行时。

数据集使用 MaxText 内置的 synthetic 数据，**不需要任何外部存储**。目标是验证
流程与性能，而不是训练出可用模型。需要真实数据集时见文末
[改用 Lustre 存储](#附录改用-lustre-存储)。

## 任务配置

-   序列长度（Sequence Length）：4096
-   精度：bf16
-   芯片数：128（4x4x8 拓扑，32 个 VM × 4 芯片）
-   数据集：synthetic（MaxText 内置，无外部依赖）
-   Checkpoint：关闭

## 与其他配方的关系

| | `4k-bf16-tpu7x-4x4x8/k8s` | 本配方 |
| --- | --- | --- |
| MaxText 入口 | `MaxText.train` | `src.maxtext.trainers.pre_train.train` |
| config 路径 | `MaxText/configs/base.yml` | `src/maxtext/configs/base.yml` |
| MoE 分片参数 | `fsdp_shard_on_exp=True` | `shard_exp_on_fsdp=True`（上游已重命名） |
| XLA | — | 增加 `--xla_tpu_dvfs_p_state=3` |
| 并行配置 | 相同 | 相同 |

### 为什么 MoE 分片用 `shard_exp_on_fsdp` 而不是 2D 分片

256 芯片的 `4k-bf16-tpu7x-4x8x8-lustre` 配方使用
`use_2d_fsdp_sharding=True` + `ici_fsdp_transpose_parallelism=2`。这条路径把
MoE 权重同时切分到 `fsdp` 和 `fsdp_transpose` 两个轴上，只有当
`ici_fsdp_transpose_parallelism > 1` 时才有意义。

128 芯片下只有一个 FSDP 轴（共 256 个 device），因此本配方保持
`ici_fsdp_transpose_parallelism=1` 并使用 `shard_exp_on_fsdp=True`，把 MLP 权重
的 expert 维度切分到这个单轴上。

MaxText 对该路径有硬性约束（见
[`pyconfig_deprecated.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/pyconfig_deprecated.py)）：
`num_experts` 必须能被 `ici_fsdp_parallelism` 整除，且要求
`ici_expert_parallelism = 1`、`ici_tensor_parallelism = 1`。DeepSeek V3 有 256 个
expert，此处 `ici_fsdp_parallelism` 解析为 256，三个条件全部满足。**该约束已在
128 芯片实机上验证通过**（见 [验证状态](#验证状态)）。

## 验证状态

2026-07-26 在 `us-central1-c` 的 128 芯片 Spot 节点池上实跑过本配方。

已验证：

-   4x4x8 节点池创建（32/32 节点 Ready，每节点 `google.com/tpu: 4`）
-   最新 MaxText main（`e50e39458`，2026-07-25）镜像构建
-   32 个 pod 全部调度并启动
-   `shard_exp_on_fsdp=True` 通过 MaxText 配置校验，未报 expert 整除错误
-   XLA HLO 编译与前几个训练步

未验证：

-   **稳态性能数据**。运行到 step 2 时 Spot 实例被抢占
    （`compute.instances.preempted`），而 step 0–1 属于 JIT 编译阶段，因此没有
    采集到稳态 step time / TFLOP。下方 [性能参考](#性能参考) 中的数字来自同拓扑
    的历史运行，不是本次结果。
-   Lustre 存储路径（见附录）。

## 性能参考

以下为**同拓扑、同并行配置、synthetic 数据**下的历史数据，使用的是较旧的
MaxText（`maxtext-tutorial-v1.5.0`，JAX `0.8.2.dev20251215`）。列出用于对照，
本配方在最新 MaxText 上的实测数据尚待采集。

| 配置 | 精度 | Step Time | TFLOP/s/chip | Tokens/s/chip |
| --- | --- | --- | --- | --- |
| 4x4x8（128 芯片） | bf16 | 27.00 s | 608.0 | 2,427.7 |
| 4x4x8（128 芯片） | fp8 | 22.39 s | 733.1 | 2,926.5 |

注意单位：MaxText 日志输出的 `TFLOP/s/device` 是 **per TensorCore**，而 TPU v7
每个 chip 有 2 个 TensorCore，因此 per-chip 数值是日志值的 2 倍。上表已换算为
per-chip。以 v7 每 chip BF16 峰值 2,307 TFLOPS 计，608 TFLOP/s/chip 约合 26.4%
MFU。

## 前置条件

-   **GKE 集群**：已安装并运行 [JobSet](https://jobset.sigs.k8s.io/docs/installation/)。
-   **容器镜像**：包含 MaxText 的 runner 镜像，且 GKE 集群可以拉取。构建方式见
    [容器镜像](#容器镜像)。
-   **工具**：工作机上已安装 `gcloud`、`kubectl`、`gke-gcloud-auth-plugin` 和
    `envsubst`。缺 `envsubst` 时用
    `sudo apt-get update && sudo apt-get install -y gettext-base` 安装。
-   **权限**：可对目标集群执行 `kubectl apply`，且集群有权限拉取镜像。
-   **Placement policy**：TPU v7 **不支持自动创建** placement policy，必须提前
    创建对应拓扑的 policy，否则无法创建 multi-host 节点池。

## 创建 128 芯片节点池

TPU v7 的 multi-host 节点池是 all-or-nothing 分配：32 台机器必须同时拿到，
4x4x8 拓扑要求物理相邻的完整立方体。

```bash
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE="us-central1-c"
export REGION="us-central1"

# 1. 创建 4x4x8 的 placement policy（v7 不支持自动创建，必须先建）
gcloud compute resource-policies create workload-policy tpu7x-128chip \
  --region=${REGION} --project=${PROJECT_ID} \
  --type=HIGH_THROUGHPUT --accelerator-topology=4x4x8

# 2. 创建节点池：32 个 VM × 4 芯片 = 128 芯片
gcloud container node-pools create np-tpu7x-128chip \
  --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} \
  --node-locations=${ZONE} \
  --machine-type=tpu7x-standard-4t \
  --tpu-topology=4x4x8 \
  --num-nodes=32 \
  --spot \
  --placement-policy=tpu7x-128chip \
  --disk-type=hyperdisk-balanced --disk-size=200

# 3. 确认 32 个节点全部 Ready 且 TPU 资源已注册
kubectl get nodes -l cloud.google.com/gke-nodepool=np-tpu7x-128chip \
  -o custom-columns='NODE:.metadata.name,TPU:.status.allocatable.google\.com/tpu'
```

用 Spot 时要有被抢占的预期。本次验证中节点池在运行约 10 分钟后被回收。

## 容器镜像

本配方针对 **MaxText main 分支最新版**。

```bash
export PROJECT_ID=""
export CONTAINER_REGISTRY=""   # 例如 us-docker.pkg.dev/${PROJECT_ID}/gcr.io
export CLOUD_IMAGE_NAME="${USER}-maxtext-latest"

# Python 3.11 虚拟环境
uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.11 --clear
source ${HOME}/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 记录本次构建的 commit，保证结果可复现
git rev-parse HEAD

# 1. 构建依赖镜像（MODE=nightly 自动解析最新 jax/libtpu）
bash src/dependencies/scripts/docker_build_dependency_image.sh MODE=nightly

# 2. 构建 runner 镜像 —— 这一步才会把 MaxText 源码放进 /deps
docker build --network host \
  -f src/dependencies/dockerfiles/maxtext_runner.Dockerfile \
  --build-arg BASEIMAGE=maxtext_base_image \
  --build-arg PACKAGE_DIR=src \
  -t maxtext_base_image__runner .

# 3. 推送
docker tag maxtext_base_image__runner:latest ${CONTAINER_REGISTRY}/${CLOUD_IMAGE_NAME}:runner
docker push ${CONTAINER_REGISTRY}/${CLOUD_IMAGE_NAME}:runner

deactivate
```

**注意必须推 runner 镜像，不是 base 镜像。** `maxtext_base_image` 只含依赖，
不含 MaxText 源码，用它跑会因为找不到模块而失败。

已验证版本：MaxText `e50e39458`（2026-07-25），nightly jax/libtpu，Python 3.11。

## 运行配方

### 1. 配置环境变量

```bash
export PROJECT_ID=""
export CLUSTER_NAME=""
export REGION="us-central1"
export WORKLOAD_IMAGE=""     # 上一步推送的 runner 镜像
export BASE_OUTPUT_DIR=""    # 例如 "gs://your-bucket/ds3-run"
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-deepseekv3-671b-128")-$(date +%Y%m%d-%H%M)"
```

### 2. 提交任务

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

envsubst '${BASE_OUTPUT_DIR} ${WORKLOAD_NAME} ${WORKLOAD_IMAGE}' < k8s_manifest.yaml | kubectl apply -n default -f -
```

## 监控任务

```bash
# JobSet 状态
kubectl get jobset -n default ${WORKLOAD_NAME}

# 32 个 pod 的状态分布
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} --no-headers | awk '{print $3}' | sort | uniq -c

# 跟踪第一个 pod 的日志
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default ${POD_NAME}

# 只看训练步
kubectl logs -n default ${POD_NAME} | grep 'completed step'
```

Spot 被抢占时，可以用下面的命令确认：

```bash
gcloud logging read 'protoPayload.methodName="compute.instances.preempted"' \
  --project ${PROJECT_ID} --limit 5 --format='value(timestamp,protoPayload.resourceName)'
```

## 读取结果

跳过前几步再看稳态值：step 0 是 JIT 编译（可能上百秒），如果开了 profiler，
采集步和其后一步也会偏慢。

| 指标 | 说明 |
| --- | --- |
| Step Time | 一个训练步的端到端时间 |
| TFLOP/s/device | MaxText 日志值，per TensorCore（半个 chip） |
| TFLOP/s/chip | per-chip，等于日志值 × 2 |
| MFU | per-chip 值 ÷ 2,307（v7 BF16 每 chip 峰值） |

单位关系（chip / device / TensorCore）跨代际不同，详见
[TPU 单位对照](../../../TPU-UNITS.md)。

## 清理资源

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default

# 节点池缩到 0（保留配置，之后可再扩）
gcloud container clusters resize ${CLUSTER_NAME} \
  --node-pool=np-tpu7x-128chip --num-nodes=0 \
  --region=${REGION} --project=${PROJECT_ID} --quiet

# 或删除整个节点池
gcloud container node-pools delete np-tpu7x-128chip \
  --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} --quiet
```

placement policy 是可复用资源，不产生费用，建议保留。

## 常见问题

**`file:///deps does not appear to be a Python project`**

runner 镜像的 `/deps` 下只有 `benchmarks/`、`src/`、`tests/`、`pytest.ini`，
没有 `pyproject.toml` 或 `setup.py`，所以不要执行 `pip install -e .`。
直接 `cd /deps` 后运行入口模块即可。

**`No module named ...` 或找不到 config**

config 的正确路径是 `src/maxtext/configs/base.yml`。`/deps/maxtext` 这个目录
不存在。

**assets 路径相关报错**

runner 镜像已经设好 `MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets`
（**小写** `maxtext`）和 `MAXTEXT_PKG_DIR`，不要在启动命令里手动 export 覆盖。
旧配方里写的 `/deps/src/MaxText/assets`（大写）在当前镜像中不存在。

**节点池创建后节点数为 0**

检查 placement policy 是否存在且拓扑匹配。TPU v7 不支持自动创建 policy。

## 附录：改用 Lustre 存储

如果需要用真实数据集和 checkpoint，可以挂载 Google Cloud Managed Lustre。
前提是 Lustre 实例与 GKE 集群在**同一个 VPC 网络**。

1. 启用 CSI 驱动（GKE 需 `1.34.0-gke.2201000` 或更高）：

```bash
gcloud container clusters update ${CLUSTER_NAME} \
  --location ${ZONE} --project ${PROJECT_ID} \
  --update-addons=LustreCsiDriver=ENABLED
```

2. 创建 Lustre 实例（与集群同网络），并把 AllenAI C4 数据集以 ArrayRecord
   格式传输进去。数据集与 checkpoint 共用时建议至少 36 TB。

3. 编辑 `lustre_pvc.yaml` 填入实例信息后 apply：

```bash
kubectl apply -f lustre_pvc.yaml
kubectl get pvc lustre-volume -n default   # 确认为 Bound
```

4. 修改 `k8s_manifest.yaml`：

在 `volumeMounts` 中增加

```yaml
- mountPath: /mnt/lustre
  name: lustre-volume
```

在 `volumes` 中增加

```yaml
- name: lustre-volume
  persistentVolumeClaim:
    claimName: lustre-volume
```

并把 MaxText 参数中的

```
tokenizer_path=src/maxtext/assets/tokenizer.mistral-v3 dataset_type=synthetic enable_checkpointing=False
```

替换为

```
tokenizer_path='deepseek-ai/DeepSeek-V3-Base' tokenizer_type=huggingface
enable_checkpointing=True checkpoint_storage_concurrent_gb=400 async_checkpointing=true
enable_single_replica_ckpt_restoring=true checkpoint_storage_target_data_file_size_bytes=209715200
dataset_type='grain' grain_file_type=arrayrecord
grain_train_files=${DATASET_BUCKET_MOUNTED_PATH}/multilingual-c4/array-record/c4/multilingual/3.0.1/*.arrayrecord
grain_worker_count=2 checkpoint_period=25
```

同时把 `BASE_OUTPUT_DIR` 指向 `/mnt/lustre/checkpoints`，并设置
`DATASET_BUCKET_MOUNTED_PATH`（例如 `/mnt/lustre/datasets`），envsubst 时一并传入。

**该路径尚未实机验证**，参数取自上游 256 芯片的
`4k-bf16-tpu7x-4x8x8-lustre` 配方。

## 下一步

本配方目标是提供可复现的「0 到 1」体验。更深入的定制——不同 XLA flag 调优、
自定义实验——建议直接使用 MaxText 仓库的 `benchmark_runner.py`，详见
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md)。
