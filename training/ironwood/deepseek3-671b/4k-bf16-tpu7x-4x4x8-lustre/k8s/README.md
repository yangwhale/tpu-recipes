# 在 Ironwood GKE 集群上用 Kubernetes JobSet + Lustre 预训练 deepseek3-671b

本配方说明如何在 [Ironwood GKE 集群](https://cloud.google.com/kubernetes-engine)
上运行 deepseek3-671b 的
[MaxText](https://github.com/AI-Hypercomputer/maxtext) 预训练任务：通过 apply 一份
Kubernetes manifest 部署 JobSet 资源，并使用 Google Cloud Managed Lustre 作为数据集
和 checkpoint 的主存储。

所有内容都以原生 Kubernetes 对象表达，**不依赖 XPK**。对应的 XPK 版本见
[../xpk](../xpk/README.md)。

> English version: [README.en.md](README.en.md)

## 任务配置

本任务的配置如下：

-   序列长度（Sequence Length）：4096
-   精度：bf16
-   芯片数：128（4x4x8 拓扑）
-   数据集与 checkpoint 均使用 Lustre
    -   C4 多语言数据集（约 12TB），ArrayRecord 格式

## 与其他配方的关系

本配方把 128 芯片（4x4x8）的并行配置、Lustre 存储方案和当前版本的 MaxText 运行时
组合到一起，并以原始 Kubernetes manifest 的形式表达。与相邻配方的差异：

| | `4k-bf16-tpu7x-4x4x8/k8s` | 本配方 |
| --- | --- | --- |
| MaxText 入口 | `MaxText.train` | `src.maxtext.trainers.pre_train.train` |
| MoE 分片参数 | `fsdp_shard_on_exp=True` | `shard_exp_on_fsdp=True`（上游已重命名） |
| 数据集 | 合成数据 | C4 多语言，grain / ArrayRecord |
| Checkpoint | 关闭 | 开启，异步写入 Lustre |
| 存储 | 无 | Lustre PVC 挂载到 `/mnt/lustre` |

与 256 芯片的 `4k-bf16-tpu7x-4x8x8-lustre` 配方相比，本配方保持
`ici_fsdp_transpose_parallelism=1`，并使用 `shard_exp_on_fsdp=True` 而不是
`use_2d_fsdp_sharding=True`。

原因是：2D 分片路径会把 MoE 权重同时切分到 `fsdp` 和 `fsdp_transpose` 两个轴上，
只有当 `ici_fsdp_transpose_parallelism > 1` 时才有意义。在 128 芯片下只有一个
FSDP 轴（共 256 个 device），因此改用 `shard_exp_on_fsdp=True`，把 MLP 权重的
expert 维度切分到这个单轴上。

MaxText 对该路径有硬性约束：`num_experts` 必须能被 `ici_fsdp_parallelism` 整除。
DeepSeek V3 有 256 个 expert，此处 `ici_fsdp_parallelism` 解析为 256，约束成立。
该路径同时要求 `ici_expert_parallelism = 1` 且 `ici_tensor_parallelism = 1`，
两者在本配方中都是默认值。

## 前置条件

本配方假设你已经具备以下条件：

-   **GKE 集群**：已安装并运行 [JobSet](https://jobset.sigs.k8s.io/docs/installation/)
    的 GKE 集群。
-   **容器镜像**：一个预先构建好、包含 MaxText 任务的容器镜像（例如
    `gcr.io/my-project/my-maxtext-runner:latest`），且 GKE 集群可以拉取。
-   **工具**：工作机上已安装 `gcloud`、`kubectl`、`gke-gcloud-auth-plugin` 和
    `envsubst`。如果缺少 `envsubst`，用
    `sudo apt-get update && sudo apt-get install -y gettext-base` 安装。
-   **权限**：你有权限对目标集群执行 `kubectl apply`，且集群有权限拉取容器镜像。
-   **Managed Lustre**：一个与 GKE 集群位于同一网络的 Google Cloud Managed Lustre
    实例，且集群上已启用 Managed Lustre CSI 驱动（GKE 版本需 `1.34.0-gke.2201000`
    或更高）。详见下方 [Lustre 配置](#lustre-配置)。

## 编排与部署工具

本配方使用以下组合：

-   **编排** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **预训练任务配置与部署** - 使用 Kubernetes manifest（`k8s_manifest.yaml`）
    定义并部署
    [Kubernetes JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    资源，由它管理 MaxText 预训练任务的执行。

## 容器镜像

按照 XPK 配方中
[Docker 容器镜像](../xpk/README.md#docker-容器镜像)
一节构建镜像。该节针对的是 **MaxText main 分支最新版**，也正是本 manifest 所依赖的
（`src.maxtext.trainers.pre_train.train` 入口）。该节同时提供了一组已验证可用的
固定版本作为回退方案。

## Lustre 配置

每个集群只需配置一次。

### 1. 启用 Managed Lustre CSI 驱动

```bash
gcloud container clusters update ${CLUSTER_NAME} \
  --location ${ZONE} \
  --project ${PROJECT_ID} \
  --update-addons=LustreCsiDriver=ENABLED
```

### 2. 创建 Lustre 实例并准备数据集

按照[官方文档](https://docs.cloud.google.com/managed-lustre/docs/create-instance)
创建 Lustre 实例，**必须与 GKE 集群使用同一网络**。由于该实例同时承载数据集和
checkpoint，建议容量至少 36 TB。

下载 AllenAI C4 数据集，并按照
[数据传输文档](https://docs.cloud.google.com/managed-lustre/docs/transfer-data)
以 ArrayRecord 格式传输到 Lustre 实例。

### 3. 创建 PersistentVolume 和 PersistentVolumeClaim

XPK 配方把这一步封装在 `xpk storage attach` 里，而这里直接 apply PV 和 PVC。
编辑 `lustre_pvc.yaml`，填入其中的占位符（`<INSTANCE CAPACITY SIZE>`、
`<PROJECT_ID>/<ZONE>/<LUSTRE_INSTANCE_NAME>`、`<INSTANCE IP>`、
`<FILE SYSTEM NAME>`），然后 apply：

```bash
kubectl apply -f lustre_pvc.yaml

# 提交任务前先确认 PVC 状态为 Bound
kubectl get pvc lustre-volume -n default
```

manifest 会把这个 claim 挂载到每个 worker pod 的 `/mnt/lustre`。

## 训练数据集

本配方使用 AllenAI C4 多语言数据集，通过
[grain loader](https://github.com/google/grain) 以 ArrayRecord 格式从 Lustre
挂载点读取。

## 运行配方

本配方使用 Kubernetes manifest（`k8s_manifest.yaml`）部署任务。下面的命令会设置
所需的环境变量，将其替换进 `k8s_manifest.yaml`，然后把生成的配置 apply 到集群。

### 1. 配置环境变量

打开终端，按你的实际环境设置以下变量。

**注意：**
- `k8s_manifest.yaml` 与本 README 在同一目录下。
- `WORKLOAD_IMAGE` 的构建方式见
  [Docker 容器镜像](../xpk/README.md#docker-容器镜像) 一节。

```bash
# 设置你的环境变量
export PROJECT_ID=""    # 你的 GCP 项目名
export CLUSTER_NAME=""  # 你的 GKE 集群名
export ZONE=""          # 你的 GKE 集群所在 zone
export WORKLOAD_IMAGE=""   # 例如 "gcr.io/my-project/my-maxtext-runner:latest"

# Lustre 路径。这些是 pod 内的挂载点，不是 GCS URI
export BASE_OUTPUT_DIR="/mnt/lustre/checkpoints"
export DATASET_BUCKET_MOUNTED_PATH="/mnt/lustre/datasets"

# 设置任务名（可自行修改，需保证在集群内唯一）
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-deepseekv3-671b-lustre-128")-$(date +%Y%m%d-%H%M)"
```

### 2. 提交 deepseekv3-671b 预训练任务

环境变量设置好后，执行以下命令获取集群凭证并部署 JobSet：

```bash
# 获取集群凭证
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}

# Apply manifest
envsubst '${BASE_OUTPUT_DIR} ${WORKLOAD_NAME} ${WORKLOAD_IMAGE} ${DATASET_BUCKET_MOUNTED_PATH}' < k8s_manifest.yaml | kubectl apply -n default -f -
```

## 监控任务

可以用 kubectl 查看 JobSet 状态和日志：

```bash
# 查看 JobSet 状态
kubectl get jobset -n default ${WORKLOAD_NAME}

# 获取 JobSet 中第一个 pod 的名字
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')

# 跟踪该 pod 的日志
kubectl logs -f -n default ${POD_NAME}
```

也可以在 Google Cloud Console 上查看集群和 TPU 使用情况：
`https://console.cloud.google.com/kubernetes/workload/overview?project={PROJECT_ID}`

## 清理资源

### 删除指定任务

删除本配方创建的 JobSet：

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default
```

## 查看结果

任务完成后，可以通过以下方式查看结果：

-   用 `kubectl logs` 查看任务的输出日志。
-   查看 `${BASE_OUTPUT_DIR}`（即 Lustre 上的 `/mnt/lustre/checkpoints`）下的
    checkpoint 和 TensorBoard 输出。
-   如果已配置，可在 Cloud Monitoring 中查看相关指标。

## 下一步：深入探索与定制

本配方的目标是提供一个简单、可复现的「从 0 到 1」体验，帮助你快速可靠地验证环境、
在 TPU 上跑通第一次训练。

如果需要更深入的探索——比如定制模型配置、用不同的 XLA flag 调优性能、运行自定义
实验——建议直接使用 MaxText 仓库中的 `benchmark_runner.py` 脚本。该脚本提供了
MaxText 的完整灵活性，更适合希望超越初始 benchmark、按自身需求定制任务的高级用户
和研究人员。详见
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md)。
