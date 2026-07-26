# 在 TPU Ironwood (tpu7x-4x4x8) 上训练 DeepSeek3-671B（128 芯片 / 最新 MaxText）

本配方说明如何在 [Ironwood GKE 集群](https://cloud.google.com/kubernetes-engine)
上通过 [XPK](https://github.com/AI-Hypercomputer/xpk) 运行 deepseek3-671b 的
[MaxText](https://github.com/AI-Hypercomputer/maxtext) 预训练任务。

数据集使用 MaxText 内置的 synthetic 数据，**不需要任何外部存储**。目标是验证
流程与性能，而不是训练出可用模型。需要真实数据集时见文末
[改用 Lustre 存储](#附录改用-lustre-存储)。

如果你更希望用原生 Kubernetes 对象而不是 XPK 封装——比如要把配方交付给客户、
需要对方能直接看懂到底部署了什么——请参考等价的 manifest 版本
[../k8s](../k8s/README.md)。

> English version: [README.en.md](README.en.md)

## 任务配置

本任务的配置如下：

-   序列长度（Sequence Length）：4096
-   精度：bf16
-   芯片数：128（4x4x8 拓扑，32 个 VM × 4 芯片）
-   数据集：synthetic（MaxText 内置，无外部依赖）
-   Checkpoint：关闭

## 与其他配方的关系

上游的 `4k-bf16-tpu7x-4x4x8` 配方固定在重构前的 MaxText API，因此拿不到
2026-03 之后合入的 DeepSeek V3 MoE 优化。本配方保持完全相同的 128 芯片并行
配置，但改用当前的 MaxText main 分支运行时。

| | 上游 `4k-bf16-tpu7x-4x4x8` | 本配方 |
| --- | --- | --- |
| MaxText 入口 | `MaxText.train` | `src.maxtext.trainers.pre_train.train` |
| config 路径 | `MaxText/configs/base.yml` | `src/maxtext/configs/base.yml` |
| MoE 分片参数 | `fsdp_shard_on_exp=True` | `shard_exp_on_fsdp=True`（上游已重命名） |
| XPK | 0.16.1 | 1.8.0 |
| XLA | — | 增加 `--xla_tpu_dvfs_p_state=3` |
| 并行配置 | 相同 | 相同 |

### 为什么 MoE 分片用 `shard_exp_on_fsdp` 而不是 2D 分片

256 芯片的 `4k-bf16-tpu7x-4x8x8-lustre` 配方使用
`use_2d_fsdp_sharding=True` + `ici_fsdp_transpose_parallelism=2`。这条路径把
MoE 权重同时切分到 `fsdp` 和 `fsdp_transpose` 两个轴上，只有当
`ici_fsdp_transpose_parallelism > 1` 时才有意义。

128 芯片下只有一个 FSDP 轴（共 256 个 device），因此本配方保持
`ici_fsdp_transpose_parallelism=1` 并使用 `shard_exp_on_fsdp=True`。

MaxText 对该路径有硬性约束：`num_experts` 必须能被 `ici_fsdp_parallelism`
整除，且要求 `ici_expert_parallelism = 1`、`ici_tensor_parallelism = 1`。
DeepSeek V3 有 256 个 expert，此处 `ici_fsdp_parallelism` 解析为 256，三个
条件全部满足。**该约束已在 128 芯片实机上验证通过**。

注意 `shard_exp_on_fsdp` 在旧版 MaxText 中叫 `fsdp_shard_on_exp`。

## 验证状态

2026-07-26 在 `us-central1-c` 的 128 芯片 Spot 节点池上实跑过（走的是
[../k8s](../k8s/README.md) 的 manifest 路径，MaxText 参数与本配方一致）。

已验证：4x4x8 节点池创建（32/32 Ready）、最新 MaxText main
（`e50e39458`，2026-07-25）镜像构建、32 个 worker 全部启动、
`shard_exp_on_fsdp=True` 通过配置校验、XLA 编译与前几个训练步。

未验证：**稳态性能数据**（运行到 step 2 时 Spot 被抢占，step 0–1 属于 JIT
编译阶段）；Lustre 存储路径（见附录）。

## 性能参考

以下为**同拓扑、同并行配置、synthetic 数据**下的历史数据，使用较旧的 MaxText
（`maxtext-tutorial-v1.5.0`，JAX `0.8.2.dev20251215`），列出用于对照：

| 配置 | 精度 | Step Time | TFLOP/s/chip | Tokens/s/chip |
| --- | --- | --- | --- | --- |
| 4x4x8（128 芯片） | bf16 | 27.00 s | 608.0 | 2,427.7 |
| 4x4x8（128 芯片） | fp8 | 22.39 s | 733.1 | 2,926.5 |

注意单位：MaxText 日志输出的 `TFLOP/s/device` 是 **per TensorCore**，TPU v7
每 chip 有 2 个 TensorCore，per-chip 数值是日志值的 2 倍。上表已换算为
per-chip。以 v7 每 chip BF16 峰值 2,307 TFLOPS 计，608 约合 26.4% MFU。

## 前置条件

运行本配方需要具备：

-   **GCP 项目**：已开通计费，且已加入 Ironwood 访问白名单。
-   **用户项目权限**：所用账号需要以下 IAM 角色：
    -   Artifact Registry Writer
    -   Compute Admin
    -   Google Cloud Managed Lustre Admin
    -   Kubernetes Engine Admin
    -   Logging Admin
    -   Monitoring Admin
    -   Service Account User
    -   Storage Object Viewer
    -   Vertex AI Administrator
    -   Service Usage Consumer
    -   TPU Viewer
-   **Docker**：工作机上必须安装 Docker。安装步骤见
    [安装 XPK 及依赖](#安装-xpk-及依赖) 一节。
-   **Python 3.11 虚拟环境**：必须使用 Python 3.11 虚拟环境，配置方法同样见
    [安装 XPK 及依赖](#安装-xpk-及依赖) 一节。
-   **XPK 及依赖**：按 [安装 XPK 及依赖](#安装-xpk-及依赖) 一节的步骤安装 XPK、
    `kubectl`、`kubectl-kueue` 和 `kubectl-kjob`。

## 安装 XPK 及依赖

### XPK 与依赖安装

#### Python 虚拟环境

执行以下命令创建 Python 虚拟环境：

```bash
# 安装 uv
sudo apt update
curl -LsSf https://astral.sh/uv/install.sh -o install-uv.sh
chmod +x install-uv.sh
./install-uv.sh
rm install-uv.sh
source ${HOME}/.local/bin/env

# 创建并激活 Python 3.11 虚拟环境
uv venv --seed ${HOME}/.local/bin/venv --python 3.11 --clear
source ${HOME}/.local/bin/venv/bin/activate
pip install --upgrade pip
```

#### XPK

运行 XPK 时务必确保虚拟环境处于激活状态。

安装 XPK 和相关工具：

```bash
# 如未安装 gcloud，参考 https://cloud.google.com/sdk/docs/install
# 如未安装 kubectl，参考 https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl

# 确保已登录 gcloud

# 安装 xpk
pip install xpk==1.8.0

# 安装 xpk 的前置依赖 kubectl-kueue 和 kjob（如果你是通过 pip 安装的 xpk）
curl -LsSf https://raw.githubusercontent.com/AI-Hypercomputer/xpk/refs/tags/v1.8.0/tools/install-xpk.sh -o install-xpk.sh
chmod +x install-xpk.sh
sudo ./install-xpk.sh
rm install-xpk.sh

# 按 https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin 安装 gke-gcloud-auth-plugin
```

#### Docker

按你所在组织管理员提供的方式安装 Docker。安装完成后执行：

```bash
## 配置 docker 并验证安装
gcloud auth configure-docker
sudo usermod -aG docker $USER ## 执行后需重启终端，并确保重新激活虚拟环境
docker run hello-world # 测试 docker
```

## 编排与部署工具

本配方使用以下组合：

-   **编排** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **预训练任务配置与部署** - 使用 XPK 配置并部署
    [Kubernetes JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    资源，由它管理 deepseek3-671b 任务的执行。

## 测试环境

本配方针对 tpu7x-4x4x8 优化并验证。

-   **GKE 集群**：创建 GKE 集群请参考
    [XPK 文档](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#cluster-create)，
    下面提供了一个示例命令。

### 集群创建所需的环境变量

集群创建和任务执行所需的环境变量定义在 `run_recipe.sh` 脚本开头。
**在执行 `xpk workload create` 之前**，请先打开 `run_recipe.sh`，修改其中的
`export` 语句以匹配你的环境。`PROJECT_ID`、`CLUSTER_NAME` 和 `ZONE` 在所有命令和
配置中必须保持一致。

-   `PROJECT_ID`：你的 GCP 项目名。
-   `CLUSTER_NAME`：目标集群名。
-   `ZONE`：集群所在 zone（例如 `us-central1-c`）。
-   `CONTAINER_REGISTRY`：使用的容器镜像仓库（例如 `gcr.io`）。
-   `BASE_OUTPUT_DIR`：模型训练的输出目录（例如 `"<your_lustre_instance>"`）。
-   `MAXTEXT_ROOT`：你 clone MaxText 仓库的绝对路径。
-   `WORKLOAD_IMAGE`：任务使用的 Docker 镜像。`run_recipe.sh` 中默认设为
    `${CONTAINER_REGISTRY}/${PROJECT_ID}/${USER}-deepseek-v3-runner`，与
    [Docker 容器镜像](#docker-容器镜像) 一节构建的镜像对应。
-   `WORKLOAD_NAME`：任务的唯一名称。`run_recipe.sh` 中通过以下命令生成：
    `export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-deepseekv3-671b-4096-fsdp")-$(date +%Y%m%d-%H%M)"`
-   `GKE_VERSION`：GKE 版本，需 `1.34.0-gke.2201000` 或更高。
-   `RESERVATION_NAME`：你的 TPU 预留（reservation）名称。同项目内直接用预留名；
    跨项目共享时使用
    `"projects/<project_number>/reservations/<reservation_name>"`。

### Placement policy（v7 必需）

TPU v7 **不支持自动创建** placement policy。创建 multi-host 节点池前必须先建
对应拓扑的 policy，否则节点池会创建出来但节点数为 0：

```bash
gcloud compute resource-policies create workload-policy tpu7x-128chip \
  --region=${REGION} --project=${PROJECT_ID} \
  --type=HIGH_THROUGHPUT --accelerator-topology=4x4x8
```

multi-host 节点池是 all-or-nothing 分配：32 台机器必须同时拿到，4x4x8 拓扑
要求物理相邻的完整立方体。

### XPK 集群创建示例命令

```bash
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-4x4x8 \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}
```

### 在集群上启用 Managed Lustre CSI 驱动

确保 GKE 版本为 `1.34.0-gke.2201000` 或更高。如果集群已经创建，需确认 Managed
Lustre CSI 驱动已启用。

```bash
gcloud container clusters update ${CLUSTER_NAME} \
  --location ${ZONE} \
  --project ${PROJECT_ID} \
  --update-addons=LustreCsiDriver=ENABLED
```

## Docker 容器镜像

构建自己的镜像请按本节步骤操作。如果工作机上还没装 Docker，参考前面
安装 XPK 及依赖的章节，Docker 安装包含在其中。

### 构建任务镜像的步骤

本配方针对 **MaxText main 分支最新版**，其中包含了上游 4x8x8 配方固定版本
（commit `cf051eb03`，2026-03-17）之后合入的 DeepSeek V3 MoE 优化，例如
`Overlap moe comms with collective matmul`、
`Fix ragged all-to-all with ragged buffer factor in DeepSeek-V3`
和 `Enable MoE ragged sort on TPU7X`。

已验证版本：MaxText `e50e39458`（2026-07-25），nightly jax/libtpu，
Python 3.11，XPK 1.8.0。

```bash
export CONTAINER_REGISTRY="" # 例如 us-docker.pkg.dev/${PROJECT_ID}/gcr.io
export CLOUD_IMAGE_NAME="${USER}-maxtext-latest"

# 为 Docker 构建创建并激活 Python 3.11 虚拟环境
uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.11 --clear
source ${HOME}/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 记录本次构建所用的 commit，保证结果可复现
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

**必须推 runner 镜像，不是 base 镜像。** `maxtext_base_image` 只含依赖，
不含 MaxText 源码，用它跑会因为找不到模块而失败。

仓库自带的 `docker_upload_runner.sh` 会用 `gcloud config get-value project`
推送，跨项目使用时会推错地方，建议按上面的方式手动 tag 和 push。

## 训练数据集

本配方使用 MaxText 内置的 synthetic 数据集，无需准备任何外部数据。

## 运行配方

### 配置环境变量

执行本节任何命令之前，请先按
[集群创建所需的环境变量](#集群创建所需的环境变量) 一节设置好环境变量。

### 连接到已有集群（可选）

如果想在跑 benchmark 前先连上 GKE 集群查看当前状态，可以用下面的 gcloud 命令
（注意 XPK 会自动帮你做这一步）：

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

## 获取配方

```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x8-latest/xpk
```

### 运行 deepseek3-671b 预训练任务

`run_recipe.sh` 脚本包含了启动 deepseek3-671b 预训练任务所需的全部环境变量和配置。

执行前请用 `nano ./run_recipe.sh` 编辑脚本，把环境变量改成你自己的。

### 配置并启动任务

在 MaxText 根目录下启动 DeepSeek3-671B 任务。

编辑 `run_recipe.sh`，填写文件顶部的 export 变量以匹配你的环境：

```
# 在 run_recipe.sh 中修改这几行：
export PROJECT_ID="your-project-id"
export CLUSTER_NAME="your-cluster-name"
export ZONE="your-zone"
export BASE_OUTPUT_DIR="gs://your-bucket/ds3-run"
export WORKLOAD_IMAGE="your-registry/your-maxtext-latest:runner"
```

配置并运行 benchmark：

```bash
chmod +x run_recipe.sh
nano ./run_recipe.sh
./run_recipe.sh
```

可以通过修改 `run_recipe.sh` 来定制运行：

-   **环境变量**：`PROJECT_ID`、`CLUSTER_NAME`、`ZONE`、`WORKLOAD_NAME`、
    `WORKLOAD_IMAGE`、`BASE_OUTPUT_DIR` 等定义在脚本开头，按你的环境调整。
-   **XLA Flags**：`XLA_FLAGS` 变量包含了一组针对本任务优化过的 XLA 配置，
    可以用于性能调优或调试。
-   **MaxText 参数覆盖**：`MAXTEXT_ARGS` 变量保存传给
    `python3 -m src.maxtext.trainers.pre_train.train` 的参数，包含
    `per_device_batch_size`、`max_target_length` 等模型相关配置，可以修改它们来
    尝试不同的模型配置。
-   **虚拟环境**：脚本会激活安装 XPK 时创建的虚拟环境。如果你用了别的虚拟环境，
    请修改 `run_recipe.sh` 开头的 `source` 命令。

注意：`MAXTEXT_ARGS` 中未显式覆盖的 MaxText 配置，会使用指定 `WORKLOAD_IMAGE`
中的默认值。

## 监控任务

可以用 kubectl 查看 JobSet 状态并跟踪日志：

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# 列出 pod 找到具体名字（例如 deepseek3-0-0-xxxx）
kubectl get pods | grep ${WORKLOAD_NAME}
```

然后跟踪运行中 pod 的日志（把 <POD_NAME> 换成上面找到的名字）：

```bash
kubectl logs -f <POD_NAME>
```

也可以在 Google Cloud Console 上查看集群和 TPU 使用情况。

### 跟踪任务与查看指标

执行 `xpk workload create` 之后，会返回一个 Google Cloud Console 链接用于查看
任务日志。例如：`[XPK] Follow your workload here:
https://console.cloud.google.com/kubernetes/service/${ZONE}/${PROJECT_ID}/default/${WORKLOAD_NAME}/details?project=${PROJECT_ID}`

也可以列出所有任务（`xpk workload list`）：

```bash
xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

需要更深入排查时使用 xpk inspector（`xpk inspector`）：

```bash
xpk inspector --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} [--workload ${WORKLOAD_NAME}]
```

### 清理资源

#### 删除指定任务

```bash
xpk workload delete --workload ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
# 或按条件过滤删除：
xpk workload delete --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} --filter-by-job=${USER}
```

#### 删除 XPK storage 资源

```bash
xpk storage detach lustre-volume --project ${PROJECT_ID} --cluster ${CLUSTER_NAME} --zone ${ZONE}
```

#### 删除整个 XPK 集群

```bash
xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```

## 查看结果

任务完成后，可以通过以下方式查看结果：

-   查看任务的输出日志。
-   查看 `run_recipe.sh` 中 `${BASE_OUTPUT_DIR}` 变量指定的目录下的
    TensorBoard 输出（本配方关闭了 checkpoint）。
-   如果已配置，可在 Cloud Monitoring 中查看相关指标。

## 常见问题

**`file:///deps does not appear to be a Python project`**

runner 镜像的 `/deps` 下只有 `benchmarks/`、`src/`、`tests/`、`pytest.ini`，
没有 `pyproject.toml` 或 `setup.py`，所以不要执行 `pip install -e .`。
直接 `cd /deps` 后运行入口模块即可。

**找不到 config**

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

### 1. 创建 Lustre 实例

1. 按照[官方文档](https://docs.cloud.google.com/managed-lustre/docs/create-instance)
创建新的 Lustre 实例，用于存放数据集和 checkpoint。挂载方式参考
[Compute Engine](https://docs.cloud.google.com/managed-lustre/docs/connect-from-compute-engine)
或
[Kubernetes Engine](https://docs.cloud.google.com/managed-lustre/docs/lustre-csi-driver-new-volume)。
**创建 Lustre 实例时必须使用与 GKE 集群相同的网络**。由于同一实例要同时承担数据
加载和 checkpoint 写入，建议容量至少 36 TB。

2. 在 Lustre 实例中准备数据集。本配方配置为使用 Grain loader 读取 ArrayRecord
文件，需确保数据集文件在该实例中可访问。你需要先从数据源下载 AllenAI C4 数据集，
然后按照[数据传输文档](https://docs.cloud.google.com/managed-lustre/docs/transfer-data)
将其传输到 Lustre 实例。

### 2. 挂载 Lustre 实例

Managed Lustre 可以像本地文件系统一样挂载访问，应用程序用标准文件系统语义即可
读写。需要用下面的命令为该实例创建
[XPK storage 资源](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/usage/storage.md#managed-lustre)，
才能把它挂载到 MaxText 任务中。Lustre 实例的 PV/PVC 定义见 [../k8s/lustre_pvc.yaml](../k8s/lustre_pvc.yaml)。

注意要把 yaml 中的 `volumeHandle` 改成你实际的 Lustre 实例信息。创建 Lustre 实例
和挂载 xpk storage 都是一次性配置。

```
# 设置变量
export PROJECT=""
export CLUSTER=""
export ZONE=""

# Lustre PV/PVC
xpk storage attach lustre-volume --type=lustre --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE --mount-point=/mnt/lustre --readonly=false --auto-mount=false --manifest=../k8s/lustre_pvc.yaml
```

### 3. 修改 run_recipe.sh

把 MaxText 参数中的

```
tokenizer_path=src/maxtext/assets/tokenizer.mistral-v3 dataset_type=synthetic enable_checkpointing=False
```

替换为 grain / checkpoint 相关参数，并在 `xpk workload create` 命令中加上
`--storage=$LUSTRE_VOLUME_NAME`，同时把 `BASE_OUTPUT_DIR` 指向
`/mnt/lustre/checkpoints`。完整参数见上游 256 芯片的
`4k-bf16-tpu7x-4x8x8-lustre` 配方。

**该路径尚未实机验证。**

## 下一步：深入探索与定制

本配方的目标是提供一个简单、可复现的「从 0 到 1」体验，帮助你快速可靠地验证环境、
在 TPU 上跑通第一次训练。

如果需要更深入的探索——比如定制模型配置、用不同的 XLA flag 调优性能、运行自定义
实验——建议直接使用 MaxText 仓库中的 `benchmark_runner.py` 脚本。该脚本提供了
MaxText 的完整灵活性，更适合希望超越初始 benchmark、按自身需求定制任务的高级用户
和研究人员。详见
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md)。
