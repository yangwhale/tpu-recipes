# 在 TPU Ironwood (tpu7x-4x4x8) 上用 Lustre 训练 DeepSeek3-671B

本配方说明如何在 [Ironwood GKE 集群](https://cloud.google.com/kubernetes-engine)
上通过 [XPK](https://github.com/AI-Hypercomputer/xpk) 运行 deepseek3-671b 的
[MaxText](https://github.com/AI-Hypercomputer/maxtext) 预训练任务，并使用
Google Cloud Managed Lustre 作为数据集和 checkpoint 的主存储。

如果你更希望用原生 Kubernetes 对象而不是 XPK 封装——比如要把配方交付给客户、
需要对方能直接看懂到底部署了什么——请参考等价的 manifest 版本
[../k8s](../k8s/README.md)。

> English version: [README.md](README.md)

## 任务配置

本任务的配置如下：

-   序列长度（Sequence Length）：4096
-   精度：bf16
-   芯片数：128（4x4x8 拓扑）
-   数据集与 checkpoint 均使用 Lustre
    -   C4 多语言数据集（约 12TB），ArrayRecord 格式

## 与其他配方的关系

本配方把 128 芯片（4x4x8）的并行配置、Lustre 存储方案和当前版本的 MaxText 运行时
组合到一起，派生自 `4k-bf16-tpu7x-4x8x8-lustre`，有意做了以下改动：

| 配置项 | 4x8x8（256 芯片） | 本配方（128 芯片） |
| --- | --- | --- |
| `--tpu-type` | `tpu7x-4x8x8` | `tpu7x-4x4x8` |
| `ici_fsdp_transpose_parallelism` | 2 | 1 |
| `shard_exp_on_fsdp` | False | **True** |
| `use_2d_fsdp_sharding` | True | 不设置（默认 False） |

### 为什么 MoE 分片策略不同

`use_2d_fsdp_sharding` 会把 MoE 权重**同时**切分到 `fsdp` 和 `fsdp_transpose`
两个轴上。只有当 `ici_fsdp_transpose_parallelism > 1` 时这才有意义——4x8x8
满足，本配方不满足。

在 128 芯片下只有一个 FSDP 轴（共 256 个 device），因此本配方改用
`shard_exp_on_fsdp=True`，把 MLP 权重的 expert 维度切分到这个单轴上。

MaxText 对该路径有硬性约束：`num_experts` 必须能被 `ici_fsdp_parallelism` 整除。
DeepSeek V3 有 256 个 expert，此处 `ici_fsdp_parallelism` 解析为 256，约束成立。
该路径还要求 `ici_expert_parallelism = 1` 且 `ici_tensor_parallelism = 1`，
两者在本配方中都是默认值。

注意 `shard_exp_on_fsdp` 在旧版 MaxText 中叫 `fsdp_shard_on_exp`。非 Lustre 的
`4k-bf16-tpu7x-4x4x8` 配方仍在使用旧参数名和重构前的入口
（`python3 -m MaxText.train`）；本配方使用当前的
`src.maxtext.trainers.pre_train.train` 入口。

其余所有 MaxText 参数——`per_device_batch_size=8.0`、`max_target_length=4096`、
`ici_fsdp_parallelism=-1`、`dcn_data_parallelism=-1` 以及整套 XLA flag——都与
上游对应配方保持一致，未做修改。

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

## Lustre 实例配置

### 创建 Lustre 实例

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

### 挂载 Lustre 实例

Managed Lustre 可以像本地文件系统一样挂载访问，应用程序用标准文件系统语义即可
读写。需要用下面的命令为该实例创建
[XPK storage 资源](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/usage/storage.md#managed-lustre)，
才能把它挂载到 MaxText 任务中。Lustre 实例使用本仓库中的 `lustre_pvc.yaml`。

注意要把 yaml 中的 `volumeHandle` 改成你实际的 Lustre 实例信息。创建 Lustre 实例
和挂载 xpk storage 都是一次性配置。

```
# 设置变量
export PROJECT=""
export CLUSTER=""
export ZONE=""

# Lustre PV/PVC
xpk storage attach lustre-volume --type=lustre --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE --mount-point=/mnt/lustre --readonly=false --auto-mount=false --manifest=lustre_pvc.yaml
```

## Docker 容器镜像

构建自己的镜像请按本节步骤操作。如果工作机上还没装 Docker，参考前面
安装 XPK 及依赖的章节，Docker 安装包含在其中。

### 构建任务镜像的步骤

本配方针对的是 **MaxText main 分支最新版**，其中包含了 4x8x8 配方固定版本
（commit `cf051eb03`，2026-03-17）之后合入的 DeepSeek V3 MoE 优化。这期间值得
关注的改动包括 `Overlap moe comms with collective matmul`、
`Fix ragged all-to-all with ragged buffer factor in DeepSeek-V3`
和 `Enable MoE ragged sort on TPU7X`。

推荐（最新版）：

-   MaxText 版本：`main`（已对照 `e50e39458`，2026-07-25 验证）
-   Libtpu / Jax：最新 nightly（由 `MODE=nightly` 自动解析）
-   Python：3.11
-   XPK：1.8.0

已验证可用的回退版本（与上游 4x8x8 Lustre 配方完全一致）——如果最新 main 构建
失败或出现性能回退，改用这组：

-   MaxText 版本：`maxtext-tutorial-v1.1.0-1109-gcf051eb03`
-   Libtpu 版本：`0.0.35.dev20260121+nightly`
-   Jax 版本：`0.8.1`

Docker 镜像构建命令：

```bash
export CONTAINER_REGISTRY="" # 填入你的镜像仓库
export CLOUD_IMAGE_NAME="${USER}-maxtext-runner"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

# 为 Docker 构建创建并激活 Python 3.11 虚拟环境
uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.11 --clear
source ${HOME}/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

# 确认当前虚拟环境是 Python 3.11
if [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" == "3.11" ]]; then { echo "You have the correct Python version 3.11"; } else { >&2 echo "Error: Python version must be 3.11"; false;} fi

# Clone MaxText 仓库（最新 main）
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 记录本次构建所用的 commit，保证结果可复现
git rev-parse HEAD

# 构建并上传 docker 镜像。
# MODE=nightly 不指定具体版本时会自动解析最新的 jax/libtpu nightly。
bash src/dependencies/scripts/docker_build_dependency_image.sh MODE=nightly

# --- 回退方案：固定到 4x8x8 配方使用的已验证版本 ---
# git checkout maxtext-tutorial-v1.1.0-1109-gcf051eb03
# bash src/dependencies/scripts/docker_build_dependency_image.sh \
#   MODE=nightly \
#   JAX_VERSION=0.8.1 \
#   LIBTPU_VERSION=0.0.35.dev20260121+nightly
bash src/dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

# 退出虚拟环境
deactivate
```

## 训练数据集

本配方使用 AllenAI C4 数据集，通过
[grain loader](https://github.com/google/grain) 读取。请按
[Lustre 实例配置](#lustre-实例配置) 一节的说明，确保数据集文件在 Lustre 实例中
可访问。

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
cd tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x8-lustre/xpk
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
export DATASET_BUCKET_MOUNTED_PATH="/mnt/lustre/path-to-dataset-on-lustre-instance" # /mnt/lustre/ 之后的路径要与数据集在 Lustre 实例根目录下的实际路径一致
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
-   查看 `run_recipe.sh` 中 `${BASE_OUTPUT_DIR}` 变量指定的目录（本配方指向
    Lustre 上的 `/mnt/lustre/checkpoints`）下的 checkpoint 和 TensorBoard 输出。
-   如果已配置，可在 Cloud Monitoring 中查看相关指标。

## 下一步：深入探索与定制

本配方的目标是提供一个简单、可复现的「从 0 到 1」体验，帮助你快速可靠地验证环境、
在 TPU 上跑通第一次训练。

如果需要更深入的探索——比如定制模型配置、用不同的 XLA flag 调优性能、运行自定义
实验——建议直接使用 MaxText 仓库中的 `benchmark_runner.py` 脚本。该脚本提供了
MaxText 的完整灵活性，更适合希望超越初始 benchmark、按自身需求定制任务的高级用户
和研究人员。详见
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md)。
