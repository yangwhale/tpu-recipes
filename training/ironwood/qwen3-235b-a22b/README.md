# Qwen3-235B-A22B 训练测试记录

## 模型概况

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3-235B-A22B (MoE) |
| 总参数量 | 235B |
| 激活参数量 | 22B |
| Experts | 128 total, 8 active per token |
| Decoder Layers | 94 |
| Embedding Dim | 4096 |
| 硬件 | TPU v7 (Ironwood) |
| 拓扑 | tpu7x-4x8x8 (256 chips / 512 devices / 64 hosts / 1 slice) |
| 框架 | MaxText (maxtext-tutorial-v1.5.0) |
| JAX | 0.8.2.dev20251215 |
| Libtpu | 0.0.32.dev20251215+nightly |
| XPK | 0.16.1 |

## 环境变量

使用前需要设置以下环境变量（或在脚本中填写）:

```bash
export PROJECT_ID=""           # GCP 项目 ID, e.g. "my-gcp-project"
export CLUSTER_NAME=""         # GKE 集群名称, e.g. "my-v7x-training"
export ZONE=""                 # TPU 所在 zone, e.g. "us-central1-c"
export RESERVATION_NAME=""     # TPU reservation 名称, e.g. "my-reservation"
export BASE_OUTPUT_DIR=""      # GCS 输出路径, e.g. "gs://my-bucket"
export CONTAINER_REGISTRY=""   # 容器镜像仓库, e.g. "gcr.io"
export WORKLOAD_IMAGE=""       # Docker 镜像地址, e.g. "gcr.io/${PROJECT_ID}/${USER}-qwen3-235b-a22b-runner"
```

> **提示**: 使用 `gcloud alpha compute tpus reservations list --zone=<zone>` 查询可用的 TPU v7 reservation。

## 测试步骤

### 1. 环境准备

```bash
# 安装 xpk（直接安装到 base 环境，不使用虚拟环境）
pip install xpk==0.16.1 --break-system-packages

# 安装 xpk 依赖工具 (kubectl-kueue, kubectl-kjob)
curl -LsSf https://raw.githubusercontent.com/AI-Hypercomputer/xpk/refs/tags/v0.16.1/tools/install-xpk.sh -o install-xpk.sh
chmod +x install-xpk.sh
sudo ./install-xpk.sh
rm install-xpk.sh

# 设置默认项目
gcloud config set project ${PROJECT_ID}

# 配置 Docker
gcloud auth configure-docker
sudo usermod -aG docker $USER  # 需要重新登录终端生效
```

### 2. 构建 Docker 镜像 + 创建 GKE 集群

```bash
cd ~/tpu-recipes/training/ironwood/qwen3-235b-a22b/4k-bf16-tpu7x-4x8x8/xpk

# 从 template 创建自己的脚本并填入环境变量
cp setup_training_env.sh.template setup_training_env.sh
nano setup_training_env.sh  # 填入你自己的 PROJECT_ID, CLUSTER_NAME, ZONE 等

chmod +x setup_training_env.sh
./setup_training_env.sh
```

### 3. 提交训练任务

```bash
# 从 template 创建自己的脚本并填入环境变量
cp submit_workload.sh.template submit_workload.sh
nano submit_workload.sh  # 填入你自己的 PROJECT_ID, CLUSTER_NAME, ZONE 等

chmod +x submit_workload.sh
./submit_workload.sh
```

### 4. 监控任务

```bash
# 查看 workload 状态
xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}

# 查看 pod 状态（64 个 pods = 64 hosts × 4 chips/host = 256 chips）
kubectl get pods | grep qwen3

# 查看训练日志（worker 0）
kubectl logs -f <POD_NAME>

# 在日志中搜索训练指标
kubectl logs <POD_NAME> | grep "completed step"
```

### 5. 查看 Profiler Trace

Profiler 配置为跳过前 5 步 warmup，从 step 5 开始采集 3 步 xplane trace。

> 注意: profiler 采集期间（如 step 11）step time 会显著增加，这是正常现象。

```bash
# 用 TensorBoard 查看
tensorboard --logdir=${BASE_OUTPUT_DIR}/<run_name>/
```

### 6. 清理资源

```bash
# 删除 workload
xpk workload delete --workload <WORKLOAD_NAME> \
  --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}

# 释放 TPU 但保留 GKE 集群（xpk 不支持 --num-slices=0，需直接用 gcloud）
gcloud container node-pools delete ${CLUSTER_NAME}-np-0 \
  --cluster=${CLUSTER_NAME} \
  --zone=${ZONE%%-*}-${ZONE#*-} \
  --project=${PROJECT_ID} --quiet

# 下次训练时，用 adapt 重新添加 TPU node pool
xpk cluster adapt \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-4x8x8 \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}

# 彻底删除集群（不再使用时）
xpk cluster delete --cluster ${CLUSTER_NAME} \
  --zone ${ZONE} --project ${PROJECT_ID}
```

## 测试配置

### 当前 bf16 配置 (`4k-bf16-tpu7x-4x8x8`)

| 参数 | 值 | 说明 |
|------|-----|------|
| per_device_batch_size | 16.0 | 每 device batch size |
| max_target_length | 4096 | 序列长度 |
| dtype | bfloat16 | 激活值精度 |
| weight_dtype | float32 | 权重精度 (master weights) |
| attention | flash | Flash Attention |
| remat_policy | custom | 自定义重计算策略 |
| decoder_layer_input | offload | decoder 输入 offload 到 HBM |
| sparse_matmul | True | 稀疏矩阵乘法 (MoE) |
| use_tokamax_gmm | True | TokaMax GMM 内核 |
| use_tokamax_splash | True | TokaMax Splash Attention 内核 |
| ici_fsdp_parallelism | -1 (auto) | slice 内全 FSDP |
| dcn_data_parallelism | -1 (auto) | 跨 slice 数据并行 |
| dataset_type | synthetic | 合成数据（性能测试用） |
| profiler | xplane | XLA Profiler |
| profiler_steps | 3 | 采集 3 步 profile 数据 |
| skip_first_n_steps_for_profiler | 5 | 跳过前 5 步 warmup |

## 测试结果

### 官方基线 (tpu-recipes 仓库)

| 配置 | Chips | GBS | Seq Len | Precision | Step Time (s) | TFLOPs/s/chip | Tokens/s/chip |
|------|-------|-----|---------|-----------|---------------|---------------|---------------|
| 4x8x8 | 256 | 8192 | 4096 | bf16 | 31.58 | 615.69 | 4,150.78 |
| 4x8x8 | 256 | 8192 | 4096 | fp8_full | 27.67 | 702.60 | 4,736.72 |

### 我的测试记录

| 日期 | 模型 | 配置 | Precision | Step Time (s) | TFLOPs/s/device | TFLOPs/s/chip | Tokens/s/chip | Loss (final) | 备注 |
|------|------|------|-----------|---------------|-----------------|---------------|---------------|-------------|------|
| 2026-02-05 | Qwen3-235B | 4k-bf16-4x8x8 (256 chips) | bf16 | 31.60 | 307.1 | 614.3 | 4,140.6 | 12.034 | 20 steps，与官方基线吻合 |
| 2026-02-21 | Qwen3-235B | 4k-bf16-4x8x8 (256 chips) | bf16 | 31.64 | 307.3 | 614.6 | 4,143.3 | 12.034 | 20 steps，复测与上次一致 [xprof](http://xprof.corp.google.com/trace_viewer/chrisya-10026418840583447421) |

> DeepSeek3-671B 的测试记录请参见 [deepseek3-671b/README.md](../deepseek3-671b/README.md)

### 详细训练日志 (2026-02-05)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|----------------|--------------|---------------|------|
| 0 | 99.06 | 98.1 | 196.3 | 1,323.2 | 12.439 |
| 1 | 36.85 | 263.8 | 527.6 | 3,556.9 | 12.439 |
| 2 | 31.66 | 307.1 | 614.1 | 4,140.4 | 12.398 |
| 3 | 31.66 | 307.0 | 614.1 | 4,140.0 | 12.347 |
| 4 | 31.65 | 307.1 | 614.3 | 4,141.3 | 12.313 |
| 5 | 31.66 | 307.1 | 614.2 | 4,140.6 | 12.284 |
| 6 | 31.76 | 306.1 | 612.1 | 4,126.9 | 12.256 |
| 7 | 31.64 | 307.3 | 614.6 | 4,143.2 | 12.227 |
| ... | ... | ... | ... | ... | ... |
| 19 | 31.59 | 307.7 | 615.4 | 4,145.6 | 12.034 |

- **Step 0** 耗时 99s 是因为 JIT 编译（首次编译 XLA HLO → TPU 可执行代码）
- **Step 11** 耗时 106s 是因为 profiler 正在采集 xplane trace 数据
- **稳态性能 (Step 2+)**: ~31.6 s/step, ~614 TFLOP/s/chip, ~4,140 Tokens/s/chip
- **对比官方基线**: Step Time 31.60 vs 31.58 (差距 < 0.1%)，性能一致


### 详细训练日志 - 4x8x8 bf16 (2026-02-21)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|----------------|--------------|---------------|------|
| 0 | 103.04 | 94.3 | 188.7 | 1,272.1 | 12.439 |
| 1 | 41.62 | 233.6 | 467.2 | 3,149.4 | 12.439 |
| 2 | 31.68 | 306.9 | 613.8 | 4,137.7 | 12.398 |
| 3 | 31.68 | 306.9 | 613.8 | 4,138.1 | 12.347 |
| 4 | 31.67 | 307.0 | 613.9 | 4,138.9 | 12.313 |
| 5 | 31.68 | 306.8 | 613.7 | 4,137.1 | 12.284 |
| 6 | 31.71 | 306.6 | 613.1 | 4,133.4 | 12.256 |
| 7 | 31.65 | 307.2 | 614.3 | 4,141.4 | 12.227 |
| 8 | 31.64 | 307.3 | 614.5 | 4,142.9 | 12.198 |
| 9 | 31.63 | 307.3 | 614.6 | 4,143.7 | 12.169 |
| 10 | 31.61 | 307.6 | 615.1 | 4,147.1 | 12.142 |
| 11* | 77.74 | 125.0 | 250.1 | 1,686.1 | 12.118 |
| 12 | 31.61 | 307.5 | 615.0 | 4,146.1 | 12.098 |
| 13 | 31.61 | 307.5 | 615.0 | 4,146.3 | 12.081 |
| 14 | 31.61 | 307.5 | 615.1 | 4,146.6 | 12.068 |
| 15 | 31.60 | 307.7 | 615.3 | 4,148.4 | 12.057 |
| 16 | 31.61 | 307.5 | 615.1 | 4,146.7 | 12.049 |
| 17 | 31.61 | 307.6 | 615.1 | 4,147.0 | 12.043 |
| 18 | 31.61 | 307.5 | 615.0 | 4,146.5 | 12.038 |
| 19 | 31.60 | 307.6 | 615.2 | 4,147.8 | 12.034 |

- **Step 0** 耗时 103s 是因为 JIT 编译
- **Step 11*** 耗时 77.7s 是因为 profiler 正在采集 xplane trace 数据 (~2.5x)
- **稳态性能 (Step 2-10, 12-19)**: ~31.64 s/step, ~614.6 TFLOP/s/chip, ~4,143.3 Tokens/s/chip
- **Loss 下降**: 12.398 -> 12.034 (2.9%)
- **对比 2026-02-05 测试**: Step Time 31.64 vs 31.60，TFLOP/s/chip 614.6 vs 614.3，差异 < 0.1%，高度一致
- **Xprof Trace**: [trace_viewer](http://xprof.corp.google.com/trace_viewer/chrisya-10026418840583447421) | GCS: `gs://chrisya-v7x-us-central1/chrisya-qwen3-235b-4x8x8-20260221-0241/tensorboard/step_5/plugins/profile/2026_02_21_02_56_10/`

## XPK 使用经验

### 资源管理模式对比

| 模式 | 参数 | 资源分配 | 任务完成后 | 需要 Reservation |
|------|------|---------|-----------|-----------------|
| Reservation | `--reservation=xxx` | 预先占用 | 资源仍占用 | 是 |
| Flex (DWS) | `--flex` | 按需分配 | **自动释放** | 否 |
| Spot | `--spot` | 抢占式 | 自动释放 | 否 |
| On-Demand | `--on-demand` | 按需付费 | 需手动删除 | 否 |

> **建议**: 如果不需要保证资源立即可用，使用 `--flex` 模式。任务完成后 TPU 资源会自动释放，无需手动清理。

### 集群管理

```bash
# 动态调整 TPU slice 数量（例如从 1 增加到 4）
xpk cluster adapt --cluster=${CLUSTER_NAME} --tpu-type=tpu7x-4x8x8 --num-slices=4 ...

# 查看集群资源配置
kubectl get configmap ${CLUSTER_NAME}-resources-configmap -o yaml

# 查看 node pools
gcloud container node-pools list --cluster=${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID}
```

### 已知限制

- `xpk cluster adapt` 的 `--num-slices` 最小值为 1，无法缩减到 0
- `xpk cluster delete` 会删除整个集群，无法只删除 TPU node pool
- 要"保留集群 + 释放 TPU"只能用 `gcloud container node-pools delete` 直接操作
- `xpk cluster adapt` 在 v0.16.1 存在 `memory_limit` 属性缺失的 bug，可能导致 adapt 失败
- 共享项目中 default VPC 的 IP 空间可能被耗尽，建议创建自定义 VPC：
  ```bash
  gcloud compute networks create <name> --subnet-mode=auto --mtu=8896
  # 创建集群时指定：
  xpk cluster create ... --custom-cluster-arguments='--network=<name> --subnetwork=<name>'
  ```

## 目录结构

```
qwen3-235b-a22b/
├── README.md                              # 本文件
├── 4k-bf16-tpu7x-4x8x8/                  # bf16 精度配置
│   └── xpk/
│       ├── README.md                      # 官方使用说明（详细）
│       ├── run_recipe.sh                  # 官方原版脚本
│       ├── setup_training_env.sh.template # 环境设置模板（复制后填入自己的变量）
│       └── submit_workload.sh.template    # 提交任务模板（复制后填入自己的变量）
└── 4k-fp8-tpu7x-4x8x8/                   # fp8 精度配置（待测试）
    └── ...
```

## 注意事项

- 修改 MaxText 代码后需要**重新构建 Docker 镜像**（重新运行 `setup_training_env.sh`，或手动删除 GCR 上的旧镜像后再运行）
- `steps=20` 是用于性能基准测试的短训练，正式训练需调大
- `enable_checkpointing=False`，当前配置不保存 checkpoint
- Profiler 采集期间 step time 会显著增加（~3x），这是正常现象，不影响稳态性能评估
- 一个 chip 有 2 个 device (TensorCore)，日志中的 `TFLOP/s/device` 需乘以 2 才是 per-chip 性能
- Docker 权限问题：`usermod -aG docker $USER` 后需重新登录终端或 `sg docker -c 'command'`
