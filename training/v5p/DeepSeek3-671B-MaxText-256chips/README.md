# DeepSeek V3 671B 在 v5p 256 chips 上的预训练（K8s manifest）

> English version: [README.en.md](README.en.md)
>
> 返回 [v5p 配方总览](../README.md)

本配方在 **256 个 v5p chips**（拓扑 `4x8x8`）上跑 DeepSeek V3 671B 预训练，
数据集用 synthetic，不写 checkpoint，目的是验证流程与采集性能基线。

官方配方 [DeepSeek3-671B-MaxText](../DeepSeek3-671B-MaxText/README.md) 针对
`v5p-1024`（512 chips），并且用 XPK 提交。本配方有两点不同：

1. **规模减半**：256 chips，需要相应调整 `per_device_batch_size`
2. **纯 K8s manifest**：直接 `kubectl apply`，交付给客户时能一眼看清部署了什么

单位换算（chip / device / TensorCore 三套单位的关系）见
[TPU-UNITS.md](../../TPU-UNITS.md)。简记：v5p 是 MegaCore，
**1 chip = 1 JAX device**，机型名 `v5p-1024` 里的 1024 是 TensorCore 数，
对应 512 chips。

## 前置条件

- GKE 集群，含一个 256 chips 的 v5p 节点池（64 台 VM × 4 chips，拓扑 `4x8x8`）
- 已安装 JobSet CRD
- Docker 已配置 Artifact Registry 推送权限

### 拓扑形状不要自己凑

256 chips 可以拆成多种 3D 形状，但**不等价**：

```
4x8x8   ✓ 接近立方体，对分带宽好
4x4x16  ✗ 长条形，跨长轴通信绕远
```

创建节点池时不要手动指定 placement policy，GKE 会自动生成 `COMPACT`。
手动指定反而会报 `Required field 'resource.requestedRunDuration' not specified`。

## 构建镜像

官方配方钉死了 MaxText commit 和 JAX 版本：

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout 3eb77db3c94580f56f1b738f8d254b03bd205e35
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable JAX_VERSION=0.7.0
```

**但这条命令现在直接跑会得到一个跑不起来的镜像**，原因见下方
[依赖版本漂移](#依赖版本漂移必读)。需要额外两步。

### 依赖版本漂移（必读）

`requirements.txt` 里 `flax`、`pathwaysutils` 等都是**不带版本号的**。
配方写于 2025-10，当时 pip 解析出来的版本与 JAX 0.7.0 兼容；
现在同一条命令会拉到 2026 年的最新版，与 2025-10 的代码不兼容：

```
ImportError: cannot import name 'Effect' from 'jax.extend.core'        # flax 0.12.8
ImportError: cannot import name 'ifrt_proxy' from 'jax.extend.backend' # pathwaysutils
```

逐个降级是个无底洞（改完 flax 就轮到 pathwaysutils，后面还有）。
用 `uv` 的 `--exclude-newer` 把整套依赖一次性钉回 commit 当时的快照：

```dockerfile
FROM maxtext_stable__runner
RUN pip install --no-cache-dir uv && \
    uv pip install --system --exclude-newer 2025-10-15 \
      -r /deps/requirements.txt \
      "jax==0.7.0" "jaxlib==0.7.0" "libtpu==0.0.19.1"
```

验证：

```bash
docker run --rm --entrypoint python3 <IMAGE> -c \
  "import sys; sys.path.insert(0,'/deps/src'); import MaxText.train; print('OK')"
```

### 必须用 runner 镜像

`docker_build_dependency_image.sh` 产出的是 **base** 镜像，里面**没有 MaxText 源码**，
直接用会报 `file:///deps does not appear to be a Python project`。
需要再用 `maxtext_runner.Dockerfile` 构建 runner 镜像。

该 commit 的 `maxtext_runner.Dockerfile` 有个 bug——`COPY` 的源路径写成了容器内绝对路径：

```dockerfile
COPY "${MAXTEXT_ASSETS_ROOT}" ...   # 报 "/deps/src/MaxText/test_assets": not found
```

改成相对于 build context 的路径：

```dockerfile
COPY src/MaxText/assets/       src/MaxText/assets/
COPY src/MaxText/test_assets/  src/MaxText/test_assets/
COPY . .
```

完整文件见 [maxtext_runner_fixed.Dockerfile](maxtext_runner_fixed.Dockerfile)。

## 运行

```bash
export WORKLOAD_NAME=<你的作业名>
export WORKLOAD_IMAGE=<你的 runner 镜像>
export BASE_OUTPUT_DIR=/tmp/mtout

envsubst < k8s/k8s_manifest.yaml | kubectl apply -f -
```

### `BASE_OUTPUT_DIR` 必须真实可写

这是本次调试里代价最大的一个坑，详见
[输出目录不可写会伪装成 TPU 挂死](#输出目录不可写会伪装成-tpu-挂死)。

只跑 benchmark 时用**本地路径**（如 `/tmp/mtout`）最省事——不写
checkpoint 就不需要 GCS。

若确实要写 GCS，必须同时满足：

1. bucket 真实存在
2. 节点 OAuth scope 含 `devstorage.read_write` 或 `cloud-platform`

注意 **IAM 授权不足以解决**：节点 scope 是 `devstorage.read_only` 时，
即使给了 `roles/storage.objectAdmin`，token 层面依然写不了。
现有节点池的 scope 无法修改，只能重建节点池或改用 Workload Identity。

## 监控

```bash
POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME \
      -o jsonpath='{.items[0].metadata.name}')

kubectl logs $POD | grep 'completed step'
kubectl logs $POD | grep 'Slow PjRt'      # 出现即表示卡住了
```

## 实测结果

环境：`cloud-tpu-multipod-dev`，us-central1-a，spot，2026-07-27

本配方**完整沿用官方 `deepseek3_671b_v5p_1024` 的全部 tuning params 与 33 个
XLA flag**，只把 `per_device_batch_size` 从 6 降到 4——这是 256 chips 上唯一
必须改的参数。

| 项目 | 值 |
| --- | --- |
| chips | 256（64 VM × 4） |
| JAX devices | 256 |
| 拓扑 | `4x8x8` COMPACT |
| 序列长度 | 8192 |
| `per_device_batch_size` | 4 |
| global batch size | 1024 |
| 精度 | bf16（权重 fp32） |
| step 0（含编译） | 93.0 s |
| **稳态 step 时间** | **68.70 s** |
| **TFLOP/s/device** | **134.1** |
| **MFU** | **29.2%** |
| Tokens/s/device | 477.1 |
| loss（step 0 → 15） | 12.27 → 9.9 单调下降 |

稳态从 step 3 开始，抖动 ±0.05 s。step 5–12 因 xplane profiler 开启而有额外开销，
读数时应排除（官方配方同样带 profiler，横向对比时是公平的）。

`step 1` 那行会显示 29492 TFLOP/s/device 之类的离谱数字，
这是 **JAX 异步派发**造成的假象——日志在计算真正完成前就打印了，不是真实性能。
step 6、11 出现 0.006 s 也是同一原因，它与前一步是配对的。

### `per_device_batch_size` 的上限是 4

官方在 512 chips 上用 6。256 chips 的权重分片翻倍（10.5 GB → 21 GB/device），
腾不出那么多激活空间。实测阶梯：

| pdb | 结果 |
| --- | --- |
| 6 | 编译期 OOM：`Used 112.37G of 95.74G hbm`，超 16.62 GB |
| 5 | 运行期失败：`Attempting to reserve 68.03G at the bottom of memory... 66.40G reservable` |
| **4** | **通过** |

注意 pdb=5 的失败方式与 pdb=6 不同：它通过了编译期的 95.74 GB 检查，
却卡在运行期 **bottom-of-memory 区域只有 66.40 GB 可预留**这个更紧的约束上。
两个限制不是一回事，调 batch size 时两个都要留意。

### 与官方 512 chips 的对照

| | 官方 512 chips | 本配方 256 chips |
| --- | --- | --- |
| `per_device_batch_size` | 6 | 4 |
| global batch size | 3072 | 1024 |
| TFLOP/s/device | 152.4 | 134.1 |
| MFU | 33.2% | 29.2% |

规模减半、batch 减三分之一的情况下，单卡吞吐保住了 88%。

差距主要来自 batch size：pdb 从 6 降到 4 让每步计算量减少，
collective 与计算的 overlap 变差。这是 256 chips 上权重分片翻倍带来的硬约束，
不是配置没调好。

### 参数不全会显著压低 MFU

早期一版运行漏掉了官方的若干设置，只拿到 **114.82 TFLOP/s/device（MFU 25.0%）**。
补齐后提升 **16.8%**。漏掉的是：

| 参数 | 官方 | 漏掉的那版 |
| --- | --- | --- |
| `tile_batch_seq` | 512 | 未设置 |
| `tile_embed_dim` | 1024 | 未设置 |
| `tile_mlp_dim` | 1024 | 未设置 |
| `--2a886c8_chip_config_name` | `megachip_tccontrol` | 未设置 |
| `--xla_tpu_use_tc_device_shape_on_sc` | true | 未设置 |
| `--xla_sc_enable_instruction_fusion` | false | 未设置 |
| `--xla_sc_disjoint_spmem` | false | 未设置 |
| `--xla_sc_disable_megacore_partitioning` | true | 未设置 |

后五个是 SparseCore 的**运行模式**配置。只开
`xla_tpu_enable_sparse_core_collective_offload_*` 三个 offload 开关而不配运行模式，
属于半吊子状态——这也让「关掉 SparseCore 看看」这类对照实验失去意义。

`tile_*` 三个参数在新版 MaxText 里已改名为 `wi_tile_*` / `wo_tile_*`，
在本 commit（`3eb77db3`）上是有效的，不要因为在新版报错就删掉。

**教训：从官方配方出发，只改被规模逼着改的那一个参数，不要顺手删减。**

## 踩坑速查表

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| `cannot import name 'Effect' from 'jax.extend.core'` | flax 被 pip 拉到 2026 最新版 | `uv pip install --exclude-newer 2025-10-15` |
| `cannot import name 'ifrt_proxy'` | pathwaysutils 同上 | 同上 |
| `file:///deps does not appear to be a Python project` | 用了 base 镜像而非 runner | 构建 runner 镜像 |
| `"/deps/src/MaxText/test_assets": not found` | runner Dockerfile 用了容器绝对路径 | 改成相对 build context 的路径 |
| `CompileTimeScopedVmemOom`，splash attention 需 18.12 MB | VMEM 默认 16 MB | 加 `--xla_tpu_scoped_vmem_limit_kib=81920` |
| `Keys ['base_num_decoder_layers'] are overridden by both model config and CLI` | CLI 与 model config 冲突 | 加 `override_model_config=True` |
| `Cannot remeterialize this tensor with scan_layers=True` | `decoder_layer_input=remat` 与 scan 冲突 | 用 `offload` 或 `device` |
| `Required field 'resource.requestedRunDuration' not specified` | 手动指定了 placement policy | 不要指定，让 GKE 自动生成 |
| step 1 之后停住，刷 `Slow PjRt ... CopyToMemorySpace CrossDeviceSrc` | 见下节 | 见下节 |

### 输出目录不可写会伪装成 TPU 挂死

**这是本次调试里最有价值的一条。**

现象：训练跑完 step 0、step 1 后停住，日志刷屏

```
Slow PjRt TPU operation detected: ... description=CopyToMemorySpace CrossDeviceSrc
Slow PjRt TPU operation detected: ... description=TfrtTpuLoadedExecutable::ReadyFuture
```

看起来像 TPU 硬件、ICI 互连或 XLA 编译问题。实际根因是
**`BASE_OUTPUT_DIR` 指向了一个不存在的 GCS bucket**：

```
google.api_core.exceptions.Forbidden: 403 POST .../b/<bucket>/o
"Provided scope(s) are not authorized"
```

MaxText 只在 **JAX process 0** 上写 TensorBoard。process 0 在这里抛异常后
主循环停止下发程序，其余 63 个 host 就永远等在 collective 里。

沿着表象排查会全部落空。以下变量我们逐个试过，**都不是原因**：
TPU 代际（v7x / v5p）、模型层数、MTP、规模（64 / 256 chips）、
拓扑形状、MaxText 与 JAX 版本、SparseCore 卸载开关、host offload、
残留 pod 占用 HBM、spot 抢占。

**定位方法**：把全部 64 个 pod 的日志都拉下来，统计各自的 `Slow PjRt` 条数。

```bash
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME \
  --no-headers | awk '{print $1}' > pods.txt
cat pods.txt | xargs -P 20 -I{} sh -c 'kubectl logs {} > logs/{}.log 2>/dev/null'

for f in logs/*.log; do echo "$(grep -c 'Slow PjRt' $f) $f"; done | sort -n | head
```

63 个 pod 会报同样多的挂起警告，**唯一那个不报警告的才是真正出问题的**。
因为它没卡在 collective 里——它是让别人卡住的那个。

推广一下：多机 TPU 训练里任何 **process-0 独有的 side effect** 失败
（TensorBoard、checkpoint 上传、metrics 写入、Vertex 集成），
表现出来都是 collective 挂死，而且**异常现场不在报警告的机器上**。

### spot 节点池抢占

us-central1-a 的 v5p spot 容量紧张时抢占非常密集，实测：

```
10:52 UTC   13 台
10:53 UTC   51 台   ← 整池 64 台全灭
11:00 UTC   31 台
11:01 UTC   14 台
11:02 UTC   20 台   ← 重建出来的又被抢
```

抢占同样表现为 collective 挂死（活着的 host 等一台已消失的）。
区分方法是查抢占记录：

```bash
gcloud compute operations list --project=<PROJECT> --zones=<ZONE> \
  --sort-by=~startTime --limit=400 \
  --format='value(startTime,operationType)' | grep preempted
```

另外 XPK 的 wrapper 脚本会**吞掉退出码**，训练失败时 pod 依然是
`Completed`、JobSet 依然是 `completed successfully`。
不要用 JobSet 状态判断训练是否成功，要看日志里的 `completed step`。

## 清理

```bash
kubectl delete jobset $WORKLOAD_NAME
```

删除是异步的。若要立刻提交下一个作业，**必须等 pod 全部消失**，
否则新 pod 会与仍持有 TPU 的旧 pod 抢同一批节点：

```bash
while [ "$(kubectl get pods --no-headers | wc -l)" -ne 0 ]; do sleep 10; done
```
