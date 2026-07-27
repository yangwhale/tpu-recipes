# DeepSeek V3 671B 在 v5p 512 chips 上的预训练（K8s manifest）

> English version: [README.en.md](README.en.md)
>
> 返回 [v5p 配方总览](../README.md)

本配方在 **512 个 v5p chips**（拓扑 `8x8x8`）上跑 DeepSeek V3 671B 预训练。

这是官方配方 [DeepSeek3-671B-MaxText](../DeepSeek3-671B-MaxText/README.md)
（`v5p-1024` = 512 chips）的**原样复现**，参数一个没改，只是把提交方式从 XPK
换成纯 K8s manifest。

**实测对齐官方数字，偏差 0.3% 以内**——详见[实测结果](#实测结果)。
官方配方在今天的环境下是可复现的，前提是先绕开
[依赖版本漂移](../DeepSeek3-671B-MaxText-256chips/README.md#依赖版本漂移必读)。

需要半规模的版本见
[DeepSeek3-671B-MaxText-256chips](../DeepSeek3-671B-MaxText-256chips/README.md)，
那篇记录了 256 chips 上必须改什么、代价多少，以及一份完整的踩坑速查表。

## 三个版本怎么选

| | 官方配方 | 本配方 | 256chips 配方 |
| --- | --- | --- | --- |
| chips | 512 | 512 | 256 |
| 提交方式 | XPK | K8s manifest | K8s manifest |
| `per_device_batch_size` | 6 | 6 | 4 |
| 用途 | 上游基准 | 对标官方数字 / 交付客户 | 半规模适配 |

交付客户时用 manifest 版：客户能一眼看清部署了什么，XPK 那层封装反而挡住了细节。

## 前置条件

- GKE 集群，含一个 512 chips 的 v5p 节点池（128 台 VM × 4 chips，拓扑 `8x8x8`）
- 已安装 JobSet CRD，且 **controller 处于 Running**（见
  [JobSet controller 的镜像会失效](#jobset-controller-的镜像会失效)）
- 本地装了 `envsubst`（Debian/Ubuntu 在 `gettext-base` 包里）
- 一个可用的 MaxText runner 镜像（构建方式见
  [256chips 配方的构建镜像章节](../DeepSeek3-671B-MaxText-256chips/README.md#构建镜像)，
  两个规模用的是同一个镜像）

kubectl 接上集群后自检：

```bash
kubectl get nodes -l cloud.google.com/gke-nodepool=<NODEPOOL> --no-headers | grep -c ' Ready '   # 期望 128
kubectl get pods -n jobset-system                                                                # 期望 1/1 Running
command -v envsubst                                                                              # 期望有路径
```

## 创建 512 chips 节点池

```bash
gcloud container node-pools create <NODEPOOL> \
  --cluster=<CLUSTER> --region=<REGION> --project=<PROJECT> \
  --node-locations=<ZONE> \
  --machine-type=ct5p-hightpu-4t \
  --tpu-topology=8x8x8 \
  --num-nodes=128 \
  --max-pods-per-node=16 \
  --spot
```

两个参数值得解释。

### 拓扑用 `8x8x8`

512 chips 能拆成多种 3D 形状，`8x8x8` 是正立方体，对分带宽最好。
3D torus 下越接近立方体越优，不要自己凑 `4x8x16` 这类长条形。

不要手动指定 `--placement-policy`，GKE 会自动生成 `COMPACT`；
手动指定反而会报 `Required field 'resource.requestedRunDuration' not specified`。

### `--max-pods-per-node=16` 不是可选项

**不加这个参数，128 节点的池大概率建不出来**，报的还是个看起来跟 TPU 无关的错：

```
Atomic resize failed with [IP_SPACE_EXHAUSTED_WITH_DETAILS]:
IP space of 'subnetworks/default' is exhausted.
Insufficient free IP addresses in the IP range '10.90.0.0/17'.
```

这不是 TPU 容量不足，是 **pod 次级 IP 范围耗尽**。

GKE 按 `maxPodsPerNode` 给每个节点预切一段 pod CIDR，默认 110 对应一个 `/24`
（256 个 IP）。集群的 pod 次级范围若是 `/17`（32768 个 IP），
就只够 `32768 / 256 = 128` 个节点——**整个集群**，不是单个池。
集群里已有别的节点时，再要 128 个必然超。

而 v5p 训练节点上实际占用 pod IP 的只有 2 个 pod（加训练 pod 共 3 个）：

```bash
kubectl get pods -A --field-selector spec.nodeName=<NODE> -o json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
    print(len([p for p in d['items'] if not p['spec'].get('hostNetwork')]))"
```

其余全是 hostNetwork 的 daemonset，不吃 pod IP。
所以 110 是纯浪费，设成 16 后每节点降到 `/27`（32 个 IP），
**IP 需求变为八分之一**，128 节点只要 4096 个 IP。

先查清楚自己集群的 pod 范围还剩多少：

```bash
gcloud container clusters describe <CLUSTER> --region <REGION> \
  --format='value(clusterIpv4Cidr, defaultMaxPodsConstraint.maxPodsPerNode)'
kubectl get nodes --no-headers | wc -l
```

可容纳节点数 ≈ `2^(32 - 前缀长度) / (2 × maxPodsPerNode 向上取整到 2 的幂)`。

**这个坑跟 TPU 无关，任何大规模 GKE 节点池都会遇到。**
好处是它只影响新建的池，不用去动共享子网。

### 抢不到容量是另一回事

上面是 IP 问题。若报的是 `GCE_STOCKOUT`，那才是真的没容量。
多机 TPU 是**原子分配**——128 台必须一次性全拿到，差一台就整体失败，
没有"先给你 100 台"这种事。

## 运行

```bash
export WORKLOAD_NAME=<你的作业名>
export WORKLOAD_IMAGE=<你的 runner 镜像>
export BASE_OUTPUT_DIR=/tmp/mtout

envsubst < k8s/k8s_manifest.yaml | kubectl apply -f -
```

`BASE_OUTPUT_DIR` **必须真实可写**。写不了会伪装成 TPU 挂死，
是本系列文档里代价最大的一个坑，展开见
[输出目录不可写会伪装成 TPU 挂死](../DeepSeek3-671B-MaxText-256chips/README.md#输出目录不可写会伪装成-tpu-挂死)。
只跑 benchmark 时用本地路径最省事。

## 监控

```bash
POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME \
      -o jsonpath='{.items[0].metadata.name}')

kubectl logs $POD | grep 'completed step'
kubectl logs $POD | grep 'Slow PjRt'      # 出现即表示卡住了
```

`items[0]` 取哪个 pod 都行——MaxText 在每个 host 上都打印 `completed step`，
不只 process 0。

从 `apply` 到第一个 `completed step` 约 **8 分钟**，绝大部分是 XLA 编译。
512 chips 比 256 chips 的 6–7 分钟略长。这段时间日志不动是正常的。

## 实测结果

环境：`cloud-tpu-multipod-dev`，us-central1-a，spot，2026-07-27

| 项目 | 值 |
| --- | --- |
| chips | 512（128 VM × 4） |
| JAX devices | 512 |
| 拓扑 | `8x8x8` COMPACT |
| 序列长度 | 8192 |
| `per_device_batch_size` | 6 |
| global batch size | 3072 |
| 精度 | bf16（权重 fp32） |
| step 0（含编译） | 114.2 s |
| **稳态 step 时间** | **90.41 s** |
| **TFLOP/s/device** | **152.85** |
| **MFU** | **33.3%** |
| Tokens/s/device | 543.6 |
| loss（step 0 → 16） | 12.270 → 10.436 单调下降 |

稳态取 9 个干净步（3、4、7、8、9、13、14、15、16），
区间 152.70–152.91，**抖动 0.14%**。

v5p 是 MegaCore，1 chip = 1 JAX device，所以 `TFLOP/s/device` 就是 per-chip 值。
MFU = 152.85 / 459。

### 与官方数字的对照

官方 README 贴的样例日志是 step 11，逐字段对照：

| 字段 | 官方 | 本次 | 偏差 |
| --- | --- | --- | --- |
| seconds | 90.668 | 90.41 | −0.27% |
| TFLOP/s/device | 152.415 | 152.85 | +0.29% |
| Tokens/s/device | 542.108 | 543.6 | +0.28% |
| total_weights | 25165824 | 25165824 | 完全一致 |
| loss @ step 11 | 10.989 | 10.958 | −0.3% |

`total_weights` 完全一致说明模型确实是同一个；loss 曲线也吻合。
**结论：官方数字可复现。**

注意本次 step 11 落在异步派发配对上（0.005 s），
所以吞吐用的是同轮干净步的均值，不是 step 11 本身；
只有 loss 取的 step 11 原值。

### 哪些 step 能用

| step | 状态 | 用不用 |
| --- | --- | --- |
| 0 | 含首次编译（114.2 s） | ✗ |
| 1、6、11 | JAX 异步派发假象（0.005–0.35 s，TFLOP 数虚高到六位数） | ✗ |
| 2 | 尚未收敛（116.7 s） | ✗ |
| **3、4、7、8、9、13+** | **稳态（90.4 s）** | **✓** |
| 5、10、12 | xplane profiler 写盘开销（127–181 s） | ✗ |

`step 1` 会报 39817 TFLOP/s/device，`step 6`、`step 11` 会报 0.005 s——
都是 **JAX 异步派发**造成的假象：日志在计算真正完成前就打印了。
这类行数总是与前一步配对出现，把两步加起来才是两步的真实耗时。

官方配方同样开着 profiler，横向对比时是公平的。

### 512 与 256 chips 的对照

| | 512 chips | 256 chips |
| --- | --- | --- |
| `per_device_batch_size` | 6 | 4 |
| global batch size | 3072 | 1024 |
| 稳态 step | 90.41 s | 68.70 s |
| TFLOP/s/device | 152.85 | 134.1 |
| MFU | 33.3% | 29.2% |
| 每 device 权重分片 | 10.5 GB | 21 GB |

**256 chips 拿到 512 chips 单卡吞吐的 87.7%。**

差距几乎全部来自 batch size。chips 减半后每 device 的权重分片翻倍
（10.5 → 21 GB），挤掉了激活空间，`per_device_batch_size` 只能从 6 降到 4
（实测 5 都不行）。每步计算量变少，collective 与计算的 overlap 就变差。

这是硬约束，不是调参没调好。想在 256 chips 上恢复 pdb=6，
得换更省内存的并行策略或精度，不是改几个 flag 能解决的。

## 踩坑速查表

镜像构建、输出目录、spot 抢占等通用问题见
[256chips 配方的踩坑速查表](../DeepSeek3-671B-MaxText-256chips/README.md#踩坑速查表)，
两个规模完全一样。以下是 **512 规模特有**的。

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| `IP_SPACE_EXHAUSTED_WITH_DETAILS` | pod 次级 IP 范围按 maxPods=110 每节点切 `/24`，`/17` 只够 128 节点 | 建池时加 `--max-pods-per-node=16` |
| `Already exists: .../nodePools/<name>` | 上次创建失败的池仍以 `ERROR` 状态存在 | 先 `node-pools delete` 再重建 |
| `failed calling webhook "mjobset.kb.io": no endpoints available` | JobSet controller 挂了 | 见下节 |
| `GCE_STOCKOUT` | 真的没容量，原子分配差一台都不行 | 换 zone 或等 |

### JobSet controller 的镜像会失效

提交作业时报：

```
Internal error occurred: failed calling webhook "mjobset.kb.io":
no endpoints available for service "jobset-webhook-service"
```

查 controller：

```bash
kubectl get pods -n jobset-system
# jobset-controller-manager-xxx   0/1   ErrImagePull
```

根因是 JobSet 默认装的是 **staging 镜像**：

```
us-central1-docker.pkg.dev/k8s-staging-images/jobset/jobset:v0.11.1
    → not found
```

staging registry 的 tag 会被定期回收。老节点靠本地 digest 缓存能一直跑
（我们这套跑了 103 天），**一旦 controller 被重调度到新节点就拉不到镜像**。
节点池扩缩容、节点回收都会触发。

换成正式发布的同版本镜像即可：

```bash
kubectl set image deployment/jobset-controller-manager -n jobset-system \
  manager=registry.k8s.io/jobset/jobset:v0.11.1
```

`registry.k8s.io` 是正式发布仓库，tag 不会被回收。

**这个坑会在你毫无防备时炸**——controller 在你不提交作业时坏掉也没人知道，
等你要跑训练才发现。提交前先 `kubectl get pods -n jobset-system` 看一眼。

## 清理

```bash
kubectl delete jobset $WORKLOAD_NAME
```

删除是异步的。要立刻提交下一个作业，**必须等本作业的 pod 全部消失**：

```bash
while [ "$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME \
           --no-headers 2>/dev/null | wc -l)" -ne 0 ]; do sleep 10; done
```

一定要带 label selector。共享集群上不加过滤会把别人的 pod 也数进去，
循环永远退不出来。

节点池按需删除（**spot 池留着不用也会被抢占回收，但删掉就得重新抢**）：

```bash
gcloud container node-pools delete <NODEPOOL> --cluster <CLUSTER> --region <REGION>
```

**多机 TPU 节点池不能缩容，只能整池删。** 想省钱先缩到 0 是行不通的：

```
501 Unimplemented: Multi-host TPU pool (<name>) manual resize is not supported.
```

整片 TPU 是原子分配的，"改大小"这个概念不成立——这跟 CPU/GPU 节点池的运维直觉
不一样，别浪费时间试 `clusters resize --num-nodes=0`。
