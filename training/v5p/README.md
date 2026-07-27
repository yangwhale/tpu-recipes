# TPU v5p 训练配方

> English version: [README.en.md](README.en.md)

本目录是 TPU v5p 的训练配方。若你之前只看过
[Ironwood (v7) 的配方](../ironwood/README.md)，请先读
[从 v7x 转到 v5p 要改什么](#从-v7x-转到-v5p-要改什么)——两代硬件有几处差异会让
照搬的配置直接失败或悄悄跑慢。

## 硬件规格

| 项目 | v5p | v7 (Ironwood) |
| --- | --- | --- |
| 架构 | MegaCore（2 chiplet 共享统一内存） | dual-chiplet（各自独立内存） |
| **JAX device / chip** | **1** | **2** |
| HBM / chip | 95 GB HBM2e | 192 GiB HBM3e |
| HBM 带宽 / chip | ~2.8 TB/s | 7.4 TB/s |
| bf16 峰值 / chip | 459 TFLOPS | 2306 TFLOPS |
| 互联 | ICI 3.0 | ICI 4.0 |
| SparseCore | 有 | 有（每 chiplet 2 个） |

单位换算的完整说明见 [TPU-UNITS.md](../TPU-UNITS.md)。

## 从 v7x 转到 v5p 要改什么

### 1. `chip` 与 `device` 不再是 1:2

v7 每个 chip 暴露 **2 个** JAX device，v5p 是 MegaCore，**1 chip = 1 device**。
凡是按 device 计的参数（`per_device_batch_size`、`ici_fsdp_parallelism`、
日志里的 `TFLOP/s/device`）在两代之间含义都不同。

MFU 的算法因此也不同：

```
v5p:  MFU = TFLOP/s/device        / 459
v7x:  MFU = TFLOP/s/device × 2    / 2307
```

**v7x 上那个 `× 2` 搬到 v5p 就会把 MFU 算成两倍。**

### 2. 机型名里的数字是 TensorCore 数

```
v5p-1024  = 1024 TensorCores = 512 chips = 512 JAX devices
v5p-512   =  512 TensorCores = 256 chips = 256 JAX devices
```

v5p 每 chip 物理上有 2 个 TensorCore，MegaCore 把它们合并成 1 个 JAX device。
所以**机型名的数字是你实际编程操作的 device 数的两倍**。

v7 在 GKE 上不用这套命名，直接是 `tpu7x-standard-4t`（4 chips/VM）。

### 3. HBM 少一半，batch size 要重算

v5p 单 device 95 GB，v7x 单 device 96 GB——看着接近，但 v7x 一个 chip 有
两个这样的 device，**每 chip 总量是 192 vs 95，差一倍**。

同样的 chip 数下 v5p 能放的模型状态只有 v7x 的一半。实测对照：

| | v7x 4x4x8 | v5p 4x8x8 |
| --- | --- | --- |
| chips | 128 | 256 |
| JAX devices | 256 | 256 |
| 序列长度 | 4096 | 8192 |
| `per_device_batch_size` | 8.0 | 4 |

### 4. SparseCore 的 flag 名字两代完全不同

这是最容易踩的一处：两代都有 SparseCore，但 XLA flag **不通用**。

v7x 配方用的是：

```
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
--xla_tpu_enable_concurrent_sparse_core_offloading=true
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true
```

v5p 配方用的是另一组：

```
--xla_tpu_enable_sparse_core_collective_offload_{all_gather,all_reduce,reduce_scatter}=true
--2a886c8_chip_config_name=megachip_tccontrol
--xla_tpu_use_tc_device_shape_on_sc=true
--xla_sc_enable_instruction_fusion=false
--xla_sc_disjoint_spmem=false
--xla_sc_disable_megacore_partitioning=true
```

后一组里前三个是 **offload 开关**，后五个是 **运行模式**。
只开开关不配运行模式属于半吊子状态，实测会损失约 15% 吞吐，
而且不会报任何错——详见
[256chips 配方的实测记录](DeepSeek3-671B-MaxText-256chips/README.md#参数不全会显著压低-mfu)。

### 5. 并行参数的组合不一样

v7x DeepSeek 配方用 `ici_fsdp_transpose_parallelism` 和
`ici_pipeline_parallelism`；v5p 配方用 `ici_tensor_parallelism`。
不要跨代际照抄并行配置。

### 6. placement policy 由 GKE 自动生成

v7 无法自动创建 placement policy，需要先手工建好再创建节点池。
**v5p 相反**——手动指定反而会失败：

```
Required field 'resource.requestedRunDuration' not specified
```

v5p 节点池创建时不要带 `--placement-policy`，GKE 会自动生成 `COMPACT`。

## 配方列表

### MaxText

| 配方 | 规模 | 说明 |
| --- | --- | --- |
| [DeepSeek3-671B-MaxText](DeepSeek3-671B-MaxText/README.md) | v5p-1024（512 chips） | 官方配方，XPK 提交 |
| [DeepSeek3-671B-MaxText-256chips](DeepSeek3-671B-MaxText-256chips/README.md) | 256 chips | 纯 K8s manifest，含完整踩坑记录 |
| [Llama3.1-405B-MaxText](Llama3.1-405B-MaxText/README.md) | v5p-1024 | |
| [Llama4-Scout-17B-16E-Maxtext](Llama4-Scout-17B-16E-Maxtext/README.md) | v5p-256 / 512 / 1024 | |
| [Llama4-Maverick-17B-128E-Maxtext](Llama4-Maverick-17B-128E-Maxtext/README.md) | v5p-256 | |
| [Mixtral-8X7B-Maxtext](Mixtral-8X7B-Maxtext/README.md) | | |
| [GPT3-175B-MaxText](GPT3-175B-MaxText/README.md) | | |
| [Llama2-7B-Maxtext](Llama2-7B-Maxtext/README.md) | | |

### MaxDiffusion / 其他

| 配方 | 说明 |
| --- | --- |
| [SDXL-MaxDiffusion](SDXL-MaxDiffusion/README.md) | Stable Diffusion XL |
| [Diffusion-2-MaxDiffusion](Diffusion-2-MaxDiffusion/README.md) | Stable Diffusion 2 |
| [DLRM-V2-Tensorflow](DLRM-V2-Tensorflow/README.md) | TensorFlow，非 MaxText |

## 实测数据

以下是本仓库实际跑出来的数据，非官方文档抄录。

| 模型 | chips | 拓扑 | 序列长度 | GBS | 精度 | step 时间 | TFLOP/s/device | MFU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V3 671B | 256 | 4x8x8 | 8192 | 1024 | bf16 | 68.70 s | 134.1 | 29.2% |

环境：`cloud-tpu-multipod-dev`，us-central1-a，spot，2026-07-27。
完整方法与踩坑见
[DeepSeek3-671B-MaxText-256chips](DeepSeek3-671B-MaxText-256chips/README.md)。

v5p 的 `TFLOP/s/device` 就是 per-chip 值（1 chip = 1 device），
与 [ironwood/README.md](../ironwood/README.md) 里的 `TFLOPs/sec/chip`
可以直接比较，**不需要再乘 2**。

## 通用注意事项

这几条在所有 v5p 配方上都适用，是实跑中反复撞到的：

- **`base_output_directory` 必须真实可写。** 它只在 JAX process 0 上被使用，
  写失败时表现为其余所有 host 在集合通信上挂死，日志现场全在无辜的机器上。
  只跑 benchmark 时用本地路径最省事。
- **老配方钉的 commit 配不上今天的 pip。** `requirements.txt` 里大量依赖不带版本号，
  2025 年的配方在 2026 年重新构建会拉到不兼容的新版。用
  `uv pip install --exclude-newer <日期>` 整套钉回去。
- **XPK 的 wrapper 会吞掉退出码。** 训练失败时 pod 仍是 `Completed`。
  判断成败要看日志里的 `completed step`，不要看 JobSet 状态。

展开说明见
[DeepSeek3-671B-MaxText-256chips 的踩坑速查表](DeepSeek3-671B-MaxText-256chips/README.md#踩坑速查表)。
