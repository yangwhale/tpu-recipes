# TPU 单位对照：chip、device、TensorCore

> English version: [TPU-UNITS.en.md](TPU-UNITS.en.md)

TPU 生态里有三套单位在同时使用，而且**同一个词在不同代际含义不同**。算错会导致
MFU 差一倍、内存估算差一倍、并行度配置错误。本文把这些关系钉死。

## 三套单位分别用在哪

| 场景 | 用的单位 | 例子 |
| --- | --- | --- |
| 机型名 `v5p-N` / `v6e-N` | **TensorCore** 数 | `v5p-1024` = 1024 TC |
| GKE 拓扑 `--tpu-topology` | **chip** 数 | `4x8x8` = 256 chips |
| GKE 资源 `google.com/tpu` | **chip** 数 | 每 VM 4 chips |
| MaxText 日志 `TFLOP/s/device` | **JAX device** 数 | 见下方换算 |
| `per_device_batch_size` | **JAX device** 数 | 同上 |
| `ici_fsdp_parallelism` | **JAX device** 数 | 同上 |

## chip 与 JAX device 的关系随代际变化

这是最容易出错的地方。

| 代际 | 架构 | HBM/chip | **JAX device / chip** |
| --- | --- | --- | --- |
| v4 | MegaCore | 32 GB | **1** |
| v5p | MegaCore | 95 GB | **1** |
| v6e | 单 TensorCore | 32 GB | **1** |
| **v7 (Ironwood)** | dual-chiplet | 192 GB | **2** |

**v4 / v5p 是 MegaCore**：一个 chip 内的 2 个 chiplet 共享统一内存空间，编程上合并
成一个 logical core，所以 JAX 只看到 1 个 device。

**v7 (Ironwood) 是 dual-chiplet 独立内存**：每个 chiplet 有自己的 96 GB HBM 和
TensorCore，JAX 把它们暴露为 2 个独立 device（topology 加第 4 维区分 chiplet）。

结论：**同样写 `TFLOP/s/device`，在 v5p 上指整个 chip，在 v7 上只是半个 chip。**

## 机型名的数字是 TensorCore 数

```
v5p-1024  = 1024 TensorCores = 512 chips = 512 JAX devices
v5p-512   =  512 TensorCores = 256 chips = 256 JAX devices
```

注意 v5p 每 chip 物理上有 2 个 TensorCore，但 MegaCore 模式下合并为 1 个 JAX
device。所以 **机型名里的数字是你实际能编程操作的 device 数的两倍**。

v7 的 GKE machine type 不走这套命名，直接用 `tpu7x-standard-4t`（4 chips/VM）。

## 常见换算

### MFU

```
v5p:  MFU = TFLOP/s/device        / 459     # 1 device = 1 chip
v7x:  MFU = TFLOP/s/device × 2    / 2306    # 1 chip = 2 devices
```

459 和 2306 分别是 v5p、v7 的 **per-chip** BF16 峰值 TFLOPS。

**v7 上那个 `× 2` 在 v5p 上不能做**，这是最容易搬错的一步。

### global batch size

```
GBS = per_device_batch_size × JAX device 数
```

| 配置 | chips | devices | pdb | GBS |
| --- | --- | --- | --- | --- |
| v7x 4x4x8 | 128 | 256 | 8.0 | 2048 |
| v7x 4x4x4 | 64 | 128 | 8.0 | 1024 |
| v5p 4x8x8 | 256 | 256 | 4 | 1024 |

v5p 那行的 pdb 只有 4 不是笔误。device 数虽然同为 256，但 v5p 每 device
只有 95 GB，而 v7x 每 chip 有 192 GB——同样的 device 数装不下同样的 batch。
DeepSeek V3 671B 上实测 pdb=5 就会失败，详见
[v5p 配方说明](v5p/README.md#3-hbm-少一半batch-size-要重算)。

### FSDP 分片压力

`ici_fsdp_parallelism` 按 **device** 计。设模型静态状态（权重 + 优化器）总量为 S：

```
每 device 分片 = S / ici_fsdp_parallelism
每 device 可用 HBM = HBM_per_chip / (device 数 / chip 数)
```

以 DeepSeek V3 671B（S ≈ 5.4 TB）为例：

| 配置 | devices | 分片/device | HBM/device | 余量 |
| --- | --- | --- | --- | --- |
| v7x 128 chips | 256 | 21 GB | 96 GB | 75 GB |
| v7x 64 chips | 128 | 42 GB | 96 GB | 54 GB |
| v5p 256 chips | 256 | 21 GB | 95 GB | 74 GB |

**256 v5p chips 与 128 v7x chips 等效**：device 数相同（256）、每 device 分片相同
（21 GB）、可用 HBM 接近（95 vs 96 GB）。这是跨代际做对照实验时的正确配比。

注意此处 v5p 的「HBM/device」等于整个 chip 的 95 GB，因为 1 chip = 1 device；
而 v7x 的 96 GB 只是半个 chip。

## 拓扑形状的选择

同样的 chip 数可以有多种 3D 形状，但**不等价**：

```
256 chips = 4x8x8  ✓ 接近立方体，对分带宽好
          = 4x4x16 ✗ 长条形，跨长轴通信要绕远
```

3D torus 下越接近立方体，bisection bandwidth 越高。上游配方对 256 chips 一律用
`4x8x8`，不要自己凑数。

## 检查清单

配置新规模时按顺序核对：

1. 目标 chip 数 → 查该代际的 device/chip 比 → 得出 device 数
2. 拓扑选最接近立方体的形状
3. `per_device_batch_size` 按 device 数算 GBS
4. 有 `num_experts % ici_fsdp_parallelism` 类约束的，用 **device 数**代入
5. 读日志时确认 `TFLOP/s/device` 要不要 × 2 才是 per-chip
