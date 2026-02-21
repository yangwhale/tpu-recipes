# DeepSeek3-671B 训练测试记录

## 模型概况

| 项目 | 值 |
|------|-----|
| 模型 | DeepSeek3-671B (MoE) |
| 总参数量 | 671B |
| 硬件 | TPU v7 (Ironwood) |
| 框架 | MaxText (maxtext-tutorial-v1.5.0) |
| JAX | 0.8.2.dev20251215 |
| Libtpu | 0.0.32.dev20251215+nightly |
| XPK | 0.16.1 |

## 测试结果

### 我的测试记录

| 日期 | 配置 | Precision | Step Time (s) | TFLOPs/s/device | TFLOPs/s/chip | Tokens/s/chip | Loss (final) | 备注 |
|------|------|-----------|---------------|-----------------|---------------|---------------|-------------|------|
| 2026-02-21 | 4k-bf16-4x4x8 (128 chips) | bf16 | 27.00 | 304.0 | 608.0 | 2,427.7 | 7.759 | 30 steps，profiler step 6-8，xpk 1.3.0 [xprof](http://xprof.corp.google.com/trace_viewer/chrisya-3446054219397943486) |
| 2026-02-06 | 4k-bf16-4x4x8 (128 chips) | bf16 | 27.42 | 299.3 | 598.5 | 2,389.7 | 7.759 | 30 steps，profiler step 9 |
| 2026-02-06 | 4k-bf16-4x8x8 (256 chips) | bf16 | 27.12 | 302.6 | 605.2 | 2,416.6 | 9.014 | 30 steps，与官方 recipe 参数一致 |
| 2026-02-06 | 4k-fp8-4x4x8 (128 chips) | fp8_full | 22.39 | 366.5 | 733.1 | 2,926.5 | 8.541 | 30 steps，fp8 量化，比 bf16 快 22% |
| 2026-02-07 | 4k-fp8-4x8x8 (256 chips) | fp8_full | 22.02 | 372.7 | 745.3 | 2,976.0 | 9.702 | 30 steps，fp8 4x8x8，近乎线性扩展 |
| 2026-02-07 | 4k-bf16-8x8x8 (512 chips) | bf16 | 49.70 | 165.1 | 330.2 | 1,318.6 | 10.940 | 30 steps，ici_data_parallelism=2，扩展效率低 ⚠️ |
| 2026-02-07 | 2×4x8x8 multi-slice (512 chips) | bf16 | ~530 | ~4.7 | ~9.4 | ~37.5 | - | DCN 网络未就绪，性能严重异常 ❌ |

### 详细训练日志 - 4x4x8 bf16 (128 chips, 2026-02-21)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|----------------|--------------|---------------|------|
| 0 | 143.79 | 57.1 | 114.1 | 455.8 | 12.270 |
| 1 | 71.62 | 114.6 | 229.2 | 915.1 | 12.270 |
| 2 | 26.98 | 304.2 | 608.3 | 2,429.0 | 12.117 |
| 3 | 26.99 | 304.1 | 608.1 | 2,428.2 | 11.886 |
| 4 | 27.00 | 303.9 | 607.8 | 2,427.0 | 11.622 |
| 5 | 26.99 | 304.1 | 608.2 | 2,428.3 | 11.370 |
| 6* | 27.07 | 303.1 | 606.2 | 2,420.6 | 11.120 |
| 7* | 26.99 | 304.1 | 608.1 | 2,428.1 | 10.869 |
| 8* | 27.01 | 303.9 | 607.8 | 2,426.7 | 10.618 |
| 9 | 59.66 | 137.6 | 275.1 | 1,098.5 | 10.372 |
| 10 | 26.99 | 304.1 | 608.1 | 2,428.1 | 10.133 |
| 11 | 27.00 | 304.0 | 608.0 | 2,427.6 | 9.903 |
| 12 | 27.01 | 303.8 | 607.6 | 2,426.1 | 9.684 |
| 13 | 26.99 | 304.0 | 608.1 | 2,427.9 | 9.475 |
| 14 | 26.98 | 304.1 | 608.3 | 2,428.7 | 9.277 |
| 15 | 27.00 | 304.0 | 608.0 | 2,427.4 | 9.092 |
| 16 | 27.00 | 303.9 | 607.8 | 2,427.0 | 8.919 |
| 17 | 27.00 | 304.0 | 607.9 | 2,427.3 | 8.759 |
| 18 | 26.99 | 304.1 | 608.2 | 2,428.4 | 8.613 |
| 19 | 27.01 | 303.9 | 607.7 | 2,426.6 | 8.479 |
| 20 | 27.01 | 303.8 | 607.7 | 2,426.2 | 8.358 |
| 21 | 26.99 | 304.0 | 608.1 | 2,428.0 | 8.250 |
| 22 | 27.00 | 303.9 | 607.9 | 2,427.2 | 8.154 |
| 23 | 27.00 | 304.0 | 608.0 | 2,427.7 | 8.070 |
| 24 | 26.99 | 304.1 | 608.2 | 2,428.3 | 7.998 |
| 25 | 26.99 | 304.1 | 608.2 | 2,428.2 | 7.935 |
| 26 | 27.00 | 304.0 | 607.9 | 2,427.3 | 7.881 |
| 27 | 27.00 | 304.0 | 608.0 | 2,427.5 | 7.835 |
| 28 | 26.98 | 304.2 | 608.4 | 2,429.1 | 7.795 |
| 29 | 26.99 | 304.1 | 608.1 | 2,428.2 | 7.759 |

- **Step 0** 耗时 144s 是 JIT 编译
- **Step 6-8** (标 *) 是 profiler 采集步骤（profiler_steps=3, skip_first_n_steps_for_profiler=5）
- **Step 9** 耗时 60s 是 profiler xplane dump 影响
- **稳态性能 (排除 Step 0-1, 6-9)**: ~27.00 s/step, ~608.0 TFLOP/s/chip, ~2,428 Tokens/s/chip
- **Loss 下降**: 12.270 -> 7.759 (36.7%)
- **对比 2026-02-06 同配置**: step time 快了 1.5%，TFLOP/s/chip 提升 1.6%（608 vs 598.5）
- **环境差异**: xpk 1.3.0 (vs 0.16.1)，Kueue v0.15.2 (vs v0.14.3)
- **Xprof Profile**: [trace_viewer](http://xprof.corp.google.com/trace_viewer/chrisya-3446054219397943486) | GCS: `gs://chrisya-v7x-us-central1/chrisya-ds3-bf16-4x4x8-20260221-0144/tensorboard/plugins/profile/2026_02_21_02_32_08/`

### 详细训练日志 - 8x8x8 bf16 (512 chips, 2026-02-07)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|----------------|--------------|---------------|------|
| 0 | 122.84 | 66.8 | 133.6 | 533.5 | 12.270 |
| 1 | 66.54 | 123.3 | 246.7 | 984.9 | 12.270 |
| 2 | 49.70 | 165.1 | 330.3 | 1,318.7 | 12.228 |
| 3 | 49.69 | 165.1 | 330.3 | 1,318.8 | 12.152 |
| 4 | 49.71 | 165.1 | 330.2 | 1,318.3 | 12.061 |
| 5 | 49.71 | 165.1 | 330.2 | 1,318.3 | 11.981 |
| 6 | 49.84 | 164.7 | 329.3 | 1,315.0 | 11.904 |
| 7 | 49.72 | 165.1 | 330.1 | 1,318.2 | 11.829 |
| 8 | 49.72 | 165.1 | 330.2 | 1,318.2 | 11.755 |
| 9 | 115.07* | 71.3 | 142.6 | 569.5 | 11.684 |
| 10 | 49.70 | 165.1 | 330.3 | 1,318.6 | 11.615 |
| ... | ... | ... | ... | ... | ... |
| 29 | 49.70 | 165.1 | 330.2 | 1,318.6 | 10.940 |

- **Step 0** 耗时 123s 是 JIT 编译
- **Step 9** 耗时 115s 是 profiler 采集 xplane 数据
- **稳态性能 (Step 2+)**: ~49.70 s/step, ~330.2 TFLOP/s/chip, ~1,319 Tokens/s/chip
- **Loss 下降**: 12.270 → 10.940 (10.8%)
- **关键发现**: per-chip 效率仅为 4x8x8 的 54.6%（330 vs 605 TFLOP/s/chip），总吞吐仅提升 9%

#### 8x8x8 扩展效率分析

DeepSeek3-671B 在 8x8x8 (512 chips) 上的扩展效率显著低于预期：

| 指标 | 4x8x8 (256 chips) | 8x8x8 (512 chips) | 变化 |
|------|-------------------|-------------------|------|
| Step Time | 27.12s | 49.70s | +83.3% (慢了 83%) |
| TFLOP/s/chip | 605.2 | 330.2 | -45.4% |
| Tokens/s/chip | 2,416.6 | 1,318.6 | -45.4% |
| Total Tokens/s | 618,650 | 675,123 | +9.1% (仅提升 9%) |

> **根因分析**:
> - DeepSeek3-671B 的模型 tensor 某维度大小为 512（与 MoE 专家数相关）
> - 4x8x8 (512 devices) 的 `fsdp × fsdp_transpose = 256 × 2 = 512`，刚好等于 tensor 维度
> - 8x8x8 (1024 devices) 超过了这个上限，无法进一步增加 FSDP 分片
> - 被迫使用 `ici_data_parallelism=2` 来消化多余的设备，引入了 all-reduce 通信开销
> - 结论：**DeepSeek3-671B 的 FSDP 并行上限为 512 devices (256 chips)**，超过此限需要其他并行策略（专家并行、张量并行或多 slice DCN）

### Multi-Slice DCN 测试 - 2×4x8x8 (512 chips, 2026-02-07) ❌

#### 测试动机

8x8x8 单 slice 因 FSDP 分片上限导致 45% 效率损失。Multi-slice DCN 方案可以保持每个 slice 内的满血 FSDP 效率（fsdp=256, fsdp_transpose=2），仅在 slice 间通过 DCN 同步梯度。

#### 配置

- 拓扑：2 × tpu7x-4x8x8（2 slices × 256 chips = 512 chips）
- `dcn_data_parallelism=2`（跨 2 个 slice 数据并行）
- `dcn_pipeline_parallelism=1`
- slice 内 ICI 布局与单 slice 4x8x8 完全一致

#### HBM OOM 调试历程

Multi-slice 训练使用 per-device 方式计算 HBM，TPU v7 每 chip 192GB 但有 2 个 TensorCore（device），所以每 device 只有 ~94.75GB（扣除系统保留）。

| batch_size | HBM 使用 | HBM 限制 | 结果 |
|-----------|---------|---------|------|
| 8.0 | 102.62 GB | 94.75 GB | OOM (超 7.88G) |
| 6.0 | 96.25 GB | 94.75 GB | OOM (超 1.50G) |
| 5.0 | 79.30 GB | 94.75 GB | 编译通过 |

注：添加了 `--xla_tpu_enable_scheduler_memory_pressure_tracking=true` 和 `--xla_latency_hiding_scheduler_rerun=2` XLA flags 优化显存调度，但 batch_size=8.0 和 6.0 仍然 OOM。最终 batch_size=5.0 成功编译。

#### 性能结果

| Step | 耗时 (s) | TFLOP/s/device | 说明 |
|------|---------|----------------|------|
| 0 | 134.545 | 38.1 | JIT 编译 |
| 1 | 546.956 | 9.4 | DCN 通信严重阻塞 |
| 2 | 526.438 | 9.7 | DCN 通信严重阻塞 |

- 预期 step time ~27s，实际 ~530s（慢了 ~20x）
- 预期 TFLOP/s/device ~300，实际 ~9.5（仅为预期的 3%）
- XLA 日志中 `CopyToMemorySpace CrossDeviceSrc` 操作耗时 8m44s，确认瓶颈在 DCN 跨 slice 通信

#### 根因分析 — 集群网络基础设施缺失

对比 Google 内部正常运行的 multi-slice 集群（bodaborg）发现，chrisya 集群缺少关键网络配置：

| 功能 | bodaborg (正常) | chrisya (异常) |
|------|----------------|---------------|
| Multi-networking | enableMultiNetworking: true | 未启用 |
| 网卡 | 双 NIC（eth0 管理 + eth1 高速 DCN） | 单 NIC |
| Dataplane | ADVANCED_DATAPATH (Dataplane V2 / Cilium) | LEGACY (kube-proxy) |
| sliceControllerConfig | 已启用 | 未配置 |
| 机器类型 | tpu7x-standard-4t | tpu7x-ultranet-4t |
| MegaScale gRPC 接口 | 配置了 megascale_grpc_interface_prefixes | 未配置 |
| TCP rmem 调优 | DaemonSet 设置 tcp_rmem | 未配置 |

**关键发现**：
1. Multi-slice DCN 通过 host-mediated gRPC（MegaScale 协议）通信，不是 chip-to-chip 直连
2. 需要双网卡（eth0 管理流量，eth1 专用于 DCN 高速通信）
3. `enableMultiNetworking` 和 `ADVANCED_DATAPATH` 是集群级配置，**创建后不可更改**，需要重建集群
4. bodaborg 使用 `tpu7x-standard-4t`（非 ultranet），说明 iRDMA/ultranet 不是 multi-slice 的必要条件

#### 结论

Multi-slice DCN 训练需要集群级别的网络基础设施支持，不能在现有 chrisya 集群上简单启用。测试暂停，等待集群重建或新集群部署。

### 详细训练日志 - 4x8x8 fp8 (256 chips, 2026-02-07)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|----------------|--------------|---------------|------|
| 0 | 137.68 | 59.6 | 119.2 | 476.0 | 12.270 |
| 1 | 35.98 | 228.1 | 456.2 | 1,821.6 | 12.270 |
| 2 | 22.01 | 372.9 | 745.8 | 2,978.0 | 12.170 |
| 3 | 22.04 | 372.4 | 744.9 | 2,974.2 | 12.018 |
| 4 | 22.05 | 372.2 | 744.3 | 2,971.9 | 11.850 |
| 5 | 22.01 | 372.8 | 745.6 | 2,977.1 | 11.695 |
| 6 | 22.13 | 370.8 | 741.7 | 2,961.3 | 11.545 |
| 7 | 22.02 | 372.8 | 745.5 | 2,976.7 | 11.399 |
| 8 | 22.05 | 372.1 | 744.3 | 2,971.8 | 11.256 |
| 9 | 90.74* | 90.4 | 180.9 | 722.2 | 11.118 |
| 10 | 22.01 | 372.9 | 745.7 | 2,977.5 | 10.985 |
| 11 | 22.01 | 372.8 | 745.7 | 2,977.3 | 10.857 |
| 12 | 22.01 | 372.8 | 745.7 | 2,977.3 | 10.736 |
| ... | ... | ... | ... | ... | ... |
| 29 | 22.01 | 372.8 | 745.6 | 2,977.0 | 9.702 |

- **Step 0** 耗时 138s 是 JIT 编译
- **Step 9** 耗时 91s 是 profiler 采集 xplane 数据
- **稳态性能 (Step 2+)**: ~22.02 s/step, ~745.3 TFLOP/s/chip, ~2,976 Tokens/s/chip
- **Loss 下降**: 12.270 → 9.702 (20.9%)

### 全配置对比总表

| 配置 | Chips | Precision | Step Time (s) | TFLOP/s/chip | Tokens/s/chip | Total Tokens/s | 相对 bf16 4x4x8 |
|------|-------|-----------|---------------|--------------|---------------|----------------|-----------------|
| 4x4x8 | 128 | bf16 | 27.00 | 608.0 | 2,427.7 | 310,746 | 基准 (2026-02-21) |
| 4x8x8 | 256 | bf16 | 27.12 | 605.2 | 2,416.6 | 618,650 | +102% (扩展) |
| 4x4x8 | 128 | fp8 | 22.39 | 733.1 | 2,926.5 | 374,752 | +22.5% (fp8) |
| 4x8x8 | 256 | fp8 | 22.02 | 745.3 | 2,976.0 | 761,848 | +149% (fp8+扩展) |
| 8x8x8 | 512 | bf16 | 49.70 | 330.2 | 1,318.6 | 675,123 | +121% (扩展效率低) ⚠️ |
| 2×4x8x8 DCN | 512 | bf16 | ~530 | ~9.4 | ~37.5 | ~19,200 | DCN 网络未就绪 ❌ |

> **关键结论**:
> 1. **fp8 量化提升 ~23%**: 相同拓扑下，fp8 比 bf16 快 22-23%
> 2. **128→256 chips 近乎线性扩展**: per-chip 性能不变（~1-2% 提升），总吞吐翻倍
> 3. **组合效果**: fp8 + 4x8x8 相比 bf16 4x4x8 实现 149% 的吞吐提升（2.49x）
> 4. **512 chips 扩展瓶颈**: 8x8x8 (512 chips) per-chip 效率仅为 4x8x8 的 54.6%，因为模型 tensor 维度限制 FSDP 分片上限为 512 devices，超出部分只能用低效的 ICI 数据并行
> 5. **Multi-slice DCN 需要集群基础设施**: 2×4x8x8 测试因集群缺少 multi-networking、Dataplane V2、双网卡等关键配置而失败（530s/step vs 预期 27s），需要重建集群才能支持

### 详细训练日志 - 4x4x8 fp8 (128 chips, 2026-02-06)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|----------------|--------------|---------------|------|
| 0 | 156.71 | 52.4 | 104.7 | 418.2 | 12.270 |
| 1 | 68.30 | 120.2 | 240.3 | 959.5 | 12.270 |
| 2 | 22.38 | 366.7 | 733.4 | 2,928.2 | 12.127 |
| 3 | 22.39 | 366.5 | 733.0 | 2,926.8 | 11.914 |
| 4 | 22.38 | 366.7 | 733.3 | 2,928.1 | 11.677 |
| 5 | 22.38 | 366.6 | 733.3 | 2,927.7 | 11.457 |
| 6 | 22.51 | 364.6 | 729.2 | 2,911.6 | 11.242 |
| 7 | 22.38 | 366.6 | 733.3 | 2,927.9 | 11.030 |
| 8 | 22.39 | 366.5 | 732.9 | 2,926.4 | 10.821 |
| 9 | 70.48* | 116.4 | 232.9 | 929.8 | 10.617 |
| 10 | 22.38 | 366.7 | 733.4 | 2,928.5 | 10.421 |
| 11 | 22.39 | 366.6 | 733.2 | 2,927.7 | 10.233 |
| 12 | 22.42 | 366.1 | 732.2 | 2,923.7 | 10.054 |
| ... | ... | ... | ... | ... | ... |
| 29 | 22.39 | 366.5 | 733.1 | 2,927.0 | 8.541 |

- **Step 0** 耗时 157s 是 JIT 编译（fp8 quantization 内核编译比 bf16 更耗时）
- **Step 9** 耗时 70s 是 profiler 采集 xplane 数据
- **稳态性能 (Step 2+)**: ~22.39 s/step, ~733 TFLOP/s/chip, ~2,927 Tokens/s/chip
- **Loss 下降**: 12.270 → 8.541 (30.4%)

### bf16 vs fp8 对比 (4x4x8, 128 chips)

| 指标 | bf16 | fp8_full | 变化 |
|------|------|----------|------|
| Step Time | 27.42s | 22.39s | -18.3% (快了 22.4%) |
| TFLOP/s/chip | 598.5 | 733.1 | +22.5% |
| Tokens/s/chip | 2,389.7 | 2,926.5 | +22.5% |
| Total Tokens/s | 305,882 | 374,752 | +22.5% |
| JIT 编译时间 | 125s | 157s | +25.6% (fp8 编译更复杂) |
| Final Loss (30 steps) | 7.759 | 8.541 | fp8 loss 略高 |

> **说明**: fp8 量化使用 `quantization=fp8_full` + `use_qwix_quantization=True`，通过 fp8 矩阵乘法在 TPU v7 上获得 22.5% 的吞吐提升。
> fp8 训练的 Loss 略高于 bf16（8.541 vs 7.759），这在预期范围内，因为 fp8 精度低于 bf16，但在大规模训练中 Loss 差异会逐渐缩小。
> fp8 recipe 包含大量 tile 参数（`wi_tile_*`, `wo_tile_*`）用于优化 quantized matmul 的切块大小。

### 详细训练日志 - 4x4x8 bf16 (128 chips, 2026-02-06)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|----------------|--------------|---------------|------|
| 0 | 125.34 | 65.5 | 131.0 | 522.9 | 12.270 |
| 1 | 32.91 | 249.4 | 498.7 | 1,991.3 | 12.270 |
| 2 | 27.41 | 299.4 | 598.8 | 2,390.9 | 12.117 |
| 3 | 27.42 | 299.3 | 598.7 | 2,390.4 | 11.886 |
| 4 | 27.42 | 299.3 | 598.7 | 2,390.4 | 11.622 |
| 5 | 27.43 | 299.2 | 598.5 | 2,389.7 | 11.370 |
| 6 | 27.57 | 297.7 | 595.3 | 2,377.1 | 11.120 |
| 7 | 27.42 | 299.2 | 598.5 | 2,389.7 | 10.869 |
| 8 | 27.44 | 299.0 | 598.1 | 2,388.1 | 10.618 |
| 9 | 80.72* | 101.7 | 203.3 | 811.9 | 10.372 |
| 10 | 27.42 | 299.3 | 598.6 | 2,390.3 | 10.133 |
| ... | ... | ... | ... | ... | ... |
| 29 | 27.42 | 299.3 | 598.5 | 2,389.7 | 7.759 |

- **Step 0** 耗时 125s 是 JIT 编译（首次编译 XLA HLO → TPU 可执行代码）
- **Step 9** 耗时 81s 是 profiler 采集 xplane 数据
- **稳态性能 (Step 2+)**: ~27.4 s/step, ~598.5 TFLOP/s/chip, ~2,390 Tokens/s/chip
- **Loss 下降**: 12.270 → 7.759 (36.7%)，合成数据上收敛明显

### 详细训练日志 - 4x8x8 (256 chips, 2026-02-06)

| Step | 耗时 (s) | TFLOP/s/device | TFLOP/s/chip | Tokens/s/chip | Loss |
|------|---------|----------------|--------------|---------------|------|
| 0 | 131.08 | 62.6 | 125.2 | 500.0 | 12.270 |
| 1 | 35.64 | 230.3 | 460.5 | 1,838.8 | 12.270 |
| 2 | 27.12 | 302.6 | 605.3 | 2,416.8 | 12.161 |
| 3 | 27.13 | 302.5 | 605.1 | 2,416.0 | 11.991 |
| 4 | 27.11 | 302.7 | 605.3 | 2,417.0 | 11.795 |
| 5 | 27.12 | 302.7 | 605.3 | 2,417.0 | 11.608 |
| 6 | 27.26 | 301.1 | 602.2 | 2,404.6 | 11.424 |
| 7 | 27.12 | 302.7 | 605.3 | 2,417.0 | 11.241 |
| 8 | 27.12 | 302.7 | 605.3 | 2,416.8 | 11.061 |
| 9 | 91.78* | 89.4 | 178.8 | 714.0 | 10.885 |
| 10 | 27.12 | 302.6 | 605.2 | 2,416.2 | 10.715 |
| 11 | 27.12 | 302.6 | 605.2 | 2,416.6 | 10.550 |
| ... | ... | ... | ... | ... | ... |
| 29 | 27.11 | 302.7 | 605.4 | 2,417.2 | 9.014 |

- **Step 0** 耗时 131s 是 JIT 编译
- **Step 9** 耗时 92s 是 profiler 采集 xplane 数据
- **稳态性能 (Step 2+)**: ~27.12 s/step, ~605.2 TFLOP/s/chip, ~2,417 Tokens/s/chip
- **Loss 下降**: 12.270 → 9.014 (26.5%)
- **参数与官方 `run_recipe.sh` 完全一致**（仅额外添加了 profiler 配置）

### 4x4x8 vs 4x8x8 对比

| 指标 | 4x4x8 (128 chips) | 4x8x8 (256 chips) | 说明 |
|------|-------------------|-------------------|------|
| Step Time | 27.42s | 27.12s | 4x8x8 略快 (~1%) |
| TFLOP/s/chip | 598.5 | 605.2 | 4x8x8 略高 (~1%) |
| Tokens/s/chip | 2,389.7 | 2,416.6 | per-chip 吞吐接近 (~1%) |
| GBS | 2,048 | 4,096 | 4x8x8 翻倍 |
| Total Tokens/s | 305,882 | 618,650 | 4x8x8 翻倍，近乎线性扩展 |
| ici_fsdp_transpose_parallelism | 1 | 2 | 4x8x8 使用 2D FSDP |
| fsdp_shard_on_exp | True | False | 分片策略不同 |

> **说明**: 芯片数从 128 翻倍到 256 后，GBS 和总吞吐量均翻倍，per-chip 吞吐几乎不变（~1% 提升），表现出接近线性的扩展效率。
> 4x8x8 使用 `ici_fsdp_transpose_parallelism=2` 和 `use_2d_fsdp_sharding=True`，采用 2D FSDP 分片策略。

## 测试配置

### bf16 配置关键参数

**共同参数（4x4x8 和 4x8x8）:**
- per_device_batch_size: 8.0
- max_target_length: 4096
- attention: flash
- remat_policy: custom
- decoder_layer_input: offload
- sparse_matmul: True
- megablox: True
- use_tokamax_gmm: True
- use_tokamax_splash: True
- opt_type: adamw
- mu_dtype/grad_dtype: bfloat16
- dataset_type: synthetic
- steps: 30

**4x4x8 特有参数:**
- ici_fsdp_transpose_parallelism: 1
- fsdp_shard_on_exp: True

**4x8x8 特有参数:**
- ici_fsdp_transpose_parallelism: 2
- fsdp_shard_on_exp: False
- use_2d_fsdp_sharding: True

### fp8 配置关键参数 (4x4x8)

**与 bf16 相同的参数**: per_device_batch_size, max_target_length, attention, remat_policy 等

**fp8 特有参数:**
- quantization: fp8_full
- use_qwix_quantization: True
- weight_quantization_calibration_method: fixed,-224,224
- act_quantization_calibration_method: fixed,-224,224
- wi_tile_* / wo_tile_*: 共 24 个 tile 参数，控制 quantized matmul 的切块大小
- ici_fsdp_transpose_parallelism: 1 (与 bf16 4x4x8 相同)
- fsdp_shard_on_exp: True

**fp8 4x8x8 与 fp8 4x4x8 的差异:**
- ici_fsdp_transpose_parallelism: 2 (4x4x8 是 1)
- moe_fsdp_use_two_stage_all_gather: True (4x4x8 无此参数)
- use_max_logit_estimate: 22 (4x4x8 是 -1)
- attn_logits_soft_cap: 15 (4x4x8 无此参数)
- XLA_FLAGS 更多（增加了 `data_parallel_opt`, `ici_rs_pipelining` 等 flags）

## 目录结构

```
deepseek3-671b/
├── README.md                              # 本文件
├── 4k-bf16-tpu7x-4x4x8/                  # bf16 精度, 4x4x8 拓扑
│   └── xpk/
│       ├── README.md                      # 官方使用说明
│       ├── run_recipe.sh                  # 官方原版脚本
│       └── submit_deepseek3.sh            # 自定义提交脚本
├── 4k-bf16-tpu7x-4x8x8/                  # bf16 精度, 4x8x8 拓扑
│   └── xpk/
│       ├── README.md                      # 官方使用说明
│       ├── run_recipe.sh                  # 官方原版脚本
│       └── submit_deepseek3_4x8x8.sh     # 自定义提交脚本
├── 4k-fp8-tpu7x-4x4x8/                   # fp8 精度, 4x4x8 ✅ 已测试
│   └── xpk/
│       ├── run_recipe.sh                  # 官方原版脚本
│       └── submit_deepseek3_fp8.sh        # 自定义提交脚本
├── 4k-fp8-tpu7x-4x8x8/                   # fp8 精度, 4x8x8 ✅ 已测试
│   └── xpk/
│       ├── run_recipe.sh                  # 官方原版脚本
│       └── submit_deepseek3_fp8_4x8x8.sh # 自定义提交脚本
├── 4k-bf16-tpu7x-8x8x8/                  # bf16 精度, 8x8x8 ⚠️ 扩展效率低
│   ├── k8s/
│   │   ├── README.md                      # 官方 K8s 部署说明
│   │   └── k8s_manifest.yaml              # 官方 K8s manifest
│   └── xpk/
│       ├── README.md                      # 官方使用说明
│       ├── run_recipe.sh                  # 官方原版脚本
│       └── submit_deepseek3_8x8x8.sh     # 自定义提交脚本 (ici_data_parallelism=2)
├── 4k-bf16-tpu7x-8x8x16/                 # bf16 精度, 8x8x16（待测试）
│   ├── k8s/
│   │   ├── README.md                      # 官方 K8s 部署说明
│   │   └── k8s_manifest.yaml              # 官方 K8s manifest
│   └── xpk/
│       ├── README.md                      # 官方使用说明
│       ├── run_recipe.sh                  # 官方原版脚本
│       └── submit_deepseek3_8x8x16.sh    # 自定义提交脚本
└── 4k-bf16-tpu7x-2x4x8x8/                # bf16 精度, 2×4x8x8 multi-slice DCN ❌ 集群不支持
    └── xpk/
        └── submit_deepseek3_2x4x8x8.sh   # Multi-slice DCN 提交脚本（需集群重建）
```

## 注意事项

- 一个 chip 有 2 个 device (TensorCore)，日志中的 `TFLOP/s/device` 和 `Tokens/s/device` 均需乘以 2 才是 per-chip 性能
- 4x4x8 的日志 `Tokens/s/device` 已经是 per-chip 值（无需再乘 2），但 4x8x8 的需要乘 2
- Profiler 采集期间 step time 会显著增加（~3x），这是正常现象，不影响稳态性能评估
- `enable_checkpointing=False`，当前配置不保存 checkpoint
- 修改 MaxText 代码后需要重新构建 Docker 镜像
