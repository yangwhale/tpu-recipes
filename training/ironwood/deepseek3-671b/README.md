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
| 2026-02-06 | 4k-bf16-4x4x8 (128 chips) | bf16 | 27.42 | 299.3 | 598.5 | 2,389.7 | 7.759 | 30 steps，profiler step 9 |
| 2026-02-06 | 4k-bf16-4x8x8 (256 chips) | bf16 | 27.12 | 302.6 | 605.2 | 2,416.6 | 9.014 | 30 steps，与官方 recipe 参数一致 |

### 详细训练日志 - 4x4x8 (128 chips, 2026-02-06)

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
├── 4k-fp8-tpu7x-4x4x8/                   # fp8 精度, 4x4x8（待测试）
└── 4k-fp8-tpu7x-4x8x8/                   # fp8 精度, 4x8x8（待测试）
```

## 注意事项

- 一个 chip 有 2 个 device (TensorCore)，日志中的 `TFLOP/s/device` 和 `Tokens/s/device` 均需乘以 2 才是 per-chip 性能
- 4x4x8 的日志 `Tokens/s/device` 已经是 per-chip 值（无需再乘 2），但 4x8x8 的需要乘 2
- Profiler 采集期间 step time 会显著增加（~3x），这是正常现象，不影响稳态性能评估
- `enable_checkpointing=False`，当前配置不保存 checkpoint
- 修改 MaxText 代码后需要重新构建 Docker 镜像
