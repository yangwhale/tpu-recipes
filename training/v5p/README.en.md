# TPU v5p training recipes

> 中文版：[README.md](README.md)

This directory holds the TPU v5p training recipes. If you have only worked with
the [Ironwood (v7) recipes](../ironwood/README.md) before, read
[what changes when moving from v7x to v5p](#what-changes-when-moving-from-v7x-to-v5p)
first — several differences between the two generations will either break a
copied config outright or silently slow it down.

## Hardware

| Item | v5p | v7 (Ironwood) |
| --- | --- | --- |
| Architecture | MegaCore (2 chiplets, unified memory) | dual-chiplet (independent memory) |
| **JAX devices / chip** | **1** | **2** |
| HBM / chip | 95 GB HBM2e | 192 GiB HBM3e |
| HBM bandwidth / chip | ~2.8 TB/s | 7.4 TB/s |
| bf16 peak / chip | 459 TFLOPS | 2306 TFLOPS |
| Interconnect | ICI 3.0 | ICI 4.0 |
| SparseCore | yes | yes (2 per chiplet) |

Full unit conversions are in [TPU-UNITS.en.md](../TPU-UNITS.en.md).

## What changes when moving from v7x to v5p

### 1. chip-to-device is no longer 1:2

A v7 chip exposes **2** JAX devices. v5p uses MegaCore, so **1 chip = 1 device**.
Every parameter counted in devices — `per_device_batch_size`,
`ici_fsdp_parallelism`, the `TFLOP/s/device` in the logs — means something
different across the two generations.

MFU is therefore computed differently:

```
v5p:  MFU = TFLOP/s/device        / 459
v7x:  MFU = TFLOP/s/device × 2    / 2306
```

**Carrying the `× 2` over from v7x doubles your reported MFU on v5p.**

### 2. The number in the machine name counts TensorCores

```
v5p-1024  = 1024 TensorCores = 512 chips = 512 JAX devices
v5p-512   =  512 TensorCores = 256 chips = 256 JAX devices
```

A v5p chip physically has 2 TensorCores; MegaCore merges them into one JAX
device. So **the number in the machine name is twice the device count you
actually program against**.

v7 does not use this naming in GKE — it uses `tpu7x-standard-4t` (4 chips/VM).

### 3. Half the HBM, so batch size must be recomputed

v5p gives 95 GB per device and v7x gives 96 GB per device — close. But a v7x
chip holds two such devices, so **per chip it is 192 vs 95, a 2× difference**.

At the same chip count, v5p fits half the model state. Measured comparison:

| | v7x 4x4x8 | v5p 4x8x8 |
| --- | --- | --- |
| chips | 128 | 256 |
| JAX devices | 256 | 256 |
| Sequence length | 4096 | 8192 |
| `per_device_batch_size` | 8.0 | 4 |

### 4. SparseCore flags are named entirely differently

This is the easiest trap: both generations have SparseCore, but the XLA flags
**do not carry over**.

The v7x recipes use:

```
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
--xla_tpu_enable_concurrent_sparse_core_offloading=true
--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true
--xla_tpu_use_single_sparse_core_for_all_gather_offload=true
```

The v5p recipes use a different set:

```
--xla_tpu_enable_sparse_core_collective_offload_{all_gather,all_reduce,reduce_scatter}=true
--2a886c8_chip_config_name=megachip_tccontrol
--xla_tpu_use_tc_device_shape_on_sc=true
--xla_sc_enable_instruction_fusion=false
--xla_sc_disjoint_spmem=false
--xla_sc_disable_megacore_partitioning=true
```

In that second set the first three are **offload switches** and the last five
configure the **operating mode**. Enabling the switches without the operating
mode is a half-applied configuration: measured cost is roughly 15% throughput,
and it raises no error. See
[the measured record in the 256-chip recipe](DeepSeek3-671B-MaxText-256chips/README.en.md#an-incomplete-parameter-set-costs-real-mfu).

### 5. Different parallelism knobs

The only knob shared by both generations is `ici_fsdp_parallelism=-1`:

| Parameter | v7x DeepSeek | v5p DeepSeek 256chips |
| --- | --- | --- |
| `ici_fsdp_parallelism` | `-1` | `-1` |
| `ici_fsdp_transpose_parallelism` | `1` (4x4x8) / `2` (4x8x8) | unused |
| `ici_pipeline_parallelism` | `1` | unused |
| `ici_tensor_parallelism` | unused | `1` |
| `dcn_data_parallelism` | `-1` | unused |
| `dcn_pipeline_parallelism` | `1` | unused |

Two things that are easy to miss:

- `ici_fsdp_transpose_parallelism` **varies with topology** (1 for 4x4x8, 2 for
  4x8x8). It is not a constant — do not carry it across topologies
- `dcn_*` are cross-slice dimensions. The v7x recipes carry them because they
  are organised as multi-slice; this v5p recipe is single-slice and has no
  such layer

Do not copy a parallelism config across generations.
(Sources: `ironwood/deepseek3-671b/*/k8s/k8s_manifest.yaml` and the
[256chips manifest](DeepSeek3-671B-MaxText-256chips/k8s/k8s_manifest.yaml).)

### 6. GKE generates the placement policy for you

v7 cannot auto-create placement policies — you must create one by hand before
creating the node pool. **v5p is the opposite**: specifying one manually fails
with

```
Required field 'resource.requestedRunDuration' not specified
```

Omit `--placement-policy` when creating a v5p node pool; GKE generates a
`COMPACT` one.

The "generates one for you" half has been verified at two scales — pass only
`--tpu-topology`, no placement arguments, and `describe` afterwards shows
`type: COMPACT` either way:

| Node pool | topology passed in | Generated by GKE |
| --- | --- | --- |
| 64 VMs (multi-host) | `4x8x8` | `tpuTopology: 4x8x8, type: COMPACT` |
| 1 VM (single-host) | `2x2x1` | `tpuTopology: 2x2x1, type: COMPACT` |

Verify with:

```bash
gcloud container node-pools describe <NODEPOOL> --cluster <CLUSTER> \
  --region <REGION> --format='yaml(placementPolicy)'
```

## Recipes

### MaxText

| Recipe | Scale | Notes |
| --- | --- | --- |
| [DeepSeek3-671B-MaxText](DeepSeek3-671B-MaxText/README.md) | v5p-1024 (512 chips) | Upstream recipe, submitted via XPK |
| [DeepSeek3-671B-MaxText-256chips](DeepSeek3-671B-MaxText-256chips/README.en.md) | 256 chips | Plain K8s manifest, with a full troubleshooting record |
| [Llama3.1-405B-MaxText](Llama3.1-405B-MaxText/README.md) | v5p-1024 | |
| [Llama4-Scout-17B-16E-Maxtext](Llama4-Scout-17B-16E-Maxtext/README.md) | v5p-256 / 512 / 1024 | |
| [Llama4-Maverick-17B-128E-Maxtext](Llama4-Maverick-17B-128E-Maxtext/README.md) | v5p-256 | |
| [Mixtral-8X7B-Maxtext](Mixtral-8X7B-Maxtext/README.md) | | |
| [GPT3-175B-MaxText](GPT3-175B-MaxText/README.md) | | |
| [Llama2-7B-Maxtext](Llama2-7B-Maxtext/README.md) | | |

### MaxDiffusion and others

| Recipe | Notes |
| --- | --- |
| [SDXL-MaxDiffusion](SDXL-MaxDiffusion/README.md) | Stable Diffusion XL |
| [Diffusion-2-MaxDiffusion](Diffusion-2-MaxDiffusion/README.md) | Stable Diffusion 2 |
| [DLRM-V2-Tensorflow](DLRM-V2-Tensorflow/README.md) | TensorFlow, not MaxText |

## Measured results

Numbers actually produced in this repository, not copied from upstream docs.

| Model | chips | Topology | Seq len | GBS | Precision | Step time | TFLOP/s/device | MFU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V3 671B | 256 | 4x8x8 | 8192 | 1024 | bf16 | 68.70 s | 134.1 | 29.2% |

Environment: `cloud-tpu-multipod-dev`, us-central1-a, spot, 2026-07-27. Full
method and troubleshooting in
[DeepSeek3-671B-MaxText-256chips](DeepSeek3-671B-MaxText-256chips/README.en.md).

On v5p, `TFLOP/s/device` is already a per-chip number (1 chip = 1 device), so it
compares directly against `TFLOPs/sec/chip` in
[ironwood/README.md](../ironwood/README.md) — **no × 2 needed**.

## General notes

These apply to every v5p recipe here and were hit repeatedly in practice:

- **`base_output_directory` must be genuinely writable.** It is used only on JAX
  process 0. When the write fails, the symptom is every other host hanging in a
  collective, with the log evidence sitting on the innocent machines. For a pure
  benchmark run, a local path is simplest.
- **An old pinned commit does not match today's pip.** Many entries in
  `requirements.txt` carry no version, so rebuilding a 2025 recipe in 2026 pulls
  incompatible releases. Pin the whole set back with
  `uv pip install --exclude-newer <date>`.
- **XPK's wrapper swallows the exit code.** Pods still show `Completed` on a
  failed run. Judge success by `completed step` in the logs, not by JobSet status.

Expanded in
[the 256-chip recipe's troubleshooting table](DeepSeek3-671B-MaxText-256chips/README.en.md#troubleshooting-quick-reference).
