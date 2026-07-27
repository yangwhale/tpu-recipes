# TPU units: chip, device, TensorCore

> 中文版：[TPU-UNITS.md](TPU-UNITS.md)

Three different units are in use across the TPU stack, and **the same word means
different things across generations**. Getting it wrong gives you a 2× error in
MFU, a 2× error in memory estimates, or a broken parallelism config. This page
pins the relationships down.

## Where each unit is used

| Context | Unit | Example |
| --- | --- | --- |
| Machine name `v5p-N` / `v6e-N` | **TensorCores** | `v5p-1024` = 1024 TC |
| GKE `--tpu-topology` | **chips** | `4x8x8` = 256 chips |
| GKE `google.com/tpu` resource | **chips** | 4 chips per VM |
| MaxText log `TFLOP/s/device` | **JAX devices** | see conversion below |
| `per_device_batch_size` | **JAX devices** | same |
| `ici_fsdp_parallelism` | **JAX devices** | same |

## chips-to-devices varies by generation

This is where mistakes happen.

| Generation | Architecture | HBM/chip | **JAX devices / chip** |
| --- | --- | --- | --- |
| v4 | MegaCore | 32 GB | **1** |
| v5p | MegaCore | 95 GB | **1** |
| v6e | single TensorCore | 32 GB | **1** |
| **v7 (Ironwood)** | dual-chiplet | 192 GB | **2** |

**v4 / v5p use MegaCore**: the two chiplets in a chip share a unified memory
space and are programmed as one logical core, so JAX sees a single device.

**v7 (Ironwood) has independent per-chiplet memory**: each chiplet has its own
96 GB HBM and TensorCore, and JAX exposes them as two separate devices (topology
gains a 4th dimension to distinguish chiplets).

Bottom line: **`TFLOP/s/device` means a whole chip on v5p, but half a chip on
v7.**

## The number in the machine name is TensorCores

```
v5p-1024  = 1024 TensorCores = 512 chips = 512 JAX devices
v5p-512   =  512 TensorCores = 256 chips = 256 JAX devices
```

A v5p chip physically has 2 TensorCores, but MegaCore merges them into one JAX
device. So **the number in the machine name is twice the device count you
actually program against**.

v7 does not use this naming in GKE; it uses machine types like
`tpu7x-standard-4t` (4 chips per VM).

## Common conversions

### MFU

```
v5p:  MFU = TFLOP/s/device        / 459     # 1 device = 1 chip
v7x:  MFU = TFLOP/s/device × 2    / 2306    # 1 chip = 2 devices
```

459 and 2306 are the **per-chip** BF16 peaks in TFLOPS for v5p and v7.

**The `× 2` that applies on v7 must not be carried over to v5p** — this is the
easiest step to copy incorrectly.

### Global batch size

```
GBS = per_device_batch_size × number of JAX devices
```

| Config | chips | devices | pdb | GBS |
| --- | --- | --- | --- | --- |
| v7x 4x4x8 | 128 | 256 | 8.0 | 2048 |
| v7x 4x4x4 | 64 | 128 | 8.0 | 1024 |
| v5p 8x8x8 | 512 | 512 | 6 | 3072 |
| v5p 4x8x8 | 256 | 256 | 4 | 1024 |

The pdb of 4 on the `v5p 4x8x8` row is not a typo. It has the same 256 devices
as `v7x 4x4x8`, but a v5p device has 95 GB while a v7x *chip* has 192 GB — the
same device count does not hold the same batch. On DeepSeek V3 671B, pdb=5
already fails.

Contrast `v5p 8x8x8`: doubling the devices halves the weight shard per device,
which is what allows pdb to return to 6. **The pdb ceiling is set by shard
pressure per device, not by chip count.** See
[the v5p notes](v5p/README.en.md#3-half-the-hbm-so-batch-size-must-be-recomputed).

### FSDP shard pressure

`ici_fsdp_parallelism` counts **devices**. For a model whose static state
(weights + optimizer) totals S:

```
shard per device   = S / ici_fsdp_parallelism
HBM per device     = HBM_per_chip / (devices per chip)
```

For DeepSeek V3 671B (S ≈ 5.4 TB):

| Config | devices | shard/device | HBM/device | Headroom |
| --- | --- | --- | --- | --- |
| v7x 128 chips | 256 | 21 GB | 96 GB | 75 GB |
| v7x 64 chips | 128 | 42 GB | 96 GB | 54 GB |
| v5p 256 chips | 256 | 21 GB | 95 GB | 74 GB |

**256 v5p chips are equivalent to 128 v7x chips**: same device count (256), same
shard per device (21 GB), near-identical HBM (95 vs 96 GB). This is the correct
ratio for cross-generation comparisons.

Note the v5p "HBM/device" here is the full 95 GB of a chip, since 1 chip = 1
device, whereas the v7x 96 GB is only half a chip.

## Choosing the topology shape

The same chip count admits several 3D shapes, and they are **not equivalent**:

```
256 chips = 4x8x8  ✓ close to a cube, good bisection bandwidth
          = 4x4x16 ✗ elongated, traffic across the long axis travels further
```

On a 3D torus, closer to a cube means higher bisection bandwidth. Upstream
recipes always use `4x8x8` for 256 chips — don't improvise a factorization.

## Checklist

When configuring a new scale, verify in this order:

1. Target chip count → look up devices-per-chip for that generation → device count
2. Pick the topology shape closest to a cube
3. Compute GBS from `per_device_batch_size` × device count
4. For constraints like `num_experts % ici_fsdp_parallelism`, substitute the
   **device** count
5. When reading logs, confirm whether `TFLOP/s/device` needs × 2 to become
   per-chip
