# DeepSeek V3 32-layer proxy pretraining on Ironwood (64 chips / Kubernetes JobSet)

This recipe runs a **32-layer reduced version** of DeepSeek V3 (~221B parameters)
on 64 chips (4x4x4) of an [Ironwood GKE cluster](https://cloud.google.com/kubernetes-engine),
using [MaxText](https://github.com/AI-Hypercomputer/maxtext) main, deployed by
applying a Kubernetes manifest.

No XPK, no external storage. The XPK equivalent lives in [../xpk](../xpk/README.en.md).

> 中文版：[README.md](README.md)

## Why reduce layers

Full DeepSeek V3 is 61 layers, 671B parameters. It OOMs on 64 chips:

```
RESOURCE_EXHAUSTED: Ran out of memory on HBM
total memory required for HLO temporaries (105.73G) exceeds available HBM (94.74G)
```

This is not TPU-specific. The same wall shows up on A4X (GB200) with Megatron:
the worst-case buffers for 61 layers under CUDA graph exceed 184 GB HBM, and the
model has to be cut to 32 layers there too. Same root cause: **layers × 256
experts** in weights and buffers exceeds per-device capacity.

Reducing layers is the standard proxy-model approach. NVIDIA does the same in
Megatron-LM's GB200 regression tests (`deepseekv3_proxy_..._gb_200_release`,
reduced to 14 layers).

## Proxy model principle

**Reduce depth only, never width.**

| Parameter | Full model | This recipe | Changed |
| --- | --- | --- | --- |
| `base_num_decoder_layers` | 61 | **32** | yes |
| `num_experts` | 256 | 256 | **no** |
| `base_emb_dim` (H) | 7168 | 7168 | no |
| `base_mlp_dim` | 18432 | 18432 | no |
| `base_moe_mlp_dim` | 2048 | 2048 | no |
| `q_lora_rank` / `kv_lora_rank` | 1536 / 512 | 1536 / 512 | no |
| `first_num_dense_layers` | 3 | 3 | no |
| `mtp_num_layers` | 1 | 1 | no |

Per-layer compute shapes, MoE routing behaviour, MLA structure and communication
patterns stay identical to the real model, so per-layer kernel behaviour and
communication volume remain representative and can be extrapolated.

### Why `num_experts` must stay at 256

Two independent reasons:

1. **Comparability**: the expert count drives MoE routing distribution and
   all-to-all volume. Changing it makes per-layer communication unrepresentative.
2. **Hard constraint**: this recipe uses `shard_exp_on_fsdp=True`, and MaxText
   requires `num_experts % ici_fsdp_parallelism == 0`. At 64 chips
   `ici_fsdp_parallelism` resolves to 128, so `256 % 128 = 0` holds. With 64
   experts, `64 % 128 ≠ 0` and config validation fails outright.

Note this differs from NVIDIA's GB200 proxy, which cuts experts to 64 — that
config uses expert parallelism (EP16), so the constraints differ.

## Workload details

-   Sequence length: 4096
-   Precision: bf16
-   Chips: 64 (4x4x4 topology, 16 VMs × 4 chips)
-   Model: DeepSeek V3 32-layer proxy, ~221B parameters
-   MTP: 1 layer, loss scaling 0.1
-   `per_device_batch_size`: 2.0 (global batch = 2.0 × 128 devices = 256)
-   Dataset: synthetic (built into MaxText)
-   Checkpointing: disabled

### Why per_device_batch_size is 2.0

As the FSDP axis shrinks with chip count, each device carries a *larger* weight
shard:

| Chips | Devices | Weight shard/device | Headroom in 96 GB |
| --- | --- | --- | --- |
| 128 | 256 | 21 GB | 75 GB |
| **64** | **128** | **42 GB** | **54 GB** |

Per-device weight pressure at 64 chips is double that of 128 chips, so
activation memory has to shrink. Measured: 4.0 still falls 864 MB short; 2.0
passes (see [memory convergence](#memory-convergence)).

## Prerequisites

-   **GKE cluster** with [JobSet](https://jobset.sigs.k8s.io/docs/installation/).
-   **Container image**: a MaxText runner image — see
    [container image](../../4k-bf16-tpu7x-4x4x8-latest/k8s/README.en.md#container-image).
    It must be the **runner** image; the base image has no MaxText source.
-   **Tools**: `gcloud`, `kubectl`, `gke-gcloud-auth-plugin`, `envsubst`.
-   **Placement policy**: TPU v7 cannot auto-create one; create the 4x4x4 policy
    first.

## Create the 64-chip node pool

### Fixed vs autoscaling: prefer fixed when capacity is tight

Measured behaviour: in a saturated spot pool, a **fixed-size node pool obtains
capacity more reliably than an autoscaling one**.

It comes down to timing. A fixed pool issues the atomic allocation request at
creation time; an autoscaling pool only requests capacity after pods go Pending,
by which point the window is often gone. With the same placement policy and
topology, the autoscaling pool repeatedly hit `GCE out of resources` while the
fixed pool succeeded on the first attempt.

The tradeoff is that idle nodes still bill — delete the pool when done.

```bash
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE="us-central1-c"
export REGION="us-central1"

# 1. Create the 4x4x4 placement policy (v7 cannot auto-create it)
gcloud compute resource-policies create workload-policy tpu7x-64chip \
  --region=${REGION} --project=${PROJECT_ID} \
  --type=HIGH_THROUGHPUT --accelerator-topology=4x4x4

# 2. Create a fixed node pool: 16 VMs × 4 chips = 64 chips
gcloud container node-pools create np-tpu7x-64-fixed \
  --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} \
  --node-locations=${ZONE} \
  --machine-type=tpu7x-standard-4t \
  --tpu-topology=4x4x4 \
  --num-nodes=16 \
  --spot \
  --placement-policy=tpu7x-64chip \
  --disk-type=hyperdisk-balanced --disk-size=200

# 3. Confirm all 16 nodes are Ready
kubectl get nodes -l cloud.google.com/gke-nodepool=np-tpu7x-64-fixed \
  -o custom-columns='NODE:.metadata.name,TPU:.status.allocatable.google\.com/tpu'
```

Multi-host pools are all-or-nothing: all 16 VMs must land at once, since 4x4x4
requires a physically contiguous cube. Failures surface as
`Atomic resize failed with [GCE_STOCKOUT]`.

## Run the recipe

```bash
export PROJECT_ID=""
export CLUSTER_NAME=""
export REGION="us-central1"
export WORKLOAD_IMAGE=""     # runner image
export BASE_OUTPUT_DIR=""    # e.g. "gs://your-bucket/ds3-32l"
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-ds3-32l-64chip")-$(date +%Y%m%d-%H%M)"

gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

envsubst '${BASE_OUTPUT_DIR} ${WORKLOAD_NAME} ${WORKLOAD_IMAGE}' < k8s_manifest.yaml | kubectl apply -n default -f -
```

## Monitor

```bash
# Status across the 16 pods
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} --no-headers | awk '{print $3}' | sort | uniq -c

POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')

# Training steps only (XLA compilation logs are extremely verbose)
kubectl logs -n default ${POD_NAME} | grep 'completed step'

# Check for OOM
kubectl logs -n default ${POD_NAME} | grep RESOURCE_EXHAUSTED
```

## Reading the results

Step 0 is JIT compilation (can be hundreds of seconds); skip the early steps.

| Metric | Meaning |
| --- | --- |
| Step time | End-to-end time of one training step |
| TFLOP/s/device | As logged by MaxText, per TensorCore (half a chip) |
| TFLOP/s/chip | Per chip = logged value × 2 |
| MFU | Per-chip value ÷ 2,307 (v7 BF16 per-chip peak) |

**Note**: this is a 32-layer proxy, roughly half the full model. Its step time
and throughput are **not directly comparable to the 61-layer model**. Use them
for comparisons between 32-layer configurations, or to assess per-layer
efficiency.

## Metrics reference

MaxText prints one structured line per training step via `metric_logger.py`:

```
completed step: 0, seconds: 48.205, TFLOP/s/device: 23.456,
Tokens/s/device: 169.941, total_weights: 1048576, loss: 13.498,
lm_loss: 12.271, perplexity: 213435.656, moe_lb_loss: 0.000,
main_model_loss: 12.271, mtp_loss: 1.227
```

| Field | Meaning | Note |
| --- | --- | --- |
| `seconds` | End-to-end time of this step | step 0 includes JIT compilation |
| `TFLOP/s/device` | Per TensorCore | **per-chip = value × 2** |
| `Tokens/s/device` | Per TensorCore | **per-chip = value × 2** |
| `total_weights` | Effective tokens this step | = global_batch × seq_len |
| `loss` | Total loss | = `lm_loss` + `mtp_loss_scaling_factor` × `mtp_loss` |
| `lm_loss` | Language modelling loss | |
| `main_model_loss` | Main model loss (excludes MTP) | usually equals `lm_loss` |
| `mtp_loss` | MTP layer loss | **non-zero proves MTP is active** |
| `moe_lb_loss` | MoE load-balancing loss | 0 when `use_random_routing=True` |
| `perplexity` | Perplexity | = exp(lm_loss) |

Sanity-check the loss formula against the measured data:
`12.271 + 0.1 × 1.227 = 13.394`, close to the printed `13.498` (the remainder
comes from other terms), confirming `mtp_loss_scaling_factor=0.1` is in effect.

### MFU

```
MFU = (TFLOP/s/device × 2) / 2307
```

2307 is the TPU v7 per-chip BF16 peak in TFLOPS. The `× 2` is because a v7 chip
exposes 2 JAX devices.

**That `× 2` applies to v7 only.** v5p / v4 use MegaCore with 1 chip = 1 device
and convert differently. For cross-generation work, read
[TPU units](../../../../TPU-UNITS.en.md) first.

## Validation status

Run on a 64-chip fixed Spot node pool in `us-central1-c` on 2026-07-27.

### Confirmed working

| Item | Result |
| --- | --- |
| 4x4x4 fixed node pool | 16/16 Ready, allocated on first attempt |
| MaxText main `e50e39458` runner image | works |
| `override_model_config=True` + 32 layers | config validation passes |
| `shard_exp_on_fsdp=True` (256 experts) | passes |
| MTP, 1 layer | active, `mtp_loss: 1.227` |
| Memory | no OOM at `per_device_batch_size=2.0` |

### Memory convergence

This is how the parameters were arrived at:

| Layers | `per_device_batch_size` | Result |
| --- | --- | --- |
| 61 | 4.0 | OOM, 11 GB short (105.73G / 94.74G) |
| 32 | 4.0 | OOM, **864 MB** short (95.58G / 94.74G) |
| **32** | **2.0** | **passes** |

At 32 layers the gap is under 1 GB and halving the batch closes it. **Reducing
layers further yields little and degrades comparability with the real model**,
so 32 layers is the reasonable choice at this scale.

### Unresolved: TPU stall after step 1

Step 0 completes and emits full metrics, but the run hangs when advancing to
step 1:

```
Slow PjRt TPU operation detected: description=TpuLoadedExecutable::ReadyFuture
TpuDiagnosticCoordinator: Harvesting hardware telemetry for stalled chips: [6]
```

Reproduced on two runs, with a **different chip stalling each time** (chip 6,
then chip 9), after deleting the faulty node and having the MIG rebuild all 16
VMs. So this is not a single-machine hardware fault but a configuration or
software issue. Candidate directions:

-   Interaction of `use_random_routing=True` with MTP in MoE routing
-   Applicability of some XLA flag at 32 layers (the flag set comes from the
    61-layer recipe)
-   `remat_policy=custom` + `decoder_layer_input=offload` at this scale

**Steady-state step time / TFLOP has therefore not been captured.** The step 0
figure of 23.456 TFLOP/s/device includes JIT compilation and is not a
performance reference.

## Clean up

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default

# A fixed node pool never scales down on its own — delete it
gcloud container node-pools delete np-tpu7x-64-fixed \
  --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} --quiet
```

The placement policy is a reusable, free resource — keep it.

## Troubleshooting

**OOM (`RESOURCE_EXHAUSTED`)**

Check the gap first — the log reports `required (X) exceeds available (94.74G)`:

-   A few GB short: halve `per_device_batch_size`
-   More than 2× short: reduce layers further, or use more chips

Do not reduce `num_experts` to save memory — it breaks the `shard_exp_on_fsdp`
divisibility constraint.

**Node pool created but shows 0 nodes**

Check the placement policy exists and its topology matches. TPU v7 cannot
auto-create one.

**Other startup errors**

`pip install -e .` failures, config not found, assets path errors — see
[troubleshooting in the 4x4x8-latest recipe](../../4k-bf16-tpu7x-4x4x8-latest/k8s/README.en.md#troubleshooting).
