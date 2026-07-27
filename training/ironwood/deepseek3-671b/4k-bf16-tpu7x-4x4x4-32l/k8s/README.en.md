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
-   `per_device_batch_size`: 4.0 (global batch = 4.0 × 128 devices = 512)
-   Dataset: synthetic (built into MaxText)
-   Checkpointing: disabled

### Why per_device_batch_size is 4.0

As the FSDP axis shrinks with chip count, each device carries a *larger* weight
shard:

| Chips | Devices | Weight shard/device | Headroom in 96 GB |
| --- | --- | --- | --- |
| 128 | 256 | 21 GB | 75 GB |
| **64** | **128** | **42 GB** | **54 GB** |

Per-device weight pressure at 64 chips is double that of 128 chips, so
`per_device_batch_size` drops from 8.0 to 4.0 to shrink activation memory.

## Prerequisites

-   **GKE cluster** with [JobSet](https://jobset.sigs.k8s.io/docs/installation/).
-   **Container image**: a MaxText runner image — see
    [container image](../4k-bf16-tpu7x-4x4x8-latest/k8s/README.en.md#container-image).
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

## Validation status

<!-- BENCHMARK-PLACEHOLDER -->
Pending. Parts of the reasoning behind these parameters have been verified on
hardware:

-   A 64-chip 4x4x4 fixed node pool can be created (16/16 Ready)
-   The MaxText main (`e50e39458`, 2026-07-25) runner image works
-   `shard_exp_on_fsdp=True` passes config validation at 128 chips
-   61 layers with `per_device_batch_size=4.0` OOMs at 64 chips
    (105.73G > 94.74G) — the direct motivation for this reduced-layer recipe

Steady-state performance for the 32-layer configuration has not been captured
yet; it will be filled in once measured.

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
[troubleshooting in the 4x4x8-latest recipe](../4k-bf16-tpu7x-4x4x8-latest/k8s/README.en.md#troubleshooting).
