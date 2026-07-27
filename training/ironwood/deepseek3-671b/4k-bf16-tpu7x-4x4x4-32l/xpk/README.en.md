# DeepSeek V3 32-layer proxy pretraining on Ironwood (64 chips / XPK)

This recipe runs a **32-layer reduced version** of DeepSeek V3 (~221B parameters)
on 64 chips (4x4x4) of an Ironwood GKE cluster via
[XPK](https://github.com/AI-Hypercomputer/xpk), using MaxText main.

If you are handing this to a customer who needs to read exactly what gets
deployed, use the plain-manifest version in [../k8s](../k8s/README.en.md).

> 中文版：[README.md](README.md)

## Rationale for the model configuration

Why layers are reduced, the "reduce depth, never width" proxy principle, and why
`num_experts` must stay at 256 — see
[the k8s README](../k8s/README.en.md#why-reduce-layers). Only the resulting
parameters are listed here.

```
base_num_decoder_layers=32     # full model is 61
num_experts=256                # unchanged, see constraint
mtp_num_layers=1
mtp_loss_scaling_factor=0.1
per_device_batch_size=4.0      # per-device weight shard at 64 chips is 2× that of 128 chips
```

All other parameters (parallelism, XLA flags, attention config) are identical to
[4k-bf16-tpu7x-4x4x8-latest](../../4k-bf16-tpu7x-4x4x8-latest/xpk/README.en.md).

## Workload details

-   Sequence length: 4096
-   Precision: bf16
-   Chips: 64 (4x4x4 topology, 16 VMs × 4 chips)
-   Model: DeepSeek V3 32-layer proxy, ~221B parameters
-   Dataset: synthetic (no external dependency)
-   Checkpointing: disabled

## Prerequisites and environment setup

XPK installation, Docker configuration and image build steps are the same as in
[4k-bf16-tpu7x-4x4x8-latest](../../4k-bf16-tpu7x-4x4x8-latest/xpk/README.en.md).
The image must be the **runner** image; the base image has no MaxText source.

## Create the cluster

TPU v7 cannot auto-create placement policies, so create the 4x4x4 one first:

```bash
gcloud compute resource-policies create workload-policy tpu7x-64chip \
  --region=${REGION} --project=${PROJECT_ID} \
  --type=HIGH_THROUGHPUT --accelerator-topology=4x4x4
```

```bash
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-4x4x4 \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}
```

When capacity is tight, a fixed node count obtains capacity more reliably than
autoscaling — see
[the k8s README](../k8s/README.en.md#fixed-vs-autoscaling-prefer-fixed-when-capacity-is-tight).

## Run

```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x4-32l/xpk

# Edit the export variables at the top of run_recipe.sh
nano ./run_recipe.sh

chmod +x run_recipe.sh
./run_recipe.sh
```

## Monitor

```bash
xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}

kubectl get pods | grep ${WORKLOAD_NAME}
kubectl logs <POD_NAME> | grep 'completed step'
kubectl logs <POD_NAME> | grep RESOURCE_EXHAUSTED
```

## Reading the results

**This is a 32-layer proxy; its numbers are not directly comparable to the
61-layer model.**

Metric definitions and unit conversion (`TFLOP/s/device` is per TensorCore;
per-chip is 2×) are in
[the k8s README](../k8s/README.en.md#reading-the-results).

## Validation status

<!-- BENCHMARK-PLACEHOLDER -->
Pending. See [the k8s README](../k8s/README.en.md#validation-status) for what has
been verified.

## Clean up

```bash
xpk workload delete --workload ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```
