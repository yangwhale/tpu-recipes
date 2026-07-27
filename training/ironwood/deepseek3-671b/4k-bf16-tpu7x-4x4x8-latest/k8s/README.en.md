# Pretrain deepseek3-671b on Ironwood GKE clusters with Kubernetes JobSet (128 chips / latest MaxText)

This recipe outlines the steps for running a deepseek3-671b
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by applying
a Kubernetes manifest to deploy a JobSet resource.

Everything is expressed as plain Kubernetes objects — no XPK required. The
equivalent XPK-based recipe lives in [../xpk](../xpk/README.en.md).

> 中文版：[README.md](README.md)

## What this recipe is for

The upstream `4k-bf16-tpu7x-4x4x8` recipe is pinned to the pre-refactor MaxText
API (`MaxText.train` entrypoint, `fsdp_shard_on_exp` flag), so it cannot pick up
the DeepSeek V3 MoE optimizations merged after 2026-03. This recipe keeps the
exact same 128-chip parallelism configuration but runs on the current MaxText
main branch.

It uses MaxText's built-in synthetic dataset and therefore needs **no external
storage**. The goal is to validate the flow and measure performance, not to
produce a usable model. For real data, see
[Appendix: switching to Lustre](#appendix-switching-to-lustre).

## Workload details

-   Sequence length: 4096
-   Precision: bf16
-   Chips: 128 (4x4x8 topology, 32 VMs × 4 chips)
-   Dataset: synthetic (built into MaxText, no external dependency)
-   Checkpointing: disabled

## Relationship to the other recipes

| | `4k-bf16-tpu7x-4x4x8/k8s` | This recipe |
| --- | --- | --- |
| MaxText entrypoint | `MaxText.train` | `src.maxtext.trainers.pre_train.train` |
| Config path | `MaxText/configs/base.yml` | `src/maxtext/configs/base.yml` |
| MoE sharding flag | `fsdp_shard_on_exp=True` | `shard_exp_on_fsdp=True` (renamed upstream) |
| XLA | — | adds `--xla_tpu_dvfs_p_state=3` |
| Parallelism | same | same |

### Why MoE sharding uses `shard_exp_on_fsdp` rather than 2D sharding

The 256-chip `4k-bf16-tpu7x-4x8x8-lustre` recipe uses
`use_2d_fsdp_sharding=True` with `ici_fsdp_transpose_parallelism=2`. That path
shards MoE weights across both the `fsdp` and `fsdp_transpose` axes and is only
meaningful when `ici_fsdp_transpose_parallelism > 1`.

At 128 chips there is a single FSDP axis of 256 devices, so this recipe keeps
`ici_fsdp_transpose_parallelism=1` and uses `shard_exp_on_fsdp=True`, sharding
the expert dimension of the MLP weights along that single axis.

MaxText enforces hard constraints on this path (see
[`pyconfig_deprecated.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/pyconfig_deprecated.py)):
`num_experts` must be divisible by `ici_fsdp_parallelism`, and both
`ici_expert_parallelism` and `ici_tensor_parallelism` must be 1. DeepSeek V3 has
256 experts and `ici_fsdp_parallelism` resolves to 256, so all three hold.
**This has been verified on real 128-chip hardware** (see
[Validation status](#validation-status)).

## Validation status

Run on a 128-chip Spot node pool in `us-central1-c` on 2026-07-26.

Verified:

-   4x4x8 node pool creation (32/32 nodes Ready, `google.com/tpu: 4` per node)
-   Image build from MaxText main (`e50e39458`, 2026-07-25)
-   All 32 pods scheduled and started
-   `shard_exp_on_fsdp=True` passes MaxText config validation — no expert
    divisibility error
-   XLA HLO compilation and the first training steps

Not verified:

-   **Steady-state performance.** The Spot instances were preempted
    (`compute.instances.preempted`) at step 2, and steps 0–1 are JIT
    compilation, so no steady-state step time / TFLOP was captured. The numbers
    under [Performance reference](#performance-reference) come from historical
    runs on the same topology, not from this run.
-   The Lustre storage path (see appendix).

## Performance reference

Historical data on the **same topology, same parallelism, synthetic dataset**,
using an older MaxText (`maxtext-tutorial-v1.5.0`, JAX `0.8.2.dev20251215`).
Listed for comparison; measurements for this recipe on the latest MaxText are
still pending.

| Config | Precision | Step time | TFLOP/s/chip | Tokens/s/chip |
| --- | --- | --- | --- | --- |
| 4x4x8 (128 chips) | bf16 | 27.00 s | 608.0 | 2,427.7 |
| 4x4x8 (128 chips) | fp8 | 22.39 s | 733.1 | 2,926.5 |

Mind the units: MaxText logs `TFLOP/s/device` **per TensorCore**, and a TPU v7
chip has 2 TensorCores, so per-chip values are 2× the logged value. The table
above is already per-chip. Against a v7 per-chip BF16 peak of 2,307 TFLOPS,
608 TFLOP/s/chip is roughly 26.4% MFU.

## Prerequisites

-   **GKE cluster** with [JobSet](https://jobset.sigs.k8s.io/docs/installation/)
    installed and running.
-   **Container image**: a MaxText runner image the cluster can pull. See
    [Container image](#container-image).
-   **Tools**: `gcloud`, `kubectl`, `gke-gcloud-auth-plugin` and `envsubst` on
    your workstation. If `envsubst` is missing, install it with
    `sudo apt-get update && sudo apt-get install -y gettext-base`.
-   **Permissions**: you can run `kubectl apply` against the cluster, and the
    cluster can pull the image.
-   **Placement policy**: TPU v7 **does not support auto-creating** placement
    policies. You must create one for the target topology first, otherwise the
    multi-host node pool cannot be created.

## Create the 128-chip node pool

Multi-host TPU v7 node pools are allocated all-or-nothing: all 32 VMs must be
obtained at once, since a 4x4x8 topology requires a physically contiguous cube.

```bash
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE="us-central1-c"
export REGION="us-central1"

# 1. Create the 4x4x8 placement policy (v7 cannot auto-create it)
gcloud compute resource-policies create workload-policy tpu7x-128chip \
  --region=${REGION} --project=${PROJECT_ID} \
  --type=HIGH_THROUGHPUT --accelerator-topology=4x4x8

# 2. Create the node pool: 32 VMs × 4 chips = 128 chips
gcloud container node-pools create np-tpu7x-128chip \
  --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} \
  --node-locations=${ZONE} \
  --machine-type=tpu7x-standard-4t \
  --tpu-topology=4x4x8 \
  --num-nodes=32 \
  --spot \
  --placement-policy=tpu7x-128chip \
  --disk-type=hyperdisk-balanced --disk-size=200

# 3. Confirm all 32 nodes are Ready with TPU resources registered
kubectl get nodes -l cloud.google.com/gke-nodepool=np-tpu7x-128chip \
  -o custom-columns='NODE:.metadata.name,TPU:.status.allocatable.google\.com/tpu'
```

Expect preemption when using Spot. During validation the node pool was reclaimed
after roughly 10 minutes.

## Container image

This recipe targets the **latest MaxText main branch**.

```bash
export PROJECT_ID=""
export CONTAINER_REGISTRY=""   # e.g. us-docker.pkg.dev/${PROJECT_ID}/gcr.io
export CLOUD_IMAGE_NAME="${USER}-maxtext-latest"

# Python 3.11 virtual environment
uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.11 --clear
source ${HOME}/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# Record the exact commit you build from, so results stay reproducible
git rev-parse HEAD

# 1. Build the dependency image (MODE=nightly resolves the latest jax/libtpu)
bash src/dependencies/scripts/docker_build_dependency_image.sh MODE=nightly

# 2. Build the runner image — this is the step that puts MaxText into /deps
docker build --network host \
  -f src/dependencies/dockerfiles/maxtext_runner.Dockerfile \
  --build-arg BASEIMAGE=maxtext_base_image \
  --build-arg PACKAGE_DIR=src \
  -t maxtext_base_image__runner .

# 3. Push
docker tag maxtext_base_image__runner:latest ${CONTAINER_REGISTRY}/${CLOUD_IMAGE_NAME}:runner
docker push ${CONTAINER_REGISTRY}/${CLOUD_IMAGE_NAME}:runner

deactivate
```

**Push the runner image, not the base image.** `maxtext_base_image` contains
only dependencies, not the MaxText source, and will fail with missing modules.

Verified with MaxText `e50e39458` (2026-07-25), nightly jax/libtpu, Python 3.11.

## Run the recipe

### 1. Configure environment variables

```bash
export PROJECT_ID=""
export CLUSTER_NAME=""
export REGION="us-central1"
export WORKLOAD_IMAGE=""     # the runner image pushed above
export BASE_OUTPUT_DIR=""    # e.g. "gs://your-bucket/ds3-run"
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-deepseekv3-671b-128")-$(date +%Y%m%d-%H%M)"
```

### 2. Submit the workload

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

envsubst '${BASE_OUTPUT_DIR} ${WORKLOAD_NAME} ${WORKLOAD_IMAGE}' < k8s_manifest.yaml | kubectl apply -n default -f -
```

## Monitor the job

```bash
# JobSet status
kubectl get jobset -n default ${WORKLOAD_NAME}

# Status distribution across the 32 pods
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} --no-headers | awk '{print $3}' | sort | uniq -c

# Follow the first pod
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f -n default ${POD_NAME}

# Training steps only
kubectl logs -n default ${POD_NAME} | grep 'completed step'
```

To confirm a Spot preemption:

```bash
gcloud logging read 'protoPayload.methodName="compute.instances.preempted"' \
  --project ${PROJECT_ID} --limit 5 --format='value(timestamp,protoPayload.resourceName)'
```

## Reading the results

Skip the early steps before reading steady-state values: step 0 is JIT
compilation (can be hundreds of seconds), and if the profiler is enabled, the
profiled steps and the one after them are also slower.

| Metric | Meaning |
| --- | --- |
| Step time | End-to-end time of one training step |
| TFLOP/s/device | As logged by MaxText, per TensorCore (half a chip) |
| TFLOP/s/chip | Per chip, i.e. the logged value × 2 |
| MFU | Per-chip value ÷ 2,307 (v7 BF16 per-chip peak) |

The chip / device / TensorCore relationship differs across generations — see
[TPU units](../../../TPU-UNITS.en.md).

## Clean up

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default

# Scale the node pool to 0 (keeps the config for later)
gcloud container clusters resize ${CLUSTER_NAME} \
  --node-pool=np-tpu7x-128chip --num-nodes=0 \
  --region=${REGION} --project=${PROJECT_ID} --quiet

# Or delete the node pool entirely
gcloud container node-pools delete np-tpu7x-128chip \
  --cluster=${CLUSTER_NAME} --region=${REGION} --project=${PROJECT_ID} --quiet
```

The placement policy is a reusable, free resource — keep it.

## Troubleshooting

**`file:///deps does not appear to be a Python project`**

The runner image's `/deps` contains only `benchmarks/`, `src/`, `tests/` and
`pytest.ini` — there is no `pyproject.toml` or `setup.py`. Do not run
`pip install -e .`; just `cd /deps` and invoke the entrypoint module.

**`No module named ...` or config not found**

The correct config path is `src/maxtext/configs/base.yml`. There is no
`/deps/maxtext` directory.

**Errors about assets paths**

The runner image already sets `MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets`
(**lowercase** `maxtext`) and `MAXTEXT_PKG_DIR`. Do not override them in the
launch command. The `/deps/src/MaxText/assets` path used by older recipes does
not exist in the current image.

**Node pool created but shows 0 nodes**

Check that the placement policy exists and its topology matches. TPU v7 cannot
auto-create one.

## Appendix: switching to Lustre

To train on a real dataset with checkpointing, mount Google Cloud Managed
Lustre. The Lustre instance must be on the **same VPC network** as the GKE
cluster.

1. Enable the CSI driver (GKE `1.34.0-gke.2201000` or later):

```bash
gcloud container clusters update ${CLUSTER_NAME} \
  --location ${ZONE} --project ${PROJECT_ID} \
  --update-addons=LustreCsiDriver=ENABLED
```

2. Create the Lustre instance on the cluster's network and transfer the AllenAI
   C4 dataset into it in ArrayRecord format. At least 36 TB is recommended when
   the instance holds both the dataset and checkpoints.

3. Fill in `lustre_pvc.yaml` and apply it:

```bash
kubectl apply -f lustre_pvc.yaml
kubectl get pvc lustre-volume -n default   # must be Bound
```

4. Edit `k8s_manifest.yaml`.

Add to `volumeMounts`:

```yaml
- mountPath: /mnt/lustre
  name: lustre-volume
```

Add to `volumes`:

```yaml
- name: lustre-volume
  persistentVolumeClaim:
    claimName: lustre-volume
```

Replace these MaxText arguments

```
tokenizer_path=src/maxtext/assets/tokenizer.mistral-v3 dataset_type=synthetic enable_checkpointing=False
```

with

```
tokenizer_path='deepseek-ai/DeepSeek-V3-Base' tokenizer_type=huggingface
enable_checkpointing=True checkpoint_storage_concurrent_gb=400 async_checkpointing=true
enable_single_replica_ckpt_restoring=true checkpoint_storage_target_data_file_size_bytes=209715200
dataset_type='grain' grain_file_type=arrayrecord
grain_train_files=${DATASET_BUCKET_MOUNTED_PATH}/multilingual-c4/array-record/c4/multilingual/3.0.1/*.arrayrecord
grain_worker_count=2 checkpoint_period=25
```

Point `BASE_OUTPUT_DIR` at `/mnt/lustre/checkpoints`, set
`DATASET_BUCKET_MOUNTED_PATH` (e.g. `/mnt/lustre/datasets`), and pass both
through `envsubst`.

**This path has not been validated on hardware**; the parameters are taken from
the upstream 256-chip `4k-bf16-tpu7x-4x8x8-lustre` recipe.

## Next steps

This recipe aims to give a reproducible "0-to-1" experience. For deeper
customization — tuning XLA flags, running custom experiments — use
`benchmark_runner.py` from the MaxText repository directly. See the
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md).
