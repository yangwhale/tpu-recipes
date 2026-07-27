# DeepSeek V3 671B pretraining on 256 v5p chips (K8s manifest)

> 中文版：[README.md](README.md)

This recipe runs DeepSeek V3 671B pretraining on **256 v5p chips**
(topology `4x8x8`) with a synthetic dataset and checkpointing disabled, to
validate the workflow and collect a performance baseline.

The upstream recipe
[DeepSeek3-671B-MaxText](../DeepSeek3-671B-MaxText/README.md) targets
`v5p-1024` (512 chips) and submits through XPK. This one differs in two ways:

1. **Half the scale**: 256 chips, which requires adjusting `per_device_batch_size`
2. **Plain K8s manifest**: `kubectl apply` directly, so a customer can see
   exactly what gets deployed

For the chip / device / TensorCore unit relationships see
[TPU-UNITS.en.md](../../ironwood/TPU-UNITS.en.md). Short version: v5p uses
MegaCore, so **1 chip = 1 JAX device**, and the 1024 in `v5p-1024` counts
TensorCores, which is 512 chips.

## Prerequisites

- A GKE cluster with a 256-chip v5p node pool (64 VMs × 4 chips, topology `4x8x8`)
- JobSet CRD installed
- Docker configured to push to Artifact Registry

### Do not improvise the topology shape

256 chips admits several 3D shapes, and they are **not equivalent**:

```
4x8x8   ✓ close to a cube, good bisection bandwidth
4x4x16  ✗ elongated, traffic across the long axis travels further
```

Do not specify a placement policy manually when creating the node pool — GKE
generates a `COMPACT` one automatically. Specifying one manually fails with
`Required field 'resource.requestedRunDuration' not specified`.

## Building the image

The upstream recipe pins both the MaxText commit and the JAX version:

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout 3eb77db3c94580f56f1b738f8d254b03bd205e35
bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable JAX_VERSION=0.7.0
```

**Running that command today produces an image that cannot start.** See
[dependency drift](#dependency-drift-read-this) below. Two extra steps are needed.

### Dependency drift (read this)

`requirements.txt` leaves `flax`, `pathwaysutils` and others **unpinned**.
The recipe was written in 2025-10, when pip resolved versions compatible with
JAX 0.7.0. Today the same command pulls 2026 releases that do not work against
the 2025-10 source:

```
ImportError: cannot import name 'Effect' from 'jax.extend.core'        # flax 0.12.8
ImportError: cannot import name 'ifrt_proxy' from 'jax.extend.backend' # pathwaysutils
```

Downgrading one package at a time is a rabbit hole — fix flax and
pathwaysutils breaks next, and there are more behind it. Use `uv` with
`--exclude-newer` to pin the whole dependency set back to a snapshot from
around the commit date:

```dockerfile
FROM maxtext_stable__runner
RUN pip install --no-cache-dir uv && \
    uv pip install --system --exclude-newer 2025-10-15 \
      -r /deps/requirements.txt \
      "jax==0.7.0" "jaxlib==0.7.0" "libtpu==0.0.19.1"
```

Verify:

```bash
docker run --rm --entrypoint python3 <IMAGE> -c \
  "import sys; sys.path.insert(0,'/deps/src'); import MaxText.train; print('OK')"
```

### You need the runner image, not the base image

`docker_build_dependency_image.sh` produces the **base** image, which contains
**no MaxText source**. Using it fails with
`file:///deps does not appear to be a Python project`. Build the runner image
on top of it with `maxtext_runner.Dockerfile`.

That commit's `maxtext_runner.Dockerfile` has a bug — the `COPY` sources are
container absolute paths:

```dockerfile
COPY "${MAXTEXT_ASSETS_ROOT}" ...   # fails: "/deps/src/MaxText/test_assets": not found
```

Rewrite them relative to the build context:

```dockerfile
COPY src/MaxText/assets/       src/MaxText/assets/
COPY src/MaxText/test_assets/  src/MaxText/test_assets/
COPY . .
```

The full file is at [maxtext_runner_fixed.Dockerfile](maxtext_runner_fixed.Dockerfile).

## Running

```bash
export WORKLOAD_NAME=<your job name>
export WORKLOAD_IMAGE=<your runner image>
export BASE_OUTPUT_DIR=/tmp/mtout

envsubst < k8s/k8s_manifest.yaml | kubectl apply -f -
```

### `BASE_OUTPUT_DIR` must be genuinely writable

This was the most expensive trap in this exercise — see
[an unwritable output directory masquerades as a TPU hang](#an-unwritable-output-directory-masquerades-as-a-tpu-hang).

For a pure benchmark run, a **local path** such as `/tmp/mtout` is simplest:
with checkpointing disabled there is no need for GCS at all.

If you do want GCS, both of these must hold:

1. The bucket actually exists
2. Node OAuth scopes include `devstorage.read_write` or `cloud-platform`

Note that **IAM alone is not enough**: if the node scope is
`devstorage.read_only`, granting `roles/storage.objectAdmin` still leaves the
token unable to write. Scopes cannot be changed on an existing node pool — you
must recreate the pool or switch to Workload Identity.

## Monitoring

```bash
POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME \
      -o jsonpath='{.items[0].metadata.name}')

kubectl logs $POD | grep 'completed step'
kubectl logs $POD | grep 'Slow PjRt'      # any output means it is stuck
```

## Measured results

Environment: `cloud-tpu-multipod-dev`, us-central1-a, spot, 2026-07-27

| Item | Value |
| --- | --- |
| chips | 256 (64 VMs × 4) |
| JAX devices | 256 |
| Topology | `4x8x8` COMPACT |
| Sequence length | 8192 |
| `per_device_batch_size` | 3 |
| Global batch size | 768 |
| Precision | bf16 (fp32 weights) |
| step 0 (incl. compile) | 81.8 s |
| **Steady-state step time** | **60.18 s** |
| **TFLOP/s/device** | **114.82** |
| **MFU** | **25.0%** |
| Tokens/s/device | 408.5 |
| loss (step 0 → 18) | 12.27 → 9.79, monotonically decreasing |

Steady state begins at step 3, with ±0.03 s jitter.

The `step 1` line reports something like 21671 TFLOP/s/device. That is an
artifact of **JAX async dispatch** — the log prints before the computation
actually completes. It is not real throughput.

### Comparison against upstream at 512 chips

The upstream recipe reports 152.4 TFLOP/s/device on 512 chips. This recipe
gets 114.8 on 256 chips, about 25% lower.

**This is not purely a scaling effect.** This run dropped several settings
relative to the upstream config, each of which can depress MFU:

| Parameter | Upstream | This run | Effect |
| --- | --- | --- | --- |
| `per_device_batch_size` | 6 | 3 | Half the work per step, worse overlap |
| `tile_batch_seq` | 512 | unset | megablox MoE tiling |
| `tile_embed_dim` | 1024 | unset | same |
| `tile_mlp_dim` | 1024 | unset | same |
| `--2a886c8_chip_config_name` | `megachip_tccontrol` | unset | SparseCore operating mode |
| `--xla_tpu_use_tc_device_shape_on_sc` | true | unset | same |
| `--xla_sc_enable_instruction_fusion` | false | unset | same |
| `--xla_sc_disjoint_spmem` | false | unset | same |
| `--xla_sc_disable_megacore_partitioning` | true | unset | same |

In other words: the three SparseCore *offload* switches were on, but its
operating mode was never configured — a half-applied configuration. The three
`tile_*` parameters were renamed to `wi_tile_*` / `wo_tile_*` in newer MaxText,
but on this commit (`3eb77db3`) they are valid and should not have been removed.

**Treat 114.8 as a floor, not as the ceiling for this scale.**

## Troubleshooting quick reference

| Symptom | Root cause | Fix |
| --- | --- | --- |
| `cannot import name 'Effect' from 'jax.extend.core'` | pip pulled flax 2026 release | `uv pip install --exclude-newer 2025-10-15` |
| `cannot import name 'ifrt_proxy'` | same, pathwaysutils | same |
| `file:///deps does not appear to be a Python project` | Used base image instead of runner | Build the runner image |
| `"/deps/src/MaxText/test_assets": not found` | runner Dockerfile uses container absolute paths | Make COPY sources relative to build context |
| `CompileTimeScopedVmemOom`, splash attention needs 18.12 MB | VMEM defaults to 16 MB | Add `--xla_tpu_scoped_vmem_limit_kib=81920` |
| `Keys ['base_num_decoder_layers'] are overridden by both model config and CLI` | CLI conflicts with model config | Add `override_model_config=True` |
| `Cannot remeterialize this tensor with scan_layers=True` | `decoder_layer_input=remat` conflicts with scan | Use `offload` or `device` |
| `Required field 'resource.requestedRunDuration' not specified` | Placement policy specified manually | Omit it, let GKE generate one |
| Stops after step 1, floods `Slow PjRt ... CopyToMemorySpace CrossDeviceSrc` | see below | see below |

### An unwritable output directory masquerades as a TPU hang

**This is the most valuable finding here.**

Symptom: training completes step 0 and step 1, then stops, flooding the log with

```
Slow PjRt TPU operation detected: ... description=CopyToMemorySpace CrossDeviceSrc
Slow PjRt TPU operation detected: ... description=TfrtTpuLoadedExecutable::ReadyFuture
```

This looks like a TPU hardware, ICI interconnect, or XLA compilation problem.
The actual cause was **`BASE_OUTPUT_DIR` pointing at a GCS bucket that did not
exist**:

```
google.api_core.exceptions.Forbidden: 403 POST .../b/<bucket>/o
"Provided scope(s) are not authorized"
```

MaxText writes TensorBoard output only from **JAX process 0**. Once process 0
raises here, its main loop stops issuing programs, and the other 63 hosts wait
forever in the collective.

Chasing the surface symptom leads nowhere. We eliminated all of the following,
**none of which was the cause**: TPU generation (v7x / v5p), layer count, MTP,
scale (64 / 256 chips), topology shape, MaxText and JAX versions, SparseCore
offload switches, host offload, leftover pods holding HBM, and spot preemption.

**How to locate it**: pull logs from all 64 pods and count `Slow PjRt` lines in each.

```bash
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME \
  --no-headers | awk '{print $1}' > pods.txt
cat pods.txt | xargs -P 20 -I{} sh -c 'kubectl logs {} > logs/{}.log 2>/dev/null'

for f in logs/*.log; do echo "$(grep -c 'Slow PjRt' $f) $f"; done | sort -n | head
```

63 pods report the same number of stall warnings. **The one pod that reports
none is the broken one** — it is not stuck in the collective, it is what
everyone else is stuck waiting for.

Generalizing: in multi-host TPU training, any failure in a **process-0-only
side effect** (TensorBoard, checkpoint upload, metrics writes, Vertex
integration) presents as a collective hang, and **the exception is not on the
machines that are reporting warnings**.

### Spot node pool preemption

When v5p spot capacity in us-central1-a is tight, preemption is frequent.
Observed:

```
10:52 UTC   13 VMs
10:53 UTC   51 VMs   ← the entire 64-VM pool gone
11:00 UTC   31 VMs
11:01 UTC   14 VMs
11:02 UTC   20 VMs   ← the replacements were preempted too
```

Preemption also presents as a collective hang (surviving hosts waiting on one
that no longer exists). Distinguish it by checking the operation log:

```bash
gcloud compute operations list --project=<PROJECT> --zones=<ZONE> \
  --sort-by=~startTime --limit=400 \
  --format='value(startTime,operationType)' | grep preempted
```

Also note that XPK's wrapper script **swallows the exit code**: on a failed
training run the pods still show `Completed` and the JobSet still reports
`completed successfully`. Do not use JobSet status to judge success — look for
`completed step` in the logs.

## Clean up

```bash
kubectl delete jobset $WORKLOAD_NAME
```

Deletion is asynchronous. Before submitting the next job you **must wait for
all pods to disappear**, otherwise the new pods contend for nodes whose TPUs
are still held by the old ones:

```bash
while [ "$(kubectl get pods --no-headers | wc -l)" -ne 0 ]; do sleep 10; done
```

The symptom of not waiting is mysteriously reduced HBM — for example only
66.40 GB available on v5p, where a single chip should have 95.74 GB.
