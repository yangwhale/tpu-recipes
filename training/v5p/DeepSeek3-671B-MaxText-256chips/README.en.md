# DeepSeek V3 671B pretraining on 256 v5p chips (K8s manifest)

> 中文版：[README.md](README.md)
>
> Back to the [v5p recipe index](../README.en.md)

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
[TPU-UNITS.en.md](../../TPU-UNITS.en.md). Short version: v5p uses
MegaCore, so **1 chip = 1 JAX device**, and the 1024 in `v5p-1024` counts
TensorCores, which is 512 chips.

## Prerequisites

- A GKE cluster with a 256-chip v5p node pool (64 VMs × 4 chips, topology `4x8x8`)
- JobSet CRD installed
- `envsubst` available locally (Debian/Ubuntu: the `gettext-base` package) —
  the run step needs it
- Docker configured to push to Artifact Registry (**only if you build the image
  yourself**; skip it when using a prebuilt one)

Point kubectl at the cluster first:

```bash
gcloud container clusters get-credentials <CLUSTER> \
  --region <REGION> --project <PROJECT>
```

Self-check — all three must pass before continuing:

```bash
kubectl get nodes -l cloud.google.com/gke-nodepool=<NODEPOOL> --no-headers | grep -c ' Ready '   # expect 64
kubectl get crd jobsets.jobset.x-k8s.io                                                          # must exist
command -v envsubst                                                                              # must print a path
```

### Check two node pool properties up front

Both drive later decisions and **cannot be changed after the pool is created**:

```bash
gcloud container node-pools describe <NODEPOOL> --cluster <CLUSTER> \
  --region <REGION> --format='value(config.spot, config.oauthScopes)'
```

- **Spot or not**: a spot pool gets preempted, and preemption presents as a
  collective hang — see [Spot node pool preemption](#spot-node-pool-preemption)
- **OAuth scopes**: with only `devstorage.read_only`, `BASE_OUTPUT_DIR` cannot
  be a GCS path — see
  [`BASE_OUTPUT_DIR` must be genuinely writable](#base_output_dir-must-be-genuinely-writable)

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

There are three steps in total, and **the two sections below are written in the
reverse of the execution order**. Follow this list instead:

| Step | Image produced | What it does | Section |
| --- | --- | --- | --- |
| 1 | `maxtext_base_image` | the `docker_build_dependency_image.sh` command above | this section |
| 2 | `maxtext_stable__runner` | bake the MaxText source in using the fixed runner Dockerfile | [You need the runner image](#you-need-the-runner-image-not-the-base-image) |
| 3 | final image | `FROM maxtext_stable__runner`, pin dependencies back with uv | [Dependency drift](#dependency-drift-read-this) |

Step 3's `FROM maxtext_stable__runner` consumes the output of step 2, so
**following the next section first fails: that image does not exist yet**.

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

`items[0]` picks an arbitrary pod, which is fine here: MaxText prints
`completed step` on **every** host, not just process 0. (What only happens on
process 0 is the TensorBoard write — a separate matter, see
[an unwritable output directory masquerades as a TPU hang](#an-unwritable-output-directory-masquerades-as-a-tpu-hang).)

### How long until the first step

**Expect roughly 6–7 minutes between `kubectl apply` and the first
`completed step: 0`.** The log floods with HLO dumps during that window — that
is normal XLA compilation, not a hang.

Do not read the step 0 time in the results table (~100 s) as the wait time: it
is MaxText's own timer and excludes image pull, JAX distributed init, and
compilation. Judge a real hang by `Slow PjRt`, not by elapsed time.

## Measured results

Environment: `cloud-tpu-multipod-dev`, us-central1-a, spot, 2026-07-27

This recipe keeps **every tuning param and all 33 XLA flags from the upstream
`deepseek3_671b_v5p_1024` config**, changing only `per_device_batch_size` from
6 to 4 — the one parameter that 256 chips forces you to change.

| Item | Value |
| --- | --- |
| chips | 256 (64 VMs × 4) |
| JAX devices | 256 |
| Topology | `4x8x8` COMPACT |
| Sequence length | 8192 |
| `per_device_batch_size` | 4 |
| Global batch size | 1024 |
| Precision | bf16 (fp32 weights) |
| step 0 (incl. compile) | 93.0 s |
| **Steady-state step time** | **68.70 s** |
| **TFLOP/s/device** | **134.1** |
| **MFU** | **29.2%** |
| Tokens/s/device | 477.1 |
| loss (step 0 → 15) | 12.27 → 9.9, monotonically decreasing |

Steady state begins at step 3, with ±0.06 s jitter.

**Only steps 5, 10 and 12 carry profiler overhead — not the whole 5–12 range.**
In the same run, steps 7/8/9 measured 68.82 / 68.71 / 68.71 s, indistinguishable
from steps 3/4/13 (68.71 / 68.73 / 68.73 s). They are clean steady-state points.

Use this table when reading the log:

| step | State | Use it? |
| --- | --- | --- |
| 0 | includes the first compile | ✗ |
| 1, 6, 11 | JAX async-dispatch artifact (0.005–0.3 s, inflated TFLOP) | ✗ |
| 2 | not yet converged (91.8 s) | ✗ |
| **3, 4, 7, 8, 9, 13+** | **steady state** | **✓** |
| 5, 10, 12 | xplane profiler write overhead (104–137 s) | ✗ |

The upstream config also enables the profiler, so the comparison stays fair.

The `step 1` line reports something like 29492 TFLOP/s/device. That is an
artifact of **JAX async dispatch** — the log prints before the computation
actually completes. The 0.006 s readings at steps 6 and 11 are the same
artifact; each pairs with the step before it.

### Independent reproduction (2026-07-27, different operator, same node pool)

A second person re-ran this document end to end with the same image under a
different workload name:

| Metric | Original record | Reproduction | Delta |
| --- | --- | --- | --- |
| Steady-state step time | 68.70 s | 68.71 s (step 9) / 68.82 s (step 7) | +0.02% |
| TFLOP/s/device | 134.1 | 134.08 | −0.01% |
| Tokens/s/device | 477.1 | 476.9 | −0.04% |
| loss at step 0 | 12.27 | 12.270 | identical |
| final loss | 9.9 (step 15) | 9.890 (step 12) | identical |
| step 0 | 93.0 s | 104.1 s | **+11.9%** |

**The steady-state numbers reproduce to one decimal place**, so the recipe
itself is deterministic.

The only divergence is **step 0 (93.0 → 104.1 s)**. That step includes the
first XLA compile and varies with compile cache and node state — **do not read
it as a performance metric**.

The reproduction also confirmed each of the "looks wrong but is normal"
behaviours documented here:

- step 1 reported **30767 TFLOP/s/device** (original: 29492) — async-dispatch artifact
- steps 6 and 11 at **0.005 s** — same artifact, paired with the preceding step
- steps 10 and 12 at **137.5 s / 104.3 s** — xplane profiler overhead,
  consistent with "exclude 5–12"

### `per_device_batch_size` tops out at 4

Upstream uses 6 on 512 chips. At 256 chips the weight shard doubles
(10.5 GB → 21 GB per device), leaving less room for activations. Measured ladder:

| pdb | Result |
| --- | --- |
| 6 | Compile-time OOM: `Used 112.37G of 95.74G hbm`, over by 16.62 GB |
| 5 | Runtime failure: `Attempting to reserve 68.03G at the bottom of memory... 66.40G reservable` |
| **4** | **Works** |

Note that pdb=5 fails differently from pdb=6: it passes the compile-time
95.74 GB check but hits a tighter runtime constraint — only **66.40 GB is
reservable at the bottom of memory**. These are two distinct limits; watch both
when tuning batch size.

### Comparison against upstream at 512 chips

| | Upstream, 512 chips | This recipe, 256 chips |
| --- | --- | --- |
| `per_device_batch_size` | 6 | 4 |
| Global batch size | 3072 | 1024 |
| TFLOP/s/device | 152.4 | 134.1 |
| MFU | 33.2% | 29.2% |

At half the scale and a third less batch, per-device throughput holds at 88%.

The gap comes mostly from batch size: dropping pdb from 6 to 4 reduces work per
step and degrades overlap between collectives and compute. That is a hard
constraint imposed by the doubled weight shard at 256 chips, not a tuning miss.

### An incomplete parameter set costs real MFU

An earlier run omitted several upstream settings and reached only
**114.82 TFLOP/s/device (MFU 25.0%)**. Restoring them gained **16.8%**. What
was missing:

| Parameter | Upstream | The incomplete run |
| --- | --- | --- |
| `tile_batch_seq` | 512 | unset |
| `tile_embed_dim` | 1024 | unset |
| `tile_mlp_dim` | 1024 | unset |
| `--2a886c8_chip_config_name` | `megachip_tccontrol` | unset |
| `--xla_tpu_use_tc_device_shape_on_sc` | true | unset |
| `--xla_sc_enable_instruction_fusion` | false | unset |
| `--xla_sc_disjoint_spmem` | false | unset |
| `--xla_sc_disable_megacore_partitioning` | true | unset |

The last five configure SparseCore's **operating mode**. Enabling the three
`xla_tpu_enable_sparse_core_collective_offload_*` switches without configuring
the operating mode leaves it half-applied — which also invalidates any
"disable SparseCore and compare" experiment.

The three `tile_*` parameters were renamed to `wi_tile_*` / `wo_tile_*` in newer
MaxText. On this commit (`3eb77db3`) they are valid — do not drop them just
because a newer version rejects the name.

**Lesson: start from the upstream config and change only the one parameter that
scale forces you to change. Do not trim anything else along the way.**

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
this job's pods to disappear**, otherwise the new pods contend for nodes whose
TPUs are still held by the old ones:

```bash
while [ "$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME --no-headers 2>/dev/null | wc -l)" -ne 0 ]; do sleep 10; done
```

**Do not drop the `-l jobset.sigs.k8s.io/jobset-name=...` selector.** Counting
the raw `kubectl get pods` output instead includes every unrelated pod in the
namespace — near certain on a shared cluster — and the loop never exits.
