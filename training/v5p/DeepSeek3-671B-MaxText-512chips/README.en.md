# DeepSeek V3 671B pretraining on 512 v5p chips (K8s manifest)

> 中文版：[README.md](README.md)
>
> Back to the [v5p recipe index](../README.en.md)

This recipe runs DeepSeek V3 671B pretraining on **512 v5p chips**
(topology `8x8x8`).

It is a **verbatim reproduction** of the upstream recipe
[DeepSeek3-671B-MaxText](../DeepSeek3-671B-MaxText/README.md)
(`v5p-1024` = 512 chips) — not a single parameter changed, only the submission
path swapped from XPK to a plain K8s manifest.

**The measured numbers match upstream to within 0.3%** — see
[measured results](#measured-results). The upstream recipe is reproducible
today, provided you first work around
[dependency drift](../DeepSeek3-671B-MaxText-256chips/README.en.md#dependency-drift-read-this).

For the half-scale version see
[DeepSeek3-671B-MaxText-256chips](../DeepSeek3-671B-MaxText-256chips/README.en.md),
which records what must change at 256 chips, what it costs, and carries the full
troubleshooting table.

## Which version to use

| | Upstream | This recipe | 256chips recipe |
| --- | --- | --- | --- |
| chips | 512 | 512 | 256 |
| Submission | XPK | K8s manifest | K8s manifest |
| `per_device_batch_size` | 6 | 6 | 4 |
| Purpose | Upstream baseline | Match upstream numbers / customer delivery | Half-scale adaptation |

Use a manifest for customer delivery: they can read exactly what gets deployed,
whereas the XPK wrapper hides the details.

## Prerequisites

- A GKE cluster with a 512-chip v5p node pool (128 VMs × 4 chips, topology `8x8x8`)
- JobSet CRD installed **and its controller Running** (see
  [the JobSet controller image expires](#the-jobset-controller-image-expires))
- `envsubst` locally (Debian/Ubuntu: the `gettext-base` package)
- A working MaxText runner image (build steps in
  [the 256chips recipe](../DeepSeek3-671B-MaxText-256chips/README.en.md#building-the-image);
  both scales use the same image)

After pointing kubectl at the cluster, self-check:

```bash
kubectl get nodes -l cloud.google.com/gke-nodepool=<NODEPOOL> --no-headers | grep -c ' Ready '   # expect 128
kubectl get pods -n jobset-system                                                                # expect 1/1 Running
command -v envsubst                                                                              # expect a path
```

## Creating the 512-chip node pool

```bash
gcloud container node-pools create <NODEPOOL> \
  --cluster=<CLUSTER> --region=<REGION> --project=<PROJECT> \
  --node-locations=<ZONE> \
  --machine-type=ct5p-hightpu-4t \
  --tpu-topology=8x8x8 \
  --num-nodes=128 \
  --max-pods-per-node=16 \
  --spot
```

Two of those flags deserve explanation.

### Use `8x8x8` for the topology

512 chips admits several 3D shapes; `8x8x8` is a perfect cube and gives the best
bisection bandwidth. On a 3D torus, closer to a cube is better — do not
improvise something like `4x8x16`.

Do not pass `--placement-policy`; GKE generates a `COMPACT` one. Specifying it
manually fails with
`Required field 'resource.requestedRunDuration' not specified`.

### `--max-pods-per-node=16` is not optional

**Without it, a 128-node pool will most likely fail to create**, and the error
looks unrelated to TPUs:

```
Atomic resize failed with [IP_SPACE_EXHAUSTED_WITH_DETAILS]:
IP space of 'subnetworks/default' is exhausted.
Insufficient free IP addresses in the IP range '10.90.0.0/17'.
```

This is not a TPU capacity problem — it is **pod secondary IP range exhaustion**.

GKE carves a pod CIDR per node based on `maxPodsPerNode`. The default of 110
maps to a `/24` (256 addresses). If the cluster's pod secondary range is a `/17`
(32768 addresses), that allows `32768 / 256 = 128` nodes **for the whole
cluster**, not per pool. With other nodes already present, asking for 128 more
necessarily overflows.

Meanwhile only 2 pods on a v5p training node actually consume a pod IP (3
including the training pod):

```bash
kubectl get pods -A --field-selector spec.nodeName=<NODE> -o json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
    print(len([p for p in d['items'] if not p['spec'].get('hostNetwork')]))"
```

Everything else is a hostNetwork daemonset and costs no pod IP. So 110 is pure
waste. At 16 each node drops to a `/27` (32 addresses), cutting the requirement
to **one eighth** — 128 nodes need only 4096 addresses.

Check what your cluster has left before you start:

```bash
gcloud container clusters describe <CLUSTER> --region <REGION> \
  --format='value(clusterIpv4Cidr, defaultMaxPodsConstraint.maxPodsPerNode)'
kubectl get nodes --no-headers | wc -l
```

Capacity in nodes ≈ `2^(32 − prefix) / (2 × maxPodsPerNode rounded up to a power of two)`.

**This trap has nothing to do with TPUs — any large GKE node pool hits it.** The
good news is that the fix is scoped to the new pool and requires no change to a
shared subnet.

### Losing a capacity race is a different failure

The above is an IP problem. If you get `GCE_STOCKOUT` instead, that really is a
lack of capacity. Multi-host TPU allocation is **atomic**: all 128 VMs must
arrive together, and one short means the whole request fails. There is no
partial fulfilment.

## Running

```bash
export WORKLOAD_NAME=<your job name>
export WORKLOAD_IMAGE=<your runner image>
export BASE_OUTPUT_DIR=/tmp/mtout

envsubst < k8s/k8s_manifest.yaml | kubectl apply -f -
```

`BASE_OUTPUT_DIR` **must be genuinely writable**. A failed write masquerades as
a TPU hang — the most expensive trap in this series, covered in
[an unwritable output directory masquerades as a TPU hang](../DeepSeek3-671B-MaxText-256chips/README.en.md#an-unwritable-output-directory-masquerades-as-a-tpu-hang).
For a pure benchmark, a local path is simplest.

## Monitoring

```bash
POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME \
      -o jsonpath='{.items[0].metadata.name}')

kubectl logs $POD | grep 'completed step'
kubectl logs $POD | grep 'Slow PjRt'      # any output means it is stuck
```

`items[0]` is fine — MaxText prints `completed step` on every host, not just
process 0.

Expect about **8 minutes** from `apply` to the first `completed step`, almost
all of it XLA compilation. That is slightly longer than the 6–7 minutes at 256
chips. A silent log during this window is normal.

## Measured results

Environment: `cloud-tpu-multipod-dev`, us-central1-a, spot, 2026-07-27

| Item | Value |
| --- | --- |
| chips | 512 (128 VMs × 4) |
| JAX devices | 512 |
| Topology | `8x8x8` COMPACT |
| Sequence length | 8192 |
| `per_device_batch_size` | 6 |
| Global batch size | 3072 |
| Precision | bf16 (fp32 weights) |
| step 0 (incl. compile) | 114.2 s |
| **Steady-state step time** | **90.41 s** |
| **TFLOP/s/device** | **152.85** |
| **MFU** | **33.3%** |
| Tokens/s/device | 543.6 |
| loss (step 0 → 16) | 12.270 → 10.436, monotonically decreasing |

Steady state averaged over 9 clean steps (3, 4, 7, 8, 9, 13, 14, 15, 16),
range 152.70–152.91, **0.14% jitter**.

v5p uses MegaCore, so 1 chip = 1 JAX device and `TFLOP/s/device` is already a
per-chip number. MFU = 152.85 / 459.

### Against the upstream numbers

The upstream README quotes a sample log line at step 11. Field by field:

| Field | Upstream | This run | Delta |
| --- | --- | --- | --- |
| seconds | 90.668 | 90.41 | −0.27% |
| TFLOP/s/device | 152.415 | 152.85 | +0.29% |
| Tokens/s/device | 542.108 | 543.6 | +0.28% |
| total_weights | 25165824 | 25165824 | exact match |
| loss @ step 11 | 10.989 | 10.958 | −0.3% |

The exact `total_weights` match confirms it is the same model, and the loss
curves agree. **The upstream numbers reproduce.**

Note that step 11 in this run landed on an async-dispatch pairing (0.005 s), so
the throughput figures come from the clean-step average of the same run; only
the loss is taken from step 11 itself.

### Which steps are usable

| step | State | Use it? |
| --- | --- | --- |
| 0 | Includes first compile (114.2 s) | ✗ |
| 1, 6, 11 | JAX async dispatch artifact (0.005–0.35 s, six-digit TFLOP figures) | ✗ |
| 2 | Not yet converged (116.7 s) | ✗ |
| **3, 4, 7, 8, 9, 13+** | **Steady state (90.4 s)** | **✓** |
| 5, 10, 12 | xplane profiler write overhead (127–181 s) | ✗ |

`step 1` reports 39817 TFLOP/s/device and steps 6 and 11 report 0.005 s. Both
are **JAX async dispatch** artifacts: the log prints before the computation
completes. Such lines always pair with the step before them; add the two
together to get the true cost of both.

The upstream config also enables the profiler, so the comparison is fair.

### 512 versus 256 chips

| | 512 chips | 256 chips |
| --- | --- | --- |
| `per_device_batch_size` | 6 | 4 |
| Global batch size | 3072 | 1024 |
| Steady-state step | 90.41 s | 68.70 s |
| TFLOP/s/device | 152.85 | 134.1 |
| MFU | 33.3% | 29.2% |
| Weight shard per device | 10.5 GB | 21 GB |

**256 chips reaches 87.7% of the per-device throughput of 512 chips.**

Almost all of the gap comes from batch size. Halving the chip count doubles the
weight shard per device (10.5 → 21 GB), which squeezes out activation space and
forces `per_device_batch_size` down from 6 to 4 (5 was measured to fail). Less
work per step means worse overlap between collectives and compute.

That is a hard constraint, not a tuning miss. Restoring pdb=6 at 256 chips would
require a more memory-efficient parallelism strategy or lower precision — not a
few flag changes.

## Troubleshooting quick reference

Image build, output directory, spot preemption and other shared issues are in
[the 256chips troubleshooting table](../DeepSeek3-671B-MaxText-256chips/README.en.md#troubleshooting-quick-reference)
and apply identically here. Below are the ones **specific to this scale**.

| Symptom | Root cause | Fix |
| --- | --- | --- |
| `IP_SPACE_EXHAUSTED_WITH_DETAILS` | Pod secondary range carves a `/24` per node at maxPods=110; a `/17` only covers 128 nodes | Add `--max-pods-per-node=16` at pool creation |
| `Already exists: .../nodePools/<name>` | A previously failed pool still exists in `ERROR` state | `node-pools delete` first, then recreate |
| `failed calling webhook "mjobset.kb.io": no endpoints available` | The JobSet controller is down | See below |
| `GCE_STOCKOUT` | Genuinely no capacity; atomic allocation fails if one VM is short | Try another zone, or wait |

### The JobSet controller image expires

Submitting a job reports:

```
Internal error occurred: failed calling webhook "mjobset.kb.io":
no endpoints available for service "jobset-webhook-service"
```

Check the controller:

```bash
kubectl get pods -n jobset-system
# jobset-controller-manager-xxx   0/1   ErrImagePull
```

The cause is that JobSet installs a **staging image** by default:

```
us-central1-docker.pkg.dev/k8s-staging-images/jobset/jobset:v0.11.1
    → not found
```

Tags in the staging registry are garbage-collected periodically. An existing
node keeps working off its local digest cache (ours ran for 103 days), but **the
moment the controller is rescheduled onto a fresh node the pull fails.** Node
pool scaling and node recycling both trigger this.

Switch to the officially released image of the same version:

```bash
kubectl set image deployment/jobset-controller-manager -n jobset-system \
  manager=registry.k8s.io/jobset/jobset:v0.11.1
```

`registry.k8s.io` is the release registry and its tags are not reclaimed.

**This one detonates with no warning** — the controller can break while you are
not submitting anything, and you only find out when you need to train. Run
`kubectl get pods -n jobset-system` before submitting.

## Clean up

```bash
kubectl delete jobset $WORKLOAD_NAME
```

Deletion is asynchronous. Before submitting the next job, **wait for this job's
pods to disappear**:

```bash
while [ "$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$WORKLOAD_NAME \
           --no-headers 2>/dev/null | wc -l)" -ne 0 ]; do sleep 10; done
```

The label selector matters. On a shared cluster, counting every pod in the
namespace means the loop never exits.

Delete the node pool when you are done (**a spot pool left idle will still be
reclaimed, but deleting it means racing for capacity again**):

```bash
gcloud container node-pools delete <NODEPOOL> --cluster <CLUSTER> --region <REGION>
```

**A multi-host TPU node pool cannot be scaled down — deletion is the only
option.** Scaling to zero to save money does not work:

```
501 Unimplemented: Multi-host TPU pool (<name>) manual resize is not supported.
```

A TPU slice is allocated atomically, so "resize" is not a meaningful operation
on it. This differs from CPU/GPU node pool habits — do not waste time trying
`clusters resize --num-nodes=0`.
