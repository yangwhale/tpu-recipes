# Pretrain deepseek3-671b on Ironwood GKE clusters with Kubernetes JobSet and Lustre

This recipe outlines the steps for running a deepseek3-671b
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine)
by applying a Kubernetes manifest to deploy a JobSet resource, using Google
Cloud Managed Lustre as the primary storage system for the dataset and
checkpoints.

Everything is expressed as plain Kubernetes objects — no XPK required. The
equivalent XPK-based recipe lives in [../xpk](../xpk/README.en.md).

> 中文版：[README.md](README.md)

## Workload Details

This workload is configured with the following details:

-   Sequence Length: 4096
-   Precision: bf16
-   Chips: 128 (4x4x8 topology)
-   Lustre for dataset and checkpoints
    -   C4 Multi-Lingual dataset (~12TB) with ArrayRecord format

## Relationship to the other recipes

This recipe combines the 128-chip (4x4x8) parallelism configuration with the
Lustre storage setup and the current MaxText runtime, expressed as a raw
Kubernetes manifest. Differences from the neighbouring recipes:

| | `4k-bf16-tpu7x-4x4x8/k8s` | This recipe |
| --- | --- | --- |
| MaxText entrypoint | `MaxText.train` | `src.maxtext.trainers.pre_train.train` |
| MoE sharding flag | `fsdp_shard_on_exp=True` | `shard_exp_on_fsdp=True` (renamed upstream) |
| Dataset | synthetic | C4 multilingual via grain / ArrayRecord |
| Checkpointing | disabled | enabled, async, to Lustre |
| Storage | none | Lustre PVC mounted at `/mnt/lustre` |

Compared to the 256-chip `4k-bf16-tpu7x-4x8x8-lustre` recipe, this one keeps
`ici_fsdp_transpose_parallelism=1` and uses `shard_exp_on_fsdp=True` instead of
`use_2d_fsdp_sharding=True`. The 2D path shards MoE weights across both the
`fsdp` and `fsdp_transpose` axes and is only meaningful when
`ici_fsdp_transpose_parallelism > 1`. At 128 chips there is a single FSDP axis
of 256 devices; MaxText requires `num_experts` to be divisible by
`ici_fsdp_parallelism` for `shard_exp_on_fsdp`, and DeepSeek V3 has 256
experts, so the constraint holds. That path also requires
`ici_expert_parallelism = 1` and `ici_tensor_parallelism = 1`, both defaults
here.

## Prerequisites

This recipe assumes the following prerequisites are met:

-   **GKE Cluster:** A GKE cluster with [JobSet](https://jobset.sigs.k8s.io/docs/installation/) installed and running.
-   **Container Image:** A pre-built container image (such as
    `gcr.io/my-project/my-maxtext-runner:latest`) containing the MaxText
    workload, accessible by the GKE cluster.
-   **Tools:** `gcloud`, `kubectl`, `gke-gcloud-auth-plugin`, and `envsubst`
    installed on your workstation. If `envsubst` is missing, install it with
    `sudo apt-get update && sudo apt-get install -y gettext-base`.
-   **Permissions:** You have permissions to run `kubectl apply` on the target
    cluster and the cluster has permissions to pull the container image.
-   **Managed Lustre:** A Google Cloud Managed Lustre instance on the same
    network as the GKE cluster, with the Managed Lustre CSI driver enabled on
    the cluster (GKE `1.34.0-gke.2201000` or later). See
    [Lustre setup](#lustre-setup) below.

## Orchestration and deployment tools

For this recipe, the following setup is used:

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **Pretraining job configuration and deployment** - A Kubernetes manifest
    (`k8s_manifest.yaml`) is used to define and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the execution of the MaxText pretraining workload.

## Container image

Build the workload image following the
[Docker container image](../xpk/README.en.md#docker-container-image) section of the
XPK recipe. That section targets the **latest MaxText main branch**, which is
what this manifest expects (`src.maxtext.trainers.pre_train.train` entrypoint).
A known-good pinned fallback is documented there as well.

## Lustre setup

This is a one-time setup per cluster.

### 1. Enable the Managed Lustre CSI driver

```bash
gcloud container clusters update ${CLUSTER_NAME} \
  --location ${ZONE} \
  --project ${PROJECT_ID} \
  --update-addons=LustreCsiDriver=ENABLED
```

### 2. Create the Lustre instance and stage the dataset

Create a Lustre instance following the
[official instructions](https://docs.cloud.google.com/managed-lustre/docs/create-instance),
using the **same network as the GKE cluster**. Since the instance holds both the
dataset and the checkpoints, at least 36 TB is recommended.

Download the AllenAI C4 dataset and
[transfer it to the Lustre instance](https://docs.cloud.google.com/managed-lustre/docs/transfer-data)
in ArrayRecord format.

### 3. Create the PersistentVolume and PersistentVolumeClaim

Unlike the XPK recipe, which wraps this in `xpk storage attach`, here the PV and
PVC are applied directly. Edit `lustre_pvc.yaml` and fill in the placeholders
(`<INSTANCE CAPACITY SIZE>`, `<PROJECT_ID>/<ZONE>/<LUSTRE_INSTANCE_NAME>`,
`<INSTANCE IP>`, `<FILE SYSTEM NAME>`), then apply it:

```bash
kubectl apply -f lustre_pvc.yaml

# Verify the claim is Bound before submitting the workload
kubectl get pvc lustre-volume -n default
```

The manifest mounts this claim at `/mnt/lustre` in every worker pod.

## Training dataset

This recipe uses the AllenAI C4 multilingual dataset with the
[grain loader](https://github.com/google/grain) in ArrayRecord format, read
from the Lustre mount.

## Run the recipe

This recipe uses a Kubernetes manifest (`k8s_manifest.yaml`) to deploy the
workload. The following commands will set the required environment variables,
substitute them into `k8s_manifest.yaml`, and apply the resulting
configuration to your cluster.

### 1. Configure Environment Variables

Open a terminal and set the following environment variables to match your setup.
**Note:** 
- `k8s_manifest.yaml` is in the same directory as this README.  
- For WORKLOAD_IMAGE, see [Docker container image](../xpk/README.en.md#docker-container-image) section.

```bash
# Set variables for your environment
export PROJECT_ID=""    # Your GCP project name
export CLUSTER_NAME=""  # The name of your GKE cluster
export ZONE=""          # The zone of your GKE cluster
export WORKLOAD_IMAGE=""   # e.g., "gcr.io/my-project/my-maxtext-runner:latest".

# Lustre paths. These are mount points inside the pod, not GCS URIs.
export BASE_OUTPUT_DIR="/mnt/lustre/checkpoints"
export DATASET_BUCKET_MOUNTED_PATH="/mnt/lustre/datasets"

# Set workload name (or modify as needed, make sure its unique in the cluster)
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-deepseekv3-671b-lustre-128")-$(date +%Y%m%d-%H%M)"
```

### 2. Run deepseekv3-671b Pretraining Workload

Once the environment variables are set, run the following commands to fetch
cluster credentials and deploy the JobSet:

```bash
# Fetch cluster credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}

# Apply the manifest
envsubst '${BASE_OUTPUT_DIR} ${WORKLOAD_NAME} ${WORKLOAD_IMAGE} ${DATASET_BUCKET_MOUNTED_PATH}' < k8s_manifest.yaml | kubectl apply -n default -f -
```

## Monitor the job

To monitor your job's progress, you can use kubectl to check the Jobset status
and logs:

```bash
# Check JobSet status
kubectl get jobset -n default ${WORKLOAD_NAME}

# Get the name of the first pod in the JobSet
POD_NAME=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=${WORKLOAD_NAME} -n default -o jsonpath='{.items[0].metadata.name}')

# Follow the logs of that pod
kubectl logs -f -n default ${POD_NAME}
```

You can also monitor your cluster and TPU usage through the Google Cloud
Console:
`https://console.cloud.google.com/kubernetes/workload/overview?project={PROJECT_ID}`

## Delete resources

### Delete a specific workload

To delete the JobSet created by this recipe, run:

```bash
kubectl delete jobset ${WORKLOAD_NAME} -n default
```

## Check results

After the job completes, you can check the results by:

-   Accessing output logs from your job using `kubectl logs`.
-   Inspecting the checkpoints and TensorBoard output under `${BASE_OUTPUT_DIR}`
    (`/mnt/lustre/checkpoints` on the Lustre instance).
-   Reviewing metrics in Cloud Monitoring, if configured.

## Next steps: deeper exploration and customization

This recipe is designed to provide a simple, reproducible "0-to-1" experience
for running a MaxText pre-training workload. Its primary purpose is to help you
verify your environment and achieve a first success with TPUs quickly and
reliably.

For deeper exploration, including customizing model configurations, tuning
performance with different XLA flags, and running custom experiments, we
recommend using the benchmark_runner.py script directly from the MaxText
repository. This script offers the full range of MaxText's flexibility and is
the ideal tool for power users and researchers who want to move beyond the
initial benchmark and tailor the workload to their specific needs. To learn
more, see the
[MaxText Benchmark Runner Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/Getting_Started_Benchmarking.md)
on using benchmark_runner.py for advanced benchmarking.
