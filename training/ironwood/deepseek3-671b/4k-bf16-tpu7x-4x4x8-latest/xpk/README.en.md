# Training DeepSeek3-671B on TPU Ironwood (tpu7x-4x4x8) — 128 chips / latest MaxText

This recipe outlines the steps for running a deepseek3-671b
[MaxText](https://github.com/AI-Hypercomputer/maxtext) pretraining workload on
[Ironwood GKE clusters](https://cloud.google.com/kubernetes-engine) by using
[XPK](https://github.com/AI-Hypercomputer/xpk).

It uses MaxText's built-in synthetic dataset and needs **no external storage**.
The goal is to validate the flow and measure performance, not to produce a
usable model. For real data, see
[Appendix: switching to Lustre](#appendix-switching-to-lustre).

If you prefer plain Kubernetes objects over the XPK wrapper — for example when
handing the recipe to a customer who needs to read exactly what gets deployed —
see the equivalent manifest-based recipe in [../k8s](../k8s/README.en.md).

> 中文版：[README.md](README.md)

## Workload Details

This workload is configured with the following details:

-   Sequence Length: 4096
-   Precision: bf16
-   Chips: 128 (4x4x8 topology, 32 VMs × 4 chips)
-   Dataset: synthetic (built into MaxText, no external dependency)
-   Checkpointing: disabled

## Relationship to the other recipes

The upstream `4k-bf16-tpu7x-4x4x8` recipe is pinned to the pre-refactor MaxText
API, so it cannot pick up the DeepSeek V3 MoE optimizations merged after
2026-03. This recipe keeps the exact same 128-chip parallelism configuration but
runs on the current MaxText main branch.

| | upstream `4k-bf16-tpu7x-4x4x8` | This recipe |
| --- | --- | --- |
| MaxText entrypoint | `MaxText.train` | `src.maxtext.trainers.pre_train.train` |
| Config path | `MaxText/configs/base.yml` | `src/maxtext/configs/base.yml` |
| MoE sharding flag | `fsdp_shard_on_exp=True` | `shard_exp_on_fsdp=True` (renamed upstream) |
| XPK | 0.16.1 | 1.8.0 |
| XLA | — | adds `--xla_tpu_dvfs_p_state=3` |
| Parallelism | same | same |

### Why MoE sharding uses `shard_exp_on_fsdp` rather than 2D sharding

The 256-chip `4k-bf16-tpu7x-4x8x8-lustre` recipe uses
`use_2d_fsdp_sharding=True` with `ici_fsdp_transpose_parallelism=2`. That path
shards MoE weights across both the `fsdp` and `fsdp_transpose` axes and is only
meaningful when `ici_fsdp_transpose_parallelism > 1`.

At 128 chips there is a single FSDP axis of 256 devices, so this recipe keeps
`ici_fsdp_transpose_parallelism=1` and uses `shard_exp_on_fsdp=True`.

MaxText enforces hard constraints on this path: `num_experts` must be divisible
by `ici_fsdp_parallelism`, and both `ici_expert_parallelism` and
`ici_tensor_parallelism` must be 1. DeepSeek V3 has 256 experts and
`ici_fsdp_parallelism` resolves to 256, so all three hold. **This has been
verified on real 128-chip hardware.**

Note that `shard_exp_on_fsdp` was named `fsdp_shard_on_exp` in older MaxText.

## Validation status

Run on a 128-chip Spot node pool in `us-central1-c` on 2026-07-26 (via the
[../k8s](../k8s/README.en.md) manifest path; MaxText arguments are identical).

Verified: 4x4x8 node pool creation (32/32 Ready), image build from MaxText main
(`e50e39458`, 2026-07-25), all 32 workers started, `shard_exp_on_fsdp=True`
passing config validation, XLA compilation and the first training steps.

Not verified: **steady-state performance** (Spot preempted at step 2, and steps
0–1 are JIT compilation); the Lustre storage path (see appendix).

## Performance reference

Historical data on the **same topology, same parallelism, synthetic dataset**,
using an older MaxText (`maxtext-tutorial-v1.5.0`, JAX `0.8.2.dev20251215`):

| Config | Precision | Step time | TFLOP/s/chip | Tokens/s/chip |
| --- | --- | --- | --- | --- |
| 4x4x8 (128 chips) | bf16 | 27.00 s | 608.0 | 2,427.7 |
| 4x4x8 (128 chips) | fp8 | 22.39 s | 733.1 | 2,926.5 |

Mind the units: MaxText logs `TFLOP/s/device` **per TensorCore**, and a TPU v7
chip has 2 TensorCores, so per-chip values are 2× the logged value. Against a
v7 per-chip BF16 peak of 2,307 TFLOPS, 608 is roughly 26.4% MFU.

## Prerequisites

To run this recipe, you need the following:

-   **GCP Project Setup:** Ensure you have a GCP project with billing enabled
    and are allowlisted for Ironwood access.
-   **User Project Permissions:** The account used requires the following IAM
    Roles:
    -   Artifact Registry Writer
    -   Compute Admin
    -   Google Cloud Managed Lustre Admin 
    -   Kubernetes Engine Admin
    -   Logging Admin
    -   Monitoring Admin
    -   Service Account User
    -   Storage Object Viewer
    -   Vertex AI Administrator
    -   Service Usage Consumer
    -   TPU Viewer
-   **Docker:** Docker must be installed on your workstation. Follow the steps
    in the [Install XPK and dependencies](#install-xpk-and-dependencies) section
    to install Docker.
-   **Python 3.11 Virtual Environment:** A Python
    3.11 virtual environment is required. Instructions
    for setting this up are also in the
    [Install XPK and dependencies](#install-xpk-and-dependencies) section.
-   **XPK and Dependencies:** Follow the steps in the
    [Install XPK and dependencies](#install-xpk-and-dependencies) section to
    install XPK, `kubectl`, `kubectl-kueue`, and `kubectl-kjob`.


## Install XPK and dependencies

### XPK and Dependency Installation

#### Virtual Python Environment

Run the following to create a virtual Python environment:

```bash
# Set up uv
sudo apt update
curl -LsSf https://astral.sh/uv/install.sh -o install-uv.sh
chmod +x install-uv.sh
./install-uv.sh
rm install-uv.sh
source ${HOME}/.local/bin/env

# Set up and Activate Python 3.11 virtual environment
uv venv --seed ${HOME}/.local/bin/venv --python 3.11 --clear
source ${HOME}/.local/bin/venv/bin/activate
pip install --upgrade pip
```

#### XPK

Make sure you have the virtual environment activated when running XPK.

Install XPK and necessary tools:

```bash
# Install gcloud, if not already installed, https://cloud.google.com/sdk/docs/install
# Install kubectl, if not already installed, https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_kubectl

# Ensure to log in to your gcloud

# Install latest xpk
pip install xpk==1.8.0

# Install xpk pre-reqs kubectl-kueue and kjob (if you installed xpk via pip)
curl -LsSf https://raw.githubusercontent.com/AI-Hypercomputer/xpk/refs/tags/v1.8.0/tools/install-xpk.sh -o install-xpk.sh
chmod +x install-xpk.sh
sudo ./install-xpk.sh
rm install-xpk.sh

# Follow https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl#install_plugin to install gke-gcloud-auth-plugin
```

#### Docker

Install Docker using instructions provided by your administrator. Once
installed, run the following commands:

```bash
## Configure docker and test installation
gcloud auth configure-docker
sudo usermod -aG docker $USER ## relaunch the terminal and make sure you have the virtual environment activated after running this command
docker run hello-world # Test docker
```

## Orchestration and deployment tools

For this recipe, the following setup is used:

-   **Orchestration** -
    [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
-   **Pretraining job configuration and deployment** - XPK is used to configure
    and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the execution of the deepseek3-671b workload.

## Test environment

This recipe is optimized for and tested with tpu7x-4x4x8.

-   **GKE cluster** To create your GKE cluster, use the XPK instructions.
    [XPK instructions](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#cluster-create).
    A sample command to create an XPK cluster is provided below.

### Environment Variables for Cluster Creation

The environment variables required for cluster creation and workload execution
are defined at the beginning of the `run_recipe.sh` script. **Before running the
`xpk workload create` command**, please open `run_recipe.sh` and modify the
`export` statements to set these variables to match your environment. It is
crucial to use consistent values for `PROJECT_ID`, `CLUSTER_NAME`, and `ZONE`
across all commands and configurations.

-   `PROJECT_ID`: Your GCP project name.
-   `CLUSTER_NAME`: The target cluster name.
-   `ZONE`: The zone for your cluster (e.g., `us-central1-c`).
-   `CONTAINER_REGISTRY`: The container registry to use (e.g., `gcr.io`).
-   `BASE_OUTPUT_DIR`: Output directory for model training (e.g.,
    `"<your_lustre_instance>"`).
-   `MAXTEXT_ROOT`: The absolute path where you cloned the MaxText repository.
-   `WORKLOAD_IMAGE`: The Docker image for the workload. This is set in
    `run_recipe.sh` to
    `${CONTAINER_REGISTRY}/${PROJECT_ID}/${USER}-deepseek-v3-runner` by
    default, matching the image built in the
    [Docker container image](#docker-container-image) section.
-   `WORKLOAD_NAME`: A unique name for your workload. This is set in
    `run_recipe.sh` using the following command:
    `export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-deepseekv3-671b-4096-fsdp")-$(date +%Y%m%d-%H%M)"`
-   `GKE_VERSION`: The GKE version, `1.34.0-gke.2201000` or later.
-   `RESERVATION_NAME`: Your TPU reservation name. Use the reservation name if
    within the same project. For a shared project, use
    `"projects/<project_number>/reservations/<reservation_name>"`.

### Placement policy (required for v7)

TPU v7 **does not support auto-creating** placement policies. Create one for the
target topology before creating a multi-host node pool, otherwise the pool is
created but stays at 0 nodes:

```bash
gcloud compute resource-policies create workload-policy tpu7x-128chip \
  --region=${REGION} --project=${PROJECT_ID} \
  --type=HIGH_THROUGHPUT --accelerator-topology=4x4x8
```

Multi-host pools are allocated all-or-nothing: all 32 VMs must be obtained at
once, since a 4x4x8 topology requires a physically contiguous cube.

### Sample XPK Cluster Creation Command

```bash
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-4x4x8 \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}
```

### Enable Managed Lustre CSI Driver on Cluster

Ensure the GKE version is `1.34.0-gke.2201000` or later. If your GKE cluster is already created, ensure the Managed Lustre CSI driver is enabled.

```bash
gcloud container clusters update ${CLUSTER_NAME} \
  --location ${ZONE} \
  --project ${PROJECT_ID} \
  --update-addons=LustreCsiDriver=ENABLED
```

## Docker container image

To build your own image, follow the steps linked in this section. If you don't
have Docker installed on your workstation, see the section below for installing
XPK and its dependencies. Docker installation is part of this process.

### Steps for building workload image

This recipe targets the **latest MaxText main branch**, which contains DeepSeek
V3 MoE optimizations merged after the upstream 4x8x8 recipe was pinned (commit
`cf051eb03`, 2026-03-17) — for example `Overlap moe comms with collective
matmul`, `Fix ragged all-to-all with ragged buffer factor in DeepSeek-V3`, and
`Enable MoE ragged sort on TPU7X`.

Verified with MaxText `e50e39458` (2026-07-25), nightly jax/libtpu, Python 3.11,
XPK 1.8.0.

```bash
export CONTAINER_REGISTRY="" # e.g. us-docker.pkg.dev/${PROJECT_ID}/gcr.io
export CLOUD_IMAGE_NAME="${USER}-maxtext-latest"

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

The bundled `docker_upload_runner.sh` pushes to `gcloud config get-value
project`, which targets the wrong project in cross-project setups — prefer
tagging and pushing manually as above.

## Training dataset

This recipe uses MaxText's built-in synthetic dataset. No external data
preparation is required.

## Run the recipe

### Configure environment settings

Before running any commands in this section, ensure you have set the environment
variables as described in
[Environment Variables for Cluster Creation](#environment-variables-for-cluster-creation).

### Connect to an existing cluster (Optional)

If you want to connect to your GKE cluster to see its current state before
running the benchmark, you can use the following gcloud command. (Note that XPK
does this for you already):

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

## Get the recipe
```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x8-latest/xpk
```

### Run deepseek3-671b Pretraining Workload

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the deepseek3-671b pretraining workload.

Before execution, use `nano ./run_recipe.sh` to edit the script and configure the environment variables to match your specific environment.

### Configuring and Starting workload

From the MaxText root directory, start your DeepSeek3-671B workload.

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the deepseek3-671b pretraining workload.

Edit the Recipe (run_recipe.sh) and populate the exported variables at the top of the file to match your environment.

```
# In run_recipe.sh, update these lines:
export PROJECT_ID="your-project-id"
export CLUSTER_NAME="your-cluster-name"
export ZONE="your-zone"
export BASE_OUTPUT_DIR="gs://your-bucket/ds3-run"
export WORKLOAD_IMAGE="your-registry/your-maxtext-latest:runner"
```

To configure and run the benchmark:

```bash
chmod +x run_recipe.sh
nano ./run_recipe.sh
./run_recipe.sh
```

You can customize the run by modifying `run_recipe.sh`:

-   **Environment Variables:** Variables like `PROJECT_ID`, `CLUSTER_NAME`,
    `ZONE`, `WORKLOAD_NAME`, `WORKLOAD_IMAGE`, and `BASE_OUTPUT_DIR` are defined
    at the beginning of the script. Adjust these to match your environment.
-   **XLA Flags:** The `XLA_FLAGS` variable contains a set of XLA configurations
    optimized for this workload. These can be tuned for performance or
    debugging.
-   **MaxText Workload Overrides:** The `MAXTEXT_ARGS` variable holds the
    arguments passed to the `python3 -m src.maxtext.trainers.pre_train.train` command. This
    includes model-specific settings like `per_device_batch_size`,
    `max_target_length`, and others. You can modify these to experiment with
    different model configurations.
-   **Virtual Environment:** The script activates the virtual environment
    created during the
    [Install XPK and dependencies](#install-xpk-and-dependencies) steps. If you
    used a different virtual environment, modify the `source` command at the top
    of `run_recipe.sh`.

Note that any MaxText configurations not explicitly overridden in `MAXTEXT_ARGS`
are expected to use the defaults within the specified `WORKLOAD_IMAGE`.

## Monitor the job

To monitor your job's progress, you can use kubectl to check the Jobset status
and stream logs:

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# List pods to find the specific name (e.g., deepseek3-0-0-xxxx)
kubectl get pods | grep ${WORKLOAD_NAME}
```
Then, stream the logs from the running pod (replace <POD_NAME> with the name you found):

```bash
kubectl logs -f <POD_NAME>
```
You can also monitor your cluster and TPU usage through the Google Cloud
Console.

### Follow Workload and View Metrics

After running `xpk workload create`, you will get a link to the Google Cloud
Console to view your workload logs. Example: `[XPK] Follow your workload here:
https://console.cloud.google.com/kubernetes/service/${ZONE}/${PROJECT_ID}/default/${WORKLOAD_NAME}/details?project=${PROJECT_ID}`
Alternatively, list workloads: (`xpk workload list`)

```bash
xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

For more in-depth debugging, use xpk inspector: (`xpk inspector`)

```bash
xpk inspector --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} [--workload ${WORKLOAD_NAME}]
```

### Delete resources

#### Delete a specific workload

```bash
xpk workload delete --workload ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
# Or filter and delete:
xpk workload delete --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} --filter-by-job=${USER}
```

#### Delete the XPK storage resource

```bash
xpk storage detach lustre-volume --project ${PROJECT_ID} --cluster ${CLUSTER_NAME} --zone ${ZONE}
```

#### Delete the entire XPK cluster

```bash
xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```

## Check results

After the job completes, you can check the results by:

-   Accessing output logs from your job.
-   Checking the TensorBoard output under `${BASE_OUTPUT_DIR}` as set in
    `run_recipe.sh` (checkpointing is disabled in this recipe).
-   Reviewing metrics in Cloud Monitoring, if configured.

## Troubleshooting

**`file:///deps does not appear to be a Python project`**

The runner image's `/deps` contains only `benchmarks/`, `src/`, `tests/` and
`pytest.ini` — no `pyproject.toml` or `setup.py`. Do not run
`pip install -e .`; just `cd /deps` and invoke the entrypoint module.

**Config not found**

The correct config path is `src/maxtext/configs/base.yml`. There is no
`/deps/maxtext` directory.

**Errors about assets paths**

The runner image already sets `MAXTEXT_ASSETS_ROOT=/deps/src/maxtext/assets`
(**lowercase** `maxtext`) and `MAXTEXT_PKG_DIR`. Do not override them. The
`/deps/src/MaxText/assets` path used by older recipes does not exist.

**Node pool created but shows 0 nodes**

Check that the placement policy exists and its topology matches. TPU v7 cannot
auto-create one.

## Appendix: switching to Lustre

To train on a real dataset with checkpointing, mount Google Cloud Managed
Lustre. The instance must be on the **same VPC network** as the GKE cluster.

### 1. Create the Lustre instance

1. Create new Lustre instance following [instructions](https://docs.cloud.google.com/managed-lustre/docs/create-instance) to hold the dataset and checkpoints. Mount the Lustre instance on
[Compute Engine](https://docs.cloud.google.com/managed-lustre/docs/connect-from-compute-engine)
or
[Kubernetes Engine](https://docs.cloud.google.com/managed-lustre/docs/lustre-csi-driver-new-volume). It is important to use the same network as the GKE cluster when creating the Lustre instance. Since the same instance will be used for both dataloading and checkpointing, at least 36 TB of storage is recommended.

2. Prepare your dataset in the Lustre instance. This recipe is configured to use the Grain loader with ArrayRecord files. Ensure your dataset files are accessible in this instance. You would first need to download the AllenAI C4 dataset dataset from its source. Follow these [instructions](https://docs.cloud.google.com/managed-lustre/docs/transfer-data) to transfer the dataset to the Lustre instance.

### 2. Mount the Lustre instance

Managed Lustre lets you mount and access it as local file systems, so applications can read and write objects using standard file system semantics. You'll need to use the below commands to create [XPK storage resources](https://github.com/AI-Hypercomputer/xpk/blob/main/docs/usage/storage.md#managed-lustre) for the instance in order to mount it to the MaxText workload. For the lustre instance, use the PV/PVC definition in [../k8s/lustre_pvc.yaml](../k8s/lustre_pvc.yaml).
Be sure to update `volumeHandle` in the yamls with your correct lustre instance names. Creating a lustre instance and attaching xpk storage is a one time setup.
```
# Set variables
export PROJECT=""
export CLUSTER=""
export ZONE=""

# Lustre PV/PVC
xpk storage attach lustre-volume --type=lustre --project=$PROJECT --cluster=$CLUSTER --zone=$ZONE --mount-point=/mnt/lustre --readonly=false --auto-mount=false --manifest=../k8s/lustre_pvc.yaml
```

### 3. Update run_recipe.sh

Replace

```
tokenizer_path=src/maxtext/assets/tokenizer.mistral-v3 dataset_type=synthetic enable_checkpointing=False
```

with the grain / checkpoint arguments, add `--storage=$LUSTRE_VOLUME_NAME` to
`xpk workload create`, and point `BASE_OUTPUT_DIR` at `/mnt/lustre/checkpoints`.
See the upstream 256-chip `4k-bf16-tpu7x-4x8x8-lustre` recipe for the full set.

**This path has not been validated on hardware.**

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