# Sleep Infinity Testbed on Ironwood GKE clusters with XPK

This recipe launches a **sleep infinity** pod on a TPU v7 (Ironwood) single-node
slice using [XPK](https://github.com/AI-Hypercomputer/xpk). Instead of running
a training workload, the pod stays alive indefinitely so you can `kubectl exec`
into it for interactive debugging and experimentation.

## Workload Details

-   **Command:** `sleep infinity` (no training, wait for manual exec)
-   **Docker Image:** MaxText runner (JAX + Flax + TPU libs pre-installed)
-   **Topology:** 2x2x1 (4 chips, 8 devices, single node)

## Use Cases

-   Interactive JAX/TPU debugging
-   Testing custom MaxText configurations before full runs
-   Exploring the TPU v7 environment (device topology, VMEM, etc.)
-   Running one-off scripts or benchmarks manually

## Prerequisites

To run this recipe, you need the following:

-   **GCP Project Setup:** Ensure you have a GCP project with billing enabled
    and are allowlisted for Ironwood access.
-   **User Project Permissions:** The account used requires the following IAM
    Roles:
    -   Artifact Registry Writer
    -   Compute Admin
    -   Kubernetes Engine Admin
    -   Logging Admin
    -   Monitoring Admin
    -   Service Account User
    -   Storage Admin
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
pip install xpk==0.16.1

# Install xpk pre-reqs kubectl-kueue and kjob (if you installed xpk via pip)
curl -LsSf https://raw.githubusercontent.com/AI-Hypercomputer/xpk/refs/tags/v0.16.1/tools/install-xpk.sh -o install-xpk.sh
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
-   **Job deployment** - XPK is used to configure
    and deploy the
    [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset)
    resource, which manages the sleep infinity pod.

## Test environment

This recipe is designed for tpu7x-2x2x1 (single node, 4 chips).

-   **GKE cluster** To create your GKE cluster, use the XPK instructions.
    [XPK instructions](https://github.com/AI-Hypercomputer/xpk?tab=readme-ov-file#cluster-create).
    A sample command to create an XPK cluster is provided below.

### Environment Variables for Cluster Creation

The environment variables required for cluster creation and workload execution
are defined at the beginning of the `run_recipe.sh` script. **Before running the
`xpk workload create` command**, please open `run_recipe.sh` and modify the
`export` statements to set these variables to match your environment.

-   `PROJECT_ID`: Your GCP project name.
-   `CLUSTER_NAME`: The target cluster name.
-   `ZONE`: The zone for your cluster (e.g., `us-central1-c`).
-   `WORKLOAD_IMAGE`: The Docker image for the workload (MaxText runner image).
-   `WORKLOAD_NAME`: A unique name for your workload.

### Sample XPK Cluster Creation Command

```bash
xpk cluster create \
  --cluster=${CLUSTER_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --tpu-type=tpu7x-2x2x1 \
  --num-slices=1 \
  --reservation=${RESERVATION_NAME}
```

## Docker container image

This recipe reuses the standard MaxText runner image. It provides a complete
JAX/TPU environment without needing to build a custom image.

### Steps for building workload image

The following software versions are used:

-   Libtpu version: 0.0.32.dev20251215+nightly
-   Jax version: 0.8.2.dev20251215
-   Maxtext version: maxtext-tutorial-v1.5.0
-   Python: 3.11
-   XPK: 0.16.1

Docker Image Building Command:

```bash
export CONTAINER_REGISTRY="" # Initialize with your registry
export CLOUD_IMAGE_NAME="${USER}-maxtext-runner"
export WORKLOAD_IMAGE="${CONTAINER_REGISTRY}/${PROJECT_ID}/${CLOUD_IMAGE_NAME}"

# Set up and Activate Python 3.12 virtual environment for Docker build
uv venv --seed ${HOME}/.local/bin/venv-docker --python 3.12 --clear
source ${HOME}/.local/bin/venv-docker/bin/activate
pip install --upgrade pip

# Make sure you're running on a Virtual Environment with python 3.12
if [[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" == "3.12" ]]; then { echo "You have the correct Python version 3.12"; } else { >&2 echo "Error: Python version must be 3.12"; false;} fi

# Clone MaxText Repository and Checkout Recipe Branch
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git checkout maxtext-tutorial-v1.5.0

# Build and upload the docker image
bash dependencies/scripts/docker_build_dependency_image.sh \
  MODE=nightly \
  JAX_VERSION=0.8.2.dev20251215 \
  LIBTPU_VERSION=0.0.32.dev20251215+nightly
bash dependencies/scripts/docker_upload_runner.sh CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME}

# Deactivate the virtual environment
deactivate
```

## Run the recipe

### Configure environment settings

Before running any commands in this section, ensure you have set the environment
variables as described in
[Environment Variables for Cluster Creation](#environment-variables-for-cluster-creation).

### Connect to an existing cluster (Optional)

```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

## Get the recipe
```bash
cd ~
git clone https://github.com/ai-hypercomputer/tpu-recipes.git
cd tpu-recipes/training/ironwood/testbed/sleep-inf-tpu7x-2x2x1/xpk
```

### Launch the Sleep Infinity Pod

The `run_recipe.sh` script contains all the necessary environment variables and
configurations to launch the sleep infinity pod.

Before execution, use `nano ./run_recipe.sh` to edit the script and configure
the environment variables to match your specific environment.

```bash
chmod +x run_recipe.sh
nano ./run_recipe.sh
./run_recipe.sh
```

### Connect to the Pod

Once the workload is running, connect to the pod:

```bash
# Find the pod name
kubectl get pods | grep sleep-inf

# Exec into the pod
kubectl exec -it <POD_NAME> -- bash
```

### Inside the Pod

After exec-ing in, you have a full MaxText/JAX environment:

```bash
# Check TPU devices
python3 -c "import jax; print(jax.devices())"

# Run a quick MaxText training test
python3 -m MaxText.train MaxText/configs/base.yml \
  model_name=default \
  per_device_batch_size=1.0 \
  max_target_length=128 \
  steps=5 \
  dataset_type=synthetic \
  enable_checkpointing=False
```

## Monitor the job

```bash
kubectl get jobset -n default ${WORKLOAD_NAME}

# List pods
kubectl get pods | grep ${WORKLOAD_NAME}
```

### Follow Workload

```bash
xpk workload list --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
```

### Delete resources

#### Delete a specific workload

```bash
xpk workload delete --workload ${WORKLOAD_NAME} --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE}
# Or filter and delete:
xpk workload delete --cluster ${CLUSTER_NAME} --project ${PROJECT_ID} --zone ${ZONE} --filter-by-job=${USER}
```

#### Delete the entire XPK cluster

```bash
xpk cluster delete --cluster ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}
```
