#!/bin/bash

# --- Configuration ---
# Before running this script, please modify the environment variables below
# to match your specific GCP project and cluster setup.
# ---

# --- Environment Variables ---
export PROJECT_ID=""
export CLUSTER_NAME=""
export ZONE=""
export WORKLOAD_IMAGE=""
export WORKLOAD_NAME="$(printf "%.26s" "${USER//_/-}-sleep-inf-2x2x1")-$(date +%Y%m%d-%H%M)"

xpk workload create \
  --cluster=$CLUSTER_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --priority=very-high \
  --max-restarts=0 \
  --device-type=tpu7x-2x2x1 \
  --num-slices=1 \
  --docker-image="${WORKLOAD_IMAGE}" \
  --enable-debug-logs \
  --workload="${WORKLOAD_NAME}" \
  --command="echo 'Pod is ready. Use kubectl exec to connect.' && sleep infinity"
