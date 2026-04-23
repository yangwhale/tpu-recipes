#!/bin/bash

################################################################################
# SCRIPT SUMMARY: GKE TPU v7 Resource Cleanup
# 
# PURPOSE:
# This script automates the deletion and cleanup of all GCP resources created 
# during the GKE TPU v7 setup process (typically following clustersetup_gke_tpuv7.sh).
#
# RESOURCES CLEANED UP:
# 1. GKE Cluster (Zonal/Regional)
# 2. Compute Resource Policies (Workload Policies for TPU maintenance)
# 3. Cloud NAT Configuration
# 4. Cloud Router
# 5. VPC Firewall Rules
# 6. VPC Subnets
# 7. VPC Network
#
# KEY VARIABLES:
# - PROJECT_ID: The GCP project where resources reside.
# - REGION/ZONE: The geographical location of the resources.
# - RESOURCE_NAME: The base name used to identify resources (prefix matching).
#
# USAGE:
# Ensure the configuration variables below match those used during setup.
# Run: ./clustercleanup_gke_tpuv7.sh
################################################################################

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

################################################################################
# COLOR OUTPUT
################################################################################

RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

function print_header() {
    echo -e "\\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\\n"
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

################################################################################
# CONFIGURATION - Should match setup script
################################################################################

# GCP Project Configuration
export PROJECT_ID="<PROJECT_ID>"              # REQUIRED: Your GCP project ID
export REGION="<REGION>"                      # GCP region for resources
export ZONE="<ZONE>"                      # GCP zone for zonal resources
export RESOURCE_NAME="<RESOURCE_NAME>"
export CLUSTER_NAME=${RESOURCE_NAME}-gke
export NETWORK_NAME=${RESOURCE_NAME}-privatenetwork
export SUBNET_NAME=${RESOURCE_NAME}-privatesubnet
export NETWORK_FW_NAME=${RESOURCE_NAME}-privatefirewall
export ROUTER_NAME=${RESOURCE_NAME}-network
export NAT_CONFIG=${RESOURCE_NAME}-natconfig


################################################################################
# Main Cleanup Script
################################################################################

print_header "Starting resource cleanup for ${RESOURCE_NAME}"

# Set project context
print_info "Setting GCP project to ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}
print_info "Setting compute zone to ${ZONE}"
gcloud config set compute/zone ${ZONE}

# Delete GKE Cluster
print_info "Deleting GKE Cluster: ${CLUSTER_NAME}..."
gcloud container clusters delete ${CLUSTER_NAME} \
  --location=${REGION} \
  --project=${PROJECT_ID} \
  --quiet || print_warning "GKE Cluster ${CLUSTER_NAME} not found or already deleted."

# Delete Workload Policies starting with RESOURCE_NAME
print_info "Finding Workload Policies starting with ${RESOURCE_NAME}..."
POLICIES_WITH_REGIONS=$(gcloud compute resource-policies list \
  --filter="name ~ ^${RESOURCE_NAME}" \
  --format="value(name, region.basename())" \
  --project=${PROJECT_ID} 2>/dev/null || true)

if [ -n "$POLICIES_WITH_REGIONS" ]; then
    echo "Found the following policies to delete:"
    echo "$POLICIES_WITH_REGIONS"
    
    # Simple confirmation check
    echo -n "Are you sure you want to delete these workload policies? (y/n) "
    read -r confirmation
    if [[ ! "$confirmation" =~ ^[Yy]$ ]]; then
        print_info "Skipping workload policy deletion."
    else
        while read -r policy region; do
            [ -z "$policy" ] && continue
            print_info "Deleting Workload Policy: ${policy} in region ${region}..."
            gcloud compute resource-policies delete "${policy}" \
              --project="${PROJECT_ID}" \
              --region="${region}" \
              --quiet
        done <<< "$POLICIES_WITH_REGIONS"
    fi
else
    print_warning "No Workload Policies found starting with ${RESOURCE_NAME}."
fi

# Delete NAT Config
print_info "Deleting NAT Config: ${NAT_CONFIG} in router ${ROUTER_NAME}..."
gcloud compute routers nats delete "${NAT_CONFIG}" \
  --router="${ROUTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet || print_warning "NAT Config ${NAT_CONFIG} not found or already deleted."

# Delete Router
print_info "Deleting Router: ${ROUTER_NAME}..."
gcloud compute routers delete "${ROUTER_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --quiet || print_warning "Router ${ROUTER_NAME} not found or already deleted."

# Delete Firewall Rule
print_info "Deleting Firewall Rule: ${NETWORK_FW_NAME}..."
gcloud compute firewall-rules delete ${NETWORK_FW_NAME} \
  --project="${PROJECT_ID}" \
  --quiet || print_warning "Firewall Rule ${NETWORK_FW_NAME} not found or already deleted."

# Delete Subnet
print_info "Deleting Subnet: ${SUBNET_NAME}..."
gcloud compute networks subnets delete "${SUBNET_NAME}" \
  --region="${REGION}" \
  --project=$PROJECT_ID \
  --quiet || print_warning "Subnet ${SUBNET_NAME} not found or already deleted."

# Delete Network
print_info "Deleting Network: ${NETWORK_NAME}..."
gcloud compute networks delete ${NETWORK_NAME} \
  --project=${PROJECT_ID} \
  --quiet || print_warning "Network ${NETWORK_NAME} not found or already deleted."

print_success "Cleanup script finished for ${RESOURCE_NAME}"
