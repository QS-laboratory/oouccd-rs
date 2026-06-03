#!/bin/bash
# GCP Batch entrypoint for oouccd-rs
#
# The binary handles GCS output itself:
#   - reads GCS_PATH + BATCH_TASK_INDEX env vars
#   - writes results to $GCS_PATH/task_$BATCH_TASK_INDEX/results.db directly
#
# This script only needs to: download the config, then run the binary.

set -euo pipefail

echo "=== oouccd-rs batch task ==="
echo "BATCH_TASK_INDEX : ${BATCH_TASK_INDEX}"
echo "GCS_PATH         : ${GCS_PATH}"

CONFIG_GCS="${GCS_PATH}/configs/task_${BATCH_TASK_INDEX}.json"
CONFIG_LOCAL="/tmp/task_${BATCH_TASK_INDEX}.json"

# 1. Download config from GCS
echo "[1/2] Downloading config: ${CONFIG_GCS}"
gcloud storage cp "${CONFIG_GCS}" "${CONFIG_LOCAL}"

# 2. Run the simulation (binary writes results to GCS directly via GCS_PATH env var)
echo "[2/2] Running simulation..."
/app/target/release/oouccd-rs --config-filepath "${CONFIG_LOCAL}"

echo "=== Done ==="
