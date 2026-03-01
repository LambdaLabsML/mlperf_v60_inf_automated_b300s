#!/usr/bin/env bash
# =============================================================================
# MLPerf Inference Setup — GPT-OSS-120B on 8xB300
#
# Usage:
#   ./setup_mlperf.sh --work-dir /mlperf
#
# This script downloads data and models, clones the automation repo, and
# launches the benchmark pipeline. Every path is derived from --work-dir.
#
# Directory layout (created automatically):
#
#   <work-dir>/
#   ├── scratch/
#   │   ├── data/          ← downloaded dataset
#   │   └── models/
#   │       └── gpt-oss/   ← downloaded & renamed model
#   └── mlperf_v60_inf_automated_b300s/   ← automation repo
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]  $*" >&2; }
die()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
WORK_DIR=""

for arg in "$@"; do
    case "${arg}" in
        --work-dir=*) WORK_DIR="${arg#*=}" ;;
    esac
done

# Handle --work-dir <value> (space-separated)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir) WORK_DIR="${2:?--work-dir requires a value}"; shift 2 ;;
        --work-dir=*) shift ;;
        *) shift ;;
    esac
done

[[ -n "${WORK_DIR}" ]] || die "Usage: $0 --work-dir <path>"

# -----------------------------------------------------------------------------
# Derived paths (everything comes from WORK_DIR)
# -----------------------------------------------------------------------------
SCRATCH_DIR="${WORK_DIR}/scratch"
DATA_DIR="${SCRATCH_DIR}/data"
MODELS_DIR="${SCRATCH_DIR}/models"
AUTOMATION_REPO_DIR="${WORK_DIR}/mlperf_v60_inf_automated_b300s"

DOWNLOADER_URL="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"
DATA_URI="https://inference.mlcommons-storage.org/metadata/gpt-oss-data.uri"
MODEL_URI="https://inference.mlcommons-storage.org/metadata/gpt-oss-model.uri"
AUTOMATION_REPO_URL="https://github.com/LambdaLabsML/mlperf_v60_inf_automated_b300s.git"

log "Work dir     : ${WORK_DIR}"
log "Scratch dir  : ${SCRATCH_DIR}"
log "Data dir     : ${DATA_DIR}"
log "Models dir   : ${MODELS_DIR}"

# -----------------------------------------------------------------------------
# Step 1: Download data
# -----------------------------------------------------------------------------
log "Creating scratch directories..."
mkdir -p "${DATA_DIR}" "${MODELS_DIR}"

log "Downloading dataset..."
cd "${DATA_DIR}"
bash <(curl -s "${DOWNLOADER_URL}") "${DATA_URI}" \
    || die "Dataset download failed."
log "Dataset download complete."

# -----------------------------------------------------------------------------
# Step 2: Download and rename model
# -----------------------------------------------------------------------------
log "Downloading model..."
cd "${MODELS_DIR}"
bash <(curl -s "${DOWNLOADER_URL}") "${MODEL_URI}" \
    || die "Model download failed."

if [[ -d "${MODELS_DIR}/gpt-oss_model" ]]; then
    log "Renaming gpt-oss_model -> gpt-oss..."
    mv "${MODELS_DIR}/gpt-oss_model" "${MODELS_DIR}/gpt-oss"
elif [[ -d "${MODELS_DIR}/gpt-oss" ]]; then
    log "Model directory already named gpt-oss — skipping rename."
else
    die "Expected model directory not found in ${MODELS_DIR}."
fi
log "Model ready."

# -----------------------------------------------------------------------------
# Step 3: Clone automation repo and run benchmark
# -----------------------------------------------------------------------------
cd "${WORK_DIR}"

if [[ -d "${AUTOMATION_REPO_DIR}" ]]; then
    warn "Automation repo already exists — pulling latest changes."
    git -C "${AUTOMATION_REPO_DIR}" pull --ff-only || die "Failed to pull automation repo."
else
    log "Cloning automation repo..."
    git clone "${AUTOMATION_REPO_URL}" "${AUTOMATION_REPO_DIR}" \
        || die "Failed to clone automation repo."
fi

log "Launching benchmark pipeline..."
cd "${AUTOMATION_REPO_DIR}"
chmod +x automate_pass_scratch_path5.sh
./automate_pass_scratch_path5.sh \
    --work-dir "${WORK_DIR}" \
    --scratch-path "${SCRATCH_DIR}"

log "All done."
