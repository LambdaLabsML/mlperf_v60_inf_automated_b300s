#!/usr/bin/env bash
# =============================================================================
# MLPerf Inference Benchmark — GPT-OSS-120B on 8xB300
#
# Usage:
#   export WORK_DIR=/home/ubuntu/mlperf
#   curl -fsSL https://raw.githubusercontent.com/LambdaLabsML/mlperf_v60_inf_automated_b300s/master/run_gpt_oss.sh | bash -s -- --work-dir "$WORK_DIR"
#
# This script:
#   1. Downloads the GPT-OSS dataset and model
#   2. Clones the NVIDIA MLPerf partner repo
#   3. Builds the Docker image
#   4. Runs the benchmark
#
# Directory layout (created automatically):
#
#   <work-dir>/
#   ├── scratch/
#   │   ├── data/
#   │   │   └── gpt-oss/       ← preprocessed dataset
#   │   └── models/
#   │       └── gpt-oss/       ← model checkpoint
#   └── nv-mlpinf-partner/     ← NVIDIA repo
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]  $*" >&2; }
die()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; exit 1; }

cleanup_container() {
    if [[ -n "${DOCKER_CONTAINER_NAME:-}" ]]; then
        log "Stopping container '${DOCKER_CONTAINER_NAME}'..."
        docker stop "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true
        log "Container stopped."
    fi
}

build_run_args() {
    local scenario="$1"
    local test_mode="$2"
    echo "${MLPERF_RUN_ARGS_BASE} --scenarios=${scenario} --test_mode=${test_mode}"
}

build_audit_run_args() {
    local scenario="$1"
    echo "${MLPERF_AUDIT_RUN_ARGS_BASE} --scenarios=${scenario}"
}

get_docker_container_name() {
    [[ -n "${DOCKER_CONTAINER_NAME:-}" ]] || die "Container name not set. Was launch_container() called?"
    echo "${DOCKER_CONTAINER_NAME}"
}

teardown_container() {
    local container_name="$1"

    log "Stopping container '${container_name}'..."
    docker stop "${container_name}" 2>/dev/null \
        && log "Container '${container_name}' stopped." \
        || warn "Container '${container_name}' may have already exited."

    log "Removing container '${container_name}'..."
    docker rm -f "${container_name}" 2>/dev/null \
        && log "Container '${container_name}' removed." \
        || warn "Container '${container_name}' was already removed (--rm likely cleaned it up)."

    log "Verifying no residual TRT-LLM server processes remain on the host..."
    if pgrep -f trtllm_server > /dev/null 2>&1 || pgrep -f tritonserver > /dev/null 2>&1; then
        die "Residual TRT-LLM/Triton server processes detected on host after container removal. Please kill them manually before continuing."
    fi
    log "No residual TRT-LLM server processes detected."
}

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORK_DIR=""
SCRATCH_PATH=""
SKIP_DOWNLOAD=false

# Derived paths (set after argument parsing)
DATA_DIR=""
MODELS_DIR=""
DATA_ROOT=""
REPO_DIR=""
NVIDIA_DIR=""
DOCKERFILE=""

# URLs
DOWNLOADER_URL="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"
DATA_URI="https://inference.mlcommons-storage.org/metadata/gpt-oss-data.uri"
MODEL_URI="https://inference.mlcommons-storage.org/metadata/gpt-oss-model.uri"
REPO_URL="git@gitlab.com:nvidia/mlperf-inference-partner/nv-mlpinf-partner.git"

# MLPerf settings
MLPERF_SYSTEM_NAME="B300-SXM-270GBx8"

# Run configurations: "scenario:test_mode" pairs executed in order
MLPERF_RUNS=(
    "Server:PerformanceOnly"
    "Server:AccuracyOnly"
    "Offline:PerformanceOnly"
    "Offline:AccuracyOnly"
)

MLPERF_RUN_ARGS_BASE="--benchmarks=gpt-oss-120b --core_type=trtllm_endpoint"
MLPERF_AUDIT_RUN_ARGS_BASE="--benchmarks=gpt-oss-120b --core_type=trtllm_endpoint"

# TRT-LLM health check settings
TRTLLM_PORT="${TRTLLM_PORT:-30000}"
TRTLLM_HEALTH_TIMEOUT="${TRTLLM_HEALTH_TIMEOUT:-1800}"  # 30 minutes
TRTLLM_POLL_INTERVAL="${TRTLLM_POLL_INTERVAL:-10}"      # 10 seconds

# Container name (set during launch)
DOCKER_CONTAINER_NAME=""

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
parse_args() {
    for arg in "$@"; do
        case "${arg}" in
            --work-dir=*)      WORK_DIR="${arg#*=}" ;;
            --skip-download)   SKIP_DOWNLOAD=true ;;
            --work-dir|--skip-download) ;;
            *) warn "Unknown argument: ${arg}" ;;
        esac
    done

    # Handle space-separated --work-dir <value>
    local i=1
    for arg in "$@"; do
        if [[ "${arg}" == "--work-dir" ]]; then
            local next
            next=$(eval "echo \${$((i+1)):-}")
            if [[ -n "${next}" && "${next}" != --* ]]; then
                WORK_DIR="${next}"
            else
                die "--work-dir requires a value."
            fi
        fi
        ((i++))
    done

    [[ -n "${WORK_DIR}" ]] || die "Usage: $0 --work-dir <path> [--skip-download]"

    # Derive all paths from WORK_DIR
    SCRATCH_PATH="${WORK_DIR}/scratch"
    DATA_DIR="${SCRATCH_PATH}/data"
    MODELS_DIR="${SCRATCH_PATH}/models"
    DATA_ROOT="${SCRATCH_PATH}/data"
    REPO_DIR="${WORK_DIR}/nv-mlpinf-partner"
    NVIDIA_DIR="${REPO_DIR}/closed/NVIDIA"
    DOCKERFILE="${NVIDIA_DIR}/docker/Dockerfile.user"
}

# -----------------------------------------------------------------------------
# Phase 1: Download Data and Model
# -----------------------------------------------------------------------------
download_data() {
    if [[ "${SKIP_DOWNLOAD}" == true ]]; then
        log "Skipping data download (--skip-download specified)."
        return 0
    fi

    log "Creating scratch directories..."
    mkdir -p "${DATA_DIR}" "${MODELS_DIR}"

    log "Downloading dataset..."
    cd "${DATA_DIR}"
    bash <(curl -s "${DOWNLOADER_URL}") "${DATA_URI}" \
        || die "Dataset download failed."
    log "Dataset download complete."

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
}

# -----------------------------------------------------------------------------
# Phase 2: Clone NVIDIA Repo
# -----------------------------------------------------------------------------
clone_repo() {
    log "Cloning MLPerf Inference Partner repo..."
    if [[ -d "${REPO_DIR}" ]]; then
        warn "Repo already exists at ${REPO_DIR}. Pulling latest changes instead."
        git -C "${REPO_DIR}" pull --ff-only || die "Failed to pull latest changes."
    else
        git clone "${REPO_URL}" "${REPO_DIR}" || die "Failed to clone repo from ${REPO_URL}"
        cd "${REPO_DIR}" && git checkout d3091a5590a7a3dd0f4b035fa8514162bcc24ecc && cd "${WORK_DIR}"
    fi
    log "Repo ready at ${REPO_DIR}."
}

# -----------------------------------------------------------------------------
# Phase 3: Patch Dockerfile
# -----------------------------------------------------------------------------
patch_dockerfile() {
    log "Patching Dockerfile.user — commenting out 'mv /home/ubuntu' line..."

    [[ -f "${DOCKERFILE}" ]] || die "Dockerfile not found at: ${DOCKERFILE}"

    sed -i 's|    mv /home/ubuntu /home/$USER && \\|    # mv /home/ubuntu /home/$USER \&\& \\|' \
        "${DOCKERFILE}"

    if grep -q "# mv /home/ubuntu /home/\$USER" "${DOCKERFILE}"; then
        log "Patch applied successfully."
    else
        die "Patch verification failed — line may not have matched. Please check ${DOCKERFILE} manually."
    fi
}

# -----------------------------------------------------------------------------
# Phase 4: Prepare Data Directory Structure
# -----------------------------------------------------------------------------
prepare_data() {
    log "Preparing data directory structure..."
    log "  Data root: ${DATA_ROOT}"

    local src="${DATA_ROOT}/gpt-oss_data"
    local dst="${DATA_ROOT}/gpt-oss/v4"

    if [[ -d "${dst}" ]]; then
        log "Destination '${dst}' already exists — skipping data preparation."
        return 0
    fi

    [[ -d "${src}" ]] || die "Source data directory not found: ${src}"

    mkdir -p "${dst}"

    log "Moving ${src} -> ${dst}..."
    cp -r "${src}/." "${dst}/"

    local perf_dir="${dst}/perf"
    [[ -d "${perf_dir}" ]] || die "perf/ directory not found at: ${perf_dir}"
    log "Renaming files in perf/..."
    mv "${perf_dir}/input_ids_padded_perf_eval.npy" "${perf_dir}/input_ids_padded.npy" \
        || die "Failed to rename input_ids_padded_perf_eval.npy"
    mv "${perf_dir}/input_lens_perf_eval.npy"       "${perf_dir}/input_lens.npy" \
        || die "Failed to rename input_lens_perf_eval.npy"

    local acc_dir="${dst}/acc"
    [[ -d "${acc_dir}" ]] || die "acc/ directory not found at: ${acc_dir}"
    log "Renaming files in acc/..."
    mv "${acc_dir}/input_ids_padded_acc_eval.npy" "${acc_dir}/input_ids_padded.npy" \
        || die "Failed to rename input_ids_padded_acc_eval.npy"
    mv "${acc_dir}/input_lens_acc_eval.npy"       "${acc_dir}/input_lens.npy" \
        || die "Failed to rename input_lens_acc_eval.npy"

    log "Removing original source directory: ${src}..."
    rm -rf "${src}"

    log "Data preparation complete."
}

# -----------------------------------------------------------------------------
# Phase 5: Create Symlinks
# -----------------------------------------------------------------------------
symlink_build_dirs() {
    log "Creating symlinks in build directory..."

    local build_dir="${NVIDIA_DIR}/build"
    mkdir -p "${build_dir}"

    local data_link="${build_dir}/data"
    local data_src="${SCRATCH_PATH}/data"
    if [[ -L "${data_link}" ]]; then
        log "Symlink '${data_link}' already exists — skipping."
    elif [[ -e "${data_link}" ]]; then
        die "'${data_link}' exists but is not a symlink. Please remove it manually."
    else
        ln -s "${data_src}" "${data_link}" \
            || die "Failed to create symlink: ${data_link} -> ${data_src}"
        log "Created symlink: ${data_link} -> ${data_src}"
    fi

    local models_link="${build_dir}/models"
    local models_src="${SCRATCH_PATH}/models"
    if [[ -L "${models_link}" ]]; then
        log "Symlink '${models_link}' already exists — skipping."
    elif [[ -e "${models_link}" ]]; then
        die "'${models_link}' exists but is not a symlink. Please remove it manually."
    else
        ln -s "${models_src}" "${models_link}" \
            || die "Failed to create symlink: ${models_link} -> ${models_src}"
        log "Created symlink: ${models_link} -> ${models_src}"
    fi

    log "Symlinks ready."
}

# -----------------------------------------------------------------------------
# Phase 6: Setup LFS and Prebuild
# -----------------------------------------------------------------------------
setup_lfs_and_prebuild() {
    log "Changing directory to ${NVIDIA_DIR}..."
    cd "${NVIDIA_DIR}" || die "Failed to cd to ${NVIDIA_DIR}"

    log "Initializing git submodules..."
    git submodule update --init --recursive || die "git submodule update failed."

    log "Installing and pulling Git LFS..."
    git lfs install || die "git lfs install failed."
    git lfs pull    || die "git lfs pull failed."

    log "Running make prebuild (build only) for gptoss..."
    make prebuild \
        BENCHMARK=gptoss \
        ENV=release \
        DOCKER_DETACH=1 \
        || die "make prebuild failed."
    log "Docker image build complete."
}

# -----------------------------------------------------------------------------
# Phase 7: Launch Container
# -----------------------------------------------------------------------------
launch_container() {
    local run_args="$1"

    log "Launching container in background (RUN_ARGS='${run_args}')..."
    cd "${NVIDIA_DIR}" || die "Failed to cd to ${NVIDIA_DIR}"

    local uname arch docker_tag image_name container_name
    uname=$(whoami)
    arch=$(uname -p)
    docker_tag="${uname}-${arch}"
    image_name="mlperf-inference:${docker_tag}-release-latest"
    container_name="mlperf-inference-hi-$(date +%s)"

    docker image inspect "${image_name}" > /dev/null 2>&1 \
        || die "Docker image '${image_name}' not found. Did the prebuild succeed?"

    eval "$(ssh-agent -s)" > /dev/null 2>&1 || true

    docker run -d \
        --rm \
        --name "${container_name}" \
        -w /work \
        -v "$(realpath "${NVIDIA_DIR}"):/work" \
        -v "${HOME}:/mnt/${HOME}" \
        -v "${SCRATCH_PATH}:${SCRATCH_PATH}" \
        -v /etc/timezone:/etc/timezone:ro \
        -v /etc/localtime:/etc/localtime:ro \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "${SSH_AUTH_SOCK:-/dev/null}:/run/host-services/ssh-auth.sock" \
        --cap-add SYS_ADMIN \
        --cap-add SYS_TIME \
        --shm-size=32gb \
        --ulimit memlock=-1 \
        --security-opt apparmor=unconfined \
        --security-opt seccomp=unconfined \
        --user "$(id -u)" \
        --net host \
        --device /dev/fuse \
        --gpus all \
        -e SYSTEM_NAME="${MLPERF_SYSTEM_NAME}" \
        -e RUN_ARGS="${run_args}" \
        -e MLPERF_SCRATCH_PATH="${SCRATCH_PATH}" \
        -h "${container_name:0:64}" \
        --add-host "${container_name}:127.0.0.1" \
        "${image_name}" \
        sleep infinity \
        || die "Failed to launch container '${container_name}'."

    log "Container '${container_name}' launched."
    DOCKER_CONTAINER_NAME="${container_name}"
}

# -----------------------------------------------------------------------------
# Phase 8: Run Benchmarks
# -----------------------------------------------------------------------------
wait_for_container() {
    local container_name="$1"
    local retries=10
    local wait_sec=3

    log "Waiting for container '${container_name}' to be ready..."
    for ((i=1; i<=retries; i++)); do
        if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
            log "Container '${container_name}' is running."
            return 0
        fi
        log "  Attempt ${i}/${retries} — not ready yet, waiting ${wait_sec}s..."
        sleep "${wait_sec}"
    done
    die "Container '${container_name}' did not start within expected time."
}

run_in_container() {
    local container_name="$1"
    shift
    log "Executing in container '${container_name}': $*"
    docker exec "${container_name}" bash -c "$*" \
        || die "Command failed in container: $*"
}

wait_for_trtllm_server() {
    local container_name="$1"
    local host="${2:-localhost}"
    local port="${3:-${TRTLLM_PORT}}"
    local timeout="${4:-${TRTLLM_HEALTH_TIMEOUT}}"
    local poll_interval="${TRTLLM_POLL_INTERVAL}"
    local status_interval=30

    log "Waiting for TRT-LLM server at ${host}:${port} to be healthy (timeout: ${timeout}s)..."

    local start_time
    start_time=$(date +%s)

    while true; do
        local current_time elapsed http_code
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        if [[ ${elapsed} -ge ${timeout} ]]; then
            die "TRT-LLM server did not become healthy within ${timeout} seconds."
        fi

        http_code=$(docker exec "${container_name}" \
            curl -s -o /dev/null -w "%{http_code}" "http://${host}:${port}/health" 2>/dev/null || echo "000")

        if [[ "${http_code}" == "200" ]]; then
            log "TRT-LLM server is healthy after ${elapsed} seconds."
            return 0
        fi

        if [[ $((elapsed % status_interval)) -lt ${poll_interval} ]] && [[ ${elapsed} -gt 0 ]]; then
            log "  Waiting for TRT-LLM server... (${elapsed}s elapsed, HTTP: ${http_code})"
        fi

        sleep "${poll_interval}"
    done
}

run_single_benchmark() {
    local scenario="$1"
    local test_mode="$2"
    local run_args
    run_args=$(build_run_args "${scenario}" "${test_mode}")

    log "============================================================"
    log "Starting benchmark: scenario=${scenario} test_mode=${test_mode}"
    log "  RUN_ARGS: ${run_args}"
    log "============================================================"

    launch_container "${run_args}"
    local container_name
    container_name=$(get_docker_container_name)
    wait_for_container "${container_name}"

    run_in_container "${container_name}" \
        "echo 'SYSTEM_NAME='\$SYSTEM_NAME && echo 'RUN_ARGS='\$RUN_ARGS"

    log "Starting TRT-LLM servers (make run_llm_server)..."
    run_in_container "${container_name}" "cd /work && make run_llm_server" \
        || die "make run_llm_server failed."
    log "TRT-LLM servers started."

    wait_for_trtllm_server "${container_name}" "localhost" "${TRTLLM_PORT}" "${TRTLLM_HEALTH_TIMEOUT}"

    log "Running harness (make run_harness)..."
    run_in_container "${container_name}" "cd /work && make run_harness" \
        || die "make run_harness failed."
    log "Harness run complete."

    log "Benchmark complete: scenario=${scenario} test_mode=${test_mode}"
    log "============================================================"
}

run_benchmarks() {
    log "============================================================"
    log "Beginning all benchmark runs..."
    log "============================================================"

    for run in "${MLPERF_RUNS[@]}"; do
        local scenario test_mode
        scenario="${run%%:*}"
        test_mode="${run##*:}"

        run_single_benchmark "${scenario}" "${test_mode}"

        local container_name
        container_name=$(get_docker_container_name)
        teardown_container "${container_name}"
        DOCKER_CONTAINER_NAME=""
    done

    log "============================================================"
    log "All benchmark runs completed successfully."
    log "============================================================"
}

# -----------------------------------------------------------------------------
# Phase 9: Prepare Compliance Data
# -----------------------------------------------------------------------------
prepare_compliance_data() {
    log "Preparing compliance test data for TEST07..."

    local compliance_dir="${SCRATCH_PATH}/data/gpt-oss/v4/compliance/test07"
    local input_parquet="${SCRATCH_PATH}/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet"

    if [[ -d "${compliance_dir}" && -f "${compliance_dir}/input_ids_padded.npy" && -f "${compliance_dir}/input_lens.npy" ]]; then
        log "Compliance data already exists at ${compliance_dir} — skipping."
        return 0
    fi

    if [[ ! -f "${input_parquet}" ]]; then
        die "Compliance parquet file not found at: ${input_parquet}"
    fi

    log "Creating compliance data directory: ${compliance_dir}"
    mkdir -p "${compliance_dir}"

    log "Running preprocess_compliance_data.py to generate TEST07 dataset..."
    cd "${NVIDIA_DIR}" || die "Failed to cd to ${NVIDIA_DIR}"

    python3 code/gpt-oss-120b/tensorrt/preprocess_compliance_data.py \
        --input-file "${input_parquet}" \
        --output-dir "${compliance_dir}" \
        || die "preprocess_compliance_data.py failed."

    if [[ -f "${compliance_dir}/input_ids_padded.npy" && -f "${compliance_dir}/input_lens.npy" ]]; then
        log "Compliance data prepared successfully."
    else
        die "Compliance data files were not created. Check preprocess_compliance_data.py output."
    fi
}

# -----------------------------------------------------------------------------
# Phase 10: Audit Tests
# -----------------------------------------------------------------------------
run_audit_for_scenario() {
    local scenario="$1"
    local run_args
    run_args=$(build_audit_run_args "${scenario}")

    log "============================================================"
    log "Beginning audit test pipeline for scenario=${scenario}..."
    log "============================================================"

    log "Rebuilding/re-entering container for audit tests (scenario=${scenario})..."
    cd "${NVIDIA_DIR}" || die "Failed to cd to ${NVIDIA_DIR}"

    make prebuild \
        BENCHMARK=gptoss \
        ENV=release \
        DOCKER_DETACH=1 \
        || die "make prebuild (audit ${scenario}) failed."
    log "Docker image ready for audit container."

    launch_container "${run_args}"
    local audit_container
    audit_container=$(get_docker_container_name)
    wait_for_container "${audit_container}"

    log "Staging results inside container..."
    run_in_container "${audit_container}" "cd /work && make stage_results" \
        || die "make stage_results failed."
    log "Results staged."

    log "Preprocessing compliance data for TEST07 inside container..."
    run_in_container "${audit_container}" \
        "cd /work && python3 code/gpt-oss-120b/tensorrt/preprocess_compliance_data.py \
            --input-file build/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet \
            --output-dir build/data/gpt-oss/v4/compliance/test07" \
        || die "preprocess_compliance_data.py failed."
    log "Compliance data preprocessing complete."

    log "Starting TRT-LLM servers for audit tests (make run_llm_server)..."
    run_in_container "${audit_container}" "cd /work && make run_llm_server" \
        || die "make run_llm_server (audit ${scenario}) failed."
    log "TRT-LLM servers started."

    wait_for_trtllm_server "${audit_container}" "localhost" "${TRTLLM_PORT}" "${TRTLLM_HEALTH_TIMEOUT}"

    log "Running audit harness (make run_audit_harness)..."
    run_in_container "${audit_container}" "cd /work && make run_audit_harness" \
        || die "make run_audit_harness (${scenario}) failed."
    log "Audit harness run complete."

    teardown_container "${audit_container}"
    DOCKER_CONTAINER_NAME=""

    log "============================================================"
    log "Audit tests for scenario=${scenario} completed successfully."
    log "============================================================"
}

run_audit_tests() {
    log "============================================================"
    log "Beginning audit test pipeline for all scenarios..."
    log "============================================================"

    for scenario in Server Offline; do
        run_audit_for_scenario "${scenario}"
    done

    log "============================================================"
    log "All audit tests completed successfully."
    log "============================================================"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"

    trap cleanup_container EXIT

    log "=============================================="
    log "MLPerf Inference Benchmark — GPT-OSS-120B"
    log "=============================================="
    log "  Work dir     : ${WORK_DIR}"
    log "  Scratch path : ${SCRATCH_PATH}"
    log "  System       : ${MLPERF_SYSTEM_NAME}"
    log "  Benchmark runs planned:"
    for run in "${MLPERF_RUNS[@]}"; do
        log "    - ${run%%:*} / ${run##*:}"
    done
    log ""

    download_data
    clone_repo
    patch_dockerfile
    prepare_data
    symlink_build_dirs
    prepare_compliance_data
    setup_lfs_and_prebuild
    run_benchmarks
    run_audit_tests

    log "=============================================="
    log "Pipeline complete."
    log "=============================================="
}

main "$@"
