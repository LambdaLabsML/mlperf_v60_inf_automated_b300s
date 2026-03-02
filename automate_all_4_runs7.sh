#!/usr/bin/env bash
# =============================================================================
# MLPerf Inference Benchmarking - GPT-OSS-120B on 8xB300
# Working directory: /dj2
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configurable Variables
# -----------------------------------------------------------------------------
WORK_DIR=""              # Set via --work-dir argument
SCRATCH_PATH=""          # Set via --scratch-path argument
REPO_DIR=""              # Derived from WORK_DIR after argument parsing
REPO_URL="git@gitlab.com:nvidia/mlperf-inference-partner/nv-mlpinf-partner.git"
NVIDIA_DIR=""            # Derived from REPO_DIR after argument parsing
DOCKERFILE=""            # Derived from NVIDIA_DIR after argument parsing

# Data root — overridden by --toggle_test_data
DATA_ROOT=""             # Derived from SCRATCH_PATH after argument parsing

# MLPerf env exports (injected into the container)
MLPERF_SYSTEM_NAME="B300-SXM-270GBx8"

# Run configurations: "scenario:test_mode" pairs executed in order
MLPERF_RUNS=(
    "Server:PerformanceOnly"
    "Server:AccuracyOnly"
    "Offline:PerformanceOnly"
    "Offline:AccuracyOnly"
)

# Shared non-varying run args
MLPERF_RUN_ARGS_BASE="--benchmarks=gpt-oss-120b --core_type=trtllm_endpoint"

# Audit run args — no --test_mode flag, as required by MLPerf audit harness
MLPERF_AUDIT_RUN_ARGS_BASE="--benchmarks=gpt-oss-120b --core_type=trtllm_endpoint"

# Container name — set by launch_container(), consumed by downstream functions
DOCKER_CONTAINER_NAME=""

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
parse_args() {
    local toggle_test_data=false

    for arg in "$@"; do
        case "${arg}" in
            --work-dir=*)
                WORK_DIR="${arg#*=}"
                ;;
            --work-dir)
                # Handled via positional loop below
                ;;
            --scratch-path=*)
                SCRATCH_PATH="${arg#*=}"
                ;;
            --scratch-path)
                # Handled via positional loop below
                ;;
            --toggle_test_data)
                toggle_test_data=true
                ;;
            *)
                warn "Unknown argument: ${arg}"
                ;;
        esac
    done

    # Handle space-separated --key VALUE pairs via positional loop
    local i=1
    for arg in "$@"; do
        if [[ "${arg}" == "--scratch-path" ]]; then
            local next
            next=$(eval "echo \${$((i+1)):-}")
            if [[ -n "${next}" && "${next}" != --* ]]; then
                SCRATCH_PATH="${next}"
            else
                die "--scratch-path requires a value."
            fi
        fi
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

    [[ -n "${WORK_DIR}" ]]    || die "Missing required argument: --work-dir <path>"
    [[ -n "${SCRATCH_PATH}" ]] || die "Missing required argument: --scratch-path <path>"

    # Derive paths now that WORK_DIR and SCRATCH_PATH are known
    REPO_DIR="${WORK_DIR}/nv-mlpinf-partner"
    NVIDIA_DIR="${REPO_DIR}/closed/NVIDIA"
    DOCKERFILE="${NVIDIA_DIR}/docker/Dockerfile.user"

    if [[ "${toggle_test_data}" == true ]]; then
        DATA_ROOT="${WORK_DIR}/scratch_test_for_data/data"
        warn "TEST MODE: Using data root at ${DATA_ROOT}"
    else
        DATA_ROOT="${SCRATCH_PATH}/data"
    fi
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]  $*" >&2; }
die()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; exit 1; }

# Build the RUN_ARGS string for a given scenario and test mode
build_run_args() {
    local scenario="$1"
    local test_mode="$2"
    echo "${MLPERF_RUN_ARGS_BASE} --scenarios=${scenario} --test_mode=${test_mode}"
}

# Build the RUN_ARGS string for an audit run — no --test_mode flag, as required by MLPerf
build_audit_run_args() {
    local scenario="$1"
    echo "${MLPERF_AUDIT_RUN_ARGS_BASE} --scenarios=${scenario}"
}

# -----------------------------------------------------------------------------
# Pipeline Steps
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

setup_scratch_path() {
    log "Exporting MLPERF_SCRATCH_PATH=${SCRATCH_PATH}..."
    export MLPERF_SCRATCH_PATH="${SCRATCH_PATH}"
    log "MLPERF_SCRATCH_PATH set to: ${MLPERF_SCRATCH_PATH}"
}

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

prepare_data() {
    log "Preparing data directory structure..."
    log "  Data root: ${DATA_ROOT}"

    local src="${DATA_ROOT}/gpt-oss_data"
    local dst="${DATA_ROOT}/gpt-oss/v4"

    # If destination already exists, assume data has already been prepared and skip
    if [[ -d "${dst}" ]]; then
        log "Destination '${dst}' already exists — skipping data preparation."
        return 0
    fi

    [[ -d "${src}" ]] || die "Source data directory not found: ${src}"

    mkdir -p "${dst}"

    log "Moving ${src} -> ${dst}..."
    cp -r "${src}/." "${dst}/"

    # Rename files in perf/
    local perf_dir="${dst}/perf"
    [[ -d "${perf_dir}" ]] || die "perf/ directory not found at: ${perf_dir}"
    log "Renaming files in perf/..."
    mv "${perf_dir}/input_ids_padded_perf_eval.npy" "${perf_dir}/input_ids_padded.npy" \
        || die "Failed to rename input_ids_padded_perf_eval.npy"
    mv "${perf_dir}/input_lens_perf_eval.npy"       "${perf_dir}/input_lens.npy" \
        || die "Failed to rename input_lens_perf_eval.npy"

    # Rename files in acc/
    local acc_dir="${dst}/acc"
    [[ -d "${acc_dir}" ]] || die "acc/ directory not found at: ${acc_dir}"
    log "Renaming files in acc/..."
    mv "${acc_dir}/input_ids_padded_acc_eval.npy" "${acc_dir}/input_ids_padded.npy" \
        || die "Failed to rename input_ids_padded_acc_eval.npy"
    mv "${acc_dir}/input_lens_acc_eval.npy"       "${acc_dir}/input_lens.npy" \
        || die "Failed to rename input_lens_acc_eval.npy"

    log "Removing original source directory: ${src}..."
    rm -rf "${src}"

    log "Data preparation complete. Final layout:"
    find "${DATA_ROOT}/gpt-oss" -maxdepth 3 | sort | sed 's|^|  |'
}

symlink_build_dirs() {
    log "Creating symlinks in build directory..."

    local build_dir="${NVIDIA_DIR}/build"
    mkdir -p "${build_dir}"

    # Symlink data
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

    # Symlink models
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

    log "Symlinks ready:"
    log "  ${data_link}   -> $(readlink "${data_link}")"
    log "  ${models_link} -> $(readlink "${models_link}")"
}

prepare_compliance_data() {
    log "Preparing compliance test data for TEST07..."

    local compliance_dir="${SCRATCH_PATH}/data/gpt-oss/v4/compliance/test07"
    local input_parquet="${SCRATCH_PATH}/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet"

    # Check if compliance data already exists
    if [[ -d "${compliance_dir}" && -f "${compliance_dir}/input_ids_padded.npy" && -f "${compliance_dir}/input_lens.npy" ]]; then
        log "Compliance data already exists at ${compliance_dir} — skipping."
        return 0
    fi

    # Check if input parquet file exists
    if [[ ! -f "${input_parquet}" ]]; then
        die "Compliance parquet file not found at: ${input_parquet}"
    fi

    log "Creating compliance data directory: ${compliance_dir}"
    mkdir -p "${compliance_dir}"

    log "Running preprocess_compliance_data.py to generate TEST07 dataset..."
    cd "${NVIDIA_DIR}" || die "Failed to cd to ${NVIDIA_DIR}"

    # Run the preprocessing script (requires pandas and numpy, which should be available)
    python3 code/gpt-oss-120b/tensorrt/preprocess_compliance_data.py \
        --input-file "${input_parquet}" \
        --output-dir "${compliance_dir}" \
        || die "preprocess_compliance_data.py failed."

    # Verify the files were created
    if [[ -f "${compliance_dir}/input_ids_padded.npy" && -f "${compliance_dir}/input_lens.npy" ]]; then
        log "Compliance data prepared successfully:"
        ls -la "${compliance_dir}/"
    else
        die "Compliance data files were not created. Check preprocess_compliance_data.py output."
    fi
}

setup_lfs_and_prebuild() {
    log "Changing directory to ${NVIDIA_DIR}..."
    cd "${NVIDIA_DIR}" || die "Failed to cd to ${NVIDIA_DIR}"

    log "Initializing git submodules..."
    git submodule update --init --recursive || die "git submodule update failed."
    log "Submodule update complete."

    log "Installing and pulling Git LFS..."
    git lfs install || die "git lfs install failed."
    git lfs pull    || die "git lfs pull failed."
    log "Git LFS pull complete."

    # DOCKER_DETACH=1 suppresses attach_docker, so only the image gets built
    log "Running make prebuild (build only) for gptoss..."
    make prebuild \
        BENCHMARK=gptoss \
        ENV=release \
        DOCKER_DETACH=1 \
        || die "make prebuild failed."
    log "Docker image build complete."
}

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

    # Verify the image exists
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

    # Store name for downstream functions
    DOCKER_CONTAINER_NAME="${container_name}"
}

get_docker_container_name() {
    [[ -n "${DOCKER_CONTAINER_NAME:-}" ]] || die "Container name not set. Was launch_container() called?"
    echo "${DOCKER_CONTAINER_NAME}"
}

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

    # Verify no residual TRT-LLM server processes are lingering on the host
    log "Verifying no residual TRT-LLM server processes remain on the host..."
    if pgrep -f trtllm_server > /dev/null 2>&1 || pgrep -f tritonserver > /dev/null 2>&1; then
        die "Residual TRT-LLM/Triton server processes detected on host after container removal. Please kill them manually before continuing."
    fi
    log "No residual TRT-LLM server processes detected."
}

# Run a single scenario+test_mode pair end-to-end in its own container.
# Args: <scenario> <test_mode>
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

    # Verify env vars are set inside the container
    run_in_container "${container_name}" \
        "echo 'SYSTEM_NAME='\$SYSTEM_NAME && echo 'RUN_ARGS='\$RUN_ARGS"

    # Start the TRT-LLM servers that host the model
    log "Starting TRT-LLM servers (make run_llm_server)..."
    run_in_container "${container_name}" "cd /work && make run_llm_server" \
        || die "make run_llm_server failed."
    log "TRT-LLM servers started."

    # Wait 10 minutes for servers to load the model fully
    log "Waiting 10 minutes for TRT-LLM servers to load before running harness..."
    for i in $(seq 10 -1 1); do
        log "  Starting harness in ${i} minute(s)..."
        sleep 60
    done
    log "Wait complete."

    # Run the harness
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

        # Tear down between runs to ensure a clean environment for the next
        local container_name
        container_name=$(get_docker_container_name)
        teardown_container "${container_name}"
        DOCKER_CONTAINER_NAME=""
    done

    log "============================================================"
    log "All benchmark runs completed successfully."
    log "============================================================"
}

# Run the audit harness for a given scenario inside a fresh container.
# Args: <scenario>
run_audit_for_scenario() {
    local scenario="$1"
    # Audit runs must NOT include --test_mode — MLPerf requirement
    local run_args
    run_args=$(build_audit_run_args "${scenario}")

    log "============================================================"
    log "Beginning audit test pipeline for scenario=${scenario}..."
    log "============================================================"

    # ------------------------------------------------------------------
    # Step 1: Launch a fresh container for this audit run
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 2: Stage results inside the container
    # ------------------------------------------------------------------
    log "Staging results inside container..."
    run_in_container "${audit_container}" "cd /work && make stage_results" \
        || die "make stage_results failed."
    log "Results staged."

    # ------------------------------------------------------------------
    # Step 3: Preprocess compliance data for TEST07 inside container
    # ------------------------------------------------------------------
    log "Preprocessing compliance data for TEST07 inside container..."
    run_in_container "${audit_container}" \
        "cd /work && python3 code/gpt-oss-120b/tensorrt/preprocess_compliance_data.py \
            --input-file build/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet \
            --output-dir build/data/gpt-oss/v4/compliance/test07" \
        || die "preprocess_compliance_data.py failed."
    log "Compliance data preprocessing complete."

    # ------------------------------------------------------------------
    # Step 4: Start TRT-LLM servers and wait for them to be ready
    # ------------------------------------------------------------------
    log "Starting TRT-LLM servers for audit tests (make run_llm_server)..."
    run_in_container "${audit_container}" "cd /work && make run_llm_server" \
        || die "make run_llm_server (audit ${scenario}) failed."
    log "TRT-LLM servers started."

    log "Waiting 10 minutes for TRT-LLM servers to load before running audit tests..."
    for i in $(seq 10 -1 1); do
        log "  Starting audit tests in ${i} minute(s)..."
        sleep 60
    done
    log "Wait complete."

    # ------------------------------------------------------------------
    # Step 5: Run audit harness
    # ------------------------------------------------------------------
    log "Running audit harness (make run_audit_harness)..."
    run_in_container "${audit_container}" "cd /work && make run_audit_harness" \
        || die "make run_audit_harness (${scenario}) failed."
    log "Audit harness run complete."

    # ------------------------------------------------------------------
    # Step 6: Tear down audit container
    # ------------------------------------------------------------------
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

    # Run audit for both Server and Offline scenarios
    for scenario in Server Offline; do
        run_audit_for_scenario "${scenario}"
    done

    log "============================================================"
    log "All audit tests completed successfully."
    log "============================================================"
}

main() {
    parse_args "$@"

    log "Starting MLPerf Inference benchmark pipeline"
    log "  Working dir  : ${WORK_DIR}"
    log "  Scratch path : ${SCRATCH_PATH}"
    log "  Data root    : ${DATA_ROOT}"
    log "  SYSTEM_NAME  : ${MLPERF_SYSTEM_NAME}"
    log "  Benchmark runs planned:"
    for run in "${MLPERF_RUNS[@]}"; do
        log "    - ${run%%:*} / ${run##*:}"
    done

    clone_repo
    setup_scratch_path
    patch_dockerfile
    prepare_data
    symlink_build_dirs
    setup_lfs_and_prebuild
    run_benchmarks
    run_audit_tests

    log "Pipeline complete."
}

main "$@"
