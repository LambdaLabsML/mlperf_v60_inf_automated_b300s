#!/usr/bin/env bash
# =============================================================================
# MLPerf Inference Benchmarking - GPT-OSS-120B on 8xB300
# Working directory: /dj2
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configurable Variables
# -----------------------------------------------------------------------------
WORK_DIR="/dj"
SCRATCH_PATH=""          # Set via --scratch-path argument
REPO_DIR="${WORK_DIR}/nv-mlpinf-partner"
REPO_URL="git@gitlab.com:nvidia/mlperf-inference-partner/nv-mlpinf-partner.git"
NVIDIA_DIR="${REPO_DIR}/closed/NVIDIA"
DOCKERFILE="${NVIDIA_DIR}/docker/Dockerfile.user"

# Data root — overridden by --toggle_test_data
DATA_ROOT=""             # Derived from SCRATCH_PATH after argument parsing

# MLPerf env exports (injected into the container)
MLPERF_SYSTEM_NAME="B300-SXM-270GBx8"
MLPERF_RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
parse_args() {
    local toggle_test_data=false

    for arg in "$@"; do
        case "${arg}" in
            --scratch-path=*)
                SCRATCH_PATH="${arg#*=}"
                ;;
            --scratch-path)
                # Handled via shift in next iteration; use a flag approach below
                ;;
            --toggle_test_data)
                toggle_test_data=true
                ;;
            *)
                warn "Unknown argument: ${arg}"
                ;;
        esac
    done

    # Handle --scratch-path VALUE (space-separated) via positional loop
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
        ((i++))
    done

    [[ -n "${SCRATCH_PATH}" ]] || die "Missing required argument: --scratch-path <path>"

    # Derive DATA_ROOT now that SCRATCH_PATH is known
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
    log "Launching container in background..."
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
        -e RUN_ARGS="${MLPERF_RUN_ARGS}" \
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

run_benchmarks() {
    local container_name
    container_name=$(get_docker_container_name)
    wait_for_container "${container_name}"

    log "Running benchmarks inside container '${container_name}'..."

    # Verify env vars are set inside the container
    run_in_container "${container_name}" \
        "echo 'SYSTEM_NAME='\$SYSTEM_NAME && echo 'RUN_ARGS='\$RUN_ARGS"

    # Step 1: Start the TRT-LLM servers that host the model
    log "Starting TRT-LLM servers (make run_llm_server)..."
    run_in_container "${container_name}" "cd /work && make run_llm_server" \
        || die "make run_llm_server failed."
    log "TRT-LLM servers started."

    # Step 2: Wait 15 minutes for servers to load the model fully
    log "Waiting 15 minutes for TRT-LLM servers to load before running harness..."
    for i in $(seq 15 -1 1); do
        log "  Starting harness in ${i} minute(s)..."
        sleep 60
    done
    log "Wait complete."

    # Step 3: Run the performance harness
    log "Running harness (make run_harness)..."
    run_in_container "${container_name}" "cd /work && make run_harness" \
        || die "make run_harness failed."
    log "Harness run complete."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"

    log "Starting MLPerf Inference benchmark pipeline"
    log "  Working dir  : ${WORK_DIR}"
    log "  Scratch path : ${SCRATCH_PATH}"
    log "  Data root    : ${DATA_ROOT}"
    log "  SYSTEM_NAME  : ${MLPERF_SYSTEM_NAME}"
    log "  RUN_ARGS     : ${MLPERF_RUN_ARGS}"

    clone_repo
    setup_scratch_path
    patch_dockerfile
    prepare_data
    symlink_build_dirs
    setup_lfs_and_prebuild
    launch_container
    run_benchmarks

    log "Pipeline complete."
}

main "$@"
