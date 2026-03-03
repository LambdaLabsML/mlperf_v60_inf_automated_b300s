# MLPerf Inference v6.0 — GPT-OSS-120B on 8xB300

## Quick start

```bash
export WORK_DIR=/home/ubuntu/mlperf

curl -fsSL https://raw.githubusercontent.com/LambdaLabsML/mlperf_v60_inf_automated_b300s/master/run_gpt_oss.sh \
  | bash -s -- --work-dir "$WORK_DIR"
```

This single command runs the full pipeline:

1. Downloads the GPT-OSS dataset and model (~1.5 hours at ~24 MB/s)
2. Clones and pins the NVIDIA MLPerf partner repo
3. Builds the Docker image
4. Runs 4 benchmarks (Server/Offline x Performance/Accuracy)
5. Runs audit tests for Server and Offline

## Flags

| Flag | Description |
|------|-------------|
| `--work-dir <path>` | **(required)** Working directory for repo, data, and models |
| `--skip-download` | Skip data/model download (use on re-runs) |
| `--skip-benchmark` | Skip all 4 benchmark runs |
| `--skip-audit` | Skip audit tests |
| `--audit-scenarios=<list>` | Comma-separated audit scenarios (default: `Server,Offline`) |

## Examples

Re-run without re-downloading data:

```bash
curl -fsSL https://raw.githubusercontent.com/LambdaLabsML/mlperf_v60_inf_automated_b300s/master/run_gpt_oss.sh \
  | bash -s -- --work-dir "$WORK_DIR" --skip-download
```

Run only audit tests for the Server scenario:

```bash
curl -fsSL https://raw.githubusercontent.com/LambdaLabsML/mlperf_v60_inf_automated_b300s/master/run_gpt_oss.sh \
  | bash -s -- --work-dir "$WORK_DIR" --skip-download --skip-benchmark --audit-scenarios=Server
```
