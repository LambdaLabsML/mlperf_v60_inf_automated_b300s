# MLPerf Inference v6.0 — GPT-OSS-120B on 8xB300

```bash
export WORK_DIR=/home/ubuntu/mlperf

curl -fsSL https://raw.githubusercontent.com/LambdaLabsML/mlperf_v60_inf_automated_b300s/master/run_gpt_oss.sh | bash -s -- --work-dir "$WORK_DIR"
```

This downloads data/model, builds the Docker image, and runs the benchmark.

To re-run without re-downloading data:

```bash
curl -fsSL https://raw.githubusercontent.com/LambdaLabsML/mlperf_v60_inf_automated_b300s/master/run_gpt_oss.sh | bash -s -- --work-dir "$WORK_DIR" --skip-download
```
