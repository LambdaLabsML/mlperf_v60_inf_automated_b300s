# Automating MLPerf Inference v6.0 GPT-OSS-120b Workload 
DJ Matusz

# Automation script: automate_pass_scratch_path5
This is the current iteration of the script I have, specfic to running GPT-OSS-120B on the B300s.
There are four steps that need to be done by the user:

1) Create a directory on your server somewhere at the root. Let's call it `/mlperf`.
2) Within this directory, create directory scratch, such that you now have `/mlperf/scratch`.
3) Within the scratch directory, create two more directories: `/data` and `/models`
4) cd into `/mlperf/scratch/data` and run this command to install the preprocessed data:
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/gpt-oss-data.uri
```
5) Then, cd into `/mlperf/scratch/models` and run this command to install the GPT-OSS-120b model from the MLCommons checkpoint:
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/gpt-oss-model.uri
```

Finally, clone this repo into /mlperf. Make sure the script has executable privileges:
```bash
chmod +x automate_pass_scratch_path5.sh
```

Then, run the script using `./automate_pass_scratch_path5.sh --scratch-path <wherever your scratch is located>`

This script handles the very particular bugs that were faced in running the GPT-OSS-120b workload on the B300s. I'm trying to extrapolate this script
to the B200s, and hopefully using Claude Code in the future to automate the debugging of these workloads. If any problems are encountered with this
process, no matter how minute, please slack me (I'm david.matusz) or email me at david.matusz@lambdal.com
