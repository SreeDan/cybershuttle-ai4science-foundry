# CHGNet 0.3.0 inference on Georgia Tech PACE ICE

This repo is my reproducible inference workflow for CHGNet on Georgia Tech PACE ICE / AI Makerspace.

Project: CHGNet (Crystal Hamiltonian Graph Neural Network) inference  
Platform: Georgia Tech PACE ICE / AI Makerspace, Slurm  
Model/checkpoint: MPtrj-pretrained CHGNet 0.3.0, supplied by the `chgnet==0.4.2` package  
Training: None. I only ran inference with the official pretrained model.  
Status: Completed on Georgia Tech ICE on October 10, 2026. See [docs/run_report.md](docs/run_report.md) and the files under `results/`.

I saved evidence from three Slurm jobs:

| Job | ID | Purpose | Key evidence |
|---|---:|---|---|
| Smoke inference | `6150476` | Silicon physical-consistency inference on one A40 GPU | `logs/chgnet-6150476.*`, `results/slurm-6150476/` |
| Data preparation | `6150977` | Figshare MPtrj v2 download and deterministic 24-structure subset creation | `logs/mptrj-prep-6150977.*`, `results/prep_slurm_accounting.txt`, `docs/mptrj_source_manifest.json` |
| Benchmark inference | `6151243` | CHGNet inference and archived-DFT comparison on the deterministic subset | `logs/chgnet-6151243.*`, `results/slurm-6151243/` |

## 1. What this does

CHGNet is a pretrained graph neural network for materials. Given a periodic crystal structure, it predicts energy (eV/atom), atomic forces (eV/Å), stress (GPa), and site magnetic moments (μB). The goal here is reproducible inference, not retraining or fine-tuning.

I used two checks:

1. Silicon smoke test: Build a fixed silicon crystal and check finite outputs, expected shapes, translation invariance, near-zero net force, and a finite-difference force check.
2. MPtrj comparison: Run CHGNet on a deterministic 24-structure MPtrj subset and compare against archived DFT labels.

Important limitation: the 24-structure subset is sampled from the public MPtrj archive. It is not the original CHGNet held-out test set, so I do not claim the subset MAEs reproduce the paper's official test MAEs. Details are in [docs/validation.md](docs/validation.md) and [docs/provenance.md](docs/provenance.md).

## 2. Repository contents

| File/directory | Purpose |
|---|---|
| `environment.yml`, `requirements.txt` | Pinned initial software environment and Python requirements |
| `jobs/ice_gpu.sbatch` | Single-GPU Slurm batch script for both inference modes |
| `scripts/run_inference.py` | Loads official model and checkpoint and runs either smoke or MPtrj subset inference |
| `scripts/download_mptrj.py` | Retrieves pinned Figshare article version 2 and produces source SHA-256 manifest |
| `scripts/make_subset.py` | Streams the huge dataset without loading everything into RAM and makes the deterministic subset |
| `scripts/metrics.py` | Unit-consistent DFT errors (stress sign/units, missing magnetic labels) |
| `scripts/capture_environment.py` | Logs versions, installed packages, Slurm allocation, GPU information |
| `scripts/verify_artifacts.py` | Checks required execution/validation artifacts exist and show success |
| `tests/` | Fast unit tests for metrics and deterministic sampling |
| `docs/` | Provenance, validation/limitations, troubleshooting, completed run report, and report template |
| `inputs/`, `raw/`, `results/`, `logs/` | Input subset, large source archive, outputs, and Slurm stdout/stderr (run-dependent) |

## 3. Access ICE

Before starting, connect to the GT VPN if necessary. Follow the campus's current [PACE ICE instructions](https://pace.gatech.edu/ice-cluster/). A typical SSH endpoint is:

```bash
ssh YOUR_GT_USERNAME@login-ice.pace.gatech.edu
```

Clone this repository onto the cluster and run these from the repo root:

```bash
pwd
sinfo -o '%P %G %a %l'        # allowed partitions, GPU resources, wall limits
pace-quota                     # check storage allocation
module avail anaconda3        # verify installed environment module
```

PACE ICE uses Slurm (`sbatch`, `squeue`, `sacct`). Your QoS may be `pace-ice`, `coc-ice`, or another value depending on your class/account. My successful run used `coc-ice`. If your account needs a partition or account flag, add the real one assigned to you.

`jobs/ice_gpu.sbatch` requests 1 GPU, 4 CPU cores, 16 GB RAM, and 1 hour. The actual GPU, runtime, and Slurm allocation are recorded in the result folders.

## 4. Create the environment (once)

On a login node, use the installed module to create the small Python environment. Avoid running actual CHGNet inference on the login node.

```bash
module load anaconda3
# Most ICE shells support conda after loading the module:
eval "$(conda shell.bash hook)"
conda env create -f environment.yml
conda activate chgnet-repro

python --version
python -m pip check
python -m pytest -q
python - <<'PY'
import torch, chgnet
print("torch", torch.__version__, "chgnet", chgnet.__version__)
PY
```

Target: Python 3.11 and the packages listed in `environment.yml`. The environment file pins the direct dependencies. Each run also saves the full resolved package list in `pip_freeze.txt`.

If a CUDA PyTorch wheel is not selected or compatible, consult [troubleshooting](docs/troubleshooting.md) and record the repair. Do not quietly change package versions without updating these files and documenting it.

## 5. Submit the silicon smoke job

First, create the Slurm stdout/stderr directory before submitting (Slurm must open log paths before the job itself starts):

```bash
mkdir -p logs results
sbatch --qos=pace-ice jobs/ice_gpu.sbatch smoke
# CoC accounts may instead require: sbatch --qos=coc-ice jobs/ice_gpu.sbatch smoke
```

Save the job ID returned by `sbatch` (denoted `JOBID` below). Monitor and inspect:

```bash
squeue -u "$USER"
cat logs/chgnet-JOBID.out
cat logs/chgnet-JOBID.err
sacct -j JOBID \
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS -P \
  > results/slurm-JOBID/slurm_accounting.txt
python scripts/verify_artifacts.py --run results/slurm-JOBID --mode smoke
```

Substitute the actual number for `JOBID`. The generated files are:

```text
results/slurm-JOBID/
  environment.json            # hostname and allocated Slurm resources
  pip_freeze.txt               # exact installed package versions
  nvidia_smi.txt               # actual GPU and driver
  slurm_job_info.txt           # Slurm resource request and allocation
  slurm_accounting.txt         # post-completion sacct state and elapsed runtime
  artifact_checksums.sha256    # SHA-256 hashes for core run artifacts
  run_metadata.json            # loaded checkpoint SHA-256, model/software versions, runtime
  smoke_predictions.json       # Si input structure + energy, forces, stress, moments
  smoke_validation.json        # expected vs observed physical checks + pass/fail
logs/
  chgnet-JOBID.out             # stdout
  chgnet-JOBID.err             # stderr
```

The smoke job exits nonzero if one of the checks fails. This confirms the model is behaving sensibly for a simple case, but it does not by itself prove DFT accuracy.

## 6. Run the MPtrj benchmark

This is the stronger validation step because it compares against archived DFT labels. The MPtrj archive is large, about 11.35 GB, so keep it on scratch/shared storage and do not download it inside every GPU job.

Step A: Download exact archived version (once). On an appropriate data-transfer/login system with internet access:

```bash
conda activate chgnet-repro
mkdir -p /PATH/TO/YOUR/ALLOWED/SCRATCH/chgnet
python scripts/download_mptrj.py \
  --output /PATH/TO/YOUR/ALLOWED/SCRATCH/chgnet/mptrj_source
```

The script downloads Figshare article 23713842 version 2, checks the byte count, and writes a manifest with the file ID, source URL, retrieval time, and SHA-256. If ICE cannot download it directly, download it from [Figshare](https://doi.org/10.6084/m9.figshare.23713842.v2), copy it to scratch, and keep the checksum.

For my run, I used the provided CPU prep job, which does the download and subset selection together:

```bash
mkdir -p logs results
sbatch --qos=coc-ice jobs/prepare_mptrj.sbatch
```

That job stores the large source file in `$HOME/scratch/chgnet-mptrj/`, copies the manifest into `docs/mptrj_source_manifest.json`, writes `inputs/mptrj_subset.json`, and records `inputs/mptrj_subset.sha256`. If your scratch path or QoS is different, change only those site-specific parts and record it.

Step B: Create a deterministic, small subset. This uses a streaming JSON parser (`ijson`) to avoid loading the full dataset into system RAM:

```bash
python scripts/make_subset.py \
  --input /PATH/TO/YOUR/ALLOWED/SCRATCH/chgnet/mptrj_source \
  --output inputs/mptrj_subset.json \
  --count 24 --seed 42 --min-atoms 2 --max-atoms 64
sha256sum inputs/mptrj_subset.json
```

The selector chooses at most one frame per material ID. It ranks materials using SHA-256 of `42:material_id` and keeps the 24 lowest ranks. This avoids hand-picking examples. The subset JSON includes the source checksum, seed, selection method, and selected IDs.

Step C: Run benchmark inference on a GPU.

```bash
mkdir -p logs results
sbatch --qos=pace-ice jobs/ice_gpu.sbatch benchmark inputs/mptrj_subset.json
```

After completion:

```bash
cat logs/chgnet-JOBID.out
cat results/slurm-JOBID/benchmark_metrics.json
sacct -j JOBID \
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS -P \
  > results/slurm-JOBID/slurm_accounting.txt
python scripts/verify_artifacts.py --run results/slurm-JOBID --mode benchmark
```

The main outputs are `benchmark_predictions.json`, `benchmark_metrics.json`, and `artifact_checksums.sha256`. See [docs/validation.md](docs/validation.md) for the units and caveats.

## 7. Evidence to keep

For each completed Slurm job, record scheduler usage:

```bash
sacct -j JOBID \
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,MaxRSS -P \
  > results/slurm-JOBID/slurm_accounting.txt
```

Run that after the job finishes. For this submission I kept:

- Real `logs/chgnet-JOBID.out` and `.err` (even an empty `.err` is useful).
- Real `results/slurm-JOBID/` containing checksums, package versions, GPU info, predictions, scientific checks, and elapsed time.
- Real `slurm_accounting.txt` with Slurm state, allocated resources and elapsed time.
- Completed [run report](docs/run_report.md): observed metrics, resource details, reproducibility notes, validation conclusions, and genuine failures/troubleshooting.

Never commit the full 11.35 GB archive. The small subset is included here for reproducibility. If redistribution rules change, regenerate it from the documented source and checksum instead.

## 8. How I checked the results

| Test | Expected outcome | How actual outcome is recorded |
|---|---|---|
| Software environment | Pinned dependencies import and `pytest` passes | `pip_freeze.txt`, pytest stdout/log |
| Model traceability | CHGNet 0.3.0 loaded from installed checkpoint | `run_metadata.json` checkpoint path and SHA-256 |
| Inference smoke | Output energy, forces, stress, moments have correct units/shapes and are finite | `smoke_predictions.json` |
| Translation invariance | Δenergy < 0.001 eV/atom for a rigid periodic translation | `smoke_validation.json` |
| Net force consistency | ‖Σatomic forces‖ < 0.1 eV/Å | `smoke_validation.json` |
| Finite difference | Force from gradient agrees with `-(E_total(x+h)-E_total(x-h))/(2h)` | `smoke_validation.json` |
| DFT comparison | Report actual energy/force/stress/moment MAEs with valid units and counts | `benchmark_metrics.json` |
| Execution | Slurm exit code, elapsed time, compute resources, stdout/stderr | `slurm_accounting.txt`, `logs/`, `environment.json` |

The smoke-test thresholds were fixed before running. If a check fails, keep the failed output and explain it instead of changing the tolerance after the fact.

## 9. Troubleshooting, independent reproduction, and cleanup

See [docs/troubleshooting.md](docs/troubleshooting.md) for the problems I expected: QoS/account issues, GPU allocation, PyTorch CUDA, quota, and MPtrj download problems.

After finishing, cancel unused jobs, remove the large raw archive from scratch if it can be regenerated, and keep the small repo evidence needed for grading.

## 10. Primary scientific sources

- Deng et al. (2023), *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling*, Nature Machine Intelligence. https://doi.org/10.1038/s42256-023-00716-3
- Official model source: https://github.com/CederGroupHub/chgnet
- Official CHGNet 0.3.0 checkpoint and test MAEs: https://github.com/CederGroupHub/chgnet/tree/main/chgnet/pretrained/0.3.0
- MPtrj archived version 2: https://doi.org/10.6084/m9.figshare.23713842.v2
- PACE ICE: https://pace.gatech.edu/ice-cluster/

Review usage, citation and redistribution requirements in the CHGNet license, Figshare record, and Materials Project terms before publishing derived data.
