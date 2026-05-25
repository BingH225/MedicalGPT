# MedicalGPT Hopper Pipeline

## Project Goal

This project packages a fast, reproducible MedicalGPT alignment pipeline for NUS Hopper. The goal is to run all four alignment stages on the same lightweight base model and keep the outputs comparable:

1. SFT on repo-native medical instruction data
2. RM on repo-native preference data
3. RLOO on the SFT policy with the trained reward model
4. DPO on the same SFT checkpoint as a direct preference baseline

The pipeline is intentionally optimized for reproducibility on Hopper rather than leaderboard quality. It defaults to `Qwen/Qwen3.5-2B`, LoRA adapters, repo-native sample data, and one-GPU PBS jobs so that the full chain can be submitted in one pass.

## Hard Constraints

- All Hopper jobs are single-GPU jobs. The pipeline does not rely on multi-GPU training.
- All stages use the same base model family: `Qwen/Qwen3.5-2B`.
- SFT is the parent checkpoint. RM, RLOO, and DPO all branch from the same SFT lineage to keep the comparison fair.

## Why This Version

- The repository already supports `Qwen3.5` and exposes `training/supervised_finetuning.py`, `training/reward_modeling.py`, `training/ppo_training.py`, and `training/dpo_training.py`.
- The repository's "PPO" stage is actually implemented with TRL `RLOOTrainer`, so the documentation and scripts in this Hopper version use the precise name `RLOO (PPO alternative)`.
- Hopper already has PBS Pro plus `apptainer`, so the remote flow uses containerized PBS jobs instead of local Windows execution.

## Local Changes

This Hopper adaptation adds:

- `scripts/hopper/common_env.sh`
- `scripts/hopper/bootstrap_remote.sh`
- `scripts/hopper/prepare_stage_data.sh`
- `scripts/hopper/run_sft_qwen35_2b.pbs`
- `scripts/hopper/run_rm_qwen35_2b.pbs`
- `scripts/hopper/run_rloo_qwen35_2b.pbs`
- `scripts/hopper/run_dpo_qwen35_2b.pbs`
- `scripts/hopper/submit_pipeline.sh`
- this runbook

It also ignores runtime-only paths in `.gitignore`:

- `artifacts/`
- `output/`
- `runtime/`
- `data/hopper_sft/`
- `data/hopper_reward/`

## Remote Layout

The Hopper project root is:

```text
/scratch/e1561245/cot_yz/medicalgpt
```

Key directories:

```text
medicalgpt/
  artifacts/hopper/merged/
  output/hopper/
  output/hopper/pbs_logs/
  runtime/hopper/
  data/hopper_sft/
  data/hopper_reward/
  scripts/hopper/
```

## Container Strategy

The remote runtime uses `apptainer` with:

- image URI: `docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- image cache path: `runtime/hopper/pytorch_2.5.1-cuda12.4-cudnn9-runtime.sif`
- Python environment path: `.venv`

`bootstrap_remote.sh` pulls the image if needed, creates the venv inside the project root, and installs:

- `requirements.txt`
- `gradio>=3.50.2`

## Data Strategy

For the first reproducible end-to-end Hopper run, the pipeline uses repo-native sample data only:

- SFT source: `data/sft/medical_sft_1K_format.jsonl`
- RM/DPO source: `data/reward/dpo_zh_500.jsonl`

`prepare_stage_data.sh` copies those files into stage-specific directories so each training stage reads only the intended dataset. This avoids accidental mixing with the other demo files already stored in `data/sft/` and `data/reward/`.

The scripts are parameterized so that the sample counts can be increased later by replacing the copied files or overriding the `*_MAX_TRAIN_SAMPLES` environment variables.

## Stage Flow

### 1. SFT

- Base model: `Qwen/Qwen3.5-2B`
- GPU shape: `1 x GPU`
- Data: `data/hopper_sft`
- Output adapter: `output/hopper/sft_<PBS_JOBID>/adapter`
- Merged checkpoint: `artifacts/hopper/merged/sft-qwen35-2b`

Purpose:

- create the instruction-following medical base policy used by RM, RLOO, and DPO

### 2. RM

- Base model family: `Qwen/Qwen3.5-2B`
- Initialization checkpoint: merged SFT checkpoint
- GPU shape: `1 x GPU`
- Data: `data/hopper_reward`
- Output adapter: `output/hopper/rm_<PBS_JOBID>/adapter`
- Merged checkpoint: `artifacts/hopper/merged/rm-qwen35-2b`

Purpose:

- create the reward model that scores preference pairs for the online alignment stage

### 3. RLOO

- Base model family: `Qwen/Qwen3.5-2B`
- Policy init: merged SFT checkpoint
- Reward model: merged RM checkpoint
- GPU shape: `1 x GPU`
- Prompt data: `data/hopper_sft`
- Output adapter: `output/hopper/rloo_<PBS_JOBID>/adapter`
- Merged checkpoint: `artifacts/hopper/merged/rloo-qwen35-2b`

Purpose:

- run the repository's RLHF-alternative online alignment stage

Important note:

- the repository implementation is `RLOO`, not classic PPO

### 4. DPO

- Base model family: `Qwen/Qwen3.5-2B`
- Policy init: merged SFT checkpoint
- GPU shape: `1 x GPU`
- Preference data: `data/hopper_reward`
- Output adapter: `output/hopper/dpo_<PBS_JOBID>/adapter`
- Merged checkpoint: `artifacts/hopper/merged/dpo-qwen35-2b`

Purpose:

- train a direct preference model for side-by-side comparison with RLOO

## Submission Order

`scripts/hopper/submit_pipeline.sh` submits the full PBS chain with dependencies:

1. SFT
2. RM depends on SFT
3. RLOO depends on both SFT and RM
4. DPO depends on SFT

This creates a single reproducible submission point while keeping the model lineage correct.

## Reproduction Steps

From Hopper:

```bash
cd /scratch/e1561245/cot_yz/medicalgpt
bash scripts/hopper/bootstrap_remote.sh
bash scripts/hopper/submit_pipeline.sh
```

Useful monitoring commands:

```bash
qstat -u e1561245
ls -la output/hopper/pbs_logs
tail -n 200 output/hopper/pipeline_submission_*.txt
tail -n 200 output/hopper/sft_<JOBID>/train.log
tail -n 200 output/hopper/rm_<JOBID>/train.log
tail -n 200 output/hopper/rloo_<JOBID>/train.log
tail -n 200 output/hopper/dpo_<JOBID>/train.log
```

## Result Collection Plan

Each stage writes:

- `meta.log`
- `nvidia_smi.txt`
- `train.log`
- `merge.log`

The merged checkpoints are stored under:

```text
artifacts/hopper/merged/
```

Recommended final comparison prompts:

- `Child has fever. What should caregivers do first?`
- `What should a patient monitor during long-term antihypertensive treatment?`
- `What dietary advice is appropriate for a patient with diabetes?`
- `Is chest pain always a heart problem?`

Recommended evaluation dimensions:

- instruction following
- medical specificity
- safety and overclaim control
- preference alignment consistency

## Expected Results

Expected qualitative behavior:

- SFT should improve medical relevance over the raw base model
- RM should converge quickly because the preference sample is small
- RLOO should become more conservative and preference-shaped than SFT
- DPO should be simpler to run and usually more stable than RLOO at this scale

Expected engineering outcome:

- a complete Hopper-native, containerized, four-stage alignment pipeline
- merged checkpoints for `sft`, `rm`, `rloo`, and `dpo`
- a PBS submission record with job IDs and per-stage logs

## Current Remote Status

Confirmed:

- SSH access to `nus_hopper`
- remote project parent path exists: `/scratch/e1561245/cot_yz`
- project directory created: `/scratch/e1561245/cot_yz/medicalgpt`
- Hopper provides `qsub`, `apptainer`, `singularity`, and `docker`

Pending after sync:

- bootstrap the container image and venv
- submit the PBS chain
- capture actual Hopper job IDs and final run status

Submitted on 2026-05-25:

- `495384.hopper-m-02` for SFT, queue state `Q`
- `495385.hopper-m-02` for RM, queue state `H`
- `495386.hopper-m-02` for RLOO, queue state `H`
- `495387.hopper-m-02` for DPO, queue state `H`

Interpretation:

- `Q` means the SFT job is queued and waiting for resources
- `H` means the downstream jobs are held by dependency until the earlier stages complete

## Risks

- `training/ppo_training.py` depends on TRL `ModelConfig` / `RLOOConfig`; if the installed TRL release changes argument names, the RLOO PBS script may need a small flag adjustment
- the containerized venv assumes the same SIF path is reused across jobs
- the sample data is sufficient for a pipeline validation run, but not for a high-quality medical model claim
- if the compute nodes cannot reach Hugging Face at runtime, model download must be pre-warmed into `/scratch/e1561245/hf_cache`
- future bootstrap runs should keep `setuptools<82` because the current `torch 2.12.0` wheel declares that upper bound
