#!/bin/bash
#SBATCH --job-name=occ-smoke
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:25:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic/logs/%x-%j.out
set -u
cd /lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic
PY=$HOME/software/occ-venv/bin/python
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi -L
$PY -m occ.selfcheck || exit 1
$PY -m occ.train --out runs/smoke --epochs 1 --frame-step 24 --val-frame-step 40 \
    --max-hours 0.2 --warmup-steps 100
echo SMOKE_DONE
