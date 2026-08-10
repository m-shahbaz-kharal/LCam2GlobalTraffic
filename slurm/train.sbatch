#!/bin/bash
#SBATCH --job-name=occ-train
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=07:00:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic/logs/%x-%j.out
set -u
cd /lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic
PY=$HOME/software/occ-venv/bin/python
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi -L
$PY -m occ.train --out runs/r50_lss_v1 --epochs 8 --max-hours 4.6
$PY -m occ.evaluate --ckpt runs/r50_lss_v1/best.pt --out runs/r50_lss_v1/report
echo TRAIN_JOB_DONE
