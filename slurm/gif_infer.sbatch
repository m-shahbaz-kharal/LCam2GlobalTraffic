#!/bin/bash
#SBATCH --job-name=occ-gifinfer
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic/logs/%x-%j.out
set -u
cd /lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic/.claude/worktrees/gifs-viz
PY=$HOME/software/occ-venv/bin/python
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi -L
MAIN=/lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic
$PY -m occ.gif_infer --ckpt $MAIN/runs/r50_lss_v1/best.pt \
    --ep 15 --start 0 --n 40 --out $MAIN/runs/gifs/preds_ep015_f000-039.npz
echo GIF_JOB_DONE
