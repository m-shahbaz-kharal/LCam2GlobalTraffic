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
RUN=${RUN:-r50_lss_v1}                     # which training run to visualise
EP=${EP:-15}; START=${START:-0}; N=${N:-40}
OUTDIR=$MAIN/runs/clips_$RUN
mkdir -p $OUTDIR
$PY -m occ.gif_infer --ckpt $MAIN/runs/$RUN/best.pt \
    --ep $EP --start $START --n $N \
    --out $OUTDIR/preds_ep$(printf %03d $EP)_f$(printf %03d $START)-$(printf %03d $((START+N-1))).npz
echo GIF_JOB_DONE
