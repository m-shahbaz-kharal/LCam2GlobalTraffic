#!/bin/bash
#SBATCH --job-name=occ-compare
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic/logs/%x-%j.out
set -u
cd /lustre/fs1/home/mu290328/github-repos/LCam2GlobalTraffic
$HOME/software/occ-venv/bin/python -m occ.compare
echo COMPARE_JOB_DONE
