# LCam2GlobalTraffic

One network that turns a single fixed-traffic-camera picture into a **16×16 local
vehicle-occupancy grid** (4 m cells, 64×64 m ahead of the camera), plus the machinery to
compose all 141 per-camera grids into one **town-wide occupancy map**. Built on the
[UCFTCam2TWorld dataset](/lustre/fs1/home/mu290328/datasets/UCFTCam2TWorld) (CARLA Town05,
141 cameras, 16 episodes).

## Model

Efficient-SOTA camera→BEV recipe (Lift-Splat / BEVDet / FlashOcc family), single-camera:

- ImageNet-pretrained **ResNet-50** backbone, FPN-fused stride-16 features (input 960×544)
- Per-pixel **depth distribution** (64 LID bins, 2–75 m) lifts features along camera rays
- Precomputed per-camera **frustum→cell tables** (shared intrinsics; per-camera pitch/height)
  make the lift a single `index_add_`
- Small 2D conv **BEV head** → softplus → non-negative soft counts (grid sum = vehicle count)

Trained on the *visible* labels (`local_occupancy_visible.npz`) with FOV-mask-weighted MSE
(positive cells ×10) + a count-consistency L1 term. Split by episode: ep_000–012 train,
ep_013–015 validation. Horizontal-flip augmentation (mirrors the BEV target).

## Results (held-out episodes 13–15)

| | score |
|---|---|
| Occupied-cell F1 @0.25 (local) | **0.917** (P 0.930 / R 0.906) |
| Vehicles per view, mean abs. error | **0.108** |
| Views within ±0.5 vehicles | **93.0 %** |
| Town map: occupied-cell F1 | **0.808** (oracle ceiling 0.825) |
| Town vehicle count error | **1.9 %** |

Global maps are composed by splatting each camera's masked grid into the town grid,
taking the per-cell **max** over cameras (union semantics — a vehicle seen by several
cameras counts once), then scaling by a calibration constant fitted on training episodes.

## Layout

```
occ/geometry.py    frustum tables, global composition, FOV mask
occ/data.py        label cache + dataset
occ/model.py       LSSOcc network
occ/train.py       training loop + masked metrics
occ/selfcheck.py   pre-training geometry checks against dataset labels
occ/evaluate.py    final local + global evaluation, calibration, figures
slurm/             smoke.sbatch, train.sbatch
runs/r50_lss_v1/   checkpoints, log.jsonl, report/ (results.json + figures)
```

## Reproduce

```bash
PY=$HOME/software/occ-venv/bin/python
$PY -m occ.selfcheck                 # geometry vs labels; must print SELFCHECK PASS
sbatch slurm/smoke.sbatch            # ~5 min pipeline check on a GPU node
sbatch slurm/train.sbatch            # 8 epochs (~4.2 h on one H100) + evaluation
```

Cluster notes: request GPUs with untyped `--gres=gpu:1` plus `--constraint=h100`
(typed gres allocates but does not expose the device). Node `evc46`'s GPU driver was
down on 2026-08-09 (nvidia-smi finds no device) — it is excluded in the sbatch files.
