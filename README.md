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

Loss is FOV-mask-weighted MSE (positive cells ×10) + a count-consistency L1 term. Split by
episode: ep_000–012 train, ep_013–015 validation. Horizontal-flip augmentation (mirrors the
BEV target). Two models were trained, identical in every respect but the target:

| run | target | answers |
|---|---|---|
| `r50_lss_v1` | `local_occupancy_visible.npz` | what the camera can actually see |
| `r50_lss_full_v1` | `local_occupancy.npz` | every vehicle present, including occluded ones |

## Results (held-out episodes 13–15, 84,600 camera views)

| | visible-label model | full-label model |
|---|---|---|
| Occupied-cell F1 @0.25 | **0.917** (P .930 / R .906) | **0.871** (P .896 / R .847) |
| Vehicles per view, mean abs. error | **0.108** | **0.261** |
| Views within ±0.5 vehicles | **93.0 %** | **83.7 %** |
| Town map: occupied-cell F1 | **0.808** (ceiling 0.825) | **0.783** (ceiling 0.824) |
| Town vehicle count error | **1.9 %** | **2.0 %** |

Each is scored against its own target, so the columns are not directly comparable — the
full-label model is answering a strictly harder question (a quarter of the vehicles in a
typical view are hidden from that camera).

**Hidden-vehicle inference** (`occ/compare.py`) settles whether the full-label model really
infers what it cannot see. Over 8,272 held-out cells that hold a vehicle with *no visible
trace of it*:

| | hidden cells found | hidden mass recovered | visible-cell recall |
|---|---|---|---|
| visible-label model | 3.3 % | 3.6 % | 90.6 % |
| full-label model | **38.7 %** | **40.3 %** | 90.1 % |

The full-label model recovers ~40 % of what is invisible to the camera — 12× the
visible-label model's incidental rate — while giving up 0.5 pp on cells it can plainly see.

Global maps are composed by splatting each camera's masked grid into the town grid,
taking the per-cell **max** over cameras (union semantics — a vehicle seen by several
cameras counts once), then scaling by a calibration constant fitted on training episodes
(0.887 visible / 0.860 full).

## Layout

```
occ/geometry.py    frustum tables, global composition, FOV mask
occ/data.py        label cache + dataset
occ/model.py       LSSOcc network
occ/train.py       training loop + masked metrics
occ/selfcheck.py   pre-training geometry checks against dataset labels
occ/evaluate.py    final local + global evaluation, calibration, figures
occ/gif_infer.py   cache predictions for a run of sequential frames (GPU)
occ/gif_render.py  the three explanation clips, .mp4 and .gif (CPU only)
slurm/             smoke.sbatch, train.sbatch, train_full.sbatch, gif_infer.sbatch
runs/<run>/        checkpoints, log.jsonl, report/ (results.json + figures)
runs/clips_<run>/  the clips below, one folder per checkpoint
```

## Animations

Two stages, so the layout can be iterated on without a GPU: `gif_infer` writes every
camera's predicted grid for a run of consecutive frames, `gif_render` draws from that
cache. Frames in the dataset are **one simulated second apart**, so 40 frames played at
4 fps is 40 s of town traffic in a 10 s clip.

Clips live in `runs/clips_<run>/` and each frame is stamped with the checkpoint that drew
it, so the two models' clips can't be mixed up:

```bash
PY=$HOME/software/occ-venv/bin/python
CAMS=89,91,2,17,92,84,85,114        # same eight cameras for every run, so clips line up

# visible-labels model
sbatch --export=ALL,RUN=r50_lss_v1 slurm/gif_infer.sbatch      # ~1 min on an H100
$PY -m occ.gif_render --preds runs/clips_r50_lss_v1/preds_ep015_f000-039.npz \
    --out-dir runs/clips_r50_lss_v1 --labels visible --hero-cams $CAMS --cam 69

# full-labels model (infers vehicles hidden behind other vehicles)
sbatch --export=ALL,RUN=r50_lss_full_v1 slurm/gif_infer.sbatch
$PY -m occ.gif_render --preds runs/clips_r50_lss_full_v1/preds_ep015_f000-039.npz \
    --out-dir runs/clips_r50_lss_full_v1 --labels full --hero-cams $CAMS --cam 69
```

`--labels` picks the ground truth each model is judged against — `visible` compares to
`global_occupancy_visible.npy`, `full` to `global_occupancy.npy` — and rewords the
captions to match. The composition constant α is read from that run's own
`report/results.json` (0.887 visible, 0.860 full) unless `--alpha` overrides it.

Each clip is written twice: **`.mp4`** (H.264, yuv420p, 24 fps with a keyframe every
second — pausable and scrubbable, which is what you want when talking over it) and
**`.gif`** (autoplays in a README or a chat window). The mp4 is encoded from the same
true-colour frames, not transcoded from the 200-colour GIF. Source frames are *held*, not
interpolated — the data really is 1 Hz.

| file | what it shows |
|---|---|
| `town_from_cameras.{mp4,gif}` | the town map in the middle, eight camera feeds around it, a coloured beam from each feed to the slice of the map that camera fills in |
| `town_pred_vs_truth.{mp4,gif}` | composed prediction next to the ground-truth town map, same colour scale |
| `camera_069_to_grid.{mp4,gif}` | one camera picture next to its predicted 16×16 grid (cyan outline = truth) |

The cluster's `ffmpeg` modules have no libx264 (the CUDA build's `h264_nvenc` needs a GPU),
so `--formats mp4` uses a static ffmpeg at `~/software/bin/ffmpeg`, extracted from the
`imageio-ffmpeg` wheel. Point `--ffmpeg` elsewhere if you have your own.

Without `--hero-cams`, the eight cameras in the first clip are chosen automatically:
highest *churn* (occupancy mass that changes from second to second, so moving traffic
rather than parked cars), spread north–south over each half of town. Pass the same list to
every run when you want the clips to be comparable. `--cam` picks the single camera for
the third clip — camera 069 is a junction with through traffic rather than parked cars.

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
