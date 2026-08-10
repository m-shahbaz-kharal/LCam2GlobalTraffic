"""Final evaluation: local occupancy scores on val episodes, then global-map
composition scores, with an oracle (GT local maps) ceiling for comparison."""
import argparse
import json
import os
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import torch

from .data import EPISODES, N_CAMS, TRAIN_EPS, VAL_EPS, cache_paths
from .geometry import (DATA_ROOT, compose_max, compose_tables, coverage_map,
                       frustum_cell_tables, load_cameras, load_fov_mask, load_info, L)
from .model import LSSOcc
from .train import MEAN, STD, MetricAccum, OCC_THR

cv2.setNumThreads(0)

# local-grid row -> distance band (row 0 farthest);  fwd = 64 - (row+0.5)*4
BANDS = {"near_8-24m": (10, 16), "mid_24-40m": (6, 10), "far_40-64m": (0, 6)}


def frame_dir(ei, f):
    return os.path.join(DATA_ROOT, "episodes", EPISODES[ei], f"frame_{f:05d}")


class Predictor:
    def __init__(self, ckpt, device="cuda"):
        K, cams = load_cameras()
        tables, tables_flip = frustum_cell_tables(K, cams)
        self.model = LSSOcc(tables, tables_flip, pretrained=False).to(device)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"], strict=False)
        self.model.eval()
        self.device = device
        self.cam_ids = torch.arange(N_CAMS, device=device)
        self.flips = torch.zeros(N_CAMS, dtype=torch.bool, device=device)
        self.pool = ThreadPool(16)
        self.mean = MEAN.to(device)
        self.std = STD.to(device)

    @torch.no_grad()
    def predict_frame(self, ei, f):
        """All 141 camera predictions for one frame -> [141,16,16] float32 (numpy)."""
        paths = [os.path.join(frame_dir(ei, f), "rgb", f"cam_{c:03d}.jpg")
                 for c in range(N_CAMS)]
        imgs = self.pool.map(lambda p: cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1], paths)
        x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs))).to(self.device)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        x = torch.nn.functional.pad(x, (0, 0, 0, 4)).contiguous(
            memory_format=torch.channels_last)
        out = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, N_CAMS, 48):
                out.append(self.model(x[i:i + 48], self.cam_ids[i:i + 48],
                                      self.flips[i:i + 48]).float())
        return torch.cat(out).cpu().numpy()


def global_metrics(comp, gv, covered):
    err = comp.sum() - gv.sum()
    p, g = (comp >= OCC_THR), (gv >= OCC_THR)
    tp = (p & g).sum()
    f1 = 2 * tp / max(p.sum() + g.sum(), 1)
    return {
        "count_pred": float(comp.sum()), "count_gt": float(gv.sum()),
        "count_abs_err": float(abs(err)),
        "cell_mae_covered": float(np.abs(comp - gv)[covered].mean()),
        "f1": float(f1),
        "corr": float(np.corrcoef(comp.ravel(), gv.ravel())[0, 1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/r50_lss_v1/best.pt")
    ap.add_argument("--cache", default="runs/cache")
    ap.add_argument("--out", default="runs/r50_lss_v1/report")
    ap.add_argument("--alpha-frame-step", type=int, default=45)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    K, cams = load_cameras()
    info = load_info()
    fov = load_fov_mask()
    fov_t = torch.from_numpy(fov)
    comp_idx, gshape, sub = compose_tables(cams, info)
    covered = coverage_map(fov, comp_idx, gshape, sub) >= 1
    lab_path, cnt_path = cache_paths(args.cache)
    labels = np.load(lab_path, mmap_mode="r")
    pred = Predictor(args.ckpt)
    results = {"covered_cells": int(covered.sum()), "total_cells": int(covered.size)}

    # ---- calibration constant alpha, fitted on TRAIN frames only ----
    tr_frames = [(ei, f) for ei in TRAIN_EPS for f in range(0, 200, args.alpha_frame_step)]
    s_pred = s_orc = s_gt = 0.0
    for ei, f in tr_frames:
        pm = pred.predict_frame(ei, f) * fov[None]
        gt_local = np.asarray(labels[ei, f], dtype=np.float32) * fov[None]
        gv_count = float(np.load(os.path.join(frame_dir(ei, f),
                                              "global_occupancy_visible.npy")).sum())
        s_pred += compose_max(pm, comp_idx, gshape, sub).sum()
        s_orc += compose_max(gt_local, comp_idx, gshape, sub).sum()
        s_gt += gv_count
    alpha = s_gt / max(s_pred, 1e-9)
    alpha_orc = s_gt / max(s_orc, 1e-9)
    results["alpha"] = alpha
    results["alpha_oracle"] = alpha_orc
    print(f"alpha={alpha:.4f} alpha_oracle={alpha_orc:.4f}", flush=True)

    # ---- full val pass: local metrics + global composition ----
    acc = MetricAccum()
    band_acc = {b: MetricAccum() for b in BANDS}
    band_masks = {}
    for b, (r0, r1) in BANDS.items():
        m = np.zeros_like(fov)
        m[r0:r1] = fov[r0:r1]
        band_masks[b] = torch.from_numpy(m)
    gstats, ostats = [], []
    sample_preds = {}
    val_frames = [(ei, f) for ei in VAL_EPS for f in range(200)]
    for k, (ei, f) in enumerate(val_frames):
        pm_raw = pred.predict_frame(ei, f)
        t = torch.from_numpy(np.asarray(labels[ei, f], dtype=np.float32))
        p = torch.from_numpy(pm_raw)
        acc.update(p, t, fov_t)
        for b, bm in band_masks.items():
            band_acc[b].update(p, t, bm)
        pm = pm_raw * fov[None]
        gv = np.load(os.path.join(frame_dir(ei, f),
                                  "global_occupancy_visible.npy")).astype(np.float64)
        comp = compose_max(pm, comp_idx, gshape, sub) * alpha
        gstats.append(global_metrics(comp, gv, covered))
        orc = compose_max(np.asarray(labels[ei, f], np.float32) * fov[None],
                          comp_idx, gshape, sub) * alpha_orc
        ostats.append(global_metrics(orc, gv, covered))
        if (ei, f) in [(13, 100), (14, 100), (15, 50)]:
            sample_preds[(ei, f)] = (comp, gv, pm_raw)
        if k % 100 == 0:
            print(f"val frame {k}/{len(val_frames)}", flush=True)

    results["local"] = acc.results()
    results["local_by_band"] = {b: a.results() for b, a in band_acc.items()}

    def summarize(stats):
        keys = stats[0].keys()
        out = {k: float(np.mean([s[k] for s in stats])) for k in keys}
        out["count_err_pct"] = 100.0 * np.mean(
            [s["count_abs_err"] / max(s["count_gt"], 1) for s in stats])
        return out
    results["global"] = summarize(gstats)
    results["global_oracle_ceiling"] = summarize(ostats)

    with open(os.path.join(args.out, "results.json"), "w") as fp:
        json.dump(results, fp, indent=2)
    print(json.dumps(results, indent=2))

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (ei, f), (comp, gv, pm_raw) in sample_preds.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        vmax = max(gv.max(), 0.5)
        axes[0].imshow(gv, vmax=vmax, cmap="magma")
        axes[0].set_title(f"ground truth (visible) — {EPISODES[ei]} frame {f}")
        axes[1].imshow(comp, vmax=vmax, cmap="magma")
        axes[1].set_title("composed from 141 predicted local maps")
        for a in axes:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"global_{EPISODES[ei]}_f{f}.png"), dpi=130)
        plt.close(fig)
        cams_busy = np.argsort(-np.asarray(labels[ei, f]).sum(axis=(1, 2)))[:4]
        fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        for col, c in enumerate(cams_busy):
            img = cv2.imread(os.path.join(frame_dir(ei, f), "rgb", f"cam_{c:03d}.jpg"))
            axes[0, col].imshow(img[:, :, ::-1])
            axes[0, col].set_title(f"cam_{c:03d}")
            axes[1, col].imshow(np.asarray(labels[ei, f, c], np.float32) * fov,
                                vmin=0, vmax=1, cmap="magma")
            axes[1, col].set_title("truth")
            axes[2, col].imshow(pm_raw[c] * fov, vmin=0, vmax=1, cmap="magma")
            axes[2, col].set_title("prediction")
        for a in axes.ravel():
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"local_{EPISODES[ei]}_f{f}.png"), dpi=130)
        plt.close(fig)
    print("EVAL DONE")


if __name__ == "__main__":
    main()
