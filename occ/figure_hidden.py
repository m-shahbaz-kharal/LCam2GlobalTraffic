"""Figure: what each model does where vehicles are hidden from the camera.

Picks the held-out camera views with the most hidden vehicle mass and shows, per view,
the picture, what the camera can see, what is really there, and each model's answer.
"""
import argparse
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import EPISODES, cache_paths  # noqa: E402
from .evaluate import Predictor, frame_dir  # noqa: E402
from .geometry import load_fov_mask  # noqa: E402

cv2.setNumThreads(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-visible", default="runs/r50_lss_v1/best.pt")
    ap.add_argument("--ckpt-full", default="runs/r50_lss_full_v1/best.pt")
    ap.add_argument("--cache", default="runs/cache")
    ap.add_argument("--out", default="runs/r50_lss_full_v1/report/hidden_vehicles.png")
    ap.add_argument("--episode", type=int, default=14)
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--ncols", type=int, default=4)
    args = ap.parse_args()

    fov = load_fov_mask()
    lab_vis = np.load(cache_paths(args.cache, "visible")[0], mmap_mode="r")
    lab_full = np.load(cache_paths(args.cache, "full")[0], mmap_mode="r")
    ei, f = args.episode, args.frame
    v = np.asarray(lab_vis[ei, f], dtype=np.float32) * fov[None]
    fu = np.asarray(lab_full[ei, f], dtype=np.float32) * fov[None]
    hidden = np.clip(fu - v, 0.0, None)

    pv = Predictor(args.ckpt_visible).predict_frame(ei, f) * fov[None]
    pf = Predictor(args.ckpt_full).predict_frame(ei, f) * fov[None]

    cams = np.argsort(-hidden.sum(axis=(1, 2)))[:args.ncols]
    rows = ["camera picture", "visible to camera", "actually there",
            "visible-label model", "full-label model"]
    grids = [None, v, fu, pv, pf]

    fig, axes = plt.subplots(5, len(cams), figsize=(3.6 * len(cams), 15.5))
    for c_i, cam in enumerate(cams):
        for r_i, (label, g) in enumerate(zip(rows, grids)):
            ax = axes[r_i, c_i]
            if g is None:
                img = cv2.imread(os.path.join(frame_dir(ei, f), "rgb", f"cam_{cam:03d}.jpg"))
                ax.imshow(img[:, :, ::-1])
                ax.set_title(f"cam_{cam:03d} — {hidden[cam].sum():.1f} hidden",
                             fontsize=11)
            else:
                ax.imshow(g[cam], vmin=0, vmax=1, cmap="magma")
                if c_i == 0:
                    ax.set_ylabel(label, fontsize=11)
                ax.set_title(f"{g[cam].sum():.1f} vehicles", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Held-out {EPISODES[ei]} frame {f}: vehicles the camera cannot see",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)
    print("FIGURE DONE")


if __name__ == "__main__":
    main()
