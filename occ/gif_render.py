"""Render the three explanation clips (mp4 + gif) from cached per-frame predictions.

  clip 1  hero        town map in the middle, camera feeds around it, a coloured beam
                      from each feed to the slice of the town map that camera fills in
  clip 2  vs truth    composed prediction next to the ground-truth town map
  clip 3  one camera  a single camera picture next to its predicted local grid

CPU only: reads runs/gifs/preds_*.npz written by occ.gif_infer.
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

from .data import EPISODES
from .geometry import (CELL, DATA_ROOT, L, RANGE, compose_max, compose_tables,
                       load_cameras, load_fov_mask, load_info)

cv2.setNumThreads(0)

BG = "#0b0f16"          # figure background
PANEL = "#161c26"       # panel background
FG = "#dfe6f2"          # main text
DIM = "#8a97ad"         # secondary text
ROAD = "#2b3444"        # drivable surface on the town map
WALK = "#1d242f"

# occupancy heat: transparent -> amber -> white-hot (reads on a dark map)
HEAT = LinearSegmentedColormap.from_list(
    "heat", [(0.00, "#1a2233"), (0.15, "#3d2a6b"), (0.40, "#b3357a"),
             (0.70, "#ff8a3d"), (1.00, "#ffe9a8")])
TRUTH = LinearSegmentedColormap.from_list(
    "truth", [(0.00, "#1a2233"), (0.15, "#123f5a"), (0.40, "#1f8fa8"),
              (0.70, "#5fd6a8"), (1.00, "#e8fff0")])

CAM_COLORS = ["#ff6b6b", "#ffd166", "#8bd450", "#4ecdc4",
              "#5aa9ff", "#c792ea", "#ff9f68", "#7ee8fa"]

VMAX = 0.9              # colour scale top for a 4 m occupancy cell


# ------------------------------------------------------------------ geometry ----
def cam_footprint(cam, info):
    """The 90-degree, 64 m wedge a camera can label, as global-grid (col,row) points."""
    ox, oy = info["global_grid"]["origin_world_xy"]
    half = RANGE / 2.0
    # (forward, right) corners of the visible wedge inside the 64x64 m local square
    pts = [(0.0, 0.0), (half, half), (RANGE, half), (RANGE, -half), (half, -half)]
    yaw = math.radians(cam["yaw"])
    cy, sy = math.cos(yaw), math.sin(yaw)
    out = []
    for fw, rt in pts:
        wx = cam["x"] + fw * cy - rt * sy
        wy = cam["y"] + fw * sy + rt * cy
        out.append(((wx - ox) / CELL, (wy - oy) / CELL))
    return np.asarray(out)


def cam_xy_grid(cam, info):
    ox, oy = info["global_grid"]["origin_world_xy"]
    return ((cam["x"] - ox) / CELL, (cam["y"] - oy) / CELL)


def convex_hull(pts):
    pts = sorted(map(tuple, pts))
    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) -
                                   (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]
    return np.asarray(half(pts) + half(pts[::-1]))


# ------------------------------------------------------------------- drawing ----
def road_background(info):
    """meta/road_bev.png (binary drivable mask) on a dark palette, on the global grid."""
    GR, GC = info["global_grid"]["shape_rows_cols"]
    m = cv2.imread(os.path.join(DATA_ROOT, "meta", "road_bev.png"),
                   cv2.IMREAD_UNCHANGED).astype(np.float32)
    m = cv2.resize(m, (GC * 8, GR * 8), interpolation=cv2.INTER_AREA)
    m = np.clip(m * 1.6, 0, 1)[..., None]
    bg = np.asarray(matplotlib.colors.to_rgb(BG), np.float32)
    rd = np.asarray(matplotlib.colors.to_rgb(ROAD), np.float32)
    return bg * (1 - m) + rd * m


# view window: drop the empty margin around Town05 so the map fills its panel
VIEW = (12, 141, 123, 12)   # xmin, xmax, ymax, ymin  (grid cells)


def _glow(occ, k=9):
    g = cv2.GaussianBlur(np.clip(occ, 0, VMAX).astype(np.float32), (k, k), 0)
    return g / max(g.max(), 1e-6)


def draw_map(ax, occ, road, cmap=HEAT, vmax=VMAX, title=None, tcolor=FG):
    GR, GC = occ.shape
    ax.set_facecolor(BG)
    ax.imshow(road, extent=[0, GC, GR, 0], interpolation="bilinear", zorder=0)
    ax.imshow(np.ma.masked_less(_glow(occ), 0.04), extent=[0, GC, GR, 0], cmap=cmap,
              vmin=0.0, vmax=1.0, alpha=0.45, interpolation="bilinear", zorder=1)
    ax.imshow(np.ma.masked_less(occ, 0.02), extent=[0, GC, GR, 0], cmap=cmap,
              vmin=0.0, vmax=vmax, interpolation="nearest", zorder=2)
    ax.set_xlim(VIEW[0], VIEW[1])
    ax.set_ylim(VIEW[2], VIEW[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2c3547")
    if title:
        ax.set_title(title, color=tcolor, fontsize=12, pad=6)


def draw_local(ax, grid, fov, cmap=HEAT, vmax=VMAX, gt=None, lw=0.6):
    """16x16 local grid, camera at the bottom centre, forward = up."""
    ax.set_facecolor("#0b0f16")
    ax.imshow(np.ma.masked_where(fov <= 0, np.zeros_like(fov)), cmap=cmap,
              vmin=0, vmax=vmax, zorder=0)          # the labellable wedge, dark
    ax.imshow(np.ma.masked_less(grid * fov, 0.02), cmap=cmap, vmin=0, vmax=vmax,
              interpolation="nearest", zorder=1)
    if gt is not None:
        ax.contour(np.clip(gt * fov, 0, 1), levels=[0.15], colors=["#7ee8fa"],
                   linewidths=1.4, zorder=3)
    # visible-wedge outline
    ax.plot([7.5, -0.5, -0.5, 15.5, 15.5, 7.5], [15.5, 7.5, -0.5, -0.5, 7.5, 15.5],
            color="#48566d", lw=lw, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3a4557")


def panel_image(ax, img, color, label, sub=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(color)
        s.set_linewidth(2.0)
    ax.text(0.02, 0.94, label, transform=ax.transAxes, color=color, fontsize=10,
            fontweight="bold", va="top",
            bbox=dict(fc="#0b0f16cc", ec="none", pad=2.0))
    if sub:
        ax.text(0.98, 0.94, sub, transform=ax.transAxes, color=FG, fontsize=8.5,
                va="top", ha="right", bbox=dict(fc="#0b0f16cc", ec="none", pad=2.0))


FFMPEG = os.path.expanduser("~/software/bin/ffmpeg")


def save_mp4(frames, path, ms, ffmpeg=None, out_fps=24, crf=18):
    """H.264 clip, same timing as the GIF but scrubbable and pausable.

    Source frames are held (not interpolated) up to `out_fps`: the data really is one
    frame per simulated second, and inventing in-between motion would be a lie.
    """
    ffmpeg = ffmpeg or (FFMPEG if os.path.exists(FFMPEG) else "ffmpeg")
    h, w, _ = frames[0].shape
    src_fps = 1000.0 / ms
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{src_fps:g}",
        "-i", "-",
        # libx264 needs even dimensions in yuv420p; pad with the page background
        "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color={BG},fps={out_fps}",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-g", str(out_fps), "-movflags", "+faststart", path,
    ]
    import subprocess
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed: " + " ".join(cmd))
    print(f"wrote {path}  ({len(frames)} frames, {os.path.getsize(path)/1e6:.1f} MB)")


def save_clip(frames, path_noext, ms, formats, ffmpeg=None):
    if "gif" in formats:
        save_gif(frames, path_noext + ".gif", ms)
    if "mp4" in formats:
        save_mp4(frames, path_noext + ".mp4", ms, ffmpeg=ffmpeg)


def save_gif(frames, path, ms, colors=200):
    """One palette for the whole clip: no inter-frame flicker, and PIL can then encode
    only the rectangle that actually changed from frame to frame."""
    sample = np.concatenate(frames[::max(len(frames) // 6, 1)], axis=0)
    pal = Image.fromarray(sample).quantize(colors=colors, method=Image.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=ms, loop=0,
                 optimize=True, disposal=1)
    print(f"wrote {path}  ({len(frames)} frames, {os.path.getsize(path)/1e6:.1f} MB)")


def fig_to_rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    return buf


# ------------------------------------------------------------------- helpers ----
def frame_dir(ei, f):
    return os.path.join(DATA_ROOT, "episodes", EPISODES[ei], f"frame_{f:05d}")


def read_rgb(ei, f, cam):
    p = os.path.join(frame_dir(ei, f), "rgb", f"cam_{cam:03d}.jpg")
    return cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]


def pick_hero_cams(labels, ei, frames, cams, info, n_side=4):
    """Cameras with the most *moving* traffic, spread north-south on each half of town."""
    w = np.asarray(labels[ei, frames[0]:frames[-1] + 1], np.float32)
    churn = np.abs(np.diff(w, axis=0)).sum((2, 3)).mean(0)      # veh mass changing / s
    mean = w.sum((2, 3)).mean(0)
    score = churn * np.clip(mean, 0, 8)                          # busy *and* moving
    score[mean < 3.0] = 0.0                                      # skip near-empty views
    GR, GC = info["global_grid"]["shape_rows_cols"]
    pos = np.asarray([cam_xy_grid(c, info) for c in cams])       # (col, row)
    # mild nudge outwards so the eight footprints do not pile up in the centre
    off = np.abs(pos[:, 0] - GC / 2) / (GC / 2)
    score = score * (0.85 + 0.4 * off)
    chosen = []
    for side in (0, 1):
        sel = np.where(pos[:, 0] < GC / 2)[0] if side == 0 else np.where(pos[:, 0] >= GC / 2)[0]
        edges = np.linspace(pos[sel, 1].min(), pos[sel, 1].max() + 1e-6, n_side + 1)
        for b in range(n_side):
            band = sel[(pos[sel, 1] >= edges[b]) & (pos[sel, 1] < edges[b + 1])]
            if len(band) == 0:
                continue
            chosen.append((side, int(band[np.argmax(score[band])])))
    left = sorted([c for s, c in chosen if s == 0], key=lambda c: pos[c, 1])
    right = sorted([c for s, c in chosen if s == 1], key=lambda c: pos[c, 1])
    return left, right


# ---------------------------------------------------------------------- gif 1 ----
def gif_hero(P, out, ms):
    ei, frames, preds, alpha = P["ei"], P["frames"], P["preds"], P["alpha"]
    cams, info, fov = P["cams"], P["info"], P["fov"]
    road = P["road"]
    left, right = pick_hero_cams(P["labels"], ei, frames, cams, info)
    picks = left + right
    print("hero cameras:", picks)
    colors = {c: CAM_COLORS[i % len(CAM_COLORS)] for i, c in enumerate(picks)}
    allpos = np.asarray([cam_xy_grid(c, info) for c in cams])

    W, H = 16.0, 9.0
    pw = 0.185
    ph = pw * W / H * (9 / 16)              # keep the 16:9 picture square-true
    ys = np.linspace(0.855 - ph, 0.055, len(left))
    slots = ([(0.008, y) for y in ys] + [(1 - 0.008 - pw, y) for y in ys])

    out_frames = []
    for k, f in enumerate(frames):
        comp = compose_max(preds[k] * fov[None], P["comp_idx"], P["gshape"], P["sub"]) * alpha
        fig = plt.figure(figsize=(W, H), dpi=P["dpi"], facecolor=BG)
        axm = fig.add_axes([0.225, 0.045, 0.55, 0.815])
        draw_map(axm, comp, road)
        axm.scatter(allpos[:, 0], allpos[:, 1], s=3.0, c="#4a566b", marker="o",
                    zorder=3, linewidths=0)

        # beams: camera panel -> the slice of the town map that camera fills in
        for (c, (px, py)) in zip(picks, slots):
            col = colors[c]
            fp = cam_footprint(cams[c], info)
            axm.add_patch(Polygon(fp, closed=True, fc=col, ec=col, alpha=0.30,
                                  lw=1.2, zorder=4))
            axm.scatter([allpos[c, 0]], [allpos[c, 1]], s=42, c=col, marker="o",
                        edgecolors="white", linewidths=0.7, zorder=6)
            inner = px < 0.5
            ax_corner = (px + pw, py) if inner else (px, py)
            corners = [ax_corner, (ax_corner[0], py + ph)]
            fpf = axm.transData.transform(fp)
            fpf = fig.transFigure.inverted().transform(fpf)
            hull = convex_hull(np.vstack([np.asarray(corners), fpf]))
            fig.add_artist(Polygon(hull, closed=True, fc=col, ec=col, alpha=0.14,
                                   lw=1.1, zorder=0, transform=fig.transFigure,
                                   figure=fig))
            # thin leader over the map so panel and footprint are unmistakably paired
            cen = fpf.mean(0)
            fig.add_artist(matplotlib.lines.Line2D(
                [ax_corner[0], cen[0]], [py + ph / 2, cen[1]], color=col, lw=1.0,
                alpha=0.45, zorder=5, transform=fig.transFigure, figure=fig))

        # camera panels + their local occupancy inset
        for (c, (px, py)) in zip(picks, slots):
            col = colors[c]
            axp = fig.add_axes([px, py, pw, ph])
            n_here = float((preds[k, c] * fov).sum())
            panel_image(axp, read_rgb(ei, f, c), col, f"cam_{c:03d}",
                        sub=f"{n_here:.1f} veh")
            gs = 0.070
            gx = px + pw - gs * H / W - 0.006 if px < 0.5 else px + 0.006
            axg = fig.add_axes([gx, py + 0.007, gs * H / W, gs])
            draw_local(axg, preds[k, c], fov)
            for s in axg.spines.values():
                s.set_color(col)
                s.set_linewidth(1.2)

        fig.text(0.5, 0.988, "141 fixed traffic cameras  →  one live occupancy map of the town",
                 color=FG, fontsize=19, fontweight="bold", ha="center", va="top")
        fig.text(0.5, 0.947,
                 "each camera turns its own picture into a 16×16 local vehicle grid (small inset); "
                 "the coloured beam shows where that grid lands in the town map",
                 color=DIM, fontsize=11.0, ha="center", va="top")
        fig.text(0.5, 0.0135,
                 f"CARLA Town05 · held-out episode {ei} · t = {k:02d} s "
                 f"· predicted vehicles in town: {comp.sum():5.1f}",
                 color=DIM, fontsize=11, ha="center", va="bottom", family="monospace")
        fig.text(0.225, 0.868, "town occupancy map, predicted", color=FG, fontsize=12.5,
                 ha="left", va="bottom")
        _heat_legend(fig, 0.775, 0.868)
        out_frames.append(fig_to_rgb(fig))
        plt.close(fig)
        print(f"hero frame {k+1}/{len(frames)}", flush=True)
    save_clip(out_frames, out, ms, P["formats"], P["ffmpeg"])


def _heat_legend(fig, x, y, cmap=HEAT, label="vehicles per 4 m cell"):
    """Colour key ending at figure-x `x`, baseline `y` (same line as the panel title)."""
    ax = fig.add_axes([x - 0.10, y + 0.001, 0.10, 0.012])
    ax.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3a4557")
    fig.text(x - 0.108, y + 0.001, label, color=DIM, fontsize=9.5, ha="right",
             va="bottom")


# ---------------------------------------------------------------------- gif 2 ----
def gif_vs_truth(P, out, ms):
    ei, frames, preds, alpha = P["ei"], P["frames"], P["preds"], P["alpha"]
    fov, road = P["fov"], P["road"]
    W, H = 16.0, 8.2
    out_frames = []
    for k, f in enumerate(frames):
        comp = compose_max(preds[k] * fov[None], P["comp_idx"], P["gshape"], P["sub"]) * alpha
        gt = np.load(os.path.join(frame_dir(ei, f), "global_occupancy_visible.npy"))
        fig = plt.figure(figsize=(W, H), dpi=P["dpi"], facecolor=BG)
        axa = fig.add_axes([0.035, 0.055, 0.44, 0.79])
        axb = fig.add_axes([0.525, 0.055, 0.44, 0.79])
        draw_map(axa, comp, road, cmap=HEAT)
        draw_map(axb, gt, road, cmap=HEAT)
        axa.set_title("predicted — composed from 141 camera views", color="#ffb26b",
                      fontsize=15, pad=10)
        axb.set_title("ground truth — vehicles the cameras can see", color="#7ee8fa",
                      fontsize=15, pad=10)
        for ax, v, lab in ((axa, comp.sum(), "predicted"), (axb, gt.sum(), "actual")):
            ax.text(0.99, 0.015, f"{lab} vehicles: {v:5.1f}", transform=ax.transAxes,
                    color=FG, fontsize=12, ha="right", va="bottom", family="monospace",
                    bbox=dict(fc="#0b0f16cc", ec="none", pad=3.0))
        err = 100 * abs(comp.sum() - gt.sum()) / max(gt.sum(), 1)
        fig.text(0.5, 0.975, "town occupancy: prediction vs. ground truth",
                 color=FG, fontsize=20, fontweight="bold", ha="center", va="top")
        fig.text(0.5, 0.929,
                 "141 single-camera predictions splatted into one 136×153 grid of 4 m cells "
                 "(per-cell max, so a vehicle two cameras both see is counted once)",
                 color=DIM, fontsize=11, ha="center", va="top")
        fig.text(0.5, 0.013,
                 f"held-out episode {ei} · t = {k:02d} s · count error {err:4.1f} %",
                 color=DIM, fontsize=12, ha="center", va="bottom", family="monospace")
        _heat_legend(fig, 0.60, 0.877)
        out_frames.append(fig_to_rgb(fig))
        plt.close(fig)
        print(f"vs-truth frame {k+1}/{len(frames)}", flush=True)
    save_clip(out_frames, out, ms, P["formats"], P["ffmpeg"])


# ---------------------------------------------------------------------- gif 3 ----
def gif_one_cam(P, out, ms, cam):
    ei, frames, preds = P["ei"], P["frames"], P["preds"]
    fov, labels = P["fov"], P["labels"]
    W, H = 15.0, 6.4
    out_frames = []
    for k, f in enumerate(frames):
        img = read_rgb(ei, f, cam)
        pr = preds[k, cam]
        gt = np.asarray(labels[ei, f, cam], np.float32)
        fig = plt.figure(figsize=(W, H), dpi=P["dpi"], facecolor=BG)
        axi = fig.add_axes([0.028, 0.075, 0.545, 0.75])
        axg = fig.add_axes([0.638, 0.075, 0.75 * H / W, 0.75])
        axi.imshow(img)
        axi.set_xticks([])
        axi.set_yticks([])
        for s in axi.spines.values():
            s.set_color("#3a4557")
        axi.set_title(f"camera {cam:03d} — what the camera sees", color=FG, fontsize=15,
                      pad=9)
        draw_local(axg, pr, fov, gt=gt)
        axg.set_title("predicted local occupancy (bird's eye, 64 × 64 m)", color=FG,
                      fontsize=15, pad=9)
        # orientation cues on the bird's-eye panel: distance ahead of the camera
        for row, lab in ((-0.5, "64 m"), (7.5, "32 m"), (15.5, "0 m")):
            axg.text(-0.9, row, lab, color=DIM, fontsize=9.5, ha="right", va="center")
        axg.text(7.5, 16.1, "▲ camera", color=DIM, fontsize=10, ha="center", va="top",
                 clip_on=False)
        fig.text(0.5, 0.975,
                 "one camera picture  →  a 16 × 16 vehicle-occupancy grid, every second",
                 color=FG, fontsize=19, fontweight="bold", ha="center", va="top")
        fig.text(0.5, 0.915,
                 "orange = model prediction   ·   cyan outline = ground truth   ·   "
                 "grey outline = the 90° wedge this camera can label",
                 color=DIM, fontsize=11, ha="center", va="top")
        fig.text(0.5, 0.018,
                 f"held-out episode {ei} · t = {k:02d} s · "
                 f"predicted {float((pr*fov).sum()):4.1f} vehicles, actual {float((gt*fov).sum()):4.1f}",
                 color=DIM, fontsize=12, ha="center", va="bottom", family="monospace")
        out_frames.append(fig_to_rgb(fig))
        plt.close(fig)
        print(f"one-cam frame {k+1}/{len(frames)}", flush=True)
    save_clip(out_frames, out, ms, P["formats"], P["ffmpeg"])


# ------------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="runs/gifs/preds_ep015_f000-039.npz")
    ap.add_argument("--labels-cache", default="runs/cache/labels_visible.npy")
    ap.add_argument("--alpha", type=float, default=0.8867)
    ap.add_argument("--out-dir", default="runs/gifs")
    ap.add_argument("--dpi", type=int, default=85)
    ap.add_argument("--ms", type=int, default=250)
    ap.add_argument("--cam", type=int, default=69)
    ap.add_argument("--which", default="hero,truth,cam")
    ap.add_argument("--limit", type=int, default=0, help="render only the first N frames")
    ap.add_argument("--formats", default="mp4,gif",
                    help="mp4 (scrubbable, pausable) and/or gif (autoplays anywhere)")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg binary for the mp4 encode")
    ap.add_argument("--gt-instead-of-pred", action="store_true",
                    help="layout preview without a trained model")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    K, cams = load_cameras()
    info = load_info()
    fov = load_fov_mask()
    comp_idx, gshape, sub = compose_tables(cams, info)
    labels = np.load(args.labels_cache, mmap_mode="r")

    z = np.load(args.preds)
    ei = int(z["episode"])
    frames = [int(x) for x in z["frames"]]
    preds = np.asarray(z["pred"], np.float32)
    if args.gt_instead_of_pred:
        preds = np.asarray(labels[ei, frames[0]:frames[-1] + 1], np.float32)
    if args.limit:
        frames = frames[:args.limit]
        preds = preds[:args.limit]

    P = dict(ei=ei, frames=frames, preds=preds, alpha=args.alpha, cams=cams, info=info,
             fov=fov, comp_idx=comp_idx, gshape=gshape, sub=sub, labels=labels,
             road=road_background(info), dpi=args.dpi,
             formats=[f.strip() for f in args.formats.split(",") if f.strip()],
             ffmpeg=args.ffmpeg)

    which = args.which.split(",")
    if "hero" in which:
        gif_hero(P, os.path.join(args.out_dir, "town_from_cameras"), args.ms)
    if "truth" in which:
        gif_vs_truth(P, os.path.join(args.out_dir, "town_pred_vs_truth"), args.ms)
    if "cam" in which:
        gif_one_cam(P, os.path.join(args.out_dir, f"camera_{args.cam:03d}_to_grid"),
                    args.ms, args.cam)
    print("GIF_RENDER_DONE")


if __name__ == "__main__":
    main()
