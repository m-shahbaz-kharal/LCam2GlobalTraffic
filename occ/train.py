"""Train the single-image -> visible local occupancy model."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import OccDataset, TRAIN_EPS, VAL_EPS, build_cache, cache_paths
from .geometry import (DATA_ROOT, frustum_cell_tables, load_cameras, load_fov_mask)
from .model import LSSOcc

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OCC_THR = 0.25          # a cell counts as "occupied" at a quarter vehicle of mass
POS_THR = 0.02          # target mass above this = positive cell for loss weighting


def to_gpu_images(img_u8, device):
    x = img_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - MEAN.to(device)) / STD.to(device)
    x = F.pad(x, (0, 0, 0, 4))                      # 540 -> 544 rows (bottom)
    return x.contiguous(memory_format=torch.channels_last)


def losses(pred, target, mask, pos_weight=9.0, count_weight=0.05):
    w = mask * (1.0 + pos_weight * (target > POS_THR).float())
    l_occ = ((pred - target).pow(2) * w).sum() / w.sum().clamp(min=1.0)
    cnt_p = (pred * mask).sum(dim=(1, 2))
    cnt_t = (target * mask).sum(dim=(1, 2))
    l_cnt = (cnt_p - cnt_t).abs().mean()
    return l_occ + count_weight * l_cnt, l_occ.detach(), l_cnt.detach()


class MetricAccum:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.abs_err = 0.0
        self.cells = 0
        self.cnt_err = 0.0
        self.cnt_within_half = 0
        self.views = 0

    def update(self, pred, target, mask):
        m = mask > 0.5
        p = (pred >= OCC_THR) & m
        g = (target >= OCC_THR) & m
        self.tp += (p & g).sum().item()
        self.fp += (p & ~g).sum().item()
        self.fn += (~p & g).sum().item()
        self.abs_err += ((pred - target).abs() * mask).sum().item()
        self.cells += mask.sum().item() * pred.shape[0] if mask.dim() == 2 else mask.sum().item()
        ce = ((pred * mask).sum(dim=(1, 2)) - (target * mask).sum(dim=(1, 2))).abs()
        self.cnt_err += ce.sum().item()
        self.cnt_within_half += (ce <= 0.5).sum().item()
        self.views += pred.shape[0]

    def results(self):
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return {
            "precision": prec, "recall": rec, "f1": f1,
            "cell_mae": self.abs_err / max(self.cells, 1),
            "count_mae": self.cnt_err / max(self.views, 1),
            "count_within_0.5": self.cnt_within_half / max(self.views, 1),
            "views": self.views,
        }


@torch.no_grad()
def validate(model, loader, mask_gpu, device):
    model.eval()
    acc = MetricAccum()
    for batch in loader:
        x = to_gpu_images(batch["img"], device)
        target = batch["target"].to(device, non_blocking=True)
        cam = batch["cam"].to(device)
        flip = batch["flip"].to(device).bool()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, cam, flip)
        acc.update(pred.float(), target, mask_gpu)
    model.train()
    return acc.results()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--cache", default="runs/cache")
    ap.add_argument("--out", default="runs/r50_lss_v1")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--val-frame-step", type=int, default=5)
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.jsonl"), "a", buffering=1)

    def log(rec):
        rec["t"] = round(time.time() - t_start, 1)
        logf.write(json.dumps(rec) + "\n")
        print(rec, flush=True)

    t_start = time.time()
    build_cache(args.cache, args.data_root)

    K, cams = load_cameras(args.data_root)
    tables, tables_flip = frustum_cell_tables(K, cams)
    fov = load_fov_mask(args.data_root)
    mask_gpu = torch.from_numpy(fov).to(device)

    train_ds = OccDataset(args.cache, TRAIN_EPS, args.data_root,
                          frame_step=args.frame_step, hflip=True)
    val_ds = OccDataset(args.cache, VAL_EPS, args.data_root,
                        frame_step=args.val_frame_step, hflip=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=True, prefetch_factor=6)
    val_ld = DataLoader(val_ds, batch_size=96, shuffle=False, num_workers=8,
                        pin_memory=True, persistent_workers=True)
    log({"event": "data", "train_views": len(train_ds), "val_views": len(val_ds)})

    model = LSSOcc(tables, tables_flip).to(device)
    model = model.to(memory_format=torch.channels_last)

    backbone_names = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")
    decay, no_decay, decay_bb, no_decay_bb = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = n.startswith(backbone_names)
        is_nd = p.dim() <= 1
        [[decay, no_decay], [decay_bb, no_decay_bb]][is_bb][is_nd].append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
        {"params": decay_bb, "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": no_decay_bb, "lr": args.lr_backbone, "weight_decay": 0.0},
    ])
    total_steps = args.epochs * (len(train_ds) // args.batch_size)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        prog = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch, gstep, best_f1 = 0, 0, -1.0
    last_path = os.path.join(args.out, "last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, gstep, best_f1 = ck["epoch"] + 1, ck["gstep"], ck["best_f1"]
        log({"event": "resume", "epoch": start_epoch, "gstep": gstep})

    def save(epoch, path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "gstep": gstep,
                    "best_f1": best_f1, "args": vars(args)}, path)

    budget_s = args.max_hours * 3600
    stop = False
    for epoch in range(start_epoch, args.epochs):
        t_ep = time.time()
        seen = 0
        for it, batch in enumerate(train_ld):
            x = to_gpu_images(batch["img"], device)
            target = batch["target"].to(device, non_blocking=True)
            cam = batch["cam"].to(device)
            flip = batch["flip"].to(device).bool()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x, cam, flip)
            loss, l_occ, l_cnt = losses(pred.float(), target, mask_gpu)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            gstep += 1
            seen += x.shape[0]
            if it % 50 == 0:
                ips = seen / (time.time() - t_ep)
                log({"event": "step", "epoch": epoch, "it": it, "gstep": gstep,
                     "loss": round(loss.item(), 5), "l_occ": round(l_occ.item(), 5),
                     "l_cnt": round(l_cnt.item(), 4),
                     "lr": sched.get_last_lr()[0], "img_per_s": round(ips, 1)})
            if it % 200 == 0 and time.time() - t_start > budget_s:
                log({"event": "time_budget_stop", "epoch": epoch, "it": it})
                stop = True
                break
        m = validate(model, val_ld, mask_gpu, device)
        log({"event": "val", "epoch": epoch, **{k: round(v, 4) for k, v in m.items()}})
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            save(epoch, os.path.join(args.out, "best.pt"))
        save(epoch, last_path)
        if stop:
            break
    log({"event": "done", "best_f1": round(best_f1, 4),
         "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)})


if __name__ == "__main__":
    main()
