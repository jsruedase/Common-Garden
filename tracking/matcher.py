#!/usr/bin/env python3
"""
JardínComún — Tracking Matcher (Tier-1: oracle detections) + visual confirmation
================================================================================

Seeds a template from date 1, then for every later date:
  1. register the frame to the template with CORRESPONDENCE-FREE trimmed ICP on
     the pot centroids (using the bed_position ids to register would be circular);
  2. Hungarian-assign plant detections to template nodes within a distance gate;
  3. propagate the id to matched plants, ENROLL unmatched detections as new ids,
     mark unmatched template nodes as absent.

It scores against the ground-truth bed_position (never seen by the matcher) and
renders two kinds of visual:
  - a per-terrace GRID (one small panel per date, registered frame), and
  - a per-image OVERLAY (one full-size figure per date) with the assigned id
    drawn at each plant centroid, on the REAL photo when available, else on a
    schematic canvas.
Colours in both:  green = correct · red = wrong · blue = new (enrolled) ·
grey x = template node absent this date.

This is Tier-1: detections are the ground-truth instances, so it measures the
matcher + registration alone, independent of the segmentation model.

Usage:
    python matcher.py                       # all terrace_*.json in ../data/annotations
    python matcher.py terrace_1.json        # one terrace
    python matcher.py /path/to/images       # a directory arg = real-image folder
    python matcher.py terrace_7.json imgs/  # combine
Figures are written to ../data/figs (grids) and ../data/figs/overlays (per-image overlays).
"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostic import load_terrace, umeyama, nn_spacing

PLANT, POT = "planta", "matera"
COLORS = {"correct": "#1a9850", "wrong": "#d73027", "new": "#4575b4", "seed": "#555555"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images"
FIGS_DIR = DATA_DIR / "figs"


# ------------------------------- geometry --------------------------------- #
def apply(s, R, t, P):
    return (s * (R @ P.T)).T + t


def icp(src, dst, iters=50, trim=0.7):
    """Correspondence-free similarity src->dst via trimmed ICP."""
    if len(src) < 2 or len(dst) < 2:
        return 1.0, np.eye(2), np.zeros(2)
    s, R, t = 1.0, np.eye(2), dst.mean(0) - src.mean(0)
    for _ in range(iters):
        P = apply(s, R, t, src)
        D = np.linalg.norm(P[:, None] - dst[None], axis=2)
        j = D.argmin(1)
        d = D[np.arange(len(src)), j]
        keep = d <= np.quantile(d, trim)
        if keep.sum() < 2:
            break
        try:
            s, R, t = umeyama(src[keep], dst[j][keep])
        except ValueError:
            break
    return s, R, t


# ------------------------------- matcher ---------------------------------- #
def run_terrace(frames):
    frames = sorted(frames, key=lambda f: f["seq"])
    f1 = frames[0]
    plants1 = [(i, x, y) for c, i, x, y in f1["dets"] if c == PLANT]
    pots1 = np.array([[x, y] for c, i, x, y in f1["dets"] if c == POT], float)

    template = {i: np.array([x, y], float) for i, x, y in plants1}
    date1_ids = set(template)
    gate = 0.5 * nn_spacing(np.array(list(template.values())))
    new_ctr = 0

    # each panel point: (true_id, pred_id, X_reg, Y_reg, x_orig, y_orig)
    panels = [dict(date=f1["date"], pts=[(i, i, x, y, x, y) for i, x, y in plants1], absent=[])]
    obs = [(i, i) for i, x, y in plants1]

    for f in frames[1:]:
        plantsB = [(i, x, y) for c, i, x, y in f["dets"] if c == PLANT]
        potsB = np.array([[x, y] for c, i, x, y in f["dets"] if c == POT], float)

        src = potsB if len(potsB) >= 3 else np.array([[x, y] for _, x, y in plantsB], float)
        dst = pots1 if len(pots1) >= 3 else np.array(list(template.values()), float)
        s, R, t = icp(src, dst)

        det_xy = np.array([[x, y] for _, x, y in plantsB], float)
        P = apply(s, R, t, det_xy) if len(det_xy) else np.zeros((0, 2))

        tkeys = list(template)
        T = np.array([template[k] for k in tkeys], float)
        pred = [None] * len(plantsB)
        if len(P) and len(T):
            C = np.linalg.norm(P[:, None] - T[None], axis=2)
            for r, c in zip(*linear_sum_assignment(C)):
                if C[r, c] <= gate:
                    pred[r] = tkeys[c]

        seen, pts = set(), []
        for k, (true_id, ox, oy) in enumerate(plantsB):
            X, Y = P[k]
            pid = pred[k]
            if pid is None:
                new_ctr += 1
                pid = f"NEW{new_ctr}"
            template[pid] = np.array([X, Y])
            seen.add(pid)
            pts.append((true_id, pid, X, Y, ox, oy))
            obs.append((true_id, pid))
        absent = [(k, *template[k]) for k in tkeys if k not in seen]
        panels.append(dict(date=f["date"], pts=pts, absent=absent))

    return dict(panels=panels, obs=obs, date1_ids=date1_ids, gate=gate)


# ------------------------------- metrics ---------------------------------- #
def classify(true_id, pred_id, date1_ids):
    if pred_id == true_id:
        return "correct"
    if true_id in date1_ids:
        return "wrong"
    return "new" if str(pred_id).startswith("NEW") else "wrong"


def idf1(obs):
    from collections import Counter
    pair = Counter(obs)
    gts = sorted({t for t, _ in obs}); prs = sorted({p for _, p in obs})
    gi = {g: a for a, g in enumerate(gts)}; pi = {p: b for b, p in enumerate(prs)}
    W = np.zeros((len(gts), len(prs)))
    for (g, p), n in pair.items():
        W[gi[g], pi[p]] = n
    ri, ci = linear_sum_assignment(-W)
    idtp = sum(W[r, c] for r, c in zip(ri, ci))
    total = len(obs)
    return 2 * idtp / (2 * idtp + (total - idtp) + (total - idtp))


def id_switches(panels):
    seq = {}
    for p in panels:
        for true_id, pred_id, *_ in p["pts"]:
            seq.setdefault(true_id, []).append(pred_id)
    return sum(sum(a != b for a, b in zip(s, s[1:])) for s in seq.values())


def score(res):
    d1 = res["date1_ids"]
    later = [(t, p) for pnl in res["panels"][1:] for (t, p, *_ ) in pnl["pts"]]
    kinds = [classify(t, p, d1) for t, p in later]
    core = [(t, p) for t, p in later if t in d1]
    return dict(
        idf1=idf1(res["obs"]),
        acc=sum(k != "wrong" for k in kinds) / max(1, len(kinds)),
        core_acc=sum(t == p for t, p in core) / max(1, len(core)),
        switches=id_switches(res["panels"]),
        enrolled=sum(k == "new" for k in kinds),
        confused=sum(k == "wrong" for k in kinds),
    )


# --------------------------- visualisation -------------------------------- #
def short(i):
    return str(i)[-6:]


def draw_terrace(res, terr, out_dir):
    panels, d1 = res["panels"], res["date1_ids"]
    n = len(panels); cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), squeeze=False)
    for idx, pnl in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        first = idx == 0; ok = tot = 0
        for true_id, pred_id, X, Y, *_ in pnl["pts"]:
            kind = "seed" if first else classify(true_id, pred_id, d1)
            if not first:
                tot += 1; ok += kind != "wrong"
            ax.scatter(X, Y, s=26, c=COLORS[kind], edgecolors="white", linewidths=0.4, zorder=3)
            ax.text(X + 6, Y, short(pred_id), fontsize=5.2, color=COLORS[kind], va="center", zorder=4)
        for k, X, Y in pnl["absent"]:
            ax.scatter(X, Y, s=30, marker="x", c="#bbbbbb", linewidths=0.8, zorder=2)
        ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(pnl["date"].replace("_", " ") + ("  (template)" if first else f"   acc {ok/max(1,tot):.0%}"), fontsize=9)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    m = score(res)
    fig.suptitle(f"{terr}   IDF1 {m['idf1']:.2f}   core-track {m['core_acc']:.0%}   "
                 f"switches {m['switches']}   enrolled {m['enrolled']}   confused {m['confused']}", fontsize=12, y=0.998)
    _legend(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    out = os.path.join(out_dir, f"tracking_{terr}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def _legend(fig):
    h = [plt.Line2D([], [], marker="o", ls="", mec="white", mfc=c, label=l)
         for l, c in [("correct", COLORS["correct"]), ("wrong", COLORS["wrong"]), ("new (enrolled)", COLORS["new"])]]
    h.append(plt.Line2D([], [], marker="x", ls="", c="#bbbbbb", label="absent"))
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=9, frameon=False)


def find_image(images_dir, terr, date):
    """Locate the raw photo for a (terrace, date), tolerant of hash prefixes/ext."""
    if not images_dir or not os.path.isdir(images_dir):
        return None
    tag = f"{terr}_{date}".lower()
    for f in os.listdir(images_dir):
        if tag in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(images_dir, f)
    return None


def overlay_one(pnl, terr, d1, image_path, out_path, is_template):
    """One full-size overlay for a single date: assigned ids at real pixel
    positions, on the photo if given, else on a schematic canvas."""
    pts = pnl["pts"]
    xs = [px[4] for px in pts]; ys = [px[5] for px in pts]

    if image_path:
        from PIL import Image
        img = Image.open(image_path); W, H = img.size
        fig, ax = plt.subplots(figsize=(W / 200, H / 200))
        ax.imshow(img)
        boxed = True
    else:
        pad = 0.05 * (max(max(xs) - min(xs), max(ys) - min(ys)) or 100)
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.add_patch(plt.Rectangle((min(xs) - pad, min(ys) - pad),
                                   (max(xs) - min(xs)) + 2 * pad, (max(ys) - min(ys)) + 2 * pad,
                                   fc="#f2f2ef", ec="none", zorder=0))
        ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.invert_yaxis(); ax.set_aspect("equal")
        boxed = False

    ok = tot = 0
    for true_id, pred_id, X, Y, ox, oy in pts:
        kind = "seed" if is_template else classify(true_id, pred_id, d1)
        if not is_template:
            tot += 1; ok += kind != "wrong"
        c = COLORS[kind]
        ax.scatter(ox, oy, s=48, facecolors="none", edgecolors=c, linewidths=1.8, zorder=3)
        ax.text(ox + 12, oy, short(pred_id), fontsize=7, color="white" if boxed else c,
                va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc=c, ec="none", alpha=0.75) if boxed else None)
    ax.set_xticks([]); ax.set_yticks([])
    ttl = f"{terr}  {pnl['date'].replace('_', ' ')}"
    ttl += "  — template" if is_template else f"   acc {ok/max(1,tot):.0%}"
    if not image_path:
        ttl += "   (no image — schematic)"
    ax.set_title(ttl, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)


def overlay_all(res, terr, frames, images_dir, out_dir):
    """Separate function: render one overlay per date for a terrace."""
    od = os.path.join(out_dir, "overlays"); os.makedirs(od, exist_ok=True)
    d1 = res["date1_ids"]; made = []
    seqs = [f["seq"] for f in sorted(frames, key=lambda f: f["seq"])]
    for idx, (seq, pnl) in enumerate(zip(seqs, res["panels"])):
        img = find_image(images_dir, terr, pnl["date"])
        out = os.path.join(od, f"{terr}_{seq}_{pnl['date']}.png")
        overlay_one(pnl, terr, d1, img, out, is_template=(idx == 0))
        made.append((out, img is not None))
    return made


# --------------------------------- main ----------------------------------- #
def main():
    args = sys.argv[1:]
    images_dir = str(IMAGES_DIR)
    paths = [a for a in args if a.endswith(".json")] or sorted(str(p) for p in ANNOTATIONS_DIR.glob("terrace_*.json"))
    if not paths:
        print(f"No terrace_*.json found in {ANNOTATIONS_DIR}."); sys.exit(1)
    out_dir = str(FIGS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'terrace':8s} {'IDF1':>6s} {'core-track':>11s} {'switches':>9s} {'enrolled':>9s} {'confused':>9s}")
    n_real = n_schem = 0
    for p in paths:
        frames = load_terrace(p)
        if len(frames) < 2:
            continue
        res = run_terrace(frames)
        terr = sorted(frames, key=lambda f: f["seq"])[0]["terrace"]
        m = score(res)
        print(f"{terr:8s} {m['idf1']:6.2f} {m['core_acc']:10.0%} {m['switches']:9d} {m['enrolled']:9d} {m['confused']:9d}")
        draw_terrace(res, terr, out_dir)
        for _, real in overlay_all(res, terr, frames, images_dir, out_dir):
            n_real += real; n_schem += not real
    print(f"\noverlays written to {out_dir}/overlays/  ({n_real} on real images, {n_schem} schematic)")
    print(f"grids written to {out_dir}/")


if __name__ == "__main__":
    main()