#!/usr/bin/env python3
"""
JardínComún — Blended Tracker (geometry + frozen DINOv3 appearance)
==================================================================

matcher.py's geometric pipeline with an appearance term in the assignment cost
and a multi-view gallery per identity. It seeds identities from date 1 and
propagates them forward exactly like matcher.py, then reuses matcher.py's OWN
draw_terrace / overlay_all / score so the output is identical in form — grids,
per-image overlays, IDF1 / switches — but driven by the blended cost.

    cost(det,node) = alpha*(geom_dist/gate) + (1-alpha)*(1 - appearance_sim)

Only pairs within a generous geometric gate are eligible, so a moved pot stays a
candidate while a cross-terrace hub is excluded by position. alpha=1.0 reproduces
matcher.py (geometry-only); the sweep finds the balance. Appearance is scored as
the mean top-k cosine sim over each identity's accumulated (multi-view) gallery.

Outputs, mirroring matcher.py:
  Outputs/figs/tracking_<terr>.png            per-terrace grid (one panel/date)
  Outputs/figs/Outputs/overlays/...           per-image overlays on the real photo
  Outputs/figs/tracker/sweep_<terr>.png       IDF1 vs alpha
The grid/overlays are rendered at the BEST alpha per terrace (or VIZ_ALPHA if set).

Usage:
    python tracker.py Assets/annotations/terrace_1.json Assets/images
    python tracker.py                       # all terrace_*.json + Assets/images

Requires diagnostic.py, matcher.py, dinov3_probe.py on the path, plus
torch/timm/pillow/scipy/matplotlib.
"""
import glob
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostic import load_terrace, nn_spacing
import matcher                         # reuse apply, icp, score, draw_terrace, overlay_all
import ML as probe           # reuse build_embedder, crops_for_frame (monkeypatchable)
from matcher import PLANT, POT

# ------------------------------- config ----------------------------------- #
ALPHAS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]   # 1.0 = geometry-only (matcher.py)
GATE_FRAC = 1.2       # geometric gate as multiple of spacing (wider than matcher's 0.5)
KNN = 5               # top-k over a node's gallery when scoring appearance
GALLERY_CAP = 8       # max embeddings kept per identity
USE_CSLS = True      # True for the dead/dormant terraces (ST7): demotes hub crops
VIZ_ALPHA = None      # None = draw grids/overlays at the best alpha; or fix e.g. 0.2
BIG = 1e6


# --------------------------- appearance scoring --------------------------- #
def gallery_sim(det_emb, node_gal):
    """mean top-k cosine sim of each det against each node's multi-view gallery."""
    D, N = len(det_emb), len(node_gal)
    S = np.zeros((D, N), np.float32)
    for j, G in enumerate(node_gal):
        if len(G) == 0:
            continue
        sims = det_emb @ G.T
        k = min(KNN, sims.shape[1])
        S[:, j] = np.sort(sims, axis=1)[:, -k:].mean(1)
    return S


def csls(S, k=KNN):
    if S.size == 0:
        return S
    kd, kn = min(k, S.shape[1]), min(k, S.shape[0])
    r_det = np.sort(S, axis=1)[:, -kd:].mean(1)
    r_node = np.sort(S, axis=0)[-kn:, :].mean(0)
    return 2 * S - r_det[:, None] - r_node[None, :]


# ------------------------- prep (embed once) ------------------------------ #
def prep_frames(frames, images_dir, embed):
    """Per frame: ids, original xy, REGISTERED xy (P), pot centroids, embeddings.
    Registration is alpha-independent, so it (and embedding) is done once here."""
    frames = sorted(frames, key=lambda f: f["seq"])
    prepped = []
    pots0 = None
    for n, f in enumerate(frames):
        plants = [(i, x, y) for c, i, x, y in f["dets"] if c == PLANT]
        pots = np.array([[x, y] for c, i, x, y in f["dets"] if c == POT], float)
        ids, crops = probe.crops_for_frame(f, images_dir)
        emb = embed(crops) if crops else np.zeros((0, 0), np.float32)
        xy = np.array([[x, y] for _, x, y in plants], float)
        if n == 0:
            pots0 = pots
            P = xy.copy()
        else:
            src = pots if len(pots) >= 3 else xy
            dst = pots0 if len(pots0) >= 3 else prepped[0]["xy"]
            s, R, t = matcher.icp(src, dst)
            P = matcher.apply(s, R, t, xy) if len(xy) else xy
        prepped.append(dict(seq=f["seq"], date=f["date"], terrace=f["terrace"],
                            ids=[i for i, _, _ in plants], xy=xy, P=P, emb=emb))
    return prepped


# ------------------------------- assignment ------------------------------- #
def assign(prepped, alpha):
    """Seed template from date 1, propagate forward. Returns matcher-format res."""
    f1 = prepped[0]
    if len(f1["ids"]) == 0 or len(f1["emb"]) == 0:
        return None
    gate = GATE_FRAC * nn_spacing(f1["xy"])

    node_id = list(f1["ids"])
    node_pos = [f1["xy"][k].copy() for k in range(len(f1["ids"]))]
    node_gal = [f1["emb"][k:k + 1].copy() for k in range(len(f1["ids"]))]
    date1_ids = set(f1["ids"])
    new_ctr = 0

    panels = [dict(date=f1["date"],
                   pts=[(i, i, x, y, x, y) for i, (x, y) in zip(f1["ids"], f1["xy"])],
                   absent=[])]
    obs = [(i, i) for i in f1["ids"]]

    for fr in prepped[1:]:
        ids, xy, P, emb = fr["ids"], fr["xy"], fr["P"], fr["emb"]
        if len(ids) == 0:
            panels.append(dict(date=fr["date"], pts=[],
                               absent=[(node_id[j], *node_pos[j]) for j in range(len(node_id))]))
            continue

        n_pre = len(node_id)                                   # nodes existing before enrol
        T = np.array(node_pos)
        Dgeom = np.linalg.norm(P[:, None] - T[None], axis=2)
        within = Dgeom <= gate

        use_app = alpha < 1.0 and len(emb) == len(ids) and len(emb)
        if use_app:
            S = gallery_sim(emb, node_gal)
            if USE_CSLS:
                S = csls(S)
            appcost = 1.0 - S
        else:
            appcost = np.zeros_like(Dgeom)

        C = np.where(within, alpha * (Dgeom / gate) + (1 - alpha) * appcost, BIG)
        pred = [None] * len(ids)
        if C.size:
            for r, c in zip(*linear_sum_assignment(C)):
                if within[r, c]:
                    pred[r] = c

        seen, pts = set(), []
        for k, tid in enumerate(ids):
            c = pred[k]
            if c is None:
                new_ctr += 1
                node_id.append(f"NEW{new_ctr}")
                node_pos.append(P[k].copy())
                node_gal.append(emb[k:k + 1].copy() if len(emb) == len(ids) else np.zeros((0, 0)))
                pid = node_id[-1]
            else:
                pid = node_id[c]
                node_pos[c] = P[k]
                if len(emb) == len(ids):
                    node_gal[c] = np.vstack([node_gal[c], emb[k:k + 1]])[-GALLERY_CAP:]
            seen.add(pid)
            pts.append((tid, pid, P[k][0], P[k][1], xy[k][0], xy[k][1]))
            obs.append((tid, pid))
        absent = [(node_id[j], *node_pos[j]) for j in range(n_pre) if node_id[j] not in seen]
        panels.append(dict(date=fr["date"], pts=pts, absent=absent))

    return dict(panels=panels, obs=obs, date1_ids=date1_ids, gate=gate)


# ------------------------------- sweep plot ------------------------------- #
def draw_sweep(terr, sweep, out_dir):
    a = [r["alpha"] for r in sweep]; f1 = [r["idf1"] for r in sweep]
    geom = next(r for r in sweep if r["alpha"] == 1.0)
    best = max(sweep, key=lambda r: r["idf1"])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(a, f1, "-o", color="#4575b4")
    ax.scatter([geom["alpha"]], [geom["idf1"]], color="#555", zorder=5, label=f"geometry-only {geom['idf1']:.2f}")
    ax.scatter([best["alpha"]], [best["idf1"]], color="#1a9850", zorder=5, label=f"best α={best['alpha']} {best['idf1']:.2f}")
    ax.set_xlabel("α   (1 = geometry-only → 0 = appearance-only)")
    ax.set_ylabel("IDF1"); ax.invert_xaxis()
    ax.set_title(f"{terr}   blend lift {best['idf1'] - geom['idf1']:+.2f} IDF1")
    ax.legend(frameon=False, fontsize=9); fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"sweep_{terr}.png"); fig.savefig(out, dpi=130); plt.close(fig)
    return out


# --------------------------------- main ----------------------------------- #
def main():
    args = sys.argv[1:]
    images_dir = next((a for a in args if os.path.isdir(a)), "Assets/images")
    paths = [a for a in args if a.endswith(".json")] or sorted(glob.glob("Assets/annotations/terrace_*.json"))
    if not paths:
        print("No terrace_*.json found."); sys.exit(1)

    embed = probe.build_embedder()
    os.makedirs("Outputs/figs", exist_ok=True)
    print(f"\n{'terrace':8s} {'α':>4s} {'IDF1':>6s} {'core':>6s} {'switch':>7s} {'conf':>5s} {'enrl':>5s}")

    lifts = []
    for p in paths:
        frames = load_terrace(p)
        if len(frames) < 2:
            continue
        prepped = prep_frames(frames, images_dir, embed)     # embed ONCE, reuse across α
        terr = prepped[0]["terrace"]

        sweep = []
        for a in ALPHAS:
            res = assign(prepped, a)
            if res is None:
                break
            m = matcher.score(res)                            # matcher.py's OWN scoring
            sweep.append(dict(alpha=a, res=res, **m))
            print(f"{terr:8s} {a:4.1f} {m['idf1']:6.2f} {m['core_acc']:6.0%} "
                  f"{m['switches']:7d} {m['confused']:5d} {m['enrolled']:5d}"
                  f"{'  <- geom' if a == 1.0 else ''}")
        if not sweep:
            print(f"{terr:8s}  (skipped — no crops/photos on date 1)")
            continue

        # choose alpha for the visuals and render with matcher.py's own functions
        chosen = (next(r for r in sweep if r["alpha"] == VIZ_ALPHA) if VIZ_ALPHA is not None
                  else max(sweep, key=lambda r: r["idf1"]))
        matcher.draw_terrace(chosen["res"], terr, "Outputs/figs")                     # grid
        matcher.overlay_all(chosen["res"], terr, frames, images_dir, "Outputs/figs")  # overlays
        draw_sweep(terr, sweep, "Outputs/figs/tracker")

        geom = next(r for r in sweep if r["alpha"] == 1.0)
        lifts.append((terr, geom["idf1"], chosen["idf1"], chosen["alpha"]))
        print(f"         -> visuals drawn at α={chosen['alpha']}  (IDF1 {chosen['idf1']:.2f})\n")

    if lifts:
        print(f"{'terrace':8s} {'geom':>6s} {'viz':>6s} {'α':>4s} {'lift':>6s}")
        for terr, g, b, a in lifts:
            print(f"{terr:8s} {g:6.2f} {b:6.2f} {a:4.1f} {b - g:+6.2f}")
        gm = np.mean([g for _, g, _, _ in lifts]); bm = np.mean([b for _, _, b, _ in lifts])
        print(f"{'MEAN':8s} {gm:6.2f} {bm:6.2f} {'':4s} {bm - gm:+6.2f}")
        print("\ngrids -> Outputs/figs/tracking_*.png   overlays -> Outputs/figs/Outputs/overlays/")
        print("sweeps -> Outputs/figs/tracker/")


if __name__ == "__main__":
    main()