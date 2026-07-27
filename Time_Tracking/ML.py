#!/usr/bin/env python3
"""
JardínComún — DINOv3 Appearance Probe (retrieval + separability)
================================================================

The question this answers is NOT "build a tracker". It is the prior step:
does a frozen visual embedding carry identity on THIS data at all, before you
wire it into the matcher? It re-uses the same data path as matcher.py
(diagnostic.load_terrace) but ignores geometry entirely.

For each terrace:
  1. crop every instance from the real photo (pot-centred by default — the bag
     is more distinctive than the near-identical rosette);
  2. embed each crop with a FROZEN DINOv3 (no training — that's the whole point,
     it sidesteps the "too few observations per id" wall);
  3. GALLERY = date-1 crops (known ids). QUERY = every later-date crop.
     Nearest gallery embedding by cosine similarity = predicted id.
  4. score top-1 / top-5 retrieval against bed_position, and — the number that
     actually decides whether this is worth pursuing — the SEPARABILITY between
     the same-identity and different-identity similarity distributions (AUC).

Read it like this:
  * top-1 high  → appearance alone tracks; a tracker is worth building.
  * AUC ~0.5, histograms overlapping → appearance can't tell your plants apart;
    stop here, you learned it in one afternoon, and lean on geometry.
  * in between → appearance is a tiebreaker, not a standalone signal → blend it
    into matcher.py's cost, don't replace geometry with it.

Output mirrors matcher.py: a printed table + per-terrace figures in Outputs/figs.

Usage:
    python dinov3_probe.py                      # all terrace_*.json + Assets/images
    python dinov3_probe.py terrace_1.json imgs/ # one terrace, explicit image dir

Requires: torch, timm, pillow, scipy, matplotlib  (pip install torch timm pillow)
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from diagnostic import load_terrace, nn_spacing

# ------------------------------- config ----------------------------------- #
# The one thing to verify on your machine: the exact timm checkpoint tag.
#   python -c "import timm; print([m for m in timm.list_models('*dinov3*')])"
# If DINOv3 isn't in your timm yet, the DINOv2 fallback below is a safe swap and
# still answers the same question.
MODEL = "vit_base_patch16_dinov3.lvd1689m"
FALLBACK = "vit_base_patch14_dinov2.lvd142m"

PLANT, POT = "planta", "matera"
CROP_FRAC = 1.0        # crop side as a multiple of nearest-neighbour spacing
POT_CENTRED = True     # centre each crop on the plant's nearest pot, not the rosette
COLORS = {"same": "#1a9850", "diff": "#d73027", "correct": "#1a9850", "wrong": "#d73027"}


# =======================  ~20-LINE CORE EXPERIMENT  ======================== #
# Everything below this block is I/O and plotting to match matcher.py's output.
# The actual "swap geometry for a model" experiment is only this:
def build_embedder():
    import timm, torch
    name = MODEL
    try:
        net = timm.create_model(name, pretrained=True, num_classes=0)
    except Exception as e:
        print(f"[warn] '{MODEL}' unavailable ({type(e).__name__}); falling back to {FALLBACK}")
        name = FALLBACK
        net = timm.create_model(name, pretrained=True, num_classes=0)
    net.eval()
    cfg = timm.data.resolve_data_config({}, model=net)
    tf = timm.data.create_transform(**cfg)

    @torch.no_grad()
    def embed(crops):                                   # list[PIL] -> [N,D] unit vectors
        if not crops:
            return np.zeros((0, net.num_features), np.float32)
        x = torch.stack([tf(c.convert("RGB")) for c in crops])
        e = net(x).numpy()
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    print(f"[model] {name}  (dim={net.num_features})")
    return embed


def retrieve(query_emb, query_ids, gallery_emb, gallery_ids):
    """Nearest-neighbour id per query + the same/different similarity pairs."""
    S = query_emb @ gallery_emb.T                        # cosine sim (unit vectors)
    order = np.argsort(-S, axis=1)
    gids = np.asarray(gallery_ids)
    top1 = gids[order[:, 0]]
    top5 = gids[order[:, :5]]
    same, diff = [], []
    for r, tid in enumerate(query_ids):
        hit = gids == tid
        if hit.any():
            same.append(S[r, hit].max())
            diff.extend(S[r, ~hit].tolist())
    return top1, top5, np.array(same), np.array(diff)
# ==========================  END CORE EXPERIMENT  ========================== #


# ------------------------------- cropping --------------------------------- #
def find_image(images_dir, terr, date):
    if not images_dir or not os.path.isdir(images_dir):
        return None
    tag = f"{terr}_{date}".lower()
    for f in os.listdir(images_dir):
        if tag in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(images_dir, f)
    return None


def crops_for_frame(frame, images_dir):
    """Return (ids, crops) for a frame's plants, cropped from its real photo.
    Missing photo -> ([],[]) so the terrace is skipped with a clear message."""
    img_path = find_image(images_dir, frame["terrace"], frame["date"])
    if img_path is None:
        return [], []
    img = Image.open(img_path).convert("RGB")

    plants = [(i, x, y) for c, i, x, y in frame["dets"] if c == PLANT]
    pots = np.array([[x, y] for c, i, x, y in frame["dets"] if c == POT], float)
    allc = np.array([[x, y] for _, x, y in plants], float)
    if len(allc) == 0:
        return [], []
    half = 0.5 * CROP_FRAC * nn_spacing(allc if len(allc) > 1 else np.vstack([allc, allc + 1]))

    ids, crops = [], []
    for i, x, y in plants:
        cx, cy = x, y
        if POT_CENTRED and len(pots):
            cx, cy = pots[np.argmin(np.hypot(pots[:, 0] - x, pots[:, 1] - y))]
        box = (cx - half, cy - half, cx + half, cy + half)
        ids.append(i)
        crops.append(img.crop(box))
    return ids, crops


# ------------------------------- per terrace ------------------------------ #
def run_terrace(frames, images_dir, embed):
    frames = sorted(frames, key=lambda f: f["seq"])
    gids, gcrops = crops_for_frame(frames[0], images_dir)
    if not gcrops:
        return None
    gemb = embed(gcrops)
    gallery_set = set(gids)

    q_true, q_pred, q_ok = [], [], []
    same_all, diff_all = [], []
    overlay = None
    for k, f in enumerate(frames[1:]):
        ids, crops = crops_for_frame(f, images_dir)
        if not crops:
            continue
        qemb = embed(crops)
        # restrict scoring to queries whose true id is in the gallery (mirrors core_acc)
        keep = [j for j, tid in enumerate(ids) if tid in gallery_set]
        if not keep:
            continue
        qe = qemb[keep]; qi = [ids[j] for j in keep]
        top1, top5, same, diff = retrieve(qe, qi, gemb, gids)
        for tid, p1, p5 in zip(qi, top1, top5):
            q_true.append(tid); q_pred.append(p1)
            q_ok.append(tid == p1)
        same_all.extend(same.tolist()); diff_all.extend(diff.tolist())
        # keep the first later date for a retrieval overlay on the real photo
        if overlay is None:
            plants = {i: (x, y) for c, i, x, y in f["dets"] if c == PLANT}
            overlay = dict(frame=f, ids=qi, top1=top1,
                           xy=[plants[i] for i in qi],
                           img=find_image(images_dir, f["terrace"], f["date"]))
    if not q_true:
        return None

    same_all = np.array(same_all); diff_all = np.array(diff_all)
    return dict(
        terr=frames[0]["terrace"],
        top1=float(np.mean(q_ok)),
        same=same_all, diff=diff_all,
        auc=_auc(same_all, diff_all),
        n=len(q_true),
        overlay=overlay,
    )


# ------------------------------- metrics ---------------------------------- #
def _auc(pos, neg):
    """P(sim(same) > sim(diff)); 0.5 = no separability, 1.0 = perfect. Ties=0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


# ------------------------------- plotting --------------------------------- #
def draw_separability(res, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    lo = min(res["same"].min(), res["diff"].min())
    hi = max(res["same"].max(), res["diff"].max())
    bins = np.linspace(lo, hi, 40)
    ax.hist(res["diff"], bins=bins, color=COLORS["diff"], alpha=0.55,
            density=True, label=f"different id (n={len(res['diff'])})")
    ax.hist(res["same"], bins=bins, color=COLORS["same"], alpha=0.65,
            density=True, label=f"same id (n={len(res['same'])})")
    ax.axvline(res["same"].mean(), color=COLORS["same"], ls="--", lw=1)
    ax.axvline(res["diff"].mean(), color=COLORS["diff"], ls="--", lw=1)
    ax.set_xlabel("cosine similarity to gallery crop")
    ax.set_ylabel("density"); ax.set_yticks([])
    ax.set_title(f"{res['terr']}   top-1 {res['top1']:.0%}   "
                 f"separability AUC {res['auc']:.2f}   (n={res['n']} queries)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(out_dir, f"separability_{res['terr']}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def draw_overlay(res, out_dir):
    ov = res["overlay"]
    if ov is None or ov["img"] is None:
        return None
    img = Image.open(ov["img"]).convert("RGB"); W, H = img.size
    fig, ax = plt.subplots(figsize=(W / 200, H / 200))
    ax.imshow(img)
    for tid, pred, (x, y) in zip(ov["ids"], ov["top1"], ov["xy"]):
        c = COLORS["correct"] if pred == tid else COLORS["wrong"]
        ax.scatter(x, y, s=48, facecolors="none", edgecolors=c, linewidths=1.8, zorder=3)
        ax.text(x + 12, y, str(pred)[-6:], fontsize=7, color="white", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc=c, ec="none", alpha=0.75))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{ov['frame']['terrace']}  {ov['frame']['date'].replace('_',' ')}"
                 f"   nearest-neighbour id  (green=correct, red=wrong)", fontsize=10)
    fig.tight_layout()
    od = os.path.join(out_dir, "overlays"); os.makedirs(od, exist_ok=True)
    out = os.path.join(od, f"retrieval_{ov['frame']['terrace']}_{ov['frame']['date']}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


# --------------------------------- main ----------------------------------- #
def main():
    args = sys.argv[1:]
    images_dir = next((a for a in args if os.path.isdir(a)), "Assets/images")
    paths = [a for a in args if a.endswith(".json")] or sorted(glob.glob("Assets/annotations/terrace_*.json"))
    if not paths:
        print("No terrace_*.json found."); sys.exit(1)

    os.makedirs("Outputs/figs", exist_ok=True)
    embed = build_embedder()

    print(f"\n{'terrace':8s} {'top1':>6s} {'AUC':>6s} {'same':>6s} {'diff':>6s} {'nQ':>5s}")
    rows = []
    for p in paths:
        frames = load_terrace(p)
        if len(frames) < 2:
            continue
        res = run_terrace(frames, images_dir, embed)
        if res is None:
            print(f"{load_terrace(p)[0]['terrace']:8s}  (skipped — no crops/photos)")
            continue
        print(f"{res['terr']:8s} {res['top1']:6.0%} {res['auc']:6.2f} "
              f"{res['same'].mean():6.2f} {res['diff'].mean():6.2f} {res['n']:5d}")
        draw_separability(res, "Outputs/figs")
        draw_overlay(res, "Outputs/figs")
        rows.append(res)

    if rows:
        t1 = np.mean([r["top1"] for r in rows]); au = np.nanmean([r["auc"] for r in rows])
        print(f"\n{'MEAN':8s} {t1:6.0%} {au:6.2f}")
        print("\nRead-out:")
        print("  AUC ~0.50 and overlapping histograms -> appearance can't separate your")
        print("     plants; keep geometry, don't build an appearance tracker.")
        print("  AUC >~0.85 and high top-1 -> appearance carries identity; worth blending")
        print("     into matcher.py's cost as (1-cos) alongside the geometric distance.")
    print("\nfigures in Outputs/figs/  (separability_*.png, overlays/retrieval_*.png)")


if __name__ == "__main__":
    main()