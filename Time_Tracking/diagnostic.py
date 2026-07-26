#!/usr/bin/env python3
"""
JardínComún — Tracking Feasibility Diagnostic
=============================================

Reads the Label Studio annotation exports (one JSON per terrace, each a list of
dated captures) and reports, per terrace, the numbers that decide whether
identity-by-position is viable BEFORE any matcher is built:

    s        container spacing  (median nearest-neighbour distance within a date)
    epsilon  registration residual (jitter of a STABLE container after the
             between-date camera pose is removed)
    R = s/e  separability:  >5 trivial | 2-5 workable | <2 geometry insufficient
    gate     recommended matching radius  = min(3*epsilon, s/2)
    movement fraction of plants that genuinely relocate between dates
    churn    plants that appear / disappear vs the first date

Pose and real movement are separated with no raw image and no reference marker:
each later date is aligned to the first via a similarity transform fitted with
RANSAC over the pot centroids, so the stable majority defines the frame and
movers fall out as outliers. Identity is the linked bed_position textarea (shared by a bed
plant and pot); class is the region label (Matera/Planta). Coords are percent.

Usage:
    python diagnostic.py                     # all terrace_*.json in this folder
    python diagnostic.py terrace_1.json ...  # or explicit files
"""
import glob
import json
import os
import re
import sys

import numpy as np

REG_CLASS = "matera"      # register on pots (one per bed, clean centres)
PLANT_CLASS = "planta"    # measure movement on plants
_FN = re.compile(r"-(\d+)-(ST\d+)_(.+)\.jpe?g$", re.I)


# ----------------------------- geometry ----------------------------------- #
def umeyama(src, dst):
    """Similarity mapping src->dst:  dst ~= s * R @ src + t."""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var = (sc ** 2).sum() / n
    if var < 1e-12:
        raise ValueError("coincident points")
    U, S, Vt = np.linalg.svd((dc.T @ sc) / n)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / var)
    return s, R, mu_d - s * (R @ mu_s)


def resid(src, dst, s, R, t):
    return np.linalg.norm(dst - ((s * (R @ src.T)).T + t), axis=1)


def ransac(src, dst, iters=2000, seed=0):
    """Robust similarity src->dst. Returns (s,R,t). Threshold self-scales (MAD)."""
    n = len(src)
    if n < 3:
        return umeyama(src, dst)
    try:
        r0 = resid(src, dst, *umeyama(src, dst))
        thr = max(3 * 1.4826 * np.median(np.abs(r0 - np.median(r0))), 1e-6)
    except ValueError:
        thr = 1.0
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        i = rng.choice(n, 2, replace=False)
        try:
            p = umeyama(src[i], dst[i])
        except ValueError:
            continue
        m = resid(src, dst, *p) <= thr
        if m.sum() > best_n:
            best, best_n = m, int(m.sum())
    if best is None or best.sum() < 2:
        return umeyama(src, dst)
    return umeyama(src[best], dst[best])


def nn_spacing(pts):
    if len(pts) < 2:
        return float("nan")
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))


# ------------------------------ loading ----------------------------------- #
def load_terrace(path):
    frames = []
    for task in json.load(open(path)):
        m = _FN.search(task.get("file_upload", ""))
        seq = int(m.group(1)) if m else 0
        terrace = m.group(2) if m else os.path.basename(path)
        date = m.group(3) if m else task.get("file_upload", "?")
        anns = task.get("annotations") or []
        if not anns:
            continue
        result = anns[0].get("result", [])
        # region id -> bed_position (identity lives in a linked textarea)
        bedpos = {}
        for r in result:
            if r.get("from_name") == "bed_position" and r.get("type") == "textarea":
                txt = (r.get("value") or {}).get("text") or []
                if txt and str(txt[0]).strip():
                    bedpos[r.get("id")] = str(txt[0]).strip()
        seen, dets = set(), []
        for r in result:
            ty = r.get("type")
            if ty not in ("ellipselabels", "polygonlabels"):
                continue
            bid = bedpos.get(r.get("id"))
            if not bid:
                continue
            lbl = r["value"].get("ellipselabels") or r["value"].get("polygonlabels") or ["?"]
            cls = str(lbl[0]).lower()
            key = (cls, bid)
            if key in seen:            # dedupe within a frame (ST1 date-2 dup pots)
                continue
            seen.add(key)
            W = r.get("original_width") or 1
            H = r.get("original_height") or 1
            v = r["value"]
            if ty == "ellipselabels":
                cx, cy = v["x"] / 100 * W, v["y"] / 100 * H
            else:
                p = np.asarray(v["points"], float)
                cx, cy = p[:, 0].mean() / 100 * W, p[:, 1].mean() / 100 * H
            dets.append((cls, bid, cx, cy))
        frames.append(dict(terrace=terrace, date=date, seq=seq, dets=dets))
    frames.sort(key=lambda f: f["seq"])
    return frames


# ------------------------------ per-pair ---------------------------------- #
def diagnose_pair(A, B):
    da = {(c, i): (x, y) for c, i, x, y in A["dets"]}
    db = {(c, i): (x, y) for c, i, x, y in B["dets"]}
    common = sorted(set(da) & set(db))
    if len(common) < 3:
        return None

    reg = [k for k in common if k[0] == REG_CLASS] or common
    s, R, t = ransac(np.array([db[k] for k in reg]), np.array([da[k] for k in reg]))

    srcB = np.array([db[k] for k in common])
    dstA = np.array([da[k] for k in common])
    res = resid(srcB, dstA, s, R, t)

    spacing = nn_spacing(np.array([da[k] for k in reg]))
    scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
    # a mover changes bed (residual ~ spacing), not jitter -> gate sits above jitter
    gate = max(5 * scale, 0.4 * (spacing if spacing == spacing else 5 * scale))
    stable = res <= gate

    plants = [k for k in common if k[0] == PLANT_CLASS] or common
    movers = [k for k, ok in zip(common, stable) if not ok and k[0] == PLANT_CLASS]
    ap = {k for k in db if k[0] == PLANT_CLASS} - {k for k in da if k[0] == PLANT_CLASS}
    dis = {k for k in da if k[0] == PLANT_CLASS} - {k for k in db if k[0] == PLANT_CLASS}
    return dict(eps=res[stable], spacing=spacing,
                n_plants=len(plants), n_movers=len(movers),
                appeared=len(ap), disappeared=len(dis))


def band(R):
    if R != R:
        return "n/a"
    if R > 5:
        return "TRIVIAL   (expect IDF1 > 0.95)"
    if R >= 2:
        return "WORKABLE  (gating critical; IDF1 ~0.85-0.95)"
    return "INSUFFICIENT (geometry alone won't cut it)"


def report(frames):
    fs = sorted(frames, key=lambda f: f["seq"])
    if len(fs) < 2:
        return None
    terr = fs[0]["terrace"]
    eps, sp = [], []
    mv_n = mv_d = ap = dis = 0
    A = fs[0]
    for B in fs[1:]:
        m = diagnose_pair(A, B)
        if not m:
            continue
        eps += m["eps"].tolist()
        sp.append(m["spacing"])
        mv_n += m["n_movers"]
        mv_d += m["n_plants"]
        ap += m["appeared"]
        dis += m["disappeared"]
    if not eps or not sp:
        return None
    e = float(np.median(eps))
    s = float(np.median(sp))
    R = s / e if e > 1e-9 else float("inf")
    print(f"\n[{terr}]  {len(fs)} dates, {len(fs) - 1} pairs vs first date")
    print(f"  spacing  s        : {s:8.1f} px")
    print(f"  residual epsilon  : {e:8.1f} px   (p90 {np.percentile(eps, 90):.1f})")
    print(f"  separability R=s/e: {R:8.1f}      {band(R)}")
    print(f"  gate radius       : {min(3 * e, s / 2):8.1f} px")
    print(f"  movement          : {mv_n / max(1, mv_d):7.1%}      ({mv_n}/{mv_d} plant-obs relocated)")
    print(f"  churn vs first    : +{ap} appeared / -{dis} disappeared")
    return dict(terr=terr, eps=eps, sp=sp, mv_n=mv_n, mv_d=mv_d, ap=ap, dis=dis)


def main():
    paths = sys.argv[1:] or sorted(glob.glob("Assets/annotations/terrace_*.json"))
    if not paths:
        print("No terrace_*.json found. Pass paths, or run in the folder with them.")
        sys.exit(1)
    allr = [r for p in paths for r in [report(load_terrace(p))] if r]
    if not allr:
        return
    eps = [x for r in allr for x in r["eps"]]
    sp = [x for r in allr for x in r["sp"]]
    e, s = float(np.median(eps)), float(np.median(sp))
    R = s / e if e > 1e-9 else float("inf")
    mv_n = sum(r["mv_n"] for r in allr)
    mv_d = sum(r["mv_d"] for r in allr)
    print("\n" + "=" * 60)
    print("OVERALL")
    print(f"  spacing  s        : {s:8.1f} px")
    print(f"  residual epsilon  : {e:8.1f} px")
    print(f"  separability R    : {R:8.1f}      {band(R)}")
    print(f"  gate radius       : {min(3 * e, s / 2):8.1f} px")
    print(f"  movement          : {mv_n / max(1, mv_d):7.1%}")
    print(f"  churn             : +{sum(r['ap'] for r in allr)} / -{sum(r['dis'] for r in allr)}")


if __name__ == "__main__":
    main()
