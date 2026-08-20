"""
eval_table3.py — Métricas de superficie (Tabla 3) para el modelo YOLO26-seg.

Para cada fold LOTO (una terraza dejada fuera como test), empareja predicción↔GT y
calcula:  NSD, Penalized NSD, ASSD, Precision, Recall, F1, TP/FP/FN.
Los agrega como en la Tabla 3 e imprime la fila LaTeX lista para pegar junto a
Ellipse-RCNN y SAM-2.

Definiciones (verificadas contra las filas Ellipse-RCNN y SAM-2 del artículo):
  - TP  : predicción emparejada a un GT de la MISMA clase con IoU >= --iou (Hungarian).
  - FP  : predicción sin emparejar.   FN : GT sin emparejar.
  - Precision = TP/(TP+FP) ; Recall = TP/(TP+FN) ; F1 = 2·P·R/(P+R).
  - NSD (por fold)          = media del Normalized Surface Dice(τ) sobre los pares TP.
  - Penalized NSD (por fold)= NSD · TP/(TP+FP+FN)   (no emparejados cuentan como 0).
  - ASSD (por fold)         = media de la distancia simétrica de superficie (px) sobre TP.
  - NSD / Pen.NSD / ASSD se reportan como media ± std ENTRE folds.
  - TP/FP/FN son TOTALES (sumados sobre todos los folds); P/R/F1 se calculan de esos totales.

Se evalúan las FORMAS PARAMÉTRICAS (como en el paper): planta → convex hull de la
máscara YOLO ; matera → elipse ajustada. Usa --raw para evaluar la máscara cruda.

Requisitos: ultralytics, opencv-python, numpy, scipy.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.optimize import linear_sum_assignment

PLANT, POT = 0, 1
ELLIPSE_POINTS = 24


# ─────────────────────────────────────────────────────────────────────────────
# Formas paramétricas (mismas que postprocess.py)
# ─────────────────────────────────────────────────────────────────────────────
def to_parametric(cls: int, poly_xy: np.ndarray):
    """planta → convex hull ; matera → elipse muestreada a polígono. Devuelve (N,2) px."""
    pts = np.asarray(poly_xy, dtype=np.float32)
    if cls == PLANT:
        return cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
    if len(pts) < 5:
        return pts
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    rx, ry = d1 / 2.0, d2 / 2.0
    a = np.deg2rad(ang)
    t = np.linspace(0, 2 * np.pi, ELLIPSE_POINTS, endpoint=False)
    x = cx + rx * np.cos(t) * np.cos(a) - ry * np.sin(t) * np.sin(a)
    y = cy + rx * np.cos(t) * np.sin(a) + ry * np.sin(t) * np.cos(a)
    return np.stack([x, y], axis=1)


def rasterize(poly_xy: np.ndarray, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    if len(poly_xy) >= 3:
        cv2.fillPoly(m, [np.round(poly_xy).astype(np.int32)], 1)
    return m.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Cargar instancias GT (labels YOLO-seg) y predichas (YOLO)
# ─────────────────────────────────────────────────────────────────────────────
def load_gt(label_txt: Path, H: int, W: int, parametric: bool):
    out = []
    if not label_txt.exists():
        return out
    for line in label_txt.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 7:
            continue
        cls = int(float(p[0]))
        xy = np.array(p[1:], float).reshape(-1, 2) * [W, H]     # normalizado → px
        if parametric:
            xy = to_parametric(cls, xy)
        out.append((cls, rasterize(xy, H, W)))
    return out


def load_pred(res, H: int, W: int, parametric: bool):
    out = []
    if res.masks is None:
        return out
    classes = res.boxes.cls.cpu().numpy().astype(int)
    for poly_xy, cls in zip(res.masks.xy, classes):
        if len(poly_xy) < 3:
            continue
        xy = to_parametric(int(cls), np.asarray(poly_xy)) if parametric else np.asarray(poly_xy)
        out.append((int(cls), rasterize(xy, H, W)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Emparejamiento por IoU (Hungarian), respetando la clase
# ─────────────────────────────────────────────────────────────────────────────
def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())


def match(gt, pred, iou_thr: float):
    """Devuelve (pares_TP, n_TP, n_FP, n_FN). pares_TP = lista de (mask_gt, mask_pred)."""
    ng, npd = len(gt), len(pred)
    if ng == 0 or npd == 0:
        return [], 0, npd, ng
    iou_mat = np.zeros((ng, npd))
    for i, (cg, mg) in enumerate(gt):
        for j, (cp, mp) in enumerate(pred):
            if cg == cp:                      # solo empareja misma clase
                iou_mat[i, j] = iou(mg, mp)
    ri, cj = linear_sum_assignment(-iou_mat)  # maximiza IoU total
    pairs, matched_g, matched_p = [], set(), set()
    for i, j in zip(ri, cj):
        if iou_mat[i, j] >= iou_thr:
            pairs.append((gt[i][1], pred[j][1]))
            matched_g.add(i); matched_p.add(j)
    tp = len(pairs)
    return pairs, tp, npd - len(matched_p), ng - len(matched_g)


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de superficie: NSD(τ) y ASSD
# ─────────────────────────────────────────────────────────────────────────────
def _boundary(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    er = binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~er


def surface_dists(gt: np.ndarray, pred: np.ndarray):
    """Distancias de superficie GT→pred y pred→GT (en píxeles). None si algún borde vacío."""
    bg, bp = _boundary(gt), _boundary(pred)
    if bg.sum() == 0 or bp.sum() == 0:
        return None
    edt_p = distance_transform_edt(~bp)   # dist. a borde predicho en cada píxel
    edt_g = distance_transform_edt(~bg)   # dist. a borde GT en cada píxel
    return edt_p[bg], edt_g[bp]           # (GT→pred), (pred→GT)


def nsd(d_gp: np.ndarray, d_pg: np.ndarray, tau: float) -> float:
    n = len(d_gp) + len(d_pg)
    within = (d_gp <= tau).sum() + (d_pg <= tau).sum()
    return float(within) / n if n else 0.0


def assd(d_gp: np.ndarray, d_pg: np.ndarray) -> float:
    n = len(d_gp) + len(d_pg)
    return float(d_gp.sum() + d_pg.sum()) / n if n else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación
# ─────────────────────────────────────────────────────────────────────────────
def find_folds(folds_dir: Path):
    return sorted(p.parent for p in folds_dir.glob("*/data.yaml"))


def _image_level_std(per_image, key) -> float:
    """std MUESTRAL (ddof=1) ENTRE imágenes, descartando NaN. Idéntico al del compañero."""
    vals = np.array([r[key] for r in per_image], dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")


def evaluate(model, folds, tau, iou_thr, imgsz, conf, parametric):
    """
    Acumula sobre TODAS las imágenes de todos los folds (igual que el _aggregate del
    compañero, que recibe todo junto — NO promedia por fold):
      - all_nsd / all_assd : valor por par emparejado (pooled)  → para las MEDIAS
      - per_image          : agregados por imagen                → para las STD
      - TP/FP/FN           : totales
    media    = nanmean sobre pares pooled
    Pen.NSD  = sum(nsd) / (TP+FP+FN)
    std      = _image_level_std entre imágenes (ddof=1, NaN descartados)
    """
    all_nsd, all_assd, per_image = [], [], []
    TP = FP = FN = 0
    for fold in folds:
        img_dir = fold / "images" / "test"
        lbl_dir = fold / "labels" / "test"
        if not img_dir.exists():
            print(f"[aviso] {fold.name}: sin images/test, se omite")
            continue
        n_before = TP + FP + FN
        results = model.predict(source=str(img_dir), conf=conf, imgsz=imgsz,
                                verbose=False, stream=True)
        for res in results:
            H, W = res.orig_shape
            stem = Path(res.path).stem
            gt = load_gt(lbl_dir / f"{stem}.txt", H, W, parametric)
            pred = load_pred(res, H, W, parametric)
            pairs, tp, fp, fn = match(gt, pred, iou_thr)
            TP += tp; FP += fp; FN += fn
            img_nsd, img_assd = [], []
            for mg, mp in pairs:
                d = surface_dists(mg, mp)
                if d is None:
                    img_nsd.append(float("nan")); img_assd.append(float("nan"))
                    continue
                vn, va = nsd(d[0], d[1], tau), assd(d[0], d[1])
                all_nsd.append(vn); all_assd.append(va)
                img_nsd.append(vn); img_assd.append(va)
            denom_img = tp + fp + fn                      # como el compañero: TP+FP+FN de la imagen
            arr = np.array(img_nsd, dtype=float)
            mean_nsd_img = float(np.nanmean(arr)) if arr.size and not np.all(np.isnan(arr)) else float("nan")
            nsd_pen_img = float(np.nansum(arr) / denom_img) if denom_img else float("nan")
            mean_assd_img = (float(np.nanmean(img_assd))
                             if img_assd and not np.all(np.isnan(img_assd)) else float("nan"))
            per_image.append({"mean_nsd": mean_nsd_img,
                              "nsd_penalized": nsd_pen_img,
                              "mean_assd": mean_assd_img})
        fold_tp = (TP + FP + FN) - n_before
        print(f"  {fold.name:12} instancias(TP+FP+FN)={fold_tp}")
    return all_nsd, all_assd, per_image, TP, FP, FN


def main():
    ap = argparse.ArgumentParser(description="Tabla 3 (NSD/ASSD/P/R/F1) para YOLO26-seg")
    ap.add_argument("--weights", default=Path("Segmentation_Model/best.pt"), type=Path, help="best.pt del modelo final")
    ap.add_argument("--folds", type=Path, default=Path("datasets/obj2_loto"),
                    help="carpeta con los folds LOTO (held_ST*)")
    ap.add_argument("--tau", type=float, default=20.0,
                    help="tolerancia de NSD en PÍXELES (¡confírmala con tu compañero!)")
    ap.add_argument("--iou", type=float, default=0.5, help="umbral IoU para contar un TP")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--raw", action="store_true",
                    help="evaluar la máscara cruda de YOLO (por defecto: hull/elipse)")
    ap.add_argument("--model-name", default="YOLOv26")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"[error] no existe el modelo: {args.weights.resolve()}")
    folds = find_folds(args.folds)
    if not folds:
        raise SystemExit(f"[error] no hay folds LOTO en {args.folds.resolve()} (¿corriste build_dataset.py?)")

    parametric = not args.raw
    print(f"[info] {len(folds)} folds | τ={args.tau}px | IoU_TP={args.iou} | "
          f"formas={'hull+elipse' if parametric else 'crudas'}\n")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    all_nsd, all_assd, per_image, TP, FP, FN = evaluate(
        model, folds, args.tau, args.iou, args.imgsz, args.conf, parametric)

    if not per_image:
        raise SystemExit("[error] ninguna imagen evaluada")

    # ---- medias: nanmean sobre pares pooled; Pen.NSD = sum(nsd)/(TP+FP+FN) ----
    nsd_arr = np.array(all_nsd, dtype=float)
    assd_arr = np.array(all_assd, dtype=float)
    denom = TP + FP + FN
    nsd_m = float(np.nanmean(nsd_arr)) if nsd_arr.size else float("nan")
    assd_m = float(np.nanmean(assd_arr)) if assd_arr.size else float("nan")
    pen_m = float(np.nansum(nsd_arr) / denom) if denom else float("nan")

    # ---- std a nivel de IMAGEN (ddof=1), como el compañero ----
    nsd_s = _image_level_std(per_image, "mean_nsd")
    pen_s = _image_level_std(per_image, "nsd_penalized")
    assd_s = _image_level_std(per_image, "mean_assd")

    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    print("\n" + "=" * 60)
    print(f"TABLA 3 — media (sobre pares) ± std (entre {len(per_image)} imágenes, ddof=1)")
    print("=" * 60)
    print(f"  NSD        {nsd_m:.3f} ± {nsd_s:.3f}")
    print(f"  Pen. NSD   {pen_m:.3f} ± {pen_s:.3f}")
    print(f"  ASSD       {assd_m:.3f} ± {assd_s:.3f}")
    print(f"  Precision  {P:.3f}")
    print(f"  Recall     {R:.3f}")
    print(f"  F1         {F1:.3f}")
    print(f"  TP/FP/FN   {TP}/{FP}/{FN}")

    print("\n--- Fila LaTeX (pegar en la Tabla 3) ---")
    print(f"{args.model_name} & {nsd_m:.3f}$\\pm${nsd_s:.3f} & {pen_m:.3f}$\\pm${pen_s:.3f} & "
          f"{assd_m:.3f}$\\pm${assd_s:.3f} & {P:.3f} & {R:.3f} & {F1:.3f} & {TP}/{FP}/{FN} \\\\")


if __name__ == "__main__":
    main()