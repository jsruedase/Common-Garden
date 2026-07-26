"""
cv_run.py — Corre validación cruzada sobre los folds ya generados y agrega mean ± std.

Sirve para los dos objetivos (misma mecánica de CV, distinta definición de fold):
  Objetivo 1:  --folds datasets/obj1_forward     (forward-chaining)
  Objetivo 2:  --folds datasets/obj2_loto        (leave-one-terrace-out)

Cada subcarpeta con un data.yaml = un fold. Entrena un modelo NUEVO por fold,
lee su métrica, los descarta, y reporta el promedio y la desviación entre folds.
Eso es la estimación de CV. NO se elige "el mejor fold": se promedian.

Uso:
    python cv_run.py --folds datasets/obj2_loto
    python cv_run.py --folds datasets/obj1_forward --epochs 100 --imgsz 1280 --batch 2
    python cv_run.py --folds datasets/obj2_loto --eval-split test   # LOTO se evalúa en test
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import statistics
from pathlib import Path


def seg_metrics(metrics) -> dict:
    """mAP de máscara / caja, tolerante a versiones de Ultralytics."""
    out = {}
    try:
        out["mask_mAP50-95"] = float(metrics.seg.map)
        out["mask_mAP50"] = float(metrics.seg.map50)
    except Exception:
        pass
    try:
        out["box_mAP50-95"] = float(metrics.box.map)
    except Exception:
        pass
    return out


def find_folds(folds_dir: Path) -> list:
    """Cada subcarpeta con data.yaml es un fold (orden alfabético estable)."""
    return sorted(p.parent for p in folds_dir.glob("*/data.yaml"))


def aggregate(per_fold: list, keys=("mask_mAP50-95", "mask_mAP50", "box_mAP50-95")) -> dict:
    agg = {}
    for k in keys:
        vals = [d[k] for d in per_fold if k in d]
        if vals:
            agg[k] = (statistics.mean(vals),
                      statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return agg


def main():
    ap = argparse.ArgumentParser(description="CV runner (mean ± std) sobre folds generados")
    ap.add_argument("--folds", required=True, type=Path,
                    help="carpeta con subcarpetas de folds (obj1_forward u obj2_loto)")
    ap.add_argument("--weights", default="yolo26n-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-split", default="val", choices=["val", "test"],
                    help="split para medir cada fold (LOTO: usa 'test')")
    ap.add_argument("--project", default="runs_cv")
    ap.add_argument("--tag", default="cv", help="prefijo para nombrar las corridas")
    args = ap.parse_args()

    folds = find_folds(args.folds)
    if not folds:
        raise SystemExit(f"[error] no hay folds (carpetas con data.yaml) en {args.folds.resolve()}\n"
                         f"        ¿Corriste build_dataset.py?")
    print(f"[info] {len(folds)} folds encontrados en {args.folds.name}:")
    for f in folds:
        print("   -", f.name)

    from ultralytics import YOLO

    per_fold = []
    rows = []
    for fold in folds:
        data = fold / "data.yaml"
        if args.eval_split == "test" and "test:" not in data.read_text(encoding="utf-8"):
            print(f"[aviso] {fold.name} no tiene split 'test:'; se mide en 'val'")
            split = "val"
        else:
            split = args.eval_split

        name = f"{args.tag}_{fold.name}"
        print(f"\n===== FOLD {fold.name}  (eval={split}) =====")
        model = YOLO(args.weights)   # modelo NUEVO por fold
        model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    project=args.project, name=name,
                    degrees=180.0, fliplr=0.5, flipud=0.5, hsv_s=0.7, hsv_v=0.4)
        metrics = model.val(data=str(data), split=split,
                            project=args.project, name=f"{name}_{split}")
        m = seg_metrics(metrics)
        m_round = {k: round(v, 4) for k, v in m.items()}
        print(f"[fold {fold.name}] {m_round}")
        per_fold.append(m)
        rows.append({"fold": fold.name, "eval_split": split, **m_round})

    # ---- agregación: esto ES la estimación de CV ----
    agg = aggregate(per_fold)
    print("\n" + "=" * 48)
    print(f"RESULTADO CV — {args.folds.name}  ({len(per_fold)} folds)")
    print("=" * 48)
    for k, (mean, std) in agg.items():
        print(f"  {k:<14} {mean:.4f} ± {std:.4f}")
    print("(se reporta el promedio entre folds, NO el mejor fold)")

    # guardar CSV para el informe
    out_csv = Path(args.project) / f"{args.tag}_{args.folds.name}_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["fold", "eval_split", "mask_mAP50-95", "mask_mAP50", "box_mAP50-95"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({})
        w.writerow({"fold": "MEAN", **{k: round(v[0], 4) for k, v in agg.items()}})
        w.writerow({"fold": "STD", **{k: round(v[1], 4) for k, v in agg.items()}})
    print(f"\nResumen guardado en: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
