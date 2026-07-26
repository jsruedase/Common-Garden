"""
run_obj1_cv.py — Corre la validación cruzada del Objetivo 1 (forward-chaining) y
reporta mean ± std sobre los folds de datasets/obj1_forward.

Es un envoltorio delgado sobre cv_run.py: usa la MISMA agregación, pero con
detección de dispositivo y ajustes coherentes con train_gpu.py / train_cpu.py,
para que baste UN comando.

  - Con GPU (CUDA o ROCm):  imgsz alto, batch auto, modelo 'm' por defecto.
  - Solo CPU:               imgsz menor, batch chico, modelo 'n', y avisa que va lento.

Uso:
    python modeling/run_obj1_cv.py                         # auto-detecta GPU/CPU
    python modeling/run_obj1_cv.py --weights yolo26s-seg.pt --epochs 200
    python modeling/run_obj1_cv.py --folds datasets/obj1_forward --imgsz 1280
"""

from __future__ import annotations

import os
# reduce la fragmentación de memoria en GPU (evita muchos OOM); debe ir ANTES de importar torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import statistics
from pathlib import Path

import cv_run   # reutiliza find_folds, seg_metrics, aggregate


def detect_device():
    """Devuelve (device_str, is_gpu, descripcion)."""
    try:
        import torch
        if torch.cuda.is_available():
            is_rocm = getattr(torch.version, "hip", None) is not None
            back = "ROCm/HIP (AMD)" if is_rocm else "CUDA (NVIDIA)"
            return "0", True, f"{torch.cuda.get_device_name(0)} · {back}"
    except Exception:
        pass
    return "cpu", False, "CPU (sin GPU)"


def main():
    ap = argparse.ArgumentParser(description="CV Objetivo 1 (forward-chaining) → mean ± std")
    ap.add_argument("--folds", type=Path, default=Path("datasets/obj1_forward"))
    ap.add_argument("--weights", default=None, help="por defecto: m-seg en GPU, n-seg en CPU")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--project", default="runs_cv_obj1")
    args = ap.parse_args()

    device, is_gpu, desc = detect_device()
    print(f"[info] dispositivo: {desc}  →  device={device}")

    # defaults según dispositivo (coherentes con train_gpu.py / train_cpu.py)
    weights = args.weights or ("yolo26s-seg.pt" if is_gpu else "yolo26n-seg.pt")
    epochs = args.epochs if args.epochs is not None else (300 if is_gpu else 100)
    imgsz = args.imgsz if args.imgsz is not None else (1280 if is_gpu else 768)
    # batch fijo: el auto-batch (-1) se rompe con pocas imágenes a 1280px (estima mal y da OOM)
    batch = args.batch if args.batch is not None else (2 if is_gpu else 4)
    if not is_gpu:
        print("[nota] sin GPU: será LENTO en todos los folds. Considera GPU/Colab.")

    folds = cv_run.find_folds(args.folds)
    if not folds:
        raise SystemExit(f"[error] no hay folds en {args.folds.resolve()} (¿corriste build_dataset.py?)")
    print(f"[info] {len(folds)} folds forward-chaining: {[f.name for f in folds]}")
    print(f"[info] {weights}, epochs={epochs}, imgsz={imgsz}, batch={batch}, eval=val\n")

    from ultralytics import YOLO
    per_fold, rows = [], []
    for fold in folds:
        data = fold / "data.yaml"
        name = f"obj1_{fold.name}"
        print(f"===== {fold.name} =====")
        model = YOLO(weights)                      # modelo NUEVO por fold
        model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
                    device=device, project=args.project, name=name,
                    amp=is_gpu, cache="ram", patience=50, close_mosaic=15,
                    degrees=180.0, fliplr=0.5, flipud=0.5, hsv_s=0.7, hsv_v=0.4)
        m = model.val(data=str(data), split="val",   # forward-chaining → val
                      project=args.project, name=f"{name}_val")
        met = cv_run.seg_metrics(m)
        print(f"[{fold.name}] " + str({k: round(v, 4) for k, v in met.items()}))
        per_fold.append(met)
        rows.append((fold.name, met))

    # ---- agregación = estimación de CV (promedio, NO el mejor fold) ----
    agg = cv_run.aggregate(per_fold)
    print("\n" + "=" * 50)
    print(f"OBJETIVO 1 — forward-chaining CV  ({len(per_fold)} folds)")
    print("=" * 50)
    for k, (mean, std) in agg.items():
        print(f"  {k:<14} {mean:.4f} ± {std:.4f}")
    print("(se reporta el promedio entre folds)")

    # CSV para el informe
    import csv
    out_csv = Path(args.project) / "obj1_forward_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["mask_mAP50-95", "mask_mAP50", "box_mAP50-95"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["fold"] + keys)
        for name, met in rows:
            w.writerow([name] + [round(met.get(k, float("nan")), 4) for k in keys])
        w.writerow([])
        w.writerow(["MEAN"] + [round(agg[k][0], 4) if k in agg else "" for k in keys])
        w.writerow(["STD"] + [round(agg[k][1], 4) if k in agg else "" for k in keys])
    print(f"\nResumen: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
