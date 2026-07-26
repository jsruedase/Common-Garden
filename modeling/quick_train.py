"""
quick_train.py — Entrenamiento rápido de prueba (smoke test) para JardínComún.

NO es un entrenamiento real: son pocas épocas a baja resolución solo para
comprobar que el pipeline funciona (imágenes + máscaras cargan, YOLO corre,
se producen métricas y visualizaciones). Para resultados reales: más épocas,
imgsz alto y GPU.

Uso:
    python modeling/quick_train.py                                   # usa obj1_final
    python modeling/quick_train.py --data datasets/obj2_loto/held_ST1/data.yaml
    python modeling/quick_train.py --epochs 10 --imgsz 960 --weights yolo26s-seg.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def seg_metrics(metrics) -> dict:
    """Extrae mAP de máscara / caja de forma tolerante a versiones."""
    out = {}
    try:
        out["mask_mAP50-95"] = round(float(metrics.seg.map), 4)
        out["mask_mAP50"] = round(float(metrics.seg.map50), 4)
    except Exception:
        pass
    try:
        out["box_mAP50-95"] = round(float(metrics.box.map), 4)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description="Smoke test de YOLO-seg")
    ap.add_argument("--data", type=Path, default=Path("datasets/obj1_final/data.yaml"),
                    help="ruta al data.yaml del fold a entrenar")
    ap.add_argument("--weights", default="yolo26n-seg.pt",
                    help="pesos preentrenados (alternativa estable: yolo11n-seg.pt)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="smoke", help="nombre de la corrida")
    ap.add_argument("--project", default="runs_smoke")
    args = ap.parse_args()

    # 1) comprobaciones antes de tocar nada pesado
    if not args.data.exists():
        raise SystemExit(f"[error] no existe el data.yaml: {args.data.resolve()}\n"
                         f"        ¿Corriste build_dataset.py primero?")

    # 2) ¿hay GPU? (solo informativo; en CPU funciona pero va lento)
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU"
        print(f"[info] CUDA disponible: {cuda}  ->  entrenando en: {dev}")
        if not cuda:
            print("[info] en CPU esto será lento; para el entrenamiento real usa GPU (Colab, etc.)")
    except Exception:
        print("[info] no se pudo consultar torch; se deja que YOLO elija el dispositivo")

    # 3) ¿el fold tiene split de test?
    has_test = "test:" in args.data.read_text(encoding="utf-8")

    # 4) entrenar (pocas épocas)
    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"\n[info] smoke test: {args.epochs} épocas, imgsz={args.imgsz}, batch={args.batch}")
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=args.project, name=args.name,
        # aumentos aptos para vista cenital (se aplican solo a train)
        degrees=180.0, fliplr=0.5, flipud=0.5, hsv_s=0.7, hsv_v=0.4,
    )

    run_dir = Path(args.project) / args.name

    # 5) validación explícita sobre el split de test, si existe
    if has_test:
        print("\n[info] evaluando el split de TEST...")
        metrics = model.val(data=str(args.data), split="test",
                            project=args.project, name=f"{args.name}_test")
        print("[resultado] métricas en test:", seg_metrics(metrics))
    else:
        print("\n[info] este fold no tiene split 'test:'; mira las métricas de val.")

    # 6) dónde mirar
    print(f"\nRevisar en: {run_dir.resolve()}")
    print("  - train_batch0.jpg      → imágenes de entrenamiento con máscaras (¿ajustan bien?)")
    print("  - val_batch0_labels.jpg → ground truth de validación")
    print("  - val_batch0_pred.jpg   → predicciones (vacías al inicio es normal)")
    print("  - results.csv / results.png → curvas de pérdida y métricas por época")
    print("\nRecuerda: con pocas épocas las métricas pueden salir en 0; eso NO indica un fallo,")
    print("solo que el modelo aún no aprendió lo suficiente. Es una prueba de plomería.")


if __name__ == "__main__":
    main()