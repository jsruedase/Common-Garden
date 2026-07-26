"""
train_gpu.py — Entrenamiento REAL de YOLO26-seg en máquina con GPU.

Funciona igual en NVIDIA (CUDA) que en AMD (ROCm): con la build de PyTorch para
ROCm, la GPU AMD se ve como dispositivo 'cuda' y torch.cuda.is_available() da True,
así que no hay que cambiar nada del código — solo instalar la rueda de PyTorch
correcta (CUDA o ROCm) y pasar --device 0.

Uso:
    python modeling/train_gpu.py                                     # obj1_final
    python modeling/train_gpu.py --data datasets/obj2_loto/held_ST1/data.yaml
    python modeling/train_gpu.py --weights yolo26m-seg.pt --imgsz 1280 --batch -1
    python modeling/train_gpu.py --device 0,1                        # multi-GPU
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
from pathlib import Path


def describe_device():
    try:
        import torch
        if not torch.cuda.is_available():
            print("[AVISO] no se detecta GPU (torch.cuda.is_available()=False).")
            print("        NVIDIA -> instala PyTorch CUDA;  AMD -> instala PyTorch ROCm (en Linux).")
            print("        Mientras tanto puedes usar train_cpu.py, pero será lento.")
            return False
        is_rocm = getattr(torch.version, "hip", None) is not None
        backend = "ROCm/HIP (AMD)" if is_rocm else "CUDA (NVIDIA)"
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[info] GPU: {name}  |  {vram:.0f} GB  |  backend: {backend}")
        return True
    except Exception as e:
        print(f"[aviso] no se pudo consultar torch ({e}); se deja que YOLO elija el dispositivo")
        return True


def main():
    ap = argparse.ArgumentParser(description="Entrenamiento GPU de YOLO26-seg")
    ap.add_argument("--data", type=Path, default=Path("datasets/obj1_final/data.yaml"))
    ap.add_argument("--weights", default="yolo26s-seg.pt",
                    help="'s' es buen punto medio con pocos datos; 'm'/'l' si tienes más imágenes")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1280, help="alto para objetos pequeños (frailejones)")
    ap.add_argument("--batch", type=int, default=4,
                    help="fijo; evita el auto-batch (-1) que estima mal con pocas imágenes a 1280px")
    ap.add_argument("--device", default="0", help="'0', '0,1' para multi-GPU, o 'cpu'")
    ap.add_argument("--name", default="gpu")
    ap.add_argument("--project", default="runs_train")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"[error] no existe: {args.data.resolve()} (¿corriste build_dataset.py?)")

    describe_device()
    has_test = "test:" in args.data.read_text(encoding="utf-8")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"[info] entrenando {args.weights}: {args.epochs} épocas, imgsz={args.imgsz}, "
          f"batch={args.batch}, device={args.device}")
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=args.project, name=args.name,
        # rendimiento en GPU
        amp=True, cache="ram", workers=8, plots=True,
        # pocos datos → paciencia alta y cerrar mosaic al final
        patience=50, close_mosaic=15,
        # aumentos aptos para vista cenital (solo se aplican a train)
        degrees=180.0, fliplr=0.5, flipud=0.5, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5,
    )

    if has_test:
        print("\n[info] evaluando split de TEST...")
        m = model.val(data=str(args.data), split="test",
                      project=args.project, name=f"{args.name}_test")
        try:
            print(f"[test] mask mAP50-95={m.seg.map:.4f}  mask mAP50={m.seg.map50:.4f}  "
                  f"box mAP50-95={m.box.map:.4f}")
        except Exception:
            pass

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[listo] mejores pesos: {best.resolve()}")
    print("        siguiente paso: postprocess.py con estos pesos (convexos + elipses).")


if __name__ == "__main__":
    main()
