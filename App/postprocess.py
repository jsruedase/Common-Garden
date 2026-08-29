"""
postprocess.py — Post-procesa las predicciones de YOLO26-seg:

  - PLANTAS (clase 0): las máscaras salen como polígonos arbitrarios; se fuerza
    su envolvente CONVEXA (convex hull), acorde a la convención de anotación.
  - MATERAS (clase 1): se ajusta una ELIPSE a la máscara (centro, ejes, ángulo),
    recuperando la parametrización original sin quedarnos con el polígono.

Para cada instancia también se calcula el CENTROIDE, que es lo que alimenta la
etapa de re-identificación / seguimiento.

Salida por imagen:
  - <stem>.json  objeto con: image, image_width, image_height, e "instances"
                 (cada instancia: clase, forma [convex_polygon/ellipse], centroide).
                 El ancho/alto se guardan para poder convertir a Label Studio
                 (que usa coordenadas en % de la imagen).
  - <stem>.png   overlay para revisar visualmente (opcional, --save-vis)

Uso:
    python postprocess.py --weights best.pt --source dataset/images --save-vis
    # además, exportar tareas de Label Studio (pre-anotaciones para corregir):
    python postprocess.py --weights best.pt --source dataset/images --labelstudio
    python postprocess.py --weights best.pt --source img.JPG --labelstudio \
        --url-prefix "/data/local-files/?d=images/"
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {0: "plant", 1: "pot"}


# ─────────────────────────────────────────────────────────────────────────────
# Geometría (funciones puras, testeables sin YOLO)
# ─────────────────────────────────────────────────────────────────────────────
def convex_hull_polygon(points: np.ndarray) -> np.ndarray:
    """points (N,2) → vértices de la envolvente convexa (M,2), en orden."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)                     # (M,1,2)
    return hull.reshape(-1, 2)


def convexity_ratio(points: np.ndarray) -> float:
    """área(polígono) / área(envolvente convexa) ∈ (0,1]. 1.0 = ya era convexo."""
    pts = np.asarray(points, dtype=np.float32)
    a_poly = cv2.contourArea(pts.reshape(-1, 1, 2))
    a_hull = cv2.contourArea(convex_hull_polygon(pts).reshape(-1, 1, 2))
    return float(a_poly / a_hull) if a_hull > 0 else 1.0


def fit_ellipse(points: np.ndarray):
    """
    Ajusta una elipse a points (N,2). Devuelve dict con centro, semiejes y ángulo,
    o None si hay <5 puntos (mínimo que exige cv2.fitEllipse).
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 5:
        return None
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    # d1,d2 = diámetros en el marco local de la elipse (x local, y local) a 'angle'.
    # Se guardan los SEMIEJES crudos (rx,ry) y la rotación tal cual: así el round-trip
    # a Label Studio (x,y,radiusX,radiusY,rotation) es exacto y sin ambigüedad de eje.
    return {
        "cx": float(cx), "cy": float(cy),
        "rx": float(d1 / 2.0), "ry": float(d2 / 2.0),   # semiejes en píxeles (marco local)
        "rotation_deg": float(angle),
        "axis_major": float(max(d1, d2)), "axis_minor": float(min(d1, d2)),  # solo informativo
    }


def centroid(points: np.ndarray) -> tuple:
    """Centroide del polígono (o de los puntos si el área es ~0)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    m = cv2.moments(pts)
    if abs(m["m00"]) > 1e-6:
        return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))
    c = pts.reshape(-1, 2).mean(axis=0)
    return (float(c[0]), float(c[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Procesar una instancia
# ─────────────────────────────────────────────────────────────────────────────
def process_instance(cls: int, polygon_xy: np.ndarray) -> dict:
    name = CLASS_NAMES.get(cls, str(cls))
    if cls == 0:  # planta → convexa
        hull = convex_hull_polygon(polygon_xy)
        return {
            "class": name,
            "shape": "convex_polygon",
            "polygon": hull.round(2).tolist(),
            "convexity_ratio_before": round(convexity_ratio(polygon_xy), 4),
            "centroid": [round(v, 2) for v in centroid(hull)],
        }
    else:         # matera → elipse
        ell = fit_ellipse(polygon_xy)
        out = {"class": name, "shape": "ellipse", "ellipse": ell}
        if ell is not None:
            out["centroid"] = [round(ell["cx"], 2), round(ell["cy"], 2)]
        else:
            out["shape"] = "polygon_fallback"      # muy pocos puntos para elipse
            out["polygon"] = np.asarray(polygon_xy).round(2).tolist()
            out["centroid"] = [round(v, 2) for v in centroid(polygon_xy)]
        return out


def draw_overlay(img, instances):
    for ins in instances:
        if ins["shape"] == "convex_polygon":
            pts = np.array(ins["polygon"], np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 200, 0), 2)
        elif ins["shape"] == "ellipse" and ins["ellipse"]:
            e = ins["ellipse"]
            cv2.ellipse(img, (int(e["cx"]), int(e["cy"])),
                        (int(e["rx"]), int(e["ry"])),
                        e["rotation_deg"], 0, 360, (0, 120, 255), 2)
        cx, cy = [int(v) for v in ins["centroid"]]
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Driver (usa YOLO para predecir; import perezoso)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Export a Label Studio (pre-anotaciones para que el biólogo corrija)
# ─────────────────────────────────────────────────────────────────────────────
PLANT_LABEL = "Planta"
POT_LABEL = "Matera"
MODEL_VERSION = "yolo26m-seg"


def _rid(n: int = 10) -> str:
    """id de region estilo Label Studio (alfanumérico)."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _ls_entry(rid, from_name, type_, value, W, H):
    return {"id": rid, "type": type_, "from_name": from_name, "to_name": "image",
            "original_width": int(W), "original_height": int(H),
            "image_rotation": 0, "origin": "manual", "value": value}


def _pct(v, size):
    return round(float(v) / float(size) * 100.0, 6)


def _plant_region(inst, W, H):
    rid = _rid()
    pts = [[_pct(x, W), _pct(y, H)] for x, y in inst["polygon"]]
    geo = {"points": pts, "closed": True}
    return [
        _ls_entry(rid, "plant_polygon", "polygonlabels", {**geo, "polygonlabels": [PLANT_LABEL]}, W, H),
        _ls_entry(rid, "collection_id", "textarea", {**geo, "text": [""]}, W, H),
        _ls_entry(rid, "bed_position", "textarea", {**geo, "text": [""]}, W, H),
    ]


def _ellipse_geo(inst, W, H):
    """Geometría de elipse en % (semiejes crudos + rotación); fallback a bbox del polígono."""
    e = inst.get("ellipse")
    if e and all(k in e for k in ("cx", "cy", "rx", "ry", "rotation_deg")):
        return {"x": _pct(e["cx"], W), "y": _pct(e["cy"], H),
                "radiusX": _pct(e["rx"], W), "radiusY": _pct(e["ry"], H),
                "rotation": round(float(e["rotation_deg"]), 6)}
    poly = inst.get("polygon", [])
    if not poly:
        return None
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    rx, ry = (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2
    return {"x": _pct(cx, W), "y": _pct(cy, H),
            "radiusX": _pct(rx, W), "radiusY": _pct(ry, H), "rotation": 0}


def _pot_region(inst, W, H):
    geo = _ellipse_geo(inst, W, H)
    if geo is None:
        return []
    rid = _rid()
    return [
        _ls_entry(rid, "pot_ellipse", "ellipselabels", {**geo, "ellipselabels": [POT_LABEL]}, W, H),
        _ls_entry(rid, "collection_id", "textarea", {**geo, "text": [""]}, W, H),
        _ls_entry(rid, "bed_position", "textarea", {**geo, "text": [""]}, W, H),
    ]


def record_to_labelstudio(record: dict, url_prefix: str = "") -> dict:
    """Convierte un 'record' de postprocess en una tarea de Label Studio con pre-anotaciones."""
    W, H = record["image_width"], record["image_height"]
    result = []
    for inst in record.get("instances", []):
        if inst.get("class") == "plant" and inst.get("shape") == "convex_polygon":
            result += _plant_region(inst, W, H)
        elif inst.get("class") == "pot":
            result += _pot_region(inst, W, H)
    return {"data": {"image": url_prefix + record["image"]},
            "predictions": [{"model_version": MODEL_VERSION, "result": result}]}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# API para la app (sin argparse, rutas explícitas)
# ─────────────────────────────────────────────────────────────────────────────
def run_postprocess(weights, source, out_dir, conf=0.25, imgsz=1280,
                    save_vis=False, labelstudio=False, ls_out=None, url_prefix=""):
    """
    Corre la inferencia y el post-proceso (hulls convexos + elipses + centroides).

    weights   : ruta a best.pt
    source    : imagen o carpeta de imágenes
    out_dir   : carpeta de salida para los JSON (+ overlays si save_vis)
    labelstudio: además genera tareas de Label Studio con pre-anotaciones
    url_prefix : prefijo para data.image (p. ej. "/data/local-files/?d=images/")

    Devuelve dict con: n_images, records, per_image (stats), files, ls_files.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ls_dir = None
    if labelstudio:
        ls_dir = Path(ls_out) if ls_out else (out_dir / "labelstudio")
        ls_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(weights))
    results = model.predict(source=str(source), conf=conf, imgsz=imgsz,
                            verbose=False, stream=True)

    records, per_image, files, ls_files = [], [], [], []
    for res in results:
        stem = Path(res.path).stem
        H, W = res.orig_shape
        instances = []
        if res.masks is not None:
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for poly_xy, cls in zip(res.masks.xy, classes):
                if len(poly_xy) >= 3:
                    instances.append(process_instance(int(cls), np.asarray(poly_xy)))
        record = {"image": Path(res.path).name, "image_width": int(W),
                  "image_height": int(H), "instances": instances}
        records.append(record)
        p = out_dir / f"{stem}.json"
        p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        files.append(p)

        if ls_dir is not None:
            task = record_to_labelstudio(record, url_prefix)
            lp = ls_dir / f"{stem}_ls.json"
            lp.write_text(json.dumps([task], ensure_ascii=False, indent=2), encoding="utf-8")
            ls_files.append(lp)

        vis = None
        if save_vis:
            img = cv2.imread(res.path)
            if img is not None:
                vis = out_dir / f"{stem}.png"
                cv2.imwrite(str(vis), draw_overlay(img, instances))
        per_image.append(dict(
            image=record["image"], stem=stem,
            plants=sum(1 for i in instances if i["class"] == "plant"),
            pots=sum(1 for i in instances if i["class"] == "pot"),
            overlay=str(vis) if vis else None,
        ))

    return dict(n_images=len(records), records=records, per_image=per_image,
                files=files, ls_files=ls_files)


def main():
    ap = argparse.ArgumentParser(description="Convexos (plantas) + elipses (materas)")
    ap.add_argument("--weights", default=Path("Segmentation_Model/best.pt"), type=Path)
    ap.add_argument("--source", default=Path("Assets/images"), type=Path,
                    help="imagen o carpeta de imágenes")
    ap.add_argument("--out", default=Path("Outputs/postprocessed"), type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--save-vis", action="store_true", help="guardar overlays .png")
    ap.add_argument("--labelstudio", action="store_true",
                    help="además, exportar tareas de Label Studio (pre-anotaciones)")
    ap.add_argument("--ls-out", type=Path, default=None)
    ap.add_argument("--url-prefix", default="")
    args = ap.parse_args()

    r = run_postprocess(args.weights, args.source, args.out, args.conf, args.imgsz,
                        args.save_vis, args.labelstudio, args.ls_out, args.url_prefix)
    for d in r["per_image"]:
        print(f"  {d['image']:38s} plantas={d['plants']:3d} materas={d['pots']:3d}")
    print(f"[listo] {r['n_images']} imagen(es) → {Path(args.out).resolve()}")
    if r["ls_files"]:
        print(f"[listo] {len(r['ls_files'])} tarea(s) Label Studio → "
              f"{r['ls_files'][0].parent.resolve()}")


if __name__ == "__main__":
    main()
