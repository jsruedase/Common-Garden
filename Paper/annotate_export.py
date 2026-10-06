#!/usr/bin/env python3
"""
Paper/annotate_export.py — Pinta un export de Label Studio sobre su foto.
========================================================================

Toma el JSON que Label Studio exporta y devuelve la imagen con TODAS las
anotaciones dibujadas y, encima de cada cama, su identificador:

    export de LS  ->  [este script]  ->  <foto>_annotated.jpg

Qué dibuja:
  - PLANTA  (plant_polygon, "Planta")  contorno verde
  - MATERA  (pot_ellipse,  "Matera")   elipse azul
  - una ETIQUETA por cama con su bed_position (el id), no una por region:
    la planta y su matera comparten bed_position, asi que repetirlo dos veces
    solo ensucia. Con --labels both se agrega tambien el collection_id.
  - en rojo, las regiones SIN bed_position: son las que hay que completar en
    Label Studio antes de que el seguimiento pueda sembrarse con ellas.

La foto se busca sola en --images a partir de data.image / file_upload del
task, quitando el hash que antepone Label Studio y el sufijo "_ls" que dejan
las tareas generadas por Segmentation_Model/postprocess.py. Si no aparece, se
dibuja sobre un lienzo blanco del tamano que declara el export.

Uso:
    python Paper/annotate_export.py export.json
    python Paper/annotate_export.py export.json --labels both
    python Paper/annotate_export.py export.json --image Assets/images/1-ST6_Sep_1-5_2025.JPG
    python Paper/annotate_export.py export.json --out Paper/images/annotated

Sirve para cualquier export: anotado a mano, o las pre-anotaciones de
postprocess.py / track_to_labelstudio.py (si no hay 'annotations' usa
'predictions'). Reutiliza la conversion de coordenadas de Paper/detections.py
para que lo dibujado sea EXACTAMENTE lo que lee el tracker.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np

import detections as D
from detections import shape_points
import render

PLANT_COLOR = (90, 210, 90)      # BGR: verde
POT_COLOR = (40, 150, 255)       # BGR: naranja
NO_ID_COLOR = (60, 60, 230)      # BGR: rojo -> region sin bed_position
IMG_EXTS = {".jpg", ".jpeg", ".png"}
LS_HASH = re.compile(r"^[0-9a-f]{8}-", re.I)


# ---------------------------------------------------------------------------
# Localizar la foto del task
# ---------------------------------------------------------------------------
def _candidate_stems(task: dict) -> list:
    """Nombres plausibles de la foto, en orden de confianza."""
    raw = []
    img = (task.get("data") or {}).get("image")
    if img:
        raw.append(str(img))
    if task.get("file_upload"):
        raw.append(str(task["file_upload"]))

    stems = []
    for r in raw:
        name = Path(unquote(r).replace("\\", "/")).name
        name = LS_HASH.sub("", name)          # 37566bdc-1-ST6_... -> 1-ST6_...
        stem = Path(name).stem
        if stem.endswith("_ls"):              # 1-ST6_..._ls.json -> 1-ST6_...
            stem = stem[:-3]
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def resolve_image(task: dict, images_dir, override=None):
    """Ruta de la foto, o None si no esta en --images."""
    if override:
        p = Path(override)
        return p if p.exists() else None
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return None
    pool = [f for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
    for stem in _candidate_stems(task):
        for f in pool:
            if f.stem.lower() == stem.lower():
                return f
    return None


# ---------------------------------------------------------------------------
# Leer las regiones del task
# ---------------------------------------------------------------------------
def _texts_by_region(task: dict) -> dict:
    """{region_id: {'bed_position': ..., 'collection_id': ...}}"""
    out = {}
    for r in D._result_of(task):
        fn = r.get("from_name")
        if r.get("type") != "textarea" or fn not in ("bed_position", "collection_id"):
            continue
        txt = (r.get("value") or {}).get("text") or [""]
        out.setdefault(r.get("id"), {})[fn] = str(txt[0]).strip()
    return out


def _kinds_by_region(task: dict) -> dict:
    """{region_id: 'plant'|'pot'}"""
    out = {}
    for r in D._result_of(task):
        fn = r.get("from_name")
        if fn == "plant_polygon":
            out[r.get("id")] = "plant"
        elif fn == "pot_ellipse":
            out[r.get("id")] = "pot"
    return out


def read_regions(task: dict):
    """
    [(kind, bed, coll, shape_px)] mas el tamano de la imagen que declara el export.

    La geometria sale de detections._shapes_by_region, la MISMA conversion de
    porcentajes a pixeles que usa el tracker: si se duplicara aqui, el dibujo y
    el seguimiento podrian discrepar sin que nadie lo note.
    """
    shapes, W, H = D._shapes_by_region(task)
    texts = _texts_by_region(task)
    kinds = _kinds_by_region(task)
    regions = []
    for rid, shape in shapes.items():
        t = texts.get(rid, {})
        regions.append((kinds.get(rid, "?"), t.get("bed_position", ""),
                        t.get("collection_id", ""), shape))
    return regions, W, H


def _centroid(shape: dict) -> np.ndarray:
    pts = shape_points(shape).astype(np.float32)
    m = cv2.moments(pts.reshape(-1, 1, 2))
    if abs(m["m00"]) > 1e-6:
        return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], float)
    return pts.mean(0)


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def _label(img, centre, text, k, color=render.INK, fg=render.WHITE):
    """Etiqueta centrada en `centre`, recortada a la imagen."""
    fs = 0.95 * k
    th = max(1, int(round(2.0 * k)))
    (tw, tht), _ = cv2.getTextSize(text, render.FONT, fs, th)
    pad = int(round(9 * k))
    x = int(np.clip(centre[0] - tw / 2 - pad, 0, img.shape[1] - tw - 2 * pad - 1))
    y = int(np.clip(centre[1] - tht / 2 - pad, 0, img.shape[0] - tht - 2 * pad - 1))
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad, y + tht + 2 * pad), color, -1)
    cv2.putText(img, text, (x + pad, y + pad + tht), render.FONT, fs, fg, th, cv2.LINE_AA)


def _anchor(shape: dict, where: str, k: float) -> np.ndarray:
    """Donde cuelga la etiqueta: en el centro de la forma o justo encima."""
    if where == "center":
        return _centroid(shape)
    pts = shape_points(shape)
    return np.array([pts[:, 0].mean(), pts[:, 1].min() - 16 * k], float)


def _colour_of(kind: str, bed: str):
    if not bed:
        return NO_ID_COLOR
    return PLANT_COLOR if kind == "plant" else POT_COLOR


def annotate(img, regions, header_lines, labels="bed", label_pos="above", fill=0.30):
    """Dibuja contornos (y relleno) mas una etiqueta por cama. Devuelve la imagen."""
    img = np.ascontiguousarray(img)
    k = max(img.shape[:2]) / 2000.0          # escala de trazos y tipografia
    thick = max(2, int(round(3.0 * k)))

    # relleno primero, en una capa aparte: compuesto de golpe, las formas que se
    # solapan (planta dentro de su matera) no se oscurecen entre si
    if fill > 0:
        layer = img.copy()
        for kind, bed, _coll, shape in regions:
            render.fill_shape(layer, shape, _colour_of(kind, bed))
        img = cv2.addWeighted(layer, fill, img, 1.0 - fill, 0)

    for kind, bed, _coll, shape in regions:
        render.draw_shape(img, shape, _colour_of(kind, bed), thick)

    if labels != "none":
        # una etiqueta por cama: la planta manda, la matera solo si no hay planta
        by_bed = {}
        for kind, bed, coll, shape in regions:
            if not bed:
                _label(img, _anchor(shape, label_pos, k), "?", k, color=NO_ID_COLOR)
                continue
            if bed not in by_bed or kind == "plant":
                by_bed[bed] = (kind, coll, shape)
        for bed, (_kind, coll, shape) in by_bed.items():
            if labels == "collection":
                text = coll or "-"
            elif labels == "both" and coll:
                text = f"{bed}  {coll}"
            else:
                text = bed
            _label(img, _anchor(shape, label_pos, k), text, k)

    if header_lines:
        render._text_box(img, (int(round(30 * k)), int(round(30 * k))), header_lines, k)
    return img


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _out_stem(task: dict, image_path) -> str:
    if image_path:
        return Path(image_path).stem
    stems = _candidate_stems(task)
    return stems[0] if stems else f"task_{task.get('id', 'x')}"


def process_task(task: dict, args, log=print) -> dict:
    regions, W, H = read_regions(task)
    if not regions:
        log(f"[aviso] task {task.get('id')}: sin regiones con forma, se omite")
        return {}

    image_path = resolve_image(task, args.images, args.image)
    img = cv2.imread(str(image_path)) if image_path else None
    if img is None:
        if not (W and H):
            log(f"[aviso] task {task.get('id')}: ni foto ni tamano en el export, se omite")
            return {}
        img = np.full((H, W, 3), 245, np.uint8)   # lienzo: el dibujo igual sirve
        source = "SIN FOTO (lienzo en blanco)"
    else:
        source = Path(image_path).name
        if (W and H) and (img.shape[1], img.shape[0]) != (W, H):
            log(f"[aviso] la foto es {img.shape[1]}x{img.shape[0]} y el export dice "
                f"{W}x{H}: los porcentajes se reescalan a la foto")

    n_plant = sum(1 for k, *_ in regions if k == "plant")
    n_pot = sum(1 for k, *_ in regions if k == "pot")
    n_sin = sum(1 for _, bed, *_ in regions if not bed)
    beds = {bed for _, bed, *_ in regions if bed}
    header = [source,
              f"{len(beds)} camas   {n_plant} plantas   {n_pot} materas",
              "verde = planta    naranja = matera    etiqueta = bed_position"]
    if n_sin:
        header.append(f"{n_sin} region(es) SIN bed_position (en rojo)")

    out = annotate(img, regions, header if args.header else [],
                   labels=args.labels, label_pos=args.label_pos, fill=args.fill)
    if args.max_width and out.shape[1] > args.max_width:
        s = args.max_width / float(out.shape[1])
        out = cv2.resize(out, (int(out.shape[1] * s), int(out.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{_out_stem(task, image_path)}_annotated.{args.format}"
    params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if args.format in ("jpg", "jpeg") else []
    cv2.imwrite(str(dest), out, params)
    log(f"  {source:38s} camas={len(beds):3d} plantas={n_plant:3d} materas={n_pot:3d}"
        + (f"  SIN-ID={n_sin}" if n_sin else "") + f"  -> {dest.name}")
    return dict(task=task.get("id"), image=source, file=str(dest), beds=len(beds),
                plants=n_plant, pots=n_pot, without_id=n_sin)


def main():
    ap = argparse.ArgumentParser(
        description="Dibuja un export de Label Studio (anotaciones + bed_position) sobre su foto")
    ap.add_argument("export", type=Path, help="JSON exportado de Label Studio")
    ap.add_argument("--images", type=Path, default=D.DEFAULT_IMAGES,
                    help="carpeta donde buscar las fotos")
    ap.add_argument("--image", type=Path, default=None,
                    help="forzar ESTA foto (solo util si el export trae un task)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "images" / "annotated")
    ap.add_argument("--labels", choices=["bed", "collection", "both", "none"], default="bed",
                    help="que se escribe en la etiqueta de cada cama")
    ap.add_argument("--fill", type=float, default=0.30,
                    help="opacidad del relleno de plantas y materas (0 = solo contorno)")
    ap.add_argument("--header", action="store_true",
                    help="superponer la cartela con el nombre de la foto y el resumen")
    ap.add_argument("--label-pos", choices=["above", "center"], default="above",
                    help="etiqueta encima de la planta (por defecto) o sobre ella")
    ap.add_argument("--max-width", type=int, default=2200,
                    help="ancho maximo de salida (0 = resolucion original)")
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--quality", type=int, default=92, help="calidad JPEG")
    args = ap.parse_args()

    if not args.export.exists():
        raise SystemExit(f"[error] no existe: {args.export.resolve()}")
    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    if isinstance(tasks, dict):
        tasks = [tasks]
    if not tasks:
        raise SystemExit("[error] el export no trae ninguna tarea.")
    if args.image and len(tasks) > 1:
        raise SystemExit("[error] --image solo vale para un export de UNA tarea; "
                         "con varias, deja que se resuelvan solas desde --images.")

    print(f"[export]  {args.export.name}: {len(tasks)} tarea(s)")
    done = [r for t in tasks if (r := process_task(t, args))]
    if not done:
        raise SystemExit("[error] no se pudo dibujar ninguna tarea.")
    sin = sum(d["without_id"] for d in done)
    print(f"\n[listo] {len(done)} imagen(es) -> {Path(args.out).resolve()}")
    if sin:
        print(f"        ojo: {sin} region(es) sin bed_position, marcadas en rojo con '?'")


if __name__ == "__main__":
    main()
