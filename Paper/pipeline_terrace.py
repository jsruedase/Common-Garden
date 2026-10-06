#!/usr/bin/env python3
"""
Paper/pipeline_terrace.py — El pipeline completo sobre una terraza nueva.
========================================================================

Encadena las tres etapas del proyecto sobre una carpeta de fotos y deja, por
fecha, la imagen con las anotaciones y el identificador propagado:

    1. SEGMENTAR   YOLO-seg (best.pt) + postproceso (casco convexo / elipse)
                   -> una tarea de Label Studio por foto, sin identidades
    2. SEMBRAR     la fecha 1 recibe los bed_position: del export que anoto el
                   biologo (--seed-export) o, si no hay, ids sinteticos P01..
    3. PROPAGAR    ICP sobre las materas + Hungarian sobre las plantas llevan
                   esos ids a las fechas siguientes (el mismo codigo de la app)
    4. DIBUJAR     Paper/annotate_export.py pinta cada fecha ya identificada

Label Studio es el formato de intercambio entre etapas, no un adorno: cada paso
escribe un JSON legible que se puede abrir, revisar o reimportar, y el paso
siguiente lo lee. Si algo sale raro, se ve en que etapa paso.

AVISO sobre lo que esto mide: en una terraza sin anotar no hay con que
contrastar. Las figuras muestran lo que el pipeline DECIDIO, no si acerto. Para
numeros hace falta una terraza anotada (ver Paper/highlight_track.py, tier-1).

Uso:
    python Paper/pipeline_terrace.py --images Assets/images/ST6
    python Paper/pipeline_terrace.py --images Assets/images/ST6 \
        --seed-export "C:/.../project-2-at-2026-10-06.json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import detections as D
from detections import PLANT, POT
import annotate_export as AE
import render

import postprocess as pp              # Segmentation_Model/
import track_to_labelstudio as t2ls   # Time_Tracking/
import diagnostic                     # Time_Tracking/
from scipy.optimize import linear_sum_assignment

IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Leer/escribir tareas de Label Studio
# ---------------------------------------------------------------------------
def _regions_with_id(task: dict) -> list:
    """[(region_id, kind, bed, coll, shape_px)] — como read_regions pero con el id."""
    shapes, _W, _H = D._shapes_by_region(task)
    kinds = AE._kinds_by_region(task)
    texts = AE._texts_by_region(task)
    out = []
    for rid, shape in shapes.items():
        t = texts.get(rid, {})
        out.append((rid, kinds.get(rid, "?"), t.get("bed_position", ""),
                    t.get("collection_id", ""), shape))
    return out


def _set_text(task: dict, rid: str, field: str, value: str) -> None:
    for r in D._result_of(task):
        if r.get("id") == rid and r.get("from_name") == field and r.get("type") == "textarea":
            r.setdefault("value", {})["text"] = [value]


def _seq_of(path: Path) -> int:
    m = D.FNAME_RE.match(path.name)
    return int(m.group("seq")) if m else 0


# ---------------------------------------------------------------------------
# 1. Segmentar
# ---------------------------------------------------------------------------
def segment(images_dir: Path, weights: Path, conf: float, imgsz: int,
            dedupe=True, dup_iou=0.5, dup_contain=0.8, log=print) -> list:
    """Una tarea de Label Studio por foto, con las formas post-procesadas y sin ids."""
    from ultralytics import YOLO

    photos = sorted((f for f in Path(images_dir).iterdir() if f.suffix.lower() in IMG_EXTS),
                    key=_seq_of)
    if len(photos) < 2:
        raise SystemExit(f"[error] se necesitan >=2 fotos en {images_dir} (hay {len(photos)}).")

    log(f"[1/4] segmentando {len(photos)} fotos con {Path(weights).name} ...")
    model = YOLO(str(weights))
    tasks = []
    for f in photos:
        res = list(model.predict(source=str(f), conf=conf, imgsz=imgsz, verbose=False))[0]
        H, W = res.orig_shape
        instances = []
        if res.masks is not None and res.boxes is not None:
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for poly, c in zip(res.masks.xy, classes):
                poly = np.asarray(poly, float)
                if len(poly) >= 3:
                    instances.append(pp.process_instance(int(c), poly))
        n_dup = 0
        if dedupe:
            # dos mascaras del mismo frailejon rompen el emparejamiento uno-a-uno
            # del tracker: una se queda la identidad y la otra se enrola como NEW
            instances, n_dup = pp.dedupe_instances(instances, dup_iou, dup_contain)
        record = {"image": f.name, "image_width": int(W), "image_height": int(H),
                  "instances": instances}
        task = pp.record_to_labelstudio(record)
        n_pl = sum(1 for i in instances if i.get("class") == "plant")
        log(f"      {f.name:34s} plantas={n_pl:3d} materas={len(instances) - n_pl:3d}"
            + (f"  (-{n_dup} repetida(s))" if n_dup else ""))
        tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# 2. Sembrar la fecha 1
# ---------------------------------------------------------------------------
def seed_from_export(task: dict, seed_task: dict, gate_frac: float = 0.5, log=print) -> int:
    """
    Copia bed_position/collection_id del export ANOTADO a las plantas detectadas
    de la misma fecha, emparejando por centroide (Hungarian + gate).

    Es lo que hace el biologo en la practica: anota sobre lo que el segmentador
    detecto. Las materas no se tocan: heredan el id de su planta al propagar.
    """
    det = [(rid, AE._centroid(sh)) for rid, kind, _b, _c, sh in _regions_with_id(task)
           if kind == "plant"]
    ref = [(b, c, AE._centroid(sh)) for _rid, kind, b, c, sh in _regions_with_id(seed_task)
           if kind == "plant" and b]
    if not det or not ref:
        log("[aviso] no se pudo sembrar: faltan plantas en la deteccion o en el export")
        return 0

    Dx = np.array([p for _, p in det], float)
    Gx = np.array([p for _, _, p in ref], float)
    gate = gate_frac * diagnostic.nn_spacing(Gx)
    if not np.isfinite(gate) or gate <= 0:
        gate = np.inf
    C = np.linalg.norm(Dx[:, None] - Gx[None], axis=2)
    matched, n = set(), 0
    for r, c in zip(*linear_sum_assignment(C)):
        if C[r, c] > gate:
            continue
        rid = det[r][0]
        bed, coll, _ = ref[c]
        _set_text(task, rid, "bed_position", bed)
        if coll:
            _set_text(task, rid, "collection_id", coll)
        matched.add(r)
        n += 1

    # Detecciones de la fecha 1 que el export no respalda: casi siempre falsos
    # positivos del segmentador. Se les pone un id corto y marcado ("?1", "?2")
    # en vez de dejarlos vacios: asi load_terrace no inventa "?<id-de-region>",
    # que en la figura sale como una cadena aleatoria ilegible.
    extra = 0
    for r in range(len(det)):
        if r not in matched:
            extra += 1
            _set_text(task, det[r][0], "bed_position", f"?{extra}")

    log(f"[2/4] fecha 1 sembrada: {n}/{len(det)} plantas detectadas recibieron "
        f"bed_position (el export anotado trae {len(ref)})")
    if extra:
        log(f"      {extra} deteccion(es) sin respaldo en el export -> ?1..?{extra}")
    if n < len(ref):
        log(f"      {len(ref) - n} planta(s) anotada(s) que el segmentador no detecto")
    return n


def seed_synthetic(task: dict, log=print) -> int:
    """Sin export anotado: numera P01..Pnn en orden de lectura. Ids ARBITRARIOS."""
    plants = [(rid, AE._centroid(sh)) for rid, kind, _b, _c, sh in _regions_with_id(task)
              if kind == "plant"]
    if not plants:
        return 0
    pts = np.array([p for _, p in plants], float)
    for n, k in enumerate(D._reading_order(pts)):
        _set_text(task, plants[k][0], "bed_position", f"P{n + 1:02d}")
    log(f"[2/4] fecha 1 sembrada: {len(plants)} plantas numeradas P01..P{len(plants):02d} "
        f"(ids arbitrarios: no hay anotacion)")
    return len(plants)


# ---------------------------------------------------------------------------
# 3. Propagar
# ---------------------------------------------------------------------------
def propagate(pred_path: Path, log=print):
    """
    Lleva los ids de la fecha 1 al resto. Es EXACTAMENTE la cadena de la app:
    load_terrace(require_ids=False) -> propagate -> pots_follow_plants -> write_ids.
    Devuelve [(frame, tarea_con_ids, stats)].
    """
    frames = diagnostic.load_terrace(str(pred_path), require_ids=False, verbose=False)
    if len(frames) < 2:
        raise SystemExit("[error] el JSON de segmentacion no tiene >=2 fechas legibles.")
    maps, ordered = t2ls.propagate(frames)
    coll = t2ls.collection_lookup(ordered)

    log(f"[3/4] propagando identidades por {len(ordered)} fechas ...")
    out = []
    for f, m in zip(ordered, maps):
        full = dict(m)
        full.update(t2ls.pots_follow_plants(f, full))
        task, n = t2ls.write_ids(f, full, coll)
        new = sum(1 for v in full.values() if str(v).startswith("NEW"))
        unk = sum(1 for v in full.values() if str(v).startswith("?"))
        log(f"      {f['date'][:22]:24s} regiones con id={n:3d}"
            + (f"  nuevas={new}" if new else "") + (f"  sin-id={unk}" if unk else ""))
        out.append((f, task, dict(regions=n, new=new, unknown=unk)))
    return out


# ---------------------------------------------------------------------------
# 4. Dibujar
# ---------------------------------------------------------------------------
def draw(tasks: list, images_dir: Path, out_dir: Path, args, log=print) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[4/4] dibujando {len(tasks)} fechas ...")
    cells = []
    for frame, task, _stats in tasks:
        regions, W, H = AE.read_regions(task)
        photo = AE.resolve_image(task, images_dir)
        img = cv2.imread(str(photo)) if photo else None
        if img is None:
            img = np.full((H or 800, W or 1200, 3), 245, np.uint8)
        vis = AE.annotate(img, regions, [], labels=args.labels,
                          label_pos=args.label_pos, fill=args.fill)
        if args.max_width and vis.shape[1] > args.max_width:
            s = args.max_width / float(vis.shape[1])
            vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        name = f"{frame['seq']}-{frame['terrace']}_{frame['date']}_pipeline.jpg"
        cv2.imwrite(str(out_dir / name), vis, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        log(f"      -> {name}")
        small = cv2.resize(vis, (900, int(900 * vis.shape[0] / vis.shape[1])),
                           interpolation=cv2.INTER_AREA)
        cells.append((small, f"{frame['seq']}. {frame['date'].replace('_', ' ')}"))
    return cells


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pipeline completo (segmentar -> sembrar -> propagar -> dibujar) "
                    "sobre una carpeta de fotos de una terraza")
    ap.add_argument("--images", type=Path, required=True,
                    help="carpeta con las fotos de UNA terraza (p.ej. Assets/images/ST6)")
    ap.add_argument("--seed-export", type=Path, default=None,
                    help="export de Label Studio de la PRIMERA fecha ya anotada; "
                         "sin el, los ids de la fecha 1 son arbitrarios (P01..)")
    ap.add_argument("--weights", type=Path, default=D.DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=None,
                    help="por defecto Paper/images/pipeline_<terraza>")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-dedupe", action="store_true",
                    help="conservar las detecciones repetidas del mismo objeto")
    ap.add_argument("--dup-iou", type=float, default=0.5)
    ap.add_argument("--dup-contain", type=float, default=0.8)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--labels", choices=["bed", "collection", "both", "none"], default="bed")
    ap.add_argument("--label-pos", choices=["above", "center"], default="above")
    ap.add_argument("--fill", type=float, default=0.30)
    ap.add_argument("--max-width", type=int, default=2200)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--no-strip", action="store_true", help="no montar la rejilla resumen")
    args = ap.parse_args()

    if not args.images.is_dir():
        raise SystemExit(f"[error] no es una carpeta: {args.images.resolve()}")
    photos = sorted((f for f in args.images.iterdir() if f.suffix.lower() in IMG_EXTS),
                    key=_seq_of)
    if not photos:
        raise SystemExit(f"[error] no hay fotos en {args.images.resolve()}")
    m = D.FNAME_RE.match(photos[0].name)
    terr = f"ST{m.group('terrace')}" if m else args.images.name

    out_root = Path(args.out) if args.out else \
        Path(__file__).resolve().parent / "images" / f"pipeline_{terr}"
    seg_dir = out_root / "01_segmentacion"
    trk_dir = out_root / "02_tracking"
    fig_dir = out_root / "03_figuras"
    for d in (seg_dir, trk_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[terraza] {terr}: {len(photos)} fotos en {args.images}")

    tasks = segment(args.images, args.weights, args.conf, args.imgsz,
                    dedupe=not args.no_dedupe, dup_iou=args.dup_iou,
                    dup_contain=args.dup_contain)
    for t in tasks:
        stem = Path(t["data"]["image"]).stem
        (seg_dir / f"{stem}_ls.json").write_text(
            json.dumps([t], ensure_ascii=False, indent=2), encoding="utf-8")

    tasks.sort(key=lambda t: _seq_of(Path(t["data"]["image"])))
    if args.seed_export:
        if not args.seed_export.exists():
            raise SystemExit(f"[error] no existe: {args.seed_export.resolve()}")
        seed = json.loads(args.seed_export.read_text(encoding="utf-8"))
        if isinstance(seed, dict):
            seed = [seed]
        n_seeded = seed_from_export(tasks[0], seed[0])
        seeding = f"export anotado ({args.seed_export.name})"
    else:
        n_seeded = seed_synthetic(tasks[0])
        seeding = "ids arbitrarios P01.. (sin anotacion)"
    if not n_seeded:
        raise SystemExit("[error] la fecha 1 quedo sin identidades; no hay de donde propagar.")

    pred_path = seg_dir / f"{terr}_segmentado.json"
    pred_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    tracked = propagate(pred_path)
    for frame, task, _s in tracked:
        name = f"{frame['seq']}-{frame['terrace']}_{frame['date']}_tracked.json"
        (trk_dir / name).write_text(json.dumps([task], ensure_ascii=False, indent=2),
                                    encoding="utf-8")

    cells = draw(tracked, args.images, fig_dir, args)

    if not args.no_strip and cells:
        # dos figuras con el mismo material: la rejilla se LEE (los ids son
        # legibles), la cascada se MIRA (abre el articulo y dice "serie temporal")
        rows = [cells[i:i + 4] for i in range(0, len(cells), 4)]
        render.strip(rows, out_root / f"{terr}_pipeline.png",
                     f"{terr} - pipeline completo: segmentacion YOLO + propagacion de "
                     f"identidades   (siembra: {seeding})")
        render.cascade(cells, out_root / f"{terr}_pipeline_cascada.png")

    summary = dict(terrace=terr, photos=len(photos), seeding=seeding, seeded=n_seeded,
                   dates=[dict(seq=f["seq"], date=f["date"], **s) for f, _t, s in tracked])
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[listo] {out_root.resolve()}")
    print("        01_segmentacion/  tareas LS sin ids (salida del segmentador)")
    print("        02_tracking/      tareas LS con los ids propagados (reimportables)")
    print("        03_figuras/       una imagen por fecha")
    if not args.no_strip:
        print(f"        {terr}_pipeline.png           rejilla legible")
        print(f"        {terr}_pipeline_cascada.png   cascada de apertura")
    print("\n[ojo]   terraza sin ground truth: la figura muestra lo que el pipeline")
    print("        DECIDIO, no si acerto. Para medir hace falta una terraza anotada.")


if __name__ == "__main__":
    main()
