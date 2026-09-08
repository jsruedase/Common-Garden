#!/usr/bin/env python3
"""
Paper/detections.py — Detecciones CON FORMA para las figuras de seguimiento.
===========================================================================

El resto del repo solo necesita CENTROIDES para seguir plantas
(diagnostic.load_terrace devuelve `dets` = (clase, id, x, y)). Para DIBUJAR hace
falta además la FORMA de cada instancia, así que este módulo arma "frames
enriquecidos" con la misma clave (clase, id) que usa el matcher:

    frame = {
        seq, date, terrace, image, width, height,
        dets   : [(clase, id_provisional, x_px, y_px), ...]   <- lo que ve el tracker
        shapes : {(clase, id_provisional): forma_en_px}       <- solo para dibujar
        gt     : {(clase, id_provisional): bed_position|None} <- solo para reportar
    }

Formas (siempre en PIXELES de la imagen original):
    {"kind": "polygon", "points": (N,2)}
    {"kind": "ellipse", "cx", "cy", "rx", "ry", "rot"}   rot en grados

DOS ORIGENES DE DETECCION — el mismo tracker corre sobre ambos:

  TIER-1 (`load_tier1`)  formas del export de Label Studio. Aisla el tracker del
                         segmentador; es el regimen en el que se evaluan
                         Time_Tracking/matcher.py y tracker.py.
  TIER-2 (`load_tier2`)  formas del modelo YOLO-seg entrenado (best.pt) pasadas
                         por Segmentation_Model/postprocess.py (casco convexo
                         para plantas, elipse ajustada para materas). Es la
                         cadena real de principio a fin: aqui si aparecen
                         detecciones perdidas y falsos positivos.

En AMBOS casos los identificadores se siembran SOLO en la primera fecha; las
fechas siguientes llegan al tracker sin identidad, como en el flujo real.

Este modulo no escribe nada fuera de Paper/ y no importa nada de App/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    """Hace importables Time_Tracking/ y Segmentation_Model/ sin tocar el repo."""
    sys.dont_write_bytecode = True          # no crear/actualizar __pycache__ ajenos
    for sub in ("Time_Tracking", "Segmentation_Model"):
        p = str(REPO / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


_add_repo_paths()

import diagnostic                                    # noqa: E402  Time_Tracking/
import matcher                                       # noqa: E402
from matcher import PLANT, POT                       # noqa: E402
from scipy.optimize import linear_sum_assignment     # noqa: E402

DEFAULT_ANNOTATIONS = REPO / "Assets" / "annotations"
DEFAULT_IMAGES = REPO / "Assets" / "images"
DEFAULT_WEIGHTS = REPO / "Segmentation_Model" / "best.pt"


# ---------------------------------------------------------------------------
# Formas
# ---------------------------------------------------------------------------
def _ellipse_px(v: dict, W: int, H: int) -> dict:
    """Elipse de Label Studio (centro/radios en % de la imagen) -> pixeles."""
    return {"kind": "ellipse",
            "cx": float(v["x"]) / 100.0 * W,
            "cy": float(v["y"]) / 100.0 * H,
            "rx": float(v["radiusX"]) / 100.0 * W,
            "ry": float(v["radiusY"]) / 100.0 * H,
            "rot": float(v.get("rotation", 0.0) or 0.0)}


def _polygon_px(v: dict, W: int, H: int) -> dict:
    p = np.asarray(v["points"], float)
    return {"kind": "polygon",
            "points": np.column_stack([p[:, 0] / 100.0 * W, p[:, 1] / 100.0 * H])}


def shape_points(shape: dict, n: int = 96) -> np.ndarray:
    """Contorno de cualquier forma como (N,2) en pixeles (la elipse se muestrea)."""
    if shape["kind"] == "polygon":
        return np.asarray(shape["points"], float)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rot = np.radians(shape["rot"])
    c, s = np.cos(rot), np.sin(rot)
    lx, ly = shape["rx"] * np.cos(th), shape["ry"] * np.sin(th)
    return np.column_stack([shape["cx"] + lx * c - ly * s,
                            shape["cy"] + lx * s + ly * c])


# ---------------------------------------------------------------------------
# TIER-1 — formas del export de Label Studio
# ---------------------------------------------------------------------------
def _result_of(task: dict) -> list:
    """Anotaciones manuales; si no hay, predicciones. (Igual que en el repo.)"""
    anns = task.get("annotations") or []
    if anns and anns[0].get("result"):
        return anns[0]["result"]
    preds = task.get("predictions") or []
    return preds[0].get("result", []) if preds else []


def _shapes_by_region(task: dict):
    """{region_id: forma_px}, mas el tamano de la imagen."""
    out, W, H = {}, 0, 0
    for r in _result_of(task):
        ty = r.get("type")
        if ty not in ("ellipselabels", "polygonlabels"):
            continue
        W = int(r.get("original_width") or 0) or W
        H = int(r.get("original_height") or 0) or H
        if not (W and H):
            continue
        v = r["value"]
        out[r.get("id")] = _ellipse_px(v, W, H) if ty == "ellipselabels" else _polygon_px(v, W, H)
    return out, W, H


def load_tier1(ann_path, images_dir=DEFAULT_IMAGES) -> list:
    """
    Frames enriquecidos desde un export de Label Studio.

    Se carga con require_ids=False (modo INFERENCIA), el que conserva TODAS las
    regiones; `region_of` es el puente (clase,id) -> region, y con el se le pega
    su forma a cada deteccion: asi el dibujo y el tracker hablan de lo mismo.
    """
    frames = diagnostic.load_terrace(str(ann_path), require_ids=False, verbose=False)
    out = []
    for f in frames:
        by_rid, W, H = _shapes_by_region(f["task"])
        shapes = {}
        for key, rid in f["region_of"].items():
            s = by_rid.get(rid)
            if s is not None:
                shapes[key] = s
        # en TIER-1 el id provisional ES el bed_position anotado (salvo los "?")
        gt = {(c, i): (None if str(i).startswith("?") else str(i)) for c, i, _, _ in f["dets"]}
        out.append(dict(seq=f["seq"], date=f["date"], terrace=f["terrace"],
                        dets=f["dets"], shapes=shapes, gt=gt, width=W, height=H,
                        image=matcher.find_image(str(images_dir), f["terrace"], f["date"])))
    out.sort(key=lambda f: f["seq"])
    return out


# ---------------------------------------------------------------------------
# Terrazas SIN anotacion (p.ej. ST6/ST8): la secuencia sale de los nombres
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(r"^(?P<seq>\d+)-ST(?P<terrace>\d+)_(?P<date>.+)\.(jpe?g|png)$", re.I)


def frames_from_images(images_dir, terrace: str) -> list:
    """
    Frames "vacios" (solo foto, fecha y orden) para una terraza sin export de
    Label Studio. Es el unico camino para ST6/ST8, que tienen fotos pero no
    ground truth: el orden temporal sale del prefijo <seq> del nombre, la misma
    clave que usa el resto del repo.
    """
    want = str(terrace).upper().replace("ST", "").strip()
    out = []
    for f in sorted(Path(images_dir).iterdir()):
        m = FNAME_RE.match(f.name)
        if not m or m.group("terrace") != want:
            continue
        out.append(dict(seq=int(m.group("seq")), date=m.group("date"),
                        terrace=f"ST{m.group('terrace')}", image=str(f),
                        dets=[], shapes={}, gt={}, width=0, height=0))
    out.sort(key=lambda f: f["seq"])
    return out


# ---------------------------------------------------------------------------
# TIER-2 — formas del segmentador YOLO
# ---------------------------------------------------------------------------
def _shape_from_instance(inst: dict):
    """Instancia de Segmentation_Model/postprocess.py -> forma en px."""
    if inst.get("shape") == "ellipse" and inst.get("ellipse"):
        e = inst["ellipse"]
        return {"kind": "ellipse", "cx": e["cx"], "cy": e["cy"],
                "rx": e["rx"], "ry": e["ry"], "rot": e["rotation_deg"]}
    poly = inst.get("polygon")
    if not poly:
        return None
    return {"kind": "polygon", "points": np.asarray(poly, float)}


def _plants(frame) -> list:
    return [(i, x, y) for c, i, x, y in frame["dets"] if c == PLANT]


def match_to_gt(frame: dict, gt_frame: dict, gate_frac: float = 0.5) -> dict:
    """
    Empareja las plantas DETECTADAS con las plantas ANOTADAS de la MISMA fecha por
    centroide mas cercano (Hungarian + gate). Devuelve {id_provisional: bed_position}.

    Se usa para dos cosas, y ninguna es el seguimiento:
      - sembrar la fecha 1 (el biologo anota sobre lo que el segmentador detecto);
      - reportar si el id propagado coincide con la verdad en fechas posteriores.
    """
    det = _plants(frame)
    ref = [(i, x, y) for i, x, y in _plants(gt_frame) if not str(i).startswith("?")]
    if not det or not ref:
        return {}
    D = np.array([[x, y] for _, x, y in det], float)
    G = np.array([[x, y] for _, x, y in ref], float)
    gate = gate_frac * diagnostic.nn_spacing(G)
    if not np.isfinite(gate) or gate <= 0:
        gate = np.inf
    C = np.linalg.norm(D[:, None] - G[None], axis=2)
    out = {}
    for r, c in zip(*linear_sum_assignment(C)):
        if C[r, c] <= gate:
            out[det[r][0]] = ref[c][0]
    return out


def _rename_plants(frame: dict, mapping: dict) -> None:
    """Reescribe in-place los ids provisionales de las plantas (siembra de fecha 1)."""
    if not mapping:
        return
    frame["dets"] = [(c, mapping.get(i, i) if c == PLANT else i, x, y)
                     for c, i, x, y in frame["dets"]]
    frame["shapes"] = {((c, mapping.get(i, i)) if c == PLANT else (c, i)): s
                       for (c, i), s in frame["shapes"].items()}


def _reading_order(pts: np.ndarray) -> list:
    """Indices en orden de lectura: por filas (banda en y) y, dentro, de izquierda a derecha."""
    if len(pts) == 0:
        return []
    band = diagnostic.nn_spacing(pts)
    band = 0.6 * band if np.isfinite(band) and band > 0 else float("inf")
    rows, cur = [], None
    for i in np.argsort(pts[:, 1]):
        if cur is None or pts[i, 1] - pts[cur[0], 1] <= band:
            cur = [i] if cur is None else cur + [i]
        else:
            rows.append(cur)
            cur = [i]
    rows.append(cur)
    return [int(i) for r in rows for i in sorted(r, key=lambda k: pts[k, 0])]


def seed_synthetic(frame: dict, prefix: str = "P") -> dict:
    """
    Bautiza las plantas de la primera fecha cuando NO hay anotacion (ST6/ST8).

    Sin biologo que ponga bed_position, los identificadores tienen que salir de
    algun lado: se numeran P01..Pnn en orden de lectura (filas de arriba abajo,
    y de izquierda a derecha dentro de cada fila). Son etiquetas ARBITRARIAS —
    no son bed_position y no se pueden contrastar con nada.
    """
    plants = _plants(frame)
    if not plants:
        return {}
    pts = np.array([[x, y] for _, x, y in plants], float)
    mapping = {plants[k][0]: f"{prefix}{n + 1:02d}"
               for n, k in enumerate(_reading_order(pts))}
    _rename_plants(frame, mapping)
    return mapping


def load_tier2(gt_frames: list, images_dir=DEFAULT_IMAGES, weights=DEFAULT_WEIGHTS,
               conf: float = 0.25, imgsz: int = 1280, log=print,
               seed_from_gt: bool = True) -> list:
    """
    Corre el segmentador entrenado sobre las MISMAS fechas que `gt_frames` y
    devuelve frames enriquecidos con las formas post-procesadas.

    Requiere ultralytics + torch. La fecha 1 se siembra con los bed_position
    reales (via match_to_gt); las fechas siguientes llegan al tracker sin identidad.
    """
    from ultralytics import YOLO       # import perezoso: el tier-1 no lo necesita
    import postprocess as pp           # Segmentation_Model/postprocess.py

    usable = [f for f in gt_frames if f["image"]]
    if len(usable) < 2:
        raise RuntimeError("No hay suficientes fotos en --images para el tier-2 "
                           f"({len(usable)} encontradas).")

    log(f"[tier2] cargando {Path(weights).name} ...")
    model = YOLO(str(weights))
    frames = []
    for gf in usable:
        res = list(model.predict(source=str(gf["image"]), conf=conf, imgsz=imgsz,
                                 verbose=False))[0]
        H, W = res.orig_shape
        dets, shapes = [], {}
        if res.masks is not None and res.boxes is not None:
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for k, (poly, c) in enumerate(zip(res.masks.xy, classes)):
                poly = np.asarray(poly, float)
                if len(poly) < 3:
                    continue
                inst = pp.process_instance(int(c), poly)
                shape = _shape_from_instance(inst)
                if shape is None or "centroid" not in inst:
                    continue
                cls = PLANT if inst["class"] == "plant" else POT
                pid = f"d{k}"
                cx, cy = inst["centroid"]
                dets.append((cls, pid, float(cx), float(cy)))
                shapes[(cls, pid)] = shape
        n_pl = sum(1 for c, *_ in dets if c == PLANT)
        log(f"[tier2] {Path(gf['image']).name:34s} plantas={n_pl:3d} materas={len(dets)-n_pl:3d}")
        frames.append(dict(seq=gf["seq"], date=gf["date"], terrace=gf["terrace"],
                           dets=dets, shapes=shapes, gt={}, width=int(W), height=int(H),
                           image=gf["image"]))

    frames.sort(key=lambda f: f["seq"])
    by_seq = {f["seq"]: f for f in gt_frames}   # solo se usa si seed_from_gt

    n_pl0 = sum(1 for c, *_ in frames[0]["dets"] if c == PLANT)

    if not seed_from_gt:
        # terraza sin anotacion: los ids se inventan y NO hay verdad que contrastar
        seed = seed_synthetic(frames[0])
        log(f"[tier2] fecha 1: {len(seed)}/{n_pl0} plantas numeradas P01..P{len(seed):02d} "
            f"(ids arbitrarios: no hay bed_position)")
        return frames

    # fecha 1: el biologo pone los ids sobre lo que el segmentador detecto
    seed = match_to_gt(frames[0], by_seq[frames[0]["seq"]])
    _rename_plants(frames[0], seed)
    seeded = set(seed.values())
    frames[0]["gt"] = {(c, i): (str(i) if c == PLANT and str(i) in seeded else None)
                       for c, i, _, _ in frames[0]["dets"]}
    log(f"[tier2] fecha 1: {len(seed)}/{n_pl0} plantas detectadas recibieron bed_position")

    # fechas siguientes: la verdad se guarda SOLO para reportar aciertos
    for f in frames[1:]:
        truth = match_to_gt(f, by_seq[f["seq"]])
        f["gt"] = {(c, i): (truth.get(i) if c == PLANT else None) for c, i, _, _ in f["dets"]}
    return frames


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------
_PROVISIONAL = re.compile(r"^(\?|d\d+$)")


def seed_candidates(frames: list) -> list:
    """
    Identidades de la primera fecha que sirven como planta a seguir: las que
    quedaron SEMBRADAS. Se descartan las provisionales — "?<region>" (region sin
    bed_position en el export) y "d<k>" (deteccion de YOLO que no caso con
    ninguna planta anotada). Los ids sinteticos "P01..." si valen.
    """
    return sorted({str(i) for c, i, _, _ in frames[0]["dets"]
                   if c == PLANT and not _PROVISIONAL.match(str(i))})


def terrace_files(ann_dir=DEFAULT_ANNOTATIONS) -> list:
    return sorted(Path(ann_dir).glob("terrace_*.json"))
