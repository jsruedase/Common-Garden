#!/usr/bin/env python3
"""
JardínComún — Exportador de seguimiento a Label Studio
======================================================

Cierra el paso 7→8 del flujo del biólogo: toma el export de Label Studio de una
terraza (con bed_position anotado SOLO en la primera fecha), propaga los ids con
el matcher geométrico y devuelve tareas de Label Studio con los ids ya escritos,
listas para reimportar y corregir.

    biólogo anota fecha 1  →  este script  →  reimportar en LS  →  corregir swaps

Qué hace exactamente:
  1. carga el export en MODO INFERENCIA (conserva las regiones sin id;
     ver diagnostic.load_terrace(require_ids=False));
  2. siembra la plantilla con los bed_position reales de la primera fecha;
  3. registra cada fecha posterior a la primera (ICP sobre las materas) y asigna
     por Hungarian dentro del gate → id propagado, o NEW# si es una planta nueva;
  4. la MATERA de cada cama hereda el id de su planta más cercana (una cama =
     una planta + una matera), así el biólogo no reescribe nada dos veces;
  5. escribe bed_position y collection_id en cada región y guarda un JSON por
     fecha (una tarea por archivo, como pediste).

El collection_id se propaga desde la primera fecha: si la cama X tenía YAM-1003,
todas sus observaciones futuras lo heredan.

Uso:
    python track_to_labelstudio.py Assets/annotations/terrace_7.json
    python track_to_labelstudio.py Assets/annotations/terrace_7.json --out Outputs/ls_tracked
    python track_to_labelstudio.py Assets/annotations/terrace_7.json --combined
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from diagnostic import load_terrace, nn_spacing
import matcher
from matcher import PLANT, POT

GATE_FRAC = 0.5     # igual que matcher.py


def propagate(frames):
    """
    Propaga identidades desde la fecha 1. Devuelve, por frame, un dict
    {(clase, identidad_provisional) -> id_predicho}.
    No puntúa nada: es inferencia pura para la app.
    """
    frames = sorted(frames, key=lambda f: f["seq"])
    f1 = frames[0]
    plants1 = [(i, x, y) for c, i, x, y in f1["dets"] if c == PLANT]
    pots1 = np.array([[x, y] for c, i, x, y in f1["dets"] if c == POT], float)
    if not plants1:
        raise SystemExit("[error] la primera fecha no tiene plantas anotadas; "
                         "no hay de dónde sembrar los identificadores.")

    unknown1 = [i for i, _, _ in plants1 if str(i).startswith("?")]
    if unknown1:
        print(f"[aviso] {len(unknown1)} planta(s) de la PRIMERA fecha sin bed_position. "
              f"Se propagará un id provisional; anótalas para un seguimiento correcto.")

    template = {i: np.array([x, y], float) for i, x, y in plants1}
    gate = GATE_FRAC * nn_spacing(np.array(list(template.values())))
    new_ctr = 0
    out = []

    # fecha 1: cada detección conserva su propio id
    out.append({(PLANT, i): i for i, _, _ in plants1})

    for f in frames[1:]:
        plantsB = [(i, x, y) for c, i, x, y in f["dets"] if c == PLANT]
        potsB = np.array([[x, y] for c, i, x, y in f["dets"] if c == POT], float)

        src = potsB if len(potsB) >= 3 else np.array([[x, y] for _, x, y in plantsB], float)
        dst = pots1 if len(pots1) >= 3 else np.array(list(template.values()), float)
        s, R, t = matcher.icp(src, dst) if len(src) >= 2 and len(dst) >= 2 else (1.0, np.eye(2), np.zeros(2))

        det_xy = np.array([[x, y] for _, x, y in plantsB], float)
        P = matcher.apply(s, R, t, det_xy) if len(det_xy) else np.zeros((0, 2))

        tkeys = list(template)
        T = np.array([template[k] for k in tkeys], float)
        pred = [None] * len(plantsB)
        if len(P) and len(T):
            C = np.linalg.norm(P[:, None] - T[None], axis=2)
            for r, c in zip(*linear_sum_assignment(C)):
                if C[r, c] <= gate:
                    pred[r] = tkeys[c]

        mapping = {}
        for k, (prov_id, ox, oy) in enumerate(plantsB):
            pid = pred[k]
            if pid is None:
                new_ctr += 1
                pid = f"NEW{new_ctr}"
            template[pid] = P[k]
            mapping[(PLANT, prov_id)] = pid
        out.append(mapping)
    return out, frames


def pots_follow_plants(frame, plant_map):
    """La matera hereda el id de la planta más cercana (una cama = planta + matera)."""
    plants = [(i, x, y) for c, i, x, y in frame["dets"] if c == PLANT]
    pots = [(i, x, y) for c, i, x, y in frame["dets"] if c == POT]
    if not plants or not pots:
        return {}
    pxy = np.array([[x, y] for _, x, y in plants], float)
    out = {}
    for i, x, y in pots:
        j = int(np.argmin(np.hypot(pxy[:, 0] - x, pxy[:, 1] - y)))
        pid = plant_map.get((PLANT, plants[j][0]))
        if pid is not None:
            out[(POT, i)] = pid
    return out


def collection_lookup(frames):
    """bed_position -> collection_id, tomado de la primera fecha (donde se anotó)."""
    look = {}
    for f in frames:
        result = _result_of(f["task"])
        by_region = {}
        for r in result:
            if r.get("from_name") in ("bed_position", "collection_id") and r.get("type") == "textarea":
                txt = (r.get("value") or {}).get("text") or [""]
                by_region.setdefault(r.get("id"), {})[r["from_name"]] = str(txt[0]).strip()
        for vals in by_region.values():
            bed, coll = vals.get("bed_position", ""), vals.get("collection_id", "")
            if bed and coll and bed not in look:
                look[bed] = coll
    return look


def _result_of(task):
    anns = task.get("annotations") or []
    if anns and anns[0].get("result"):
        return anns[0]["result"]
    preds = task.get("predictions") or []
    return preds[0].get("result", []) if preds else []


def write_ids(frame, id_map, coll_look):
    """Devuelve una copia de la tarea con bed_position/collection_id rellenos."""
    task = copy.deepcopy(frame["task"])
    region_pred = {}                       # region_id -> id predicho
    for (cls, prov), pid in id_map.items():
        rid = frame["region_of"].get((cls, prov))
        if rid is not None:
            region_pred[rid] = pid

    result = _result_of(task)
    for r in result:
        rid = r.get("id")
        if rid not in region_pred:
            continue
        pid = region_pred[rid]
        if r.get("from_name") == "bed_position" and r.get("type") == "textarea":
            r.setdefault("value", {})["text"] = [pid]
        elif r.get("from_name") == "collection_id" and r.get("type") == "textarea":
            r.setdefault("value", {})["text"] = [coll_look.get(pid, "")]

    # entregar como PREDICCIÓN (pre-anotación) para que el biólogo la revise
    out = {"data": task.get("data", {}),
           "predictions": [{"model_version": "tracker-geom", "result": result}]}
    return out, len(region_pred)


def main():
    ap = argparse.ArgumentParser(description="Propaga ids y exporta tareas de Label Studio")
    ap.add_argument("--annotations", type=Path, default = Path("Assets/annotations/terrace_7.json"), help="export de LS de una terraza (terrace_*.json)")
    ap.add_argument("--out", type=Path, default=Path("Outputs/ls_tracked"))
    ap.add_argument("--combined", action="store_true",
                    help="un solo archivo con todas las fechas (por defecto: uno por fecha)")
    args = ap.parse_args()

    # MODO INFERENCIA: conserva las regiones sin bed_position
    frames = load_terrace(args.annotations, require_ids=False)
    if len(frames) < 2:
        raise SystemExit("[error] se necesitan al menos 2 fechas")

    maps, frames = propagate(frames)
    coll_look = collection_lookup(frames)
    args.out.mkdir(parents=True, exist_ok=True)

    tasks, total = [], 0
    for f, m in zip(frames, maps):
        m = dict(m)
        m.update(pots_follow_plants(f, m))          # las materas heredan el id
        task, n = write_ids(f, m, coll_look)
        tasks.append(task)
        total += n
        n_new = sum(1 for v in m.values() if str(v).startswith("NEW"))
        n_prov = sum(1 for v in m.values() if str(v).startswith("?"))
        print(f"  {f['terrace']} {f['date'][:20]:22s} regiones={n:3d}  nuevas={n_new:2d}"
              + (f"  sin-id={n_prov}" if n_prov else ""))
        if not args.combined:
            out = args.out / f"{f['seq']}-{f['terrace']}_{f['date']}_tracked.json"
            out.write_text(json.dumps([task], ensure_ascii=False, indent=2), encoding="utf-8")

    if args.combined:
        out = args.out / f"{frames[0]['terrace']}_tracked.json"
        out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[listo] {len(tasks)} tarea(s), {total} región(es) con id → {args.out.resolve()}")
    print("        reimporta en Label Studio y corrige los ids intercambiados.")


if __name__ == "__main__":
    main()
