#!/usr/bin/env python3
"""
Paper/highlight_track.py — Sigue UNA planta a lo largo de las fechas de una terraza.
===================================================================================

Simula el objetivo de seguimiento tal como se usa en la practica:

    el biologo anota los ids SOLO en la primera fecha
        -> el tracker geometrico (ICP sobre materas + Hungarian sobre plantas)
           propaga esos ids a todas las fechas siguientes
        -> se elige UNA planta al azar de la primera fecha y se dibuja, fecha a
           fecha, la instancia que LLEVA SU ID SEGUN EL TRACKER

Es decir: lo que se ilumina en las fechas 2..N no es la verdad anotada, es la
decision del algoritmo. Si el tracker se equivoca, la figura muestra la planta
equivocada — y la cartela lo dice (OK / ERROR contra el bed_position real).

Se generan dos versiones de la misma historia:
    TIER-1  detecciones = anotacion de Label Studio  (tracker aislado)
    TIER-2  detecciones = YOLO-seg best.pt + postproceso  (cadena completa)

Salida (nada se escribe fuera de Paper/images/):
    Paper/images/<ST>_<bed>/tier1_gt/<seq>_<fecha>.jpg
    Paper/images/<ST>_<bed>/tier2_yolo/<seq>_<fecha>.jpg
    Paper/images/<ST>_<bed>/strip_tier1_gt.png
    Paper/images/<ST>_<bed>/strip_tier2_yolo.png
    Paper/images/<ST>_<bed>/compare.png          las dos filas juntas
    Paper/images/<ST>_<bed>/summary.json

Uso:
    python Paper/highlight_track.py                      # terraza y planta al azar
    python Paper/highlight_track.py --seed 7
    python Paper/highlight_track.py --terrace ST7 --plant 4A-3-1
    python Paper/highlight_track.py --tiers tier1        # sin YOLO (no necesita torch)

Reutiliza sin copiarla la logica de Time_Tracking/ (load_terrace, icp, propagate)
y de Segmentation_Model/ (process_instance). No modifica nada del repo.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

import detections as D
from detections import PLANT, POT, REPO
import render

import track_to_labelstudio as t2ls      # Time_Tracking/: propagate, pots_follow_plants

TIERS = {"tier1": "tier1_gt", "tier2": "tier2_yolo"}
TIER_TITLE = {"tier1": "TIER-1  deteccion = anotacion",
              "tier2": "TIER-2  deteccion = YOLO best.pt"}


# ---------------------------------------------------------------------------
# Seguimiento de una sola identidad
# ---------------------------------------------------------------------------
def follow(frames: list, target: str) -> list:
    """
    Propaga los ids desde la fecha 1 y devuelve, por fecha, quien lleva `target`.

    La propagacion es EXACTAMENTE la de la app (track_to_labelstudio.propagate):
    ICP sin correspondencias sobre las materas + Hungarian sobre las plantas con
    gate = 0.5 * espaciamiento. La matera hereda el id de su planta mas cercana.
    """
    maps, ordered = t2ls.propagate(frames)
    steps = []
    for f, m in zip(ordered, maps):
        full = dict(m)
        full.update(t2ls.pots_follow_plants(f, full))
        holders = [key for key, pid in full.items() if pid == target]
        plant_key = next((k for k in holders if k[0] == PLANT), None)
        pot_key = next((k for k in holders if k[0] == POT), None)
        truth = f["gt"].get(plant_key) if plant_key else None
        # correct=None => indecidible (no hay anotacion con la que contrastar)
        correct = None if (plant_key is None or truth is None) else (truth == target)
        steps.append(dict(frame=f, plant_key=plant_key, pot_key=pot_key,
                          truth=truth, correct=correct))
    return steps


def coverage(frames: list) -> tuple:
    """
    {id sembrado en la fecha 1: nro de fechas posteriores en que el tracker lo asigna}.

    Sirve para el sorteo: una planta que desaparece de la terraza tras la primera
    fecha da una figura vacia, asi que por defecto se sortea entre las que el
    tracker consigue arrastrar hasta el final (--any levanta esa restriccion).
    """
    maps, ordered = t2ls.propagate(frames)
    out = {s: 0 for s in maps[0].values()}
    for m in maps[1:]:
        for v in set(m.values()):
            if v in out:
                out[v] += 1
    return out, len(ordered) - 1


def _status(step, first: bool, annotated: bool = True) -> tuple:
    """(texto de estado, aviso rojo o None) para la cartela."""
    if first:
        return ("SEMBRADA (id anotado por el biologo)" if annotated else
                "SEMBRADA (id arbitrario: terraza sin anotacion)"), None
    if step["plant_key"] is None:
        return "SIN MATCH", "El tracker no asigno este id en esta fecha"
    if step["correct"] is True:
        return "SEGUIMIENTO OK", None
    if step["correct"] is False:
        real = step["truth"]
        return (f"ERROR DE ID (en realidad es {real})",
                f"El tracker confundio la identidad: esta planta es {real}")
    return "SEGUIDA (sin anotacion que contrastar)", None


# ---------------------------------------------------------------------------
# Dibujo de una fecha
# ---------------------------------------------------------------------------
def render_step(step, target, tier, idx, n_dates, max_width, with_header=True,
                annotated=True):
    """
    Compone una fecha. `with_header=False` da la version limpia (sin cartela ni
    franja) que se usa en las tiras, donde el titulo ya lo pone matplotlib.
    """
    f = step["frame"]
    if not f["image"]:
        return None
    img = cv2.imread(str(f["image"]))
    if img is None:
        return None

    keys = {k for k in (step["plant_key"], step["pot_key"]) if k}
    tgt = [f["shapes"][k] for k in keys if k in f["shapes"]]
    others = [s for k, s in f["shapes"].items() if k not in keys]

    status, warning = _status(step, idx == 0, annotated)
    header = [
        f"{f['terrace']}   fecha {idx + 1}/{n_dates}   {f['date'].replace('_', ' ')}",
        f"planta seguida: {target}",
        f"{TIER_TITLE[tier]}",
        status,
    ] if with_header else []
    return render.spotlight(img, tgt, others, header,
                            label=target if tgt else None,
                            warning=warning if with_header else None,
                            max_width=max_width)


def run_tier(tier, frames, target, out_root, max_width, log=print, annotated=True):
    steps = follow(frames, target)
    tier_dir = out_root / TIERS[tier]
    tier_dir.mkdir(parents=True, exist_ok=True)

    cells, records = [], []
    for idx, step in enumerate(steps):
        f = step["frame"]
        img = render_step(step, target, tier, idx, len(steps), max_width,
                          annotated=annotated)
        status, _ = _status(step, idx == 0, annotated)
        rec = dict(seq=f["seq"], date=f["date"], status=status,
                   matched=step["plant_key"] is not None,
                   ground_truth=step["truth"], correct=step["correct"],
                   image=None)
        if img is None:
            log(f"  [{tier}] {f['date']:24s} sin foto en Assets/images (se omite)")
            records.append(rec)
            continue
        name = f"{f['seq']}_{f['date']}.jpg"
        cv2.imwrite(str(tier_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rec["image"] = f"{TIERS[tier]}/{name}"
        records.append(rec)
        clean = render_step(step, target, tier, idx, len(steps), 900,
                            with_header=False, annotated=annotated)
        cells.append((clean if clean is not None else img,
                      f"{f['date'].replace('_', ' ')}\n{status}"))
        log(f"  [{tier}] {f['date']:24s} {status}")

    later = [r for r in records[1:] if r["correct"] is not None]
    acc = (sum(1 for r in later if r["correct"]) / len(later)) if later else None
    return dict(records=records, cells=cells, accuracy=acc,
                n_matched=sum(1 for r in records[1:] if r["matched"]),
                n_later=len(records) - 1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Ilumina una planta y siguela por las fechas de una terraza")
    ap.add_argument("--annotations", type=Path, default=D.DEFAULT_ANNOTATIONS)
    ap.add_argument("--images", type=Path, default=D.DEFAULT_IMAGES)
    ap.add_argument("--weights", type=Path, default=D.DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "images")
    ap.add_argument("--terrace", default=None,
                    help="ST7, 7 o terrace_7.json; por defecto una al azar")
    ap.add_argument("--plant", default=None,
                    help="bed_position a seguir; por defecto una al azar de la fecha 1")
    ap.add_argument("--seed", type=int, default=None, help="semilla del sorteo")
    ap.add_argument("--any", dest="any_plant", action="store_true",
                    help="sortear entre TODAS las plantas de la fecha 1, incluidas "
                         "las que desaparecen despues (por defecto se prefieren las "
                         "que el tracker sigue hasta el final)")
    ap.add_argument("--tiers", default="tier1,tier2",
                    help="tier1 (anotacion), tier2 (YOLO) o ambos")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-width", type=int, default=1800,
                    help="ancho maximo de las imagenes de salida (0 = original)")
    ap.add_argument("--no-strip", action="store_true", help="no montar las tiras")
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        raise SystemExit(f"[error] tier desconocido: {bad} (validos: {list(TIERS)})")

    rng = random.Random(args.seed)
    files = D.terrace_files(args.annotations)
    if not files:
        raise SystemExit(f"[error] no hay terrace_*.json en {args.annotations.resolve()}")

    picked = None
    if args.terrace:
        want = str(args.terrace).lower().replace("st", "").replace(".json", "").strip("_")
        picked = next((f for f in files if f.stem.split("_")[-1] == want), None)
    else:
        picked = rng.choice(files)

    # Una terraza SIN export (ST6, ST8) es el caso real puro: no hay ground truth,
    # asi que no hay tier-1 que dibujar ni aciertos que reportar. Se sigue igual,
    # con ids inventados en la fecha 1.
    annotated = picked is not None
    if annotated:
        print(f"[terraza] {picked.name}")
        gt_frames = D.load_tier1(picked, args.images)
    else:
        label = f"ST{want}"
        gt_frames = D.frames_from_images(args.images, label)
        if not gt_frames:
            raise SystemExit(f"[error] terraza {args.terrace!r}: ni export en "
                             f"{args.annotations} ni fotos en {args.images}")
        print(f"[terraza] {label} SIN anotacion: {len(gt_frames)} fotos, solo tier-2, "
              f"ids inventados y sin verdad con la que contrastar")
        if "tier2" not in tiers:
            raise SystemExit("[error] una terraza sin anotacion solo admite --tiers tier2 "
                             "(no hay formas anotadas que dibujar).")
        tiers = ["tier2"]

    if len(gt_frames) < 2:
        raise SystemExit("[error] se necesitan al menos 2 fechas para seguir una planta.")
    print(f"[fechas]  {len(gt_frames)}: " + ", ".join(f["date"] for f in gt_frames))
    missing = [f["date"] for f in gt_frames if not f["image"]]
    if missing:
        print(f"[aviso]   sin foto en {args.images}: {missing}")

    built = {"tier1": gt_frames} if annotated else {}
    if "tier2" in tiers:
        try:
            built["tier2"] = D.load_tier2(gt_frames, args.images, args.weights,
                                          conf=args.conf, imgsz=args.imgsz,
                                          seed_from_gt=annotated)
        except Exception as e:
            if not annotated:
                raise SystemExit(f"[error] sin anotacion el tier-2 es la unica via, "
                                 f"y fallo: {type(e).__name__}: {e}")
            print(f"[aviso]   tier-2 desactivado: {type(e).__name__}: {e}")
            tiers = [t for t in tiers if t != "tier2"]

    # la planta a seguir tiene que existir en la fecha 1 de TODOS los tiers pedidos
    pools = [set(D.seed_candidates(built[t])) for t in tiers if t in built]
    common = sorted(set.intersection(*pools)) if pools else []
    if not common:
        raise SystemExit("[error] ninguna planta de la fecha 1 esta disponible en "
                         "todos los tiers pedidos (revisa --conf o usa --tiers tier1).")

    # y, salvo --any, conviene que el tracker la siga en TODAS las fechas.
    # El filtro se aplica SOLO sobre el tier de referencia (tier-1 si esta): condicionar
    # el sorteo al exito del segmentador maquillaria el tier-2 de la figura.
    cov = {t: coverage(built[t]) for t in tiers if t in built}
    ref = "tier1" if "tier1" in cov else tiers[0]
    ref_cov, ref_n = cov[ref]
    full = [b for b in common if ref_cov.get(b, 0) == ref_n]
    pool = common if (args.any_plant or not full) else full
    if not full and not args.any_plant:
        print("[aviso]   ninguna planta se sigue en todas las fechas; se sortea entre todas")

    if args.plant:
        if args.plant not in common:
            raise SystemExit(f"[error] {args.plant!r} no esta entre las candidatas: {common}")
        target = args.plant
    else:
        target = rng.choice(pool)
    print(f"[planta]  {target}   (sorteada entre {len(pool)} de las "
          f"{len(common)} plantas sembradas en la fecha 1; "
          f"criterio: seguida en las {ref_n} fechas de {ref})")

    out_root = args.out / f"{gt_frames[0]['terrace']}_{target.replace('/', '-')}"
    out_root.mkdir(parents=True, exist_ok=True)

    results, rows, row_titles = {}, [], []
    for tier in tiers:
        print(f"[{tier}]")
        r = run_tier(tier, built[tier], target, out_root, args.max_width,
                     annotated=annotated)
        results[tier] = r
        if r["cells"]:
            rows.append(r["cells"])
            acc = ("sin verdad que contrastar" if r["accuracy"] is None
                   else f"aciertos {r['accuracy']:.0%}")
            row_titles.append(f"{TIER_TITLE[tier]}\n{acc}")

    if not args.no_strip and rows:
        title = (f"{gt_frames[0]['terrace']} - seguimiento de la planta {target} "
                 f"(id sembrado solo en la primera fecha)")
        for tier, row, rt in zip(tiers, rows, row_titles):
            render.strip([row], out_root / f"strip_{TIERS[tier]}.png",
                         f"{title}\n{rt}".replace("\n", "   "))
        if len(rows) > 1:
            render.strip(rows, out_root / "compare.png", title, row_titles=row_titles)

    summary = dict(
        terrace=gt_frames[0]["terrace"],
        annotations=picked.name if annotated else None, plant=target,
        seed=args.seed, n_dates=len(gt_frames), candidates=common,
        tiers={t: dict(accuracy=results[t]["accuracy"],
                       matched=f"{results[t]['n_matched']}/{results[t]['n_later']}",
                       dates=results[t]["records"]) for t in results},
    )
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[listo] {out_root.resolve()}")
    for t in results:
        acc = results[t]["accuracy"]
        acc = "n/d" if acc is None else f"{acc:.0%}"
        print(f"  {t:6s} identificada en {results[t]['n_matched']}/{results[t]['n_later']} "
              f"fechas posteriores, aciertos {acc}")


if __name__ == "__main__":
    main()
