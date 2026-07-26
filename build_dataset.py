"""
build_dataset.py — De export de Label Studio a dataset YOLO-seg + partición + IDs.

Pipeline definitivo para JardínComún (segmentación de instancias + seguimiento).

ENTRADAS (nada más):
  --annotations   carpeta (o archivo) con los JSON de Label Studio, uno por terraza.
  --images        carpeta plana con las imágenes  (<num>-ST<terraza>_<fecha>.JPG)

Por cada imagen produce DOS cosas, porque una etiqueta YOLO no puede llevar el ID:
  1) labels/…/<stem>.txt   → YOLO-seg (clase + polígono) para ENTRENAR el segmentador.
  2) ids/…/<stem>.json     → sidecar alineado línea-a-línea con el .txt, con el
                             bed_position (clave de identidad) y collection_id, para
                             la etapa de IDENTIFICACIÓN / SEGUIMIENTO.

Particiones (idénticas a la versión anterior):
  Objetivo 1  ─ forward-chaining temporal + test final = última imagen de cada terraza.
  Objetivo 2  ─ Leave-One-Terrace-Out.

Clave de identidad = bed_position (Bd-F-C): es la única única por instancia.
collection_id (No.Coleccion) se guarda como atributo (se repite entre individuos).

Materas anotadas como elipses → se muestrean a polígono (YOLO-seg no tiene elipse).
Coordenadas de Label Studio en PORCENTAJE (0-100) → se dividen entre 100 para YOLO.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ("plant", "pot")          # 0: planta, 1: matera
CLASS_ID = {"plant": 0, "pot": 1}
ELLIPSE_POINTS = 24                     # nº de vértices al muestrear una elipse

LS_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-", re.IGNORECASE)   # hash que añade Label Studio
FNAME_RE = re.compile(
    r"^(?P<num>\d+)-ST(?P<terrace>\d+)_(?P<date>.+)\.(?P<ext>jpe?g|png)$", re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# Estructuras
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Instance:
    cls: int                    # 0 planta / 1 matera
    points: list                # [(x,y), ...] normalizados a [0,1]
    bed: str                    # bed_position  (clave de identidad)
    coll: str                   # collection_id (atributo)


@dataclass
class ImageAnn:
    path: Path                  # ruta a la imagen real en --images
    terrace: int
    num: int
    date: str
    instances: list = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.path.stem


# ─────────────────────────────────────────────────────────────────────────────
# Geometría
# ─────────────────────────────────────────────────────────────────────────────
def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def polygon_from_points(pts: list) -> list:
    """[[x%,y%],...] → [(x,y),...] en [0,1]."""
    return [(_clamp01(x / 100.0), _clamp01(y / 100.0)) for x, y in pts]


def polygon_from_ellipse(v: dict, n: int = ELLIPSE_POINTS) -> list:
    """Elipse de Label Studio (centro+radios+rotación en %) → polígono en [0,1]."""
    cx, cy = float(v["x"]), float(v["y"])
    rx, ry = float(v["radiusX"]), float(v["radiusY"])
    rot = math.radians(float(v.get("rotation", 0)))
    cosR, sinR = math.cos(rot), math.sin(rot)
    out = []
    for i in range(n):
        th = 2.0 * math.pi * i / n
        lx, ly = rx * math.cos(th), ry * math.sin(th)          # punto local (en %)
        x = cx + lx * cosR - ly * sinR
        y = cy + lx * sinR + ly * cosR
        out.append((_clamp01(x / 100.0), _clamp01(y / 100.0)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Parseo de un JSON de Label Studio
# ─────────────────────────────────────────────────────────────────────────────
def _text(v: dict) -> str:
    t = v.get("text", [])
    return (t[0] if t else "").strip()


def clean_image_name(file_upload: str) -> str:
    return LS_PREFIX_RE.sub("", file_upload)


def parse_task(task: dict, images_dir: Path, warn) -> ImageAnn | None:
    name = clean_image_name(task.get("file_upload", ""))
    m = FNAME_RE.match(name)
    if not m:
        warn(f"nombre de imagen no reconocido, se ignora la tarea: {name!r}")
        return None

    img_path = images_dir / name
    if not img_path.exists():
        warn(f"falta la imagen en --images (se omite): {name}")
        return None

    anns = task.get("annotations", [])
    if not anns:
        warn(f"{name}: sin 'annotations'")
        return None
    if len(anns) > 1:
        warn(f"{name}: {len(anns)} sets de anotación; se usa el primero")
    results = anns[0].get("result", [])

    # agrupar por region id: cada objeto = shape + collection_id + bed_position
    groups: dict = defaultdict(dict)
    for r in results:
        groups[r.get("id")][r.get("from_name")] = r

    img = ImageAnn(path=img_path, terrace=int(m["terrace"]), num=int(m["num"]), date=m["date"])
    for rid, g in groups.items():
        if "plant_polygon" in g:
            shape = g["plant_polygon"]
            pts = polygon_from_points(shape["value"]["points"])
            cls = CLASS_ID["plant"]
        elif "pot_ellipse" in g:
            shape = g["pot_ellipse"]
            pts = polygon_from_ellipse(shape["value"])
            cls = CLASS_ID["pot"]
        else:
            continue  # grupo sin forma (no debería pasar)

        if len(pts) < 3:
            warn(f"{name}: polígono degenerado (<3 puntos) id={rid}, se omite")
            continue

        bed = _text(g["bed_position"]["value"]) if "bed_position" in g else ""
        coll = _text(g["collection_id"]["value"]) if "collection_id" in g else ""
        if not bed:
            warn(f"{name}: instancia sin bed_position id={rid}")
        img.instances.append(Instance(cls=cls, points=pts, bed=bed, coll=coll))

    return img


def _poly_area(points: list) -> float:
    """Área de un polígono (fórmula del zapatero) sobre coords normalizadas."""
    n = len(points)
    a = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def dedupe_instances(instances: list, img_name: str, warn):
    """
    La identidad única es (bed_position, clase): un bed tiene a lo sumo UNA planta
    y UNA matera. Si aparecen duplicados (p.ej. las materas repetidas de la imagen 2),
    se conserva el de mayor área y se descartan los demás.
    Las instancias sin bed_position no se tocan (no se pueden desambiguar).
    Devuelve (instancias_limpias, nº_eliminadas).
    """
    groups: dict = defaultdict(list)
    no_bed: list = []
    for ins in instances:
        if ins.bed:
            groups[(ins.bed, ins.cls)].append(ins)
        else:
            no_bed.append(ins)

    kept, removed = [], 0
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
        else:
            kept.append(max(group, key=lambda i: _poly_area(i.points)))  # el más grande
            removed += len(group) - 1
    kept.extend(no_bed)
    if removed:
        warn(f"{img_name}: {removed} instancia(s) duplicada(s) por (bed_position,clase) eliminada(s)")
    return kept, removed


def parse_all(ann_path: Path, images_dir: Path, warn, dedupe: bool = True) -> dict:
    """Lee todos los JSON y devuelve {terraza: [ImageAnn ordenado por num]}."""
    files = [ann_path] if ann_path.is_file() else sorted(ann_path.glob("*.json"))
    if not files:
        raise SystemExit(f"[error] no hay JSON en {ann_path}")

    by_terrace: dict = defaultdict(list)
    total_removed = 0
    for f in files:
        tasks = json.load(open(f, encoding="utf-8"))
        if isinstance(tasks, dict):
            tasks = [tasks]
        for task in tasks:
            img = parse_task(task, images_dir, warn)
            if img is not None:
                if dedupe:
                    img.instances, r = dedupe_instances(img.instances, img.stem, warn)
                    total_removed += r
                by_terrace[img.terrace].append(img)
    for t in by_terrace:
        by_terrace[t].sort(key=lambda im: im.num)
    if dedupe and total_removed:
        print(f"[dedupe] {total_removed} anotación(es) duplicada(s) eliminada(s) en total")
    elif not dedupe:
        print("[dedupe] desactivado (--no-dedupe): se conservan los duplicados")
    return dict(sorted(by_terrace.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Control de calidad (avisa, NO borra — decisión del usuario)
# ─────────────────────────────────────────────────────────────────────────────
def quality_report(by_terrace: dict) -> None:
    print("\n=== Control de calidad ===")
    for t, imgs in by_terrace.items():
        for im in imgs:
            per_bed = defaultdict(lambda: defaultdict(int))
            for ins in im.instances:
                per_bed[ins.bed][ins.cls] += 1
            dup_pots = [b for b, c in per_bed.items() if c[CLASS_ID["pot"]] > 1]
            dup_plants = [b for b, c in per_bed.items() if c[CLASS_ID["plant"]] > 1]
            np_ = sum(1 for i in im.instances if i.cls == CLASS_ID["plant"])
            nq = sum(1 for i in im.instances if i.cls == CLASS_ID["pot"])
            flags = []
            if dup_pots:
                flags.append(f"materas duplicadas en {len(dup_pots)} bed(s)")
            if dup_plants:
                flags.append(f"plantas duplicadas en {len(dup_plants)} bed(s)")
            if np_ != nq:
                flags.append(f"plantas({np_})≠materas({nq})")
            if flags:
                print(f"  [!] ST-{t} {im.stem}: " + "; ".join(flags))
    print("=== fin control de calidad ===\n")


def tracking_summary(by_terrace: dict, out_dir: Path) -> None:
    """Verifica estabilidad del bed_position entre fechas y la guarda."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, imgs in by_terrace.items():
        dates = [im.date for im in imgs]
        presence: dict = defaultdict(dict)   # bed -> {date: class}
        for im in imgs:
            for ins in im.instances:
                if ins.bed:
                    presence[ins.bed][im.date] = CLASS_NAMES[ins.cls]
        full = sum(1 for b, dd in presence.items() if len(dd) == len(dates))
        partial = len(presence) - full
        print(f"  ST-{t}: {len(presence)} bed_position únicos | "
              f"en todas las {len(dates)} fechas: {full} | parciales: {partial}")
        (out_dir / f"tracking_ST{t}.json").write_text(
            json.dumps({"dates": dates, "beds": presence}, ensure_ascii=False, indent=2),
            encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Particiones (misma lógica de siempre)
# ─────────────────────────────────────────────────────────────────────────────
def temporal_final_split(by_terrace: dict):
    train, test = [], []
    for imgs in by_terrace.values():
        if len(imgs) < 2:
            train.extend(imgs); continue
        train.extend(imgs[:-1]); test.append(imgs[-1])
    return train, test


def temporal_forward_chaining(by_terrace: dict):
    usable = {t: im[:-1] for t, im in by_terrace.items() if len(im) >= 2}
    if not usable:
        return
    for k in range(1, max(len(v) for v in usable.values())):
        train, val = [], []
        for imgs in usable.values():
            if k < len(imgs):
                train.extend(imgs[:k]); val.append(imgs[k])
            else:
                train.extend(imgs)
        yield k, train, val


def leave_one_terrace_out(by_terrace: dict, with_val: bool = True):
    terraces = list(by_terrace.keys())
    n = len(terraces)
    for i, held in enumerate(terraces):
        val_t = terraces[(i + 1) % n] if (with_val and n > 2) else None
        train, val, test = [], [], []
        for t, imgs in by_terrace.items():
            if t == held:
                test.extend(imgs)
            elif t == val_t:
                val.extend(imgs)
            else:
                train.extend(imgs)
        yield {"test_terrace": held, "val_terrace": val_t, "train": train, "val": val, "test": test}


# ─────────────────────────────────────────────────────────────────────────────
# Materializar: imagen + label(.txt) + id(.json) por split
# ─────────────────────────────────────────────────────────────────────────────
def _place(src: Path, dst: Path, link: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        dst.symlink_to(Path(src).resolve())
    else:
        shutil.copy2(src, dst)


def _write_label(path: Path, instances: list):
    lines = []
    for ins in instances:
        coords = " ".join(f"{c:.6f}" for xy in ins.points for c in xy)
        lines.append(f"{ins.cls} {coords}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_ids(path: Path, instances: list):
    data = [{"i": i, "class": CLASS_NAMES[ins.cls],
             "bed_position": ins.bed, "collection_id": ins.coll}
            for i, ins in enumerate(instances)]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def materialize(split_map: dict, out_dir: Path, link: bool = True) -> Path:
    out_dir = Path(out_dir)
    for split, imgs in split_map.items():
        if not imgs:
            continue
        for sub in ("images", "labels", "ids"):
            (out_dir / sub / split).mkdir(parents=True, exist_ok=True)
        for im in imgs:
            _place(im.path, out_dir / "images" / split / im.path.name, link)
            _write_label(out_dir / "labels" / split / f"{im.stem}.txt", im.instances)
            _write_ids(out_dir / "ids" / split / f"{im.stem}.json", im.instances)
    return out_dir


def write_data_yaml(out_dir: Path, names=CLASS_NAMES, has_test: bool = False) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"path: {out_dir.resolve()}", "train: images/train", "val: images/val"]
    if has_test:
        lines.append("test: images/test")
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "data.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Label Studio → YOLO-seg + IDs + particiones")
    ap.add_argument("--annotations", required=True, type=Path, help="carpeta o archivo JSON de Label Studio")
    ap.add_argument("--images", required=True, type=Path, help="carpeta plana de imágenes")
    ap.add_argument("--work", default=Path("datasets"), type=Path)
    ap.add_argument("--copy", action="store_false", help="copiar en vez de symlink (Windows)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="conservar duplicados por (bed_position,clase); por defecto se eliminan")
    args = ap.parse_args()

    warnings: list = []
    warn = lambda msg: warnings.append(msg)

    by_terrace = parse_all(args.annotations, args.images, warn, dedupe=not args.no_dedupe)
    if not by_terrace:
        raise SystemExit("Abortado: no se parseó ninguna imagen (¿nombres/carpetas correctos?).")

    print("Terrazas parseadas:")
    for t, imgs in by_terrace.items():
        pl = sum(sum(1 for i in im.instances if i.cls == 0) for im in imgs)
        po = sum(sum(1 for i in im.instances if i.cls == 1) for im in imgs)
        print(f"  ST-{t}: {len(imgs)} imágenes  ({imgs[0].date} … {imgs[-1].date})  "
              f"plantas={pl} materas={po}")

    quality_report(by_terrace)

    print("Estabilidad de bed_position (seguimiento):")
    tracking_summary(by_terrace, args.work / "tracking")

    link = not args.copy

    # Objetivo 1
    print("\nOBJETIVO 1 — partición temporal (forward-chaining)")
    for k, train, val in temporal_forward_chaining(by_terrace):
        d = args.work / "obj1_forward" / f"fold_{k}"
        materialize({"train": train, "val": val}, d, link); write_data_yaml(d)
        print(f"  [fold {k}] train={len(train):>2}  val={len(val):>2}")
    train, test = temporal_final_split(by_terrace)
    d = args.work / "obj1_final"
    materialize({"train": train, "val": test, "test": test}, d, link)
    write_data_yaml(d, has_test=True)
    print(f"  [test final] train={len(train)}  test={len(test)}")

    # Objetivo 2
    print("\nOBJETIVO 2 — Leave-One-Terrace-Out")
    if len(by_terrace) < 2:
        print("  (omitido: se necesitan ≥2 terrazas; ahora hay "
              f"{len(by_terrace)})")
    else:
        for fold in leave_one_terrace_out(by_terrace):
            held = fold["test_terrace"]
            d = args.work / "obj2_loto" / f"held_ST{held}"
            materialize({"train": fold["train"], "val": fold["val"], "test": fold["test"]}, d, link)
            write_data_yaml(d, has_test=True)
            print(f"  [test ST-{held}] train={len(fold['train']):>2}  "
                  f"val={len(fold['val']):>2}  test={len(fold['test']):>2}")

    # Avisos al final, agrupados
    if warnings:
        print(f"\n=== {len(warnings)} AVISO(S) ===")
        for w in warnings[:40]:
            print("  -", w)
        if len(warnings) > 40:
            print(f"  … y {len(warnings) - 40} más")


if __name__ == "__main__":
    main()
