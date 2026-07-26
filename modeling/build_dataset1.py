"""
build_dataset.py — Common-Garden dataset builder.

Reads the Label Studio exports in data/annotations/ and the terrace photos in
data/images/, and produces everything the downstream stages consume:

  datasets/<fold>/images|labels|ids/<split>/   YOLO-seg labels + id sidecars
  datasets/tracking/tracking_ST*.json           tracking ground truth

For every image it writes THREE parallel things per split:
  labels/<stem>.txt   -> YOLO-seg (class + normalized polygon) for TRAINING.
  ids/<stem>.json     -> sidecar aligned line-for-line with the .txt, carrying
                         bed_position (identity key) + collection_id, for TRACKING.

Notes baked in (were CLI flags before, now fixed behaviour):
  * Ellipses are converted to 24-point polygons (YOLO has no ellipse primitive).
  * Duplicates are removed by (bed_position, class), keeping the largest-area
    instance (fixes the 36 duplicated pots in ST1 image 2, 2 in ST7).
  * Folds: obj1_final (final held-out), obj1_forward (forward-chaining CV),
    obj2_loto (leave-one-terrace-out CV).

Run with no arguments:
    python modeling/build_dataset.py
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.parse
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths (the only thing you edit)                                              #
# --------------------------------------------------------------------------- #
ANNOTATIONS = Path("data/annotations")   # folder of Label Studio JSON exports
IMAGES      = Path("data/images")        # folder of terrace photos
WORK        = Path("datasets")           # output root

# --------------------------------------------------------------------------- #
# Fixed config                                                                #
# --------------------------------------------------------------------------- #
CLASS_MAP = {"planta": 0, "matera": 1}   # Spanish LS categories -> YOLO class id
CLASS_NAMES = {0: "plant", 1: "pot"}
ELLIPSE_POINTS = 24                      # ellipse -> polygon resolution
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #
def img_key(name: str) -> str:
    """Canonical match key, tolerant of Label Studio hash prefixes / upload paths.

    LS stores things like '/data/upload/3/a1b2c3d4-1-ST1_Sep_1-5_2025.JPG'.
    We drop the directory + extension, strip the hash prefix by anchoring on the
    '<date>-ST<n>' pattern, and lowercase so both sides compare equal.
    """
    stem = Path(urllib.parse.unquote(str(name))).stem
    m = re.search(r"\d+-ST\d+.*", stem)
    return (m.group(0) if m else stem).lower()


def terrace_of(stem: str) -> str:
    """Extract the terrace token (e.g. 'ST1') from an image stem."""
    m = re.search(r"ST\d+", stem)
    return m.group(0) if m else "ST?"


def ellipse_to_polygon(cx, cy, rx, ry, angle_deg, n=ELLIPSE_POINTS):
    """Sample an axis-aligned/rotated ellipse into n polygon points (pixels)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = np.deg2rad(angle_deg)
    x = cx + rx * np.cos(t) * np.cos(a) - ry * np.sin(t) * np.sin(a)
    y = cy + rx * np.cos(t) * np.sin(a) + ry * np.sin(t) * np.cos(a)
    return np.stack([x, y], axis=1)


def parse_task(task: dict):
    """Return (image_field, [instances]) for one Label Studio task.

    image_field is the raw value LS stored (with hash prefix / upload path);
    the caller reduces it to a match key via img_key().
    Each instance: dict(cls, bed_position, collection_id, poly_px (N,2), area).
    """
    img_field = task.get("data", {}).get("image", "")
    anns = task.get("annotations") or []
    if not anns:
        return img_field, []

    out = []
    for r in anns[0].get("result", []):
        ty = r.get("type")
        if ty not in ("ellipselabels", "polygonlabels"):
            continue
        meta = r.get("meta") or {}
        bed = meta.get("Bd-F-C")
        if not bed:                       # region with no identity key -> skip
            continue
        cat = (meta.get("category") or "?").strip().lower()
        cls = CLASS_MAP.get(cat)
        if cls is None:
            continue
        coll = meta.get("No.Coleccion")
        W = r.get("original_width") or 1
        H = r.get("original_height") or 1
        v = r["value"]

        if ty == "ellipselabels":
            cx, cy = v["x"] / 100 * W, v["y"] / 100 * H
            rx, ry = v["radiusX"] / 100 * W, v["radiusY"] / 100 * H
            poly = ellipse_to_polygon(cx, cy, rx, ry, v.get("rotation", 0.0))
        else:
            p = np.asarray(v["points"], float)
            poly = np.stack([p[:, 0] / 100 * W, p[:, 1] / 100 * H], axis=1)

        area = float(abs(_shoelace(poly)))
        out.append(dict(cls=cls, bed_position=bed, collection_id=coll,
                        poly=poly, W=W, H=H, area=area))
    return img_field, out


def _shoelace(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def dedupe(instances):
    """Keep one instance per (bed_position, class): the largest by area."""
    best = {}
    for ins in instances:
        key = (ins["bed_position"], ins["cls"])
        if key not in best or ins["area"] > best[key]["area"]:
            best[key] = ins
    return list(best.values())


# --------------------------------------------------------------------------- #
# Writing                                                                     #
# --------------------------------------------------------------------------- #
def write_label_and_ids(stem, instances, split, fold_dir):
    """Write labels/<stem>.txt (YOLO-seg) and ids/<stem>.json (sidecar)."""
    lab_dir = fold_dir / "labels" / split
    id_dir = fold_dir / "ids" / split
    lab_dir.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)

    lines, sidecar = [], []
    for i, ins in enumerate(instances):
        W, H = ins["W"], ins["H"]
        norm = ins["poly"].copy()
        norm[:, 0] /= W
        norm[:, 1] /= H
        norm = np.clip(norm, 0.0, 1.0)
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
        lines.append(f"{ins['cls']} {coords}")
        sidecar.append(dict(i=i, **{"class": CLASS_NAMES[ins["cls"]]},
                            bed_position=ins["bed_position"],
                            collection_id=ins["collection_id"]))

    (lab_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    (id_dir / f"{stem}.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")


def link_image(stem, split, fold_dir, src_img):
    img_dir = fold_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    dst = img_dir / src_img.name
    if not dst.exists():
        shutil.copy2(src_img, dst)          # copy (portable across OSes)


def write_data_yaml(fold_dir, has_test):
    splits = "train: images/train\nval: images/val\n"
    if has_test:
        splits += "test: images/test\n"
    txt = (f"path: {fold_dir.resolve()}\n{splits}"
           f"names:\n  0: plant\n  1: pot\n")
    (fold_dir / "data.yaml").write_text(txt, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Fold construction                                                           #
# --------------------------------------------------------------------------- #
def date_index(stem: str) -> int:
    """Leading integer of the filename is the temporal order within a terrace."""
    m = re.match(r"(\d+)", Path(stem).name)
    return int(m.group(1)) if m else 0


def build_folds(by_image, img_paths):
    """
    by_image: {stem: [instances]}
    img_paths: {stem: Path}
    Builds obj1_final, obj1_forward, obj2_loto under WORK.
    """
    # group stems per terrace, ordered by date
    per_terrace = defaultdict(list)
    for stem in by_image:
        per_terrace[terrace_of(stem)].append(stem)
    for t in per_terrace:
        per_terrace[t].sort(key=date_index)

    terraces = sorted(per_terrace)

    def emit(fold_dir, assignment, has_test=False):
        for stem, split in assignment:
            write_label_and_ids(stem, by_image[stem], split, fold_dir)
            link_image(stem, split, fold_dir, img_paths[stem])
        write_data_yaml(fold_dir, has_test)

    # ---- obj1_final: test = last date per terrace, rest train, val=first held ----
    assign = []
    for t in terraces:
        stems = per_terrace[t]
        for j, s in enumerate(stems):
            if j == len(stems) - 1:
                assign.append((s, "test"))
            elif j == len(stems) - 2:
                assign.append((s, "val"))
            else:
                assign.append((s, "train"))
    emit(WORK / "obj1_final", assign, has_test=True)

    # ---- obj1_forward: forward-chaining rounds (train <=k, val k+1) ----
    max_rounds = max(len(per_terrace[t]) for t in terraces)
    for k in range(1, max_rounds):        # fold k validates round k+1
        assign = []
        for t in terraces:
            stems = per_terrace[t]
            for j, s in enumerate(stems):
                if j < k:
                    assign.append((s, "train"))
                elif j == k:
                    assign.append((s, "val"))
        if any(sp == "val" for _, sp in assign):
            emit(WORK / "obj1_forward" / f"fold_{k}", assign, has_test=False)

    # ---- obj2_loto: leave-one-terrace-out (test whole terrace, val next) ----
    if len(terraces) >= 2:
        for i, held in enumerate(terraces):
            val_t = terraces[(i + 1) % len(terraces)]
            assign = []
            for t in terraces:
                split = "test" if t == held else "val" if t == val_t else "train"
                for s in per_terrace[t]:
                    assign.append((s, split))
            emit(WORK / "obj2_loto" / f"held_{held}", assign, has_test=True)

    return per_terrace


def write_tracking_gt(per_terrace, by_image):
    """datasets/tracking/tracking_ST*.json: which beds appear on which dates."""
    out_dir = WORK / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, stems in per_terrace.items():
        record = {}   # bed_position -> {class, dates:[...]}
        for s in stems:
            for ins in by_image[s]:
                rec = record.setdefault(ins["bed_position"],
                                        dict(cls=CLASS_NAMES[ins["cls"]], dates=[]))
                rec["dates"].append(date_index(s))
        n_dates = len(stems)
        summary = dict(
            terrace=t, n_dates=n_dates, n_beds=len(record),
            beds={b: r for b, r in sorted(record.items())},
        )
        (out_dir / f"tracking_{t}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ann_files = sorted(ANNOTATIONS.glob("*.json"))
    if not ann_files:
        raise SystemExit(f"No annotation JSON found in {ANNOTATIONS.resolve()}")

    # index real images by canonical key (hash-tolerant)
    img_index = {}
    for p in IMAGES.iterdir():
        if p.suffix in IMG_EXTS:
            img_index[img_key(p.name)] = p

    by_image = {}          # real filesystem stem -> instances
    img_paths = {}         # real filesystem stem -> Path
    warnings = []
    unmatched = []
    for af in ann_files:
        tasks = json.loads(af.read_text(encoding="utf-8"))
        for task in tasks:
            img_field, instances = parse_task(task)
            key = img_key(img_field)
            if key not in img_index:
                unmatched.append(img_field or "(empty image field)")
                continue
            real = img_index[key]
            stem = real.stem                        # canonical output name
            raw = len(instances)
            instances = dedupe(instances)
            if raw != len(instances):
                warnings.append(f"{stem}: removed {raw - len(instances)} duplicate(s)")
            by_image[stem] = instances
            img_paths[stem] = real

    if not by_image:
        sample_ann = unmatched[:3]
        sample_fs = list(img_index)[:3]
        raise SystemExit(
            "No annotated image matched a file in data/images/.\n"
            f"  annotation image fields (first 3): {sample_ann}\n"
            f"  filesystem keys      (first 3): {sample_fs}\n"
            "  -> the two should share a '<date>-ST<n>_...' core; if they don't, "
            "check the 'image' field in your Label Studio export."
        )
    for miss in unmatched:
        warnings.append(f"no image on disk for annotation {miss!r}")

    per_terrace = build_folds(by_image, img_paths)
    write_tracking_gt(per_terrace, by_image)

    # ---- report ----
    print(f"[build_dataset] {len(by_image)} images across "
          f"{len(per_terrace)} terrace(s): {', '.join(sorted(per_terrace))}")
    for t in sorted(per_terrace):
        n_obj = sum(len(by_image[s]) for s in per_terrace[t])
        print(f"  {t}: {len(per_terrace[t])} dates, {n_obj} instances")
    print(f"[build_dataset] wrote folds -> {WORK.resolve()}")
    for w in warnings:
        print(f"  [!] {w}")


if __name__ == "__main__":
    main()
