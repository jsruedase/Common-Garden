#!/usr/bin/env python3
"""
Paper/highlight_figure.py — Tres figuras distintas del seguimiento de UNA planta.
================================================================================

Mismo contenido que Paper/highlight_track.py (sembrar en la fecha 1, propagar,
mirar quien lleva el id), pero compuesto para un articulo: tipografia fuera de
la foto, sin cartelas superpuestas, y el veredicto codificado en el MARCO de
cada panel en vez de en un parrafo de texto.

    A  recortes   primer plano de la planta en cada fecha
                  -> se ve el INDIVIDUO: crece, se seca, sigue siendo el mismo
    B  contexto   la terraza entera en penumbra, la planta iluminada
                  -> se ve el PROBLEMA: encontrarla entre decenas de iguales
    C  compuesta  contexto arriba + recortes abajo, en una sola figura
                  -> la que iria al articulo si solo cabe una

Marco de cada panel:  verde = id correcto · rojo = id equivocado ·
gris = el tracker no la encontro · negro = fecha 1 (sembrada, no es merito).

Uso:
    python Paper/highlight_figure.py --terrace ST4
    python Paper/highlight_figure.py --terrace ST1 --plant 5A-3-6 --style A
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import detections as D
import highlight_track as HT
import render

OK, ERR, MISS, SEED, UNK = "ok", "err", "miss", "seed", "unk"
EDGE = {OK: "#1a9850", ERR: "#d73027", MISS: "#9e9e9e", SEED: "#2b2b2b", UNK: "#4575b4"}
NAME = {OK: "id correcto", ERR: "id equivocado", MISS: "no encontrada",
        SEED: "fecha 1 (sembrada)", UNK: "sin verdad que contrastar"}
PLANT_BGR = (56, 178, 255)      # ambar
POT_BGR = (130, 214, 255)


def classify(step, first: bool) -> str:
    if first:
        return SEED
    if step["plant_key"] is None:
        return MISS
    if step["correct"] is True:
        return OK
    if step["correct"] is False:
        return ERR
    return UNK


def target_shapes(step):
    f = step["frame"]
    keys = [k for k in (step["plant_key"], step["pot_key"]) if k]
    return [f["shapes"][k] for k in keys if k in f["shapes"]]


def _draw(img, shapes, k):
    for s in shapes:                                   # halo blanco de contraste
        render.draw_shape(img, s, (255, 255, 255), max(3, int(round(9 * k))))
    for s in shapes:
        render.draw_shape(img, s, PLANT_BGR if s["kind"] == "polygon" else POT_BGR,
                          max(2, int(round(4.5 * k))))


def crop(step, out_px=560, pad=2.1):
    """Primer plano cuadrado centrado en la planta seguida, con su contorno."""
    f = step["frame"]
    img = cv2.imread(str(f["image"])) if f["image"] else None
    shapes = target_shapes(step)
    if img is None or not shapes:
        return np.full((out_px, out_px, 3), 240, np.uint8)

    H, W = img.shape[:2]
    pts = np.vstack([D.shape_points(s) for s in shapes])
    cx, cy = pts.mean(0)
    r = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2.0 * pad
    r = max(r, 0.035 * min(H, W))
    # La ventana se ENCAJA dentro de la foto en vez de rellenar el sobrante:
    # estas terrazas son panoramicas (p.ej. 3820x1088) y un recorte que se sale
    # por arriba o por abajo salia con el borde replicado, una franja chorreada.
    r = min(r, min(H, W) / 2.0)
    cx = float(np.clip(cx, r, W - r))
    cy = float(np.clip(cy, r, H - r))

    vis = img.copy()
    _draw(vis, shapes, k=out_px / (2.0 * r))           # trazo segun el zoom final
    x0, y0, side = int(cx - r), int(cy - r), int(2 * r)
    cut = vis[y0:y0 + side, x0:x0 + side]
    if cut.size == 0:
        return np.full((out_px, out_px, 3), 240, np.uint8)
    return cv2.resize(cut, (out_px, out_px), interpolation=cv2.INTER_AREA)


def context(step, max_w=1500):
    """La terraza entera en penumbra con la planta iluminada (sin texto encima)."""
    f = step["frame"]
    img = cv2.imread(str(f["image"])) if f["image"] else None
    if img is None:
        return np.full((400, 600, 3), 240, np.uint8)
    keys = {k for k in (step["plant_key"], step["pot_key"]) if k}
    tgt = [f["shapes"][k] for k in keys if k in f["shapes"]]
    others = [s for k, s in f["shapes"].items() if k not in keys]
    vis = render.spotlight(img, tgt, others, [], label=None, warning=None,
                           max_width=max_w)
    return vis


# ---------------------------------------------------------------------------
# Composicion (matplotlib: la tipografia va FUERA de la foto)
# ---------------------------------------------------------------------------
def _panel(ax, img_bgr, title, kind, lw=3.2):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(EDGE[kind]); sp.set_linewidth(lw)
    if title:
        ax.set_title(title, fontsize=8.5, pad=4, color="#222222")


def _legend(fig, kinds, y=0.02, loc="lower center"):
    seen = [k for k in (SEED, OK, ERR, MISS, UNK) if k in kinds]
    h = [plt.Line2D([], [], marker="s", ls="", ms=9, mfc="white",
                    mec=EDGE[k], mew=2.5, label=NAME[k]) for k in seen]
    fig.legend(handles=h, loc=loc, ncol=len(h), fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, y))


def _suptitle(terr, target, n_ok, n_eval, fig_w=12.0):
    """Titulo, plegado al ancho real de la figura para que no se salga."""
    import textwrap
    head = f"{terr} - seguimiento del individuo {target}"
    if n_eval:
        head += f"   —   {n_ok}/{n_eval} fechas con el id correcto"
    sub = ("id sembrado solo en la primera fecha; "
           "lo resaltado en las demas es la decision del algoritmo")
    width = max(40, int(fig_w * 13))
    return head + "\n" + "\n".join(textwrap.wrap(sub, width))


def fig_crops(steps, terr, target, out_path, dpi=200):
    """A — una fila de primeros planos."""
    n = len(steps)
    kinds = [classify(s, i == 0) for i, s in enumerate(steps)]
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2.15), dpi=dpi)
    axes = np.atleast_1d(axes)
    for ax, st, kd in zip(axes, steps, kinds):
        _panel(ax, crop(st), st["frame"]["date"].replace("_", " "), kd)
    n_eval = sum(1 for k in kinds[1:] if k in (OK, ERR))
    fig.suptitle(_suptitle(terr, target, kinds[1:].count(OK), n_eval, 1.5 * n),
                 fontsize=10)
    _legend(fig, kinds, y=0.005)
    fig.tight_layout(rect=[0, 0.09, 1, 0.86])
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def fig_context(steps, terr, target, out_path, dpi=200):
    """B — una fila con la terraza completa en penumbra."""
    n = len(steps)
    kinds = [classify(s, i == 0) for i, s in enumerate(steps)]
    panels = [context(s) for s in steps]
    ar = float(np.median([p.shape[0] / p.shape[1] for p in panels]))

    # El alto se contabiliza en PULGADAS, no a ojo: con terrazas panoramicas
    # (ar ~ 0.28) una altura estimada deja los titulos encima de la fila de arriba.
    panel_w = 2.6 if ar > 0.55 else 4.2
    cols = n if ar > 0.55 else 2                       # panoramicas: dos por fila
    rows = int(np.ceil(n / cols))
    title_h, header, footer = 0.30, 0.95, 0.45
    fig_w = panel_w * cols
    fig_h = rows * (panel_w * ar + title_h) + header + footer
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis("off"); continue
        _panel(ax, panels[i], steps[i]["frame"]["date"].replace("_", " "), kinds[i])
    n_eval = sum(1 for k in kinds[1:] if k in (OK, ERR))
    fig.suptitle(_suptitle(terr, target, kinds[1:].count(OK), n_eval, fig_w),
                 fontsize=10, y=1 - 0.18 / fig_h)
    _legend(fig, kinds, y=0.006)
    fig.tight_layout(rect=[0, footer / fig_h, 1, 1 - header / fig_h], h_pad=1.1)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


def fig_combined(steps, terr, target, out_path, dpi=200):
    """C — contexto de la primera y la ultima fecha arriba, recortes abajo."""
    n = len(steps)
    kinds = [classify(s, i == 0) for i, s in enumerate(steps)]
    top = [(0, steps[0], kinds[0]), (n - 1, steps[-1], kinds[-1])]
    ctx = [context(s, max_w=1700) for _i, s, _k in top]
    ar = float(np.median([c.shape[0] / c.shape[1] for c in ctx]))

    fig = plt.figure(figsize=(1.52 * n, 1.52 * n * ar / 2 + 2.9), dpi=dpi)
    gs = fig.add_gridspec(2, n, height_ratios=[1.52 * n * ar / 2, 2.0],
                          hspace=0.28, wspace=0.07)
    half = n // 2
    for col, ((idx, st, kd), img) in enumerate(zip(top, ctx)):
        ax = fig.add_subplot(gs[0, col * half:(col + 1) * half] if half else gs[0, col])
        _panel(ax, img, f"{'primera' if col == 0 else 'ultima'} fecha  ·  "
                        f"{st['frame']['date'].replace('_', ' ')}", kd, lw=3.6)
    for col, (st, kd) in enumerate(zip(steps, kinds)):
        _panel(fig.add_subplot(gs[1, col]), crop(st),
               st["frame"]["date"].replace("_", " "), kd)
    n_eval = sum(1 for k in kinds[1:] if k in (OK, ERR))
    fig.suptitle(_suptitle(terr, target, kinds[1:].count(OK), n_eval, 1.52 * n),
                 fontsize=10.5)
    _legend(fig, kinds, y=0.005)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.02, right=0.98)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Figuras de seguimiento de una planta")
    ap.add_argument("--terrace", default="ST4", help="ST4, 4 o terrace_4.json")
    ap.add_argument("--plant", default=None, help="bed_position; por defecto una seguida en todas")
    ap.add_argument("--annotations", type=Path, default=D.DEFAULT_ANNOTATIONS)
    ap.add_argument("--images", type=Path, default=D.DEFAULT_IMAGES)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "images" / "highlight")
    ap.add_argument("--style", choices=["A", "B", "C", "D", "all"], default="all")
    args = ap.parse_args()

    want = str(args.terrace).lower().replace("st", "").replace(".json", "").strip("_")
    path = next((f for f in D.terrace_files(args.annotations)
                 if f.stem.split("_")[-1] == want), None)
    if path is None:
        raise SystemExit(f"[error] no hay export para {args.terrace!r}")
    frames = D.load_tier1(path, args.images)
    terr = frames[0]["terrace"]

    cov, n_later = HT.coverage(frames)
    cands = D.seed_candidates(frames)
    full = [b for b in cands if cov.get(b, 0) == n_later]
    target = args.plant or (full or cands)[0]
    if target not in cands:
        raise SystemExit(f"[error] {target!r} no esta en la fecha 1")
    print(f"[{terr}] {len(frames)} fechas · individuo {target} "
          f"({len(full)}/{len(cands)} plantas se siguen en todas)")

    steps = HT.follow(frames, target)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"{terr}_{target.replace('/', '-')}"
    jobs = {"A": (fig_crops, "A_recortes"), "B": (fig_context, "B_contexto"),
            "C": (fig_combined, "C_compuesta"), "D": (fig_area, "D_area")}
    for key, (fn, name) in jobs.items():
        if args.style in (key, "all"):
            p = fn(steps, terr, target, out / f"{stem}_{name}.png")
            print(f"  {name:12s} -> {p}")


# ---------------------------------------------------------------------------
# D — area de la roseta: la aplicacion (deteccion de deterioro)
# ---------------------------------------------------------------------------
def shape_area(shape) -> float:
    return float(cv2.contourArea(D.shape_points(shape, n=256).astype(np.float32)))


def pot_ruler(frame) -> float:
    """
    Regla de la fecha: mediana del EJE MAYOR de las materas de la terraza, en px.

    Sirve para comparar areas entre fechas sin depender de la resolucion ni de
    la distancia de la camara. Tres razones para esta eleccion:
      - la matera es un objeto manufacturado de diametro constante, hay una al
        lado de cada planta y ya viene segmentada;
      - se usa el EJE MAYOR, no el area: al proyectar un circulo inclinado, el
        eje mayor conserva el diametro real y el menor se acorta con el coseno
        del angulo, asi que el area de la elipse arrastra el error de perspectiva;
      - se toma la MEDIANA de toda la terraza, no la matera de la propia planta,
        para promediar el error de anotacion de una sola elipse.
    """
    majors = [2 * max(s["rx"], s["ry"]) for (c, _i), s in frame["shapes"].items()
              if c == "matera"]
    return float(np.median(majors)) if majors else float("nan")


def area_series(steps):
    """(cruda, normalizada, regla) relativas a la primera fecha. NaN si no hay match."""
    raw, ruler = [], []
    for st in steps:
        f = st["frame"]
        pk = st["plant_key"]
        raw.append(shape_area(f["shapes"][pk]) if pk and pk in f["shapes"] else np.nan)
        ruler.append(pot_ruler(f))
    raw = np.array(raw, float)
    ruler = np.array(ruler, float)
    norm = raw / ruler ** 2
    return raw / raw[0], norm / norm[0], ruler


def fig_area(steps, terr, target, out_path, dpi=200):
    """D — contexto + recortes + la curva de area normalizada."""
    n = len(steps)
    kinds = [classify(s, i == 0) for i, s in enumerate(steps)]
    rel_raw, rel_norm, ruler = area_series(steps)
    ctx = [context(steps[0], max_w=1700), context(steps[-1], max_w=1700)]
    ar = float(np.median([c.shape[0] / c.shape[1] for c in ctx]))

    h_ctx = 1.52 * n * ar / 2
    fig = plt.figure(figsize=(1.52 * n, h_ctx + 2.0 + 2.5), dpi=dpi)
    gs = fig.add_gridspec(3, n, height_ratios=[h_ctx, 2.0, 2.5], hspace=0.42, wspace=0.07)

    half = n // 2
    for col, (img, idx) in enumerate(zip(ctx, (0, n - 1))):
        ax = fig.add_subplot(gs[0, col * half:(col + 1) * half])
        _panel(ax, img, f"{'primera' if col == 0 else 'ultima'} fecha  ·  "
                        f"{steps[idx]['frame']['date'].replace('_', ' ')}", kinds[idx], lw=3.6)
    for col, (st, kd) in enumerate(zip(steps, kinds)):
        _panel(fig.add_subplot(gs[1, col]), crop(st), "", kd)

    ax = fig.add_subplot(gs[2, :])
    x = np.arange(n)
    ax.plot(x, rel_raw, ls="--", lw=1.2, color="#bbbbbb", zorder=1,
            label="sin normalizar (px²)")
    ax.plot(x, rel_norm, ls="-", lw=1.8, color="#444444", zorder=2,
            label="normalizada por el tamano de la matera")
    for xi, yi, kd in zip(x, rel_norm, kinds):
        ax.scatter(xi, yi, s=58, color=EDGE[kd], edgecolors="white",
                   linewidths=1.4, zorder=3)
    ax.axhline(1.0, color="#dddddd", lw=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([s["frame"]["date"].replace("_", " ") for s in steps],
                       fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("area de la roseta\n(fecha 1 = 1)", fontsize=8.5)
    ax.set_ylim(0, max(1.15, float(np.nanmax(rel_norm)) * 1.1))
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    drop = 100 * (1 - rel_norm[-1])
    ax.annotate(f"-{drop:.0f}%", (x[-1], rel_norm[-1]), textcoords="offset points",
                xytext=(6, 10), fontsize=9, color="#d73027", weight="bold")
    if not np.allclose(ruler, ruler[0]):
        ax.text(0.995, 0.93, "la regla cambia en las fechas marcadas con ·",
                transform=ax.transAxes, ha="right", fontsize=7, color="#888888")
        for xi, r in zip(x, ruler):
            if abs(r - ruler[0]) > 1e-6:
                ax.text(xi, 0.02, "·", transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=16, color="#888888")

    n_eval = sum(1 for k in kinds[1:] if k in (OK, ERR))
    fig.suptitle(_suptitle(terr, target, kinds[1:].count(OK), n_eval, 1.52 * n),
                 fontsize=10.5)
    _legend(fig, kinds, y=0.905, loc="upper center")
    fig.subplots_adjust(top=0.875, bottom=0.115, left=0.07, right=0.98)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
