#!/usr/bin/env python3
"""
Paper/render.py — Figura "foco": una terraza en penumbra, una planta iluminada.
==============================================================================

Cada imagen de salida responde a una sola pregunta visual: *de todas las plantas
de la terraza, cual es LA MISMA que en la primera fecha?* Por eso el fondo se
desatura y se oscurece, y solo la planta que el tracker identifico conserva color,
contorno vivo y etiqueta con su bed_position.

Convenciones de color (BGR, porque el lienzo es OpenCV):
    ambar    la planta seguida  (contorno grueso + halo blanco)
    ambar claro  su matera (mas fina: hereda el id de la planta)
    gris     el resto de instancias, apenas insinuadas
    rojo     franja de aviso cuando el tracker NO encontro la planta esa fecha

Todo se dibuja sobre la foto ORIGINAL a resolucion completa y se reescala al
final (--max-width), para que los trazos no se vean pixelados.
"""
from __future__ import annotations

import cv2
import numpy as np

from detections import shape_points

AMBER = (56, 178, 255)          # planta seguida
AMBER_SOFT = (130, 214, 255)    # matera de esa planta
GREY = (150, 150, 150)          # resto de instancias
WHITE = (255, 255, 255)
INK = (24, 22, 20)              # fondo de las cajas de texto
RED = (60, 60, 220)

FONT = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------
def draw_shape(img, shape, color, thickness):
    if shape["kind"] == "polygon":
        pts = np.asarray(shape["points"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    else:
        cv2.ellipse(img,
                    (int(round(shape["cx"])), int(round(shape["cy"]))),
                    (max(1, int(round(shape["rx"]))), max(1, int(round(shape["ry"])))),
                    shape["rot"], 0, 360, color, thickness, cv2.LINE_AA)


def _text_box(img, org, lines, k, scale=1.0, fg=WHITE, bg=INK, alpha=0.72):
    """Caja de texto semitransparente anclada arriba-izquierda en `org`."""
    if not lines:
        return None
    fs = 0.9 * scale * k
    th = max(1, int(round(2.0 * scale * k)))
    pad = int(round(14 * scale * k))
    sizes = [cv2.getTextSize(t, FONT, fs, th)[0] for t in lines]
    w = max(s[0] for s in sizes) + 2 * pad
    lh = max(s[1] for s in sizes) + int(round(12 * scale * k))
    h = lh * len(lines) + 2 * pad - int(round(12 * scale * k))
    x, y = org
    x = int(np.clip(x, 0, max(0, img.shape[1] - w)))
    y = int(np.clip(y, 0, max(0, img.shape[0] - h)))

    patch = img[y:y + h, x:x + w]
    if patch.size:
        box = np.full_like(patch, bg, dtype=np.uint8)
        img[y:y + h, x:x + w] = cv2.addWeighted(box, alpha, patch, 1 - alpha, 0)
    for i, t in enumerate(lines):
        base = y + pad + sizes[i][1] + i * lh
        cv2.putText(img, t, (x + pad, base), FONT, fs, fg, th, cv2.LINE_AA)
    return (x, y, w, h)


def _leader_label(img, anchor, text, k, color=AMBER):
    """Etiqueta con el id, colgada a la derecha de la planta y unida por una linea."""
    fs = 1.15 * k
    th = max(2, int(round(2.4 * k)))
    (tw, tht), _ = cv2.getTextSize(text, FONT, fs, th)
    pad = int(round(16 * k))
    ax, ay = int(anchor[0]), int(anchor[1])
    bx, by = ax + int(round(105 * k)), ay - int(round(105 * k))
    bx = int(np.clip(bx, 0, img.shape[1] - tw - 2 * pad - 1))
    by = int(np.clip(by, tht + 2 * pad, img.shape[0] - 1))

    cv2.line(img, (ax, ay), (bx, by), color, max(1, int(round(2 * k))), cv2.LINE_AA)
    x0, y0 = bx, by - tht - 2 * pad
    cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, by), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x0 + pad, by - pad), FONT, fs, INK, th, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# El foco
# ---------------------------------------------------------------------------
def spotlight(img, target, others, header_lines, label=None, warning=None,
              dim=0.38, desat=0.80, halo=1.30, max_width=1800):
    """
    Compone una imagen de la terraza con `target` iluminado.

    img          foto BGR a resolucion original
    target       formas de la planta seguida (y su matera); vacio si no hubo match
    others       formas del resto de instancias detectadas esa fecha
    header_lines lineas de la caja superior izquierda (terraza, fecha, tier...)
    label        texto de la etiqueta junto a la planta (el bed_position propagado)
    warning      franja roja inferior cuando no hay match
    """
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    k = max(h, w) / 2000.0                      # escala de trazos y tipografia

    # ---- fondo: desaturado y en penumbra ----
    grey3 = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    back = cv2.addWeighted(img, 1.0 - desat, grey3, desat, 0)
    back = np.clip(back.astype(np.float32) * dim, 0, 255).astype(np.uint8)

    # ---- mascara del foco: circulo suave alrededor de la planta seguida ----
    m = np.zeros((h, w), np.float32)
    centre = radius = None
    if target:
        pts = np.vstack([shape_points(s) for s in target])
        centre = pts.mean(0)
        radius = float(np.max(np.linalg.norm(pts - centre, axis=1)))
        radius = max(radius * halo, 0.05 * min(h, w))
        cv2.circle(m, (int(centre[0]), int(centre[1])), int(radius), 1.0, -1)
        blur = int(max(9, 0.35 * radius)) | 1      # impar
        m = cv2.GaussianBlur(m, (blur, blur), 0)

    m3 = cv2.merge([m, m, m])
    out = (img.astype(np.float32) * m3 + back.astype(np.float32) * (1.0 - m3))
    out = out.astype(np.uint8)

    # ---- el resto de instancias, apenas insinuadas ----
    if others:
        ov = out.copy()
        for s in others:
            draw_shape(ov, s, GREY, max(1, int(round(1.6 * k))))
        out = cv2.addWeighted(ov, 0.20, out, 0.80, 0)

    # ---- la planta seguida ----
    if target:
        ring = int(round(radius))
        cv2.circle(out, (int(centre[0]), int(centre[1])), ring, AMBER,
                   max(1, int(round(2.0 * k))), cv2.LINE_AA)
        for s in target:                                   # halo blanco de contraste
            draw_shape(out, s, WHITE, max(3, int(round(9.0 * k))))
        for s in target:
            colour = AMBER if s["kind"] == "polygon" else AMBER_SOFT
            draw_shape(out, s, colour, max(2, int(round(4.5 * k))))
        if label:
            _leader_label(out, centre, label, k)

    # ---- cartelas ----
    margin = int(round(34 * k))
    if header_lines:
        _text_box(out, (margin, margin), header_lines, k)
    if warning:
        fs = 1.0 * k
        th = max(2, int(round(2.2 * k)))
        (tw, tht), _ = cv2.getTextSize(warning, FONT, fs, th)
        pad = int(round(16 * k))
        y1 = h - margin
        y0 = y1 - tht - 2 * pad
        cv2.rectangle(out, (margin, y0), (margin + tw + 2 * pad, y1), RED, -1)
        cv2.putText(out, warning, (margin + pad, y1 - pad), FONT, fs, WHITE, th, cv2.LINE_AA)

    if max_width and w > max_width:
        s = max_width / float(w)
        out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out


# ---------------------------------------------------------------------------
# Tiras de contacto (la figura que va al articulo)
# ---------------------------------------------------------------------------
def strip(rows, out_path, suptitle, row_titles=None, cell=520, dpi=160):
    """
    Monta la rejilla de la figura: un bloque por tier, una celda por fecha.

    La ORIENTACION se decide sola a partir del aspecto de las fotos. Las terrazas
    de este proyecto son panoramicas (p.ej. 3820x1088), asi que ponerlas en fila
    da una figura larguisima e ilegible: cuando las celdas son apaisadas, las
    fechas se apilan en vertical y los tiers quedan en columnas.

    rows       [[(imagen_bgr, titulo), ...], ...]   una lista por tier
    row_titles etiqueta de cada bloque (p.ej. "TIER-1 ... aciertos 100%")
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ar = float(np.median([im.shape[0] / im.shape[1] for r in rows for im, _ in r]))
    n_tier = len(rows)
    n_date = max(len(r) for r in rows)
    wide = ar < 0.6                       # fotos apaisadas -> fechas en vertical

    def at(tier, date):
        r = rows[tier]
        return r[date] if date < len(r) else None

    if wide:
        nrows, ncols = n_date, n_tier
        cellof = lambda i, j: at(j, i)
    else:
        nrows, ncols = n_tier, n_date
        cellof = lambda i, j: at(i, j)

    title_h = 42                                       # px reservados por celda
    # cabecera: el suptitulo siempre, y ademas las etiquetas de tier cuando estas
    # van arriba (caso apaisado). Se reserva en pulgadas y se descuenta del rect.
    header = 0.82 if (wide and row_titles) else 0.45
    fig_w = (cell * ncols + 26) / dpi
    fig_h = (cell * ar + title_h) * nrows / dpi + header
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            c = cellof(i, j)
            if c is None:
                ax.axis("off")
                continue
            im, title = c
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=8, pad=3, linespacing=1.25)

    fig.suptitle(suptitle, fontsize=11, y=1 - 0.16 / fig_h)
    fig.tight_layout(rect=[0, 0, 1, 1 - header / fig_h])

    # etiqueta de cada bloque (tier), ya con las posiciones definitivas
    if row_titles:
        for t, label in enumerate(row_titles[:n_tier]):
            if wide:                                   # el tier es una COLUMNA
                box = axes[0][t].get_position()
                fig.text(box.x0 + box.width / 2, 1 - 0.52 / fig_h, label,
                         ha="center", va="top", fontsize=9.5, linespacing=1.35)
            else:                                      # el tier es una FILA
                box = axes[t][0].get_position()
                fig.text(0.004, box.y0 + box.height / 2, label, ha="left",
                         va="center", rotation=90, fontsize=9, linespacing=1.3)

    fig.savefig(str(out_path), dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path
