#!/usr/bin/env python3
"""
JardínComún — App de seguimiento (versión 1)
============================================

Interfaz para el biólogo: sube el export de Label Studio de una terraza (con los
identificadores anotados SOLO en la primera fecha), la app los propaga a todas
las fechas y devuelve los archivos listos para reimportar en Label Studio.

    anotar fecha 1  →  [esta app]  →  reimportar en LS  →  corregir los swaps

Ejecutar:
    pip install streamlit
    streamlit run app.py

No hace falta tocar código: todo se elige en la pantalla.
"""
import io
import json
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from track_to_labelstudio import run_tracking

st.set_page_config(page_title="JardínComún — Seguimiento", page_icon="🌱", layout="wide")

# ─────────────────────────────── cabecera ──────────────────────────────── #
st.title("🌱 JardínComún — Propagación de identificadores")
st.caption("Anota los ids de la primera fecha en Label Studio, sube aquí el export "
           "de la terraza y descarga las fechas siguientes ya identificadas.")

with st.expander("¿Cómo funciona? (leer una vez)"):
    st.markdown("""
1. En **Label Studio**, anota `Bd-F-C` y `No.Coleccion` en la **primera fecha** de la terraza.
2. Exporta la terraza completa en **JSON** (el formato de siempre) y súbelo aquí.
3. La app alinea cada fecha con la primera (usando las **materas** como referencia)
   y asigna a cada planta el identificador de la cama que le corresponde.
   La matera hereda el id de su planta, y el `No.Coleccion` se propaga solo.
4. Descarga los archivos y **reimpórtalos en Label Studio**: verás los ids ya puestos
   como *predicciones*, listos para revisar y corregir los que estén cambiados.
""")

# ─────────────────────────────── entradas ──────────────────────────────── #
col_a, col_b = st.columns([2, 1])

with col_a:
    up = st.file_uploader("Export de Label Studio de la terraza (.json)", type=["json"])

with col_b:
    combined = st.radio(
        "¿Cómo quieres los archivos?",
        ["Uno por fecha", "Uno solo con todas"],
        help="Uno por fecha es más cómodo si vas revisando encuesta por encuesta.",
    ) == "Uno solo con todas"

with st.expander("Opciones avanzadas (apariencia con DINOv3)"):
    st.markdown(
        "Por defecto el emparejamiento es **solo geométrico**: rápido, sin dependencias "
        "extra y suficiente en la mayoría de terrazas. Activar la apariencia solo ayuda "
        "cuando la geometría es ambigua (plantas movidas, camas muy juntas), "
        "necesita las **fotos** y es bastante más lento."
    )
    use_app = st.checkbox("Usar también la apariencia (requiere fotos y torch/timm)")
    images_dir = st.text_input("Carpeta con las fotos", value="",
                               disabled=not use_app,
                               placeholder="/ruta/a/Assets/images")
    alpha = st.slider("α  (1 = solo geometría · 0 = solo apariencia)",
                      0.0, 1.0, 0.4, 0.2, disabled=not use_app)

run = st.button("▶  Propagar identificadores", type="primary", disabled=up is None)

# ─────────────────────────────── ejecución ─────────────────────────────── #
if run and up is not None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / (up.name or "terrace.json")
        src.write_bytes(up.getvalue())
        out = tmp / "salida"

        try:
            with st.spinner("Alineando fechas y asignando identificadores…"):
                res = run_tracking(
                    src, out_dir=out, combined=combined,
                    images_dir=(images_dir or None) if use_app else None,
                    alpha=alpha if use_app else None,
                )
        except Exception as e:                      # errores claros para el biólogo
            st.error(f"No se pudo procesar el archivo: {e}")
            st.stop()

        st.success(f"Terraza **{res['terrace']}** — {len(res['tasks'])} fechas, "
                   f"{res['n_regions']} regiones identificadas.")
        st.caption(f"Método: {res['method']}")

        # ---- resumen por fecha ----
        rows = [{"Fecha": d["date"].replace("_", " "),
                 "Regiones con id": d["regions"],
                 "Plantas nuevas": d["new"],
                 "Sin identificar": d["unknown"]} for d in res["per_date"]]
        st.subheader("Resumen por fecha")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        total_new = sum(d["new"] for d in res["per_date"])
        if total_new:
            st.info(f"Se detectaron **{total_new}** plantas que no estaban en la primera "
                    f"fecha; se les asignó un id provisional `NEW#`. Revísalas en Label "
                    f"Studio y ponles su `Bd-F-C` definitivo.")
        if any(d["unknown"] for d in res["per_date"]):
            st.warning("Hay regiones sin identificador en la primera fecha. "
                       "Anótalas en Label Studio y vuelve a subir el export para "
                       "que el seguimiento sea correcto.")

        # ---- descarga ----
        st.subheader("Descargar para Label Studio")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in res["files"]:
                z.write(f, arcname=f.name)
        st.download_button(
            "⬇  Descargar todo (.zip)", buf.getvalue(),
            file_name=f"{res['terrace']}_tracked.zip", mime="application/zip",
            type="primary",
        )
        if not combined:
            st.caption("O descarga una fecha suelta:")
            cols = st.columns(min(4, len(res["files"])) or 1)
            for i, f in enumerate(res["files"]):
                with cols[i % len(cols)]:
                    st.download_button(f.name.replace("_tracked.json", ""),
                                       f.read_bytes(), file_name=f.name,
                                       mime="application/json", key=f"dl{i}")

        st.divider()
        st.markdown("**Siguiente paso:** importa estos archivos en Label Studio "
                    "(*Import*), abre cada fecha y corrige los identificadores "
                    "que hayan quedado intercambiados.")

elif up is None:
    st.info("Sube el export de una terraza para empezar.")
