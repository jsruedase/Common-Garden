# Paper/ — figuras de seguimiento para el artículo

Genera, a partir de una terraza, la secuencia de imágenes que **ilustra el objetivo
de seguimiento**: se ilumina UNA planta y se la sigue por todas las fechas usando
el identificador que le asigna el algoritmo, no la anotación.

```
el biólogo anota los ids SOLO en la primera fecha
    -> ICP sobre las materas + Hungarian sobre las plantas propagan esos ids
    -> se ilumina, fecha a fecha, la instancia que LLEVA EL ID de la planta elegida
```

Lo que se ve resaltado en las fechas 2..N es **la decisión del algoritmo**. Si el
tracker se equivoca, la figura muestra la planta equivocada y la cartela lo dice
(`SEGUIMIENTO OK` / `ERROR DE ID (en realidad es ...)` / `SIN MATCH`).

## Uso

```bash
python Paper/highlight_track.py                       # terraza y planta al azar
python Paper/highlight_track.py --seed 11             # sorteo reproducible
python Paper/highlight_track.py --terrace ST7 --plant 4A-3-1
python Paper/highlight_track.py --tiers tier1         # sin YOLO (no necesita torch)
python Paper/highlight_track.py --any                 # sortear también entre las
                                                      # plantas que desaparecen
```

Se puede lanzar desde cualquier directorio: las rutas por defecto cuelgan de la
raíz del repo, no del CWD.

| opción | qué hace |
|---|---|
| `--terrace` | `ST7`, `7` o `terrace_7.json`; por defecto una al azar |
| `--plant` | `bed_position` a seguir; por defecto sorteada |
| `--seed` | semilla del sorteo (terraza + planta) |
| `--tiers` | `tier1`, `tier2` o ambos (por defecto) |
| `--any` | levanta el filtro que prefiere plantas seguidas en todas las fechas |
| `--conf`, `--imgsz` | umbral y resolución de YOLO (solo tier-2) |
| `--max-width` | ancho máximo de las imágenes de salida (`0` = original) |
| `--no-strip` | no montar las tiras ni `compare.png` |

## Los dos tiers

| | detecciones | qué mide |
|---|---|---|
| **TIER-1** `tier1_gt/` | polígonos y elipses del export de Label Studio | el tracker **aislado** del segmentador — el régimen en que se evalúan `Time_Tracking/matcher.py` y `tracker.py` |
| **TIER-2** `tier2_yolo/` | máscaras de `Segmentation_Model/best.pt` post-procesadas (casco convexo / elipse ajustada) | la **cadena completa**: aquí aparecen detecciones perdidas y falsos positivos |

En tier-2 la primera fecha se siembra emparejando por centroide las detecciones de
YOLO con las plantas anotadas: es lo que haría el biólogo, que anota sobre lo que
el segmentador detectó. Las fechas siguientes llegan al tracker sin identidad.

**El sorteo de la planta se condiciona solo al tier-1.** Elegirla entre las que el
segmentador sigue bien maquillaría el tier-2 de la propia figura.

## Salida

```
Paper/images/<ST>_<bed>/
├── tier1_gt/<seq>_<fecha>.jpg      una imagen por fecha (foco + cartela)
├── tier2_yolo/<seq>_<fecha>.jpg
├── strip_tier1_gt.png              contact sheet de un tier
├── strip_tier2_yolo.png
├── compare.png                     los dos tiers lado a lado  ← la figura
└── summary.json                    estado y verdad por fecha, aciertos por tier
```

Las fotos van en `.jpg` (son fotografías y pesan) y las rejillas en `.png` (llevan
texto y línea fina). La orientación de las rejillas se decide sola según el aspecto
de las fotos: estas terrazas son panorámicas (p.ej. 3820x1088), así que las fechas
se apilan en vertical y los tiers quedan en columnas.

## Estructura

| archivo | rol |
|---|---|
| `detections.py` | construye "frames enriquecidos" (detección + forma) desde Label Studio o desde YOLO |
| `render.py` | el foco (fondo desaturado en penumbra, planta iluminada) y las rejillas |
| `highlight_track.py` | driver: sortea, propaga, dibuja, resume |

## Qué NO hace

No duplica lógica de seguimiento: importa `load_terrace`, `icp` y `propagate` de
`Time_Tracking/` y `process_instance` de `Segmentation_Model/`. Tampoco escribe
fuera de `Paper/images/` — pone `sys.dont_write_bytecode = True` antes de importar
para no dejar `__pycache__` en carpetas ajenas.
