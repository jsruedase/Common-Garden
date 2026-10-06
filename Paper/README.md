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

## Terrazas sin anotación (ST6, ST8)

```bash
C:\Python314\python.exe Paper/highlight_track.py --terrace ST6
```

ST6 y ST8 tienen fotos pero **no** tienen `terrace_N.json`. Es el caso real puro, y
funciona, con tres diferencias que la figura declara en la cartela:

- **solo tier-2** — no hay formas anotadas que dibujar, así que no existe la fila
  de referencia; pedir `--tiers tier1` sobre una de estas terrazas es un error;
- **los ids se inventan** — sin biólogo que ponga `bed_position`, las plantas de la
  primera fecha se numeran `P01..Pnn` en orden de lectura (filas de arriba abajo,
  y de izquierda a derecha dentro de cada fila). Son etiquetas **arbitrarias**;
- **no hay aciertos que reportar** — sin verdad con la que contrastar, cada fecha
  solo puede decir `SEGUIDA` o `SIN MATCH`, nunca `SEGUIMIENTO OK` ni `ERROR DE ID`,
  y `accuracy` sale `null` en `summary.json`.

Dicho de otro modo: sirve para **enseñar** el seguimiento sobre datos nuevos, no para
medirlo. Para medir hace falta una terraza anotada (1-5, 7).

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

## Pintar un export sobre su foto (`annotate_export.py`)

Independiente del seguimiento: toma el JSON que exporta Label Studio y devuelve la
foto con **todas** las anotaciones dibujadas y el `bed_position` de cada cama.

```bash
python Paper/annotate_export.py export.json
python Paper/annotate_export.py export.json --labels both        # bed + collection_id
python Paper/annotate_export.py Assets/annotations/terrace_7.json  # una imagen por fecha
```

- planta (`plant_polygon`) en **verde**, matera (`pot_ellipse`) en **azul**;
- **una** etiqueta por cama, no una por región: la planta y su matera comparten
  `bed_position`, y repetirlo dos veces solo ensucia la imagen;
- las regiones **sin** `bed_position` salen en rojo con `?`, y el resumen las cuenta:
  es la lista de lo que falta completar en Label Studio;
- la foto se resuelve sola desde `data.image` / `file_upload` (quitando el hash de
  Label Studio y el sufijo `_ls` de las tareas que genera `postprocess.py`); con
  `--image` se fuerza, y si no aparece se dibuja sobre un lienzo blanco.

Acepta cualquier export: anotado a mano o pre-anotaciones (si no hay `annotations`,
usa `predictions`). Salida en `Paper/images/annotated/<foto>_annotated.jpg`.

Opciones: `--labels bed|collection|both|none`, `--label-pos above|center`
(por defecto `above`, para no tapar la roseta), `--out`, `--images`, `--max-width`,
`--format jpg|png`.

## Pipeline completo sobre una terraza nueva (`pipeline_terrace.py`)

Encadena las tres etapas sobre una carpeta de fotos y deja, por fecha, la imagen
con las anotaciones y el identificador ya propagado. Es la figura que simula el
sistema entero funcionando sobre una terraza que el modelo no vio anotada.

```bash
# con la primera fecha ya anotada en Label Studio (el flujo real)
python Paper/pipeline_terrace.py --images Assets/images/ST6     --seed-export "C:/ruta/project-2-at-2026-10-06.json"

# sin anotacion: los ids de la fecha 1 son arbitrarios (P01..)
python Paper/pipeline_terrace.py --images Assets/images/ST6
```

| paso | que hace | salida |
|---|---|---|
| 1 segmentar | YOLO-seg `best.pt` + postproceso (casco convexo / elipse) | `01_segmentacion/` |
| 2 sembrar | la fecha 1 recibe los `bed_position` del export anotado, emparejando por centroide; lo que el export no respalda queda como `?1..?n` | — |
| 3 propagar | ICP sobre materas + Hungarian sobre plantas, el **mismo** codigo de la app | `02_tracking/` |
| 4 dibujar | `annotate_export.py` sobre cada fecha | `03_figuras/` + dos figuras |

Salen **dos figuras con el mismo material**, porque hacen cosas distintas:
`<ST>_pipeline.png` es la rejilla, donde los identificadores se **leen**;
`<ST>_pipeline_cascada.png` es la baraja de fechas, que se **mira** y dice "serie
temporal" de un vistazo (el color del borde codifica el orden, y la ultima fecha
queda al frente y completa). La cascada abre el articulo; la rejilla es la que se
revisa.

**Label Studio es el formato de intercambio entre pasos**, no un adorno: cada etapa
escribe un JSON que se puede abrir, revisar o reimportar, y la siguiente lo lee. Si
algo sale raro se ve en que etapa paso, y `02_tracking/` se reimporta tal cual para
corregir a mano.

**Lo que esta figura NO es.** En una terraza sin anotar no hay con que contrastar:
muestra lo que el pipeline **decidio**, no si acerto. Los `NEW#` son plantas que el
tracker no logro casar con la plantilla y enrolo como nuevas — pueden ser plantas
realmente nuevas, o un fallo de identidad. Para numeros hace falta una terraza
anotada: ver `highlight_track.py` (tier-1).

## Estructura

| archivo | rol |
|---|---|
| `detections.py` | construye "frames enriquecidos" (detección + forma) desde Label Studio o desde YOLO |
| `annotate_export.py` | pinta un export de Label Studio (anotaciones + `bed_position`) sobre su foto |
| `pipeline_terrace.py` | encadena segmentar -> sembrar -> propagar -> dibujar sobre una terraza entera |
| `render.py` | el foco (fondo desaturado en penumbra, planta iluminada) y las rejillas |
| `highlight_track.py` | driver: sortea, propaga, dibuja, resume |

## Qué NO hace

No duplica lógica de seguimiento: importa `load_terrace`, `icp` y `propagate` de
`Time_Tracking/` y `process_instance` de `Segmentation_Model/`. Tampoco escribe
fuera de `Paper/images/` — pone `sys.dont_write_bytecode = True` antes de importar
para no dejar `__pycache__` en carpetas ajenas.
