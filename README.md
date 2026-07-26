# JardínComún — Common Garden Plant Tracking

## Goal

Track individual plants across repeated photos of common-garden terraces over time, without manual re-identification: detect each plant and its pot, then re-link the same individual across dates using geometry alone (no tags, no markers).

## Pipeline

1. **Annotation → dataset** (`Segmentation_Model/build_dataset.py`): converts Label Studio exports into a YOLO-seg dataset (polygons for plants, ellipses sampled to polygons for pots), plus an identity sidecar (`bed_position`) kept out of the YOLO labels.
2. **Segmentation training** (`Segmentation_Model/train_gpu.py`): trains a YOLO-seg model (plant vs. pot) on the built dataset.
3. **Post-processing** (`Segmentation_Model/postprocess.py`): runs the trained model on images and converts raw masks into clean shapes — convex hull for plants, fitted ellipse for pots — plus a centroid per instance.
4. **Feasibility diagnostic** (`Time_Tracking/diagnostic.py`): estimates whether centroid geometry alone (pot spacing vs. registration jitter) is enough to re-identify plants across dates.
5. **Tracking / re-identification** (`Time_Tracking/matcher.py`): registers each date's frame onto a seeded template via trimmed ICP on pot centroids, then Hungarian-matches plant centroids to template identities, enrolling new plants and marking absent ones.

## Inference

Add the image(s) to `Assets/images/`, then run post-processing with the trained weights:

```bash
python Segmentation_Model/postprocess.py --weights Segmentation_Model/best.pt --source Assets/images/<your_image>.JPG --save-vis
```

- `--source` can also be a folder to process a batch of images.
- Output: `<stem>.json` (per-instance class, shape, centroid) and, with `--save-vis`, a `<stem>.png` overlay — both written to `Outputs/postprocessed/`.
