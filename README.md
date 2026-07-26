# Common-Garden
The objective of this project is to automatically perform instance segmentation and time tracking of terraces with frailejones.

## Layout

- `data/` — raw inputs and generated figures: `images/`, `annotations/` (Label Studio exports), `figs/` (matcher output).
- `modeling/` — dataset construction and the YOLO-seg training/inference pipeline: `build_dataset.py`, `cv_run.py`, `quick_train.py`, `run_obj1_cv.py`, `train_gpu.py`, `postprocess.py`.
- `tracking/` — the identity-tracking pipeline: `diagnostic.py` (feasibility diagnostic) and `matcher.py` (the matcher itself).

Scripts read/write `data/` by path relative to their own module, so run them from anywhere, e.g. `python tracking/matcher.py` or `python modeling/build_dataset.py --annotations data/annotations --images data/images`.
