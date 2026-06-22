# AutoTS Reproduction Handoff README

This document is a classmate-facing guide for understanding and continuing the
reproduction of **Han et al. 2025, "Traffic sign localization and orientation
classification for automated map updating"**.

The short version: this project reproduces the KITTI-TS side of the paper well
for detection and localization, partially reproduces the NSAL ablation study,
generates Figure 7 and Figure 8 style artifacts, and adds an HTML map viewer.
The current orientation Table IV run does **not** match the paper's reported
AutoTS orientation result, so treat Table IV as a still-open reproduction item.

## Session Update (2026-06-09): Detected-Box Association Fix

This section records the most recent round of work. Read it first; it changes the
detected location points, the Table IV numbers, the Figure 8 values, and adds a
new per-sign popup to the map.

**Root cause found and fixed.** `select_detected_sequence()` in
`AutoTS/orientation_repro_lib.py` used to build the per-frame detected track with
a greedy frame-to-frame chain (`iou(previous_detected_box, box) > 0.5`). For
small / edge-drifting KITTI signs, consecutive detected boxes have **zero
overlap** even though the detector fires correctly (IoU 0.85+ vs GT every frame),
so the chain broke on the first step and the track collapsed to a **single
location point**. With one point, NSAL/MinCut is a no-op, which is why Figure 8
rows 1/3/4 showed identical AutoTS vs ablation errors and large localization
errors. The fix associates **each frame's detection to that frame's GT box
independently** (best-IoU > 0.5), instead of chaining to the previous detection.

**Effect on detected location points (per sign):**

| Sign | Before | After | GT points | Frames |
|---|---:|---:|---:|---:|
| train/3 | 1 | 14 | 16 | 16 |
| train/274 | 17 | 17 | 17 | 17 |
| test/2 | 1 | 8 | 8 | 8 |
| train/174 | 1 | 7 | 7 | 7 |

Dataset-wide, signs with only one detected point dropped to **3/290**; median
detected points per sign is now 9 (max 33). This was **not** a detector or data
problem; it was purely the association logic.

**Pipeline re-run after the fix (all on the rebuilt cache):**

1. Rebuilt the orientation cache with `force=True` (both splits).
2. Retrained Table IV (300 epochs, seed 42) on the new cache.
3. Regenerated Figure 8 with the retrained `AutoTS_ours_best.pth`.
4. Generated new per-sign map results with `build_map_sign_results.py`.

See the updated Table IV (Section 5), Figure 8 (Section 5), new tool (Sections 4
and 6), and Status Summary (Section 10) below for the numbers.

## 1. What This Project Contains

The repository has two main technical parts:

| Area | Main files/folders | Purpose |
|---|---|---|
| AutoTS reproduction | `AutoTS/` | Traffic sign detection, localization, orientation classification, paper-table reproduction, and visualizations. |
| Depth support | `PlaneDepth/` | PlaneDepth model code used for monocular depth estimation in the paper pipeline. |
| KITTI-TS data | `AutoTS/KITTI-TS/` | Train/test sign annotations, KITTI frame images, and COCO-style detection annotations. |
| Depth maps | `AutoTS/data/img_depth.npz` | Precomputed depth maps used by localization scripts. |
| Results | `AutoTS/results/` | Saved reproduction CSVs, checkpoints, caches, figures, metadata, and notes. |
| Web tools | `AutoTS/kitti_ts_viewer.html`, `AutoTS/kitti_ts_map.html` | Browser-based viewers for inspecting KITTI-TS signs and the generated Figure 8 artifact. |

The paper pipeline has three main tasks:

1. **Traffic sign detection**: Faster R-CNN/Detectron2 detects signs in KITTI
   images.
2. **Traffic sign localization**: depth plus camera/GPS geometry generates
   location points, then NSAL clusters them into a final sign location.
3. **Orientation classification**: a neural classifier predicts whether a sign
   is leftward, backward, or rightward using image/ROI/SIFT/location features.

This repository also contains a vendored Detectron2 codebase under
`AutoTS/detectron2/`, so many detector scripts use local Detectron2 imports
instead of a separate installed Detectron2 package.

## 2. Setup

The original `AutoTS/README.md` gives this base environment:

```bash
conda create -n autots python=3.10
conda activate autots
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
cd AutoTS
pip install -r requirements.txt
```

Important local requirements:

- The KITTI-TS data must exist under `AutoTS/KITTI-TS/`.
- The detector checkpoint should exist at `AutoTS/detector_model_0049999.pth`.
- The orientation config used for the main reproduction is
  `AutoTS/orientation_config_v2_paper.yaml`.
- Current config uses `MODEL.DEVICE: "mps"`. Change this to `"cuda"` or `"cpu"`
  if your machine does not support Apple MPS.
- The correct orientation script name is `orientation_train.py`; ignore the old
  misspelled command in the original README.

Useful dataset counts:

| Split | Signs | Left | Back | Right |
|---|---:|---:|---:|---:|
| Train | 227 | 24 | 177 | 26 |
| Test | 63 | 8 | 46 | 9 |

Because the test split is small, orientation recall moves in coarse steps. For
example, rightward recall is based on only 9 test samples.

## 3. Code Architecture

### Detection

Primary files:

- `AutoTS/eval_detection_table1.py`
- `AutoTS/train_net.py`
- `AutoTS/detector.py`
- `AutoTS/configs/TS-RCNN-FPN.yaml`

This part evaluates or trains the traffic sign detector and reports COCO-style
metrics such as AP, AP50, AP75, APs, and APm. The saved reproduction outputs are:

- `AutoTS/results_table1_detection.csv`
- `AutoTS/results_table1_full_coco.csv`
- `AutoTS/results/table1_claims_verification.md`

### Localization

Primary files:

- `AutoTS/eval_table2_localization.py`
- `AutoTS/eval_table3_ablation.py`
- `AutoTS/data/img_depth.npz`

The localization scripts use sign boxes, GPS/camera geometry, and depth values
to compute per-frame sign location points. NSAL then clusters those points with
MinCut and weighted center estimation.

Saved outputs:

- `AutoTS/results/v3_table2/results_table2.csv`
- `AutoTS/results/v3_table2/table2_claims_verification.md`
- `AutoTS/results/v4_table3/results_table3.csv`
- `AutoTS/results/v4_table3/table3_claims_verification.md`
- `AutoTS/results/paper_table2_localization.csv`
- `AutoTS/results/paper_table3_ablation.csv`

### Orientation

Primary files:

- `AutoTS/orientation_train.py`
- `AutoTS/orientation_net.py`
- `AutoTS/orientation_dataset.py`
- `AutoTS/orientation_repro_lib.py`
- `AutoTS/run_table4_orientation.py`
- `AutoTS/orientation_config_v2_paper.yaml`

The newer Table IV reproduction path is based on cached feature extraction. It
caches image features, ROI features, SIFT features, boxes, labels, categories,
locations, and image IDs once, then reuses them for all Table IV variants.

Saved outputs:

- `AutoTS/results/orientation_cache/train_features.pt`
- `AutoTS/results/orientation_cache/test_features.pt`
- `AutoTS/results/table4_orientation/table4_orientation_best.csv`
- `AutoTS/results/table4_orientation/table4_orientation_final.csv`
- `AutoTS/results/table4_orientation/paper_table4_reference.csv`
- `AutoTS/results/table4_orientation/table4_predictions_best.json`
- `AutoTS/results/table4_orientation/table4_history.json`
- `AutoTS/results/table4_orientation/table4_notes.md`

### Figures And Web Viewers

Primary files:

- `AutoTS/visualize_figure7.py`
- `AutoTS/visualize_figure8.py`
- `AutoTS/build_map_sign_results.py` (per-sign popup data generator; added
  2026-06-09)
- `AutoTS/kitti_ts_viewer.html`
- `AutoTS/kitti_ts_map.html`

Saved outputs:

- `AutoTS/results/figure7/figure7_sift_heatmaps.png`
- `AutoTS/results/figure7/figure7_sift_heatmaps_metadata.json`
- `AutoTS/results/figure8/figure8_qualitative.png`
- `AutoTS/results/figure8/figure8_qualitative_errors.csv`
- `AutoTS/results/figure8/figure8_qualitative_metadata.json`
- `AutoTS/results/map/map_sign_results.json` (per-sign popup data)

## 4. How To Run The Reproduction

Run commands from `AutoTS/` unless noted otherwise.

### Table I: Detector Evaluation

```bash
cd AutoTS
python eval_detection_table1.py
```

Expected saved outputs:

- `results_table1_detection.csv`
- `results_table1_full_coco.csv`

### Table II: Localization Baselines

```bash
cd AutoTS
python eval_table2_localization.py
```

Expected saved output:

- `results/v3_table2/results_table2.csv`

### Table III: NSAL Ablations

```bash
cd AutoTS
python eval_table3_ablation.py
```

Expected saved output:

- `results/v4_table3/results_table3.csv`

### Table IV: Orientation Ablations

Full 300-epoch run:

```bash
cd AutoTS
python run_table4_orientation.py \
  --cfg ./orientation_config_v2_paper.yaml \
  --epochs 300 \
  --seed 42 \
  --cache-dir ./results/orientation_cache \
  --out-dir ./results/table4_orientation
```

Fast smoke test:

```bash
cd AutoTS
python run_table4_orientation.py \
  --cfg ./orientation_config_v2_paper.yaml \
  --epochs 1 \
  --seed 42 \
  --cache-dir ./results/orientation_cache \
  --out-dir ./results/table4_orientation_smoke \
  --limit 8
```

Expected saved outputs:

- `results/table4_orientation/table4_orientation_best.csv`
- `results/table4_orientation/table4_orientation_final.csv`
- `results/table4_orientation/paper_table4_reference.csv`
- `results/table4_orientation/table4_notes.md`

### Figure 7: SIFT Heatmaps

```bash
cd AutoTS
python visualize_figure7.py \
  --cache-dir ./results/orientation_cache \
  --backward-id 113 \
  --rightward-id 288 \
  --out ./results/figure7/figure7_sift_heatmaps.png
```

Expected saved outputs:

- `results/figure7/figure7_sift_heatmaps.png`
- `results/figure7/figure7_sift_heatmaps_metadata.json`

### Figure 8: Qualitative Localization/Orientation Panel

```bash
cd AutoTS
python visualize_figure8.py \
  --cache-dir ./results/orientation_cache \
  --orientation-ckpt ./results/table4_orientation/AutoTS_ours_best.pth \
  --out ./results/figure8/figure8_qualitative.png \
  --web-data-out ./results/figure8/figure8_web_data.json \
  --points-csv-out ./results/figure8/figure8_location_points.csv
```

Expected saved outputs:

- `results/figure8/figure8_qualitative.png`
- `results/figure8/figure8_qualitative_errors.csv`
- `results/figure8/figure8_qualitative_metadata.json`
- `results/figure8/figure8_web_data.json`
- `results/figure8/figure8_location_points.csv`

### Per-Sign Map Popup Data

Generates the per-sign result JSON consumed by the `kitti_ts_map.html` popups
(predicted orientation, AutoTS estimated location + error, and frame-sequence
thumbnails for every sign).

```bash
cd AutoTS
python build_map_sign_results.py
```

Expected saved output:

- `results/map/map_sign_results.json`

Regenerate this whenever the cache or the orientation checkpoint changes.

### Rebuilding The Cache After The Association Fix

The detected-box association fix only takes effect once the orientation cache is
rebuilt. Force a rebuild of both splits, then retrain Table IV and regenerate the
downstream artifacts:

```bash
cd AutoTS
# 1. Force-rebuild train + test caches (runs the detector over all signs).
python -c "from orientation_repro_lib import build_or_load_cache, load_cfg; \
cfg=load_cfg('./orientation_config_v2_paper.yaml'); \
build_or_load_cache(cfg,'./results/orientation_cache','train',cfg.DATA.TRAIN_JSON,cfg.DATA.TRAIN_IMG_ROOT,None,True); \
build_or_load_cache(cfg,'./results/orientation_cache','test',cfg.DATA.TEST_JSON,cfg.DATA.TEST_IMG_ROOT,None,True)"
# 2. Retrain Table IV on the new cache (see the Table IV command above).
# 3. Regenerate Figure 8 (see the Figure 8 command above).
# 4. Regenerate per-sign popup data (see above).
```

## 5. What Was Achieved Compared With The Paper

### Table I: Traffic Sign Detector

Current saved result: `AutoTS/results_table1_detection.csv`

| Metric | Paper | Ours | Status |
|---|---:|---:|---|
| AP | 60.08 | 71.72 | Ours higher |
| AP50 | 84.27 | 85.92 | Ours higher |
| AP75 | 72.78 | 84.89 | Ours higher |
| APs | 51.91 | 69.44 | Ours higher |
| APm | 67.61 | 75.09 | Ours higher |

Verdict: **Table I is reproduced strongly**, although the higher detector
numbers may come from version/checkpoint/training differences.

### Table II: KITTI-TS Localization

Current saved result: `AutoTS/results/v3_table2/results_table2.csv`

| Method | Paper MAE | Our MAE | Paper RMSE | Our RMSE | Paper R@1m | Our R@1m | Paper R@2m | Our R@2m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GeoLocating [7] | 3.98 | 4.26 | 6.27 | 5.68 | 11.86 | 15.27 | 22.03 | 35.64 |
| GeoLocating+NSAL | 3.80 | 4.28 | 5.92 | 5.70 | 15.25 | 13.09 | 27.12 | 35.64 |
| AutoTS | 2.38 | 2.36 | 3.42 | 3.53 | 30.51 | 38.55 | 54.23 | 62.55 |

Verdict: **Table II is largely reproduced for KITTI-TS.** AutoTS remains clearly
better than the baselines and is very close to the paper in MAE/RMSE while
achieving higher recall thresholds in this run.

Not reproduced: Aalborg rows from the paper.

### Table III: NSAL Ablation

Current saved result: `AutoTS/results/v4_table3/results_table3.csv`

All-data rows:

| Method | Paper MAE | Our MAE | Paper RMSE | Our RMSE | Paper R@1m | Our R@1m | Paper R@2m | Our R@2m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AutoTS w/o MinCut | 2.51 | 2.36 | 3.74 | 3.52 | 27.11 | 38.18 | 50.87 | 62.55 |
| AutoTS w/o Weight | 2.49 | 2.41 | 3.72 | 3.59 | 28.81 | 37.82 | 49.15 | 61.45 |
| AutoTS-K-Means | 2.56 | 2.43 | 3.82 | 3.59 | 27.11 | 37.45 | 47.46 | 61.82 |
| AutoTS-DBSCAN | 2.48 | 2.22 | 3.79 | 3.19 | 28.81 | 37.45 | 52.54 | 62.55 |
| AutoTS | 2.38 | 2.36 | 3.42 | 3.53 | 30.51 | 38.55 | 54.23 | 62.55 |

Sparse rows:

| Method | Paper MAE | Our MAE | Paper RMSE | Our RMSE | Paper R@1m | Our R@1m | Paper R@2m | Our R@2m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AutoTS-K-Means* | 2.46 | 2.31 | 3.52 | 3.24 | 28.81 | 35.94 | 52.54 | 58.59 |
| AutoTS-DBSCAN* | 2.42 | 2.28 | 3.48 | 3.20 | 30.51 | 35.94 | 52.54 | 59.38 |
| AutoTS* | 2.36 | 2.28 | 3.37 | 3.18 | 32.20 | 35.94 | 54.23 | 59.38 |

Verdict: **Table III is partially reproduced.** AutoTS remains strong and beats
K-Means, but DBSCAN beats or ties NSAL in some current metrics. This means the
paper's NSAL superiority claim is not fully confirmed by the current run.

### Table IV: Orientation Classification

Current saved result: `AutoTS/results/table4_orientation/table4_orientation_best.csv`
(retrained 2026-06-09 on the fixed cache).

The table below shows the run **before** the association fix (old single-point
cache) and **after** the fix + cache rebuild + 300-epoch retrain.

| Method | Paper Acc | Acc before | Acc after | Paper mRec | mRec before | mRec after |
|---|---:|---:|---:|---:|---:|---:|
| AutoTS w/ ROI | 71.43 | 82.54 | 84.13 | 42.01 | 70.25 | 64.09 |
| AutoTS w/ ROIS | 73.02 | 77.78 | **85.71** | 55.57 | 67.15 | **71.70** |
| AutoTS w/o SIFT | 76.19 | 77.78 | 80.95 | 62.38 | 59.14 | 63.57 |
| AutoTS w/o MImg | 80.95 | 82.54 | 84.13 | 68.60 | 65.42 | 70.51 |
| AutoTS w/ BiLSTMs | 82.54 | 80.95 | 84.13 | 69.79 | 61.71 | 69.59 |
| AutoTS w/ LSTMs | 82.54 | 80.95 | 82.54 | 67.07 | 61.71 | 69.79 |
| AutoTS (ours) | 85.71 | 74.60 | 77.78 | 70.77 | 61.13 | 61.19 |
| AutoTS† (GT-box) | 88.89 | 76.19 | 79.37 | 79.11 | 60.93 | 62.84 |

Verdict: **Table IV improved but is still not fully reproduced.** The fix lifted
accuracy on all 8 variants (+1.6 to +7.9) and mRecall on 7/8. `AutoTS w/ ROIS`
now reaches `85.71 / 71.70`, matching the paper's *ours* accuracy and beating its
mRecall. But the headline `AutoTS (ours)` row only moved to `77.78 / 61.19`,
still about 8 accuracy and 10 mRecall below the paper's `85.71 / 70.77`. So the
single-point bug was real and worth fixing, but it was **not** the sole cause of
the `AutoTS (ours)` gap — that row remains a still-open Table IV item.

Important note: the paper's printed `AutoTS w/o SIFT` row is internally
inconsistent. The printed per-class recalls `37.50`, `89.13`, and `44.44` do
not average to the printed mRecall `62.38`; they average to about `57.36`.

### Figure 7

Generated output:

- `AutoTS/results/figure7/figure7_sift_heatmaps.png`
- `AutoTS/results/figure7/figure7_sift_heatmaps_metadata.json`

The script renders two `12 x 160` SIFT heatmaps, matching the intended paper
layout. However, the cached samples for test sign IDs `113` and `288` had only
`11` and `1` SIFT rows respectively, so the script pads the missing rows. The
metadata records:

- Backward ID `113`: original rows `11`, padded rows `1`
- Rightward ID `288`: original rows `1`, padded rows `11`
- Rendered rows `12`
- Feature dimension `160`

Verdict: **Figure 7 is generated, but it uses padded rows for the selected
cached IDs.**

### Figure 8

Generated outputs:

- `AutoTS/results/figure8/figure8_qualitative.png`
- `AutoTS/results/figure8/figure8_qualitative_errors.csv`
- `AutoTS/results/figure8/figure8_qualitative_metadata.json`
- `AutoTS/results/figure8/figure8_web_data.json`
- `AutoTS/results/figure8/figure8_location_points.csv`

The current Figure 8 has four qualitative rows. The PNG is still generated, but
the main inspection view is now the interactive tab at
`AutoTS/kitti_ts_map.html#figure8`. Rendered values are recomputed from the
current cache/checkpoint; paper reference values are stored in
`figure8_qualitative_metadata.json`, and row/frame-level web data is stored in
`figure8_web_data.json`. The per-coordinate point table is stored in
`figure8_location_points.csv`; each row is one concrete map point attached to a
parent sign ID such as `train/274`. The web tab shows `Paper Reference` and
`Current Reproduction` side by side so paper Figure 8 values are visible without
being confused with the current checkpoint output.

Values below are **after** the association fix + retrain (regenerated
2026-06-09). The "before" column is the old single-point run.

| Row | Split | Sign ID | GT Direction | Pred Direction | Points (before→after) | AutoTS Error (before→after) | Paper Error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | train | 3 | Backward | Backward | 1 → 14 | 9.92 → 3.09 m | 2.15 m |
| 2 | train | 274 | Backward | Backward | 17 → 17 | 1.16 → 1.16 m | 1.57 m |
| 3 | test | 2 | Rightward | Leftward | 1 → 8 | 0.77 → 0.83 m | 0.85 m |
| 4 | train | 174 | Backward | Backward | 1 → 7 | 1.19 → 0.82 m | 0.81 m |

The headline change is **Row 1** (the paper's Figure 8 row 1, the Yield sign): it
went from a wrong orientation (Leftward) and a 9.92 m error to the correct
**Backward** orientation and a **3.09 m** error, now genuinely comparable to the
paper's 2.15 m. Localization on all four rows is now within ~0 to 0.9 m of the
paper. Row 3 orientation is still wrong (Leftward vs Rightward); that is the
residual Table IV gap, not a localization issue. Rows 1 to 3 use the intended
paper-caption matches; Row 4 uses `train/174`, the nearest paper row-4 GPS/visual
match in KITTI-TS.

Verdict: **Figure 8 is now an honest qualitative reproduction with localization
close to the paper.** One caveat for demos: the "MinCut helps" gap still does not
show dramatically. Rows 1 and 2 give identical AutoTS vs w/o-MinCut errors, but
now for a legitimate reason — the recovered point clouds are clean, so MinCut has
no outliers to remove (the paper's own row 2 is also identical, 1.57 = 1.57).
Explain this truthfully rather than implying a large ablation effect.

## 6. Web Tools Created

Two HTML tools are available.

### `kitti_ts_viewer.html`

Open:

```text
AutoTS/kitti_ts_viewer.html
```

This is a lightweight dataset/sign viewer backed by generated viewer data. Use
it to inspect sign crops and metadata without running training code.

### `kitti_ts_map.html`

This is the more useful web map. It uses MapLibre and Esri satellite tiles for
the main KITTI-TS sign map, and the Figure 8 tab uses Google satellite tiles for
GT GPS panels plus Google road-map tiles for estimated-location panels.

**You must serve this page over HTTP.** Opening `kitti_ts_map.html` directly with
a `file://` path makes the Figure 8 tab and the per-sign popups fail to load
their JSON ("Interactive Figure 8 data could not be loaded (Failed to fetch)"),
because browsers block `fetch()` of local files on the `file:` protocol. Use the
local server below instead.

Start a local server:

```bash
cd AutoTS
python3 -m http.server 8765
```

Then open:

```text
http://127.0.0.1:8765/kitti_ts_map.html
http://127.0.0.1:8765/kitti_ts_map.html#figure8
http://127.0.0.1:8765/kitti_ts_map.html?view=figure8
```

Features:

- Map markers for KITTI-TS signs.
- Orientation filters for leftward, backward, and rightward signs.
- **Per-sign result popups (added 2026-06-09).** Clicking any sign marker now
  shows a mini-Figure-8 result for that sign: an image-sequence thumbnail strip,
  GT orientation vs predicted orientation with a match/mismatch mark, the AutoTS
  estimated-location absolute error, the AutoTS w/o-MinCut error, the number of
  location points, and both GT and estimated GPS. This data is loaded from
  `results/map/map_sign_results.json` (generated by `build_map_sign_results.py`).
  If that file is missing, the popup falls back to GT-only metadata.
- An interactive `Figure 8` tab loaded from
  `results/figure8/figure8_web_data.json`.
- Per-row frame cells with frame id, coordinate source, detected/GT box
  overlays, GT/predicted orientation, and current reproduction errors.
- Side-by-side `Paper Reference` and `Current Reproduction` panels. For example,
  row 3 (`test/2`) shows the paper's `0.85 m` AutoTS error, `1.13 m` w/o Weight
  error, and `Rightward -> Rightward` orientation result, while also showing the
  current reproduction values from this checkpoint.
- One satellite GT map and one road-network estimated-location map per Figure 8
  row. Clicking a frame cell updates the selected image and highlighted map
  point for that frame.
- A per-point CSV,
  `results/figure8/figure8_location_points.csv`, where each map dot has its own
  coordinate row, parent sign ID, current reproduction values, and paper
  reference values.
- The static PNG remains linked as a fallback.

The `#figure8` URL opens directly to the qualitative Figure 8 tab. The
`?view=figure8` URL is an equivalent fallback for browsers/tools that do not
preserve hash fragments during capture.

## 7. Known Missing Work And Deviations

- **Aalborg dataset is not reproduced.** This project is KITTI-TS only.
- **Table IV orientation is still below the paper for the main rows**, even after
  the association fix and retrain. Current `AutoTS (ours)` is `77.78` accuracy and
  `61.19` mRecall (up from `74.60` / `61.13`), not the paper's `85.71` and
  `70.77`. `AutoTS w/ ROIS` now matches the paper *ours* accuracy at `85.71`.
- **The GT-box orientation variant is only partially faithful.** It uses GT boxes for SIFT and
  localization points, but ROI features still come from matched detector ROI
  features because this repo exposes ROI vectors only for predicted boxes.
- **Figure 7 uses padding.** The chosen cached sign IDs did not have full
  12-frame SIFT sequences.
- **Figure 8 localization now closely matches the paper** after the fix (all four
  rows within ~0 to 0.9 m; Row 1 dropped from 9.92 m to 3.09 m and the
  orientation flipped from wrong to correct). Remaining deviation: Row 3 still
  predicts Leftward instead of the paper's Rightward (a Table IV gap, not
  localization), and the "MinCut helps" gap does not show because the recovered
  point clouds are clean (no outliers for MinCut to remove).
- **Detector/version differences matter.** The detector result is higher than
  the paper, likely due to environment, checkpoint, Detectron2, training recipe,
  or hardware differences.
- **Hardware nondeterminism matters.** Orientation training can vary across MPS,
  CUDA, and CPU, even with seed `42`.
- **The paper has at least one printed metric inconsistency.** The Table IV
  `AutoTS w/o SIFT` mRecall does not equal the mean of the printed recalls.

## 8. Quick File Checklist

Run this from the project root:

```bash
test -f AutoTS/results_table1_detection.csv
test -f AutoTS/results/v3_table2/results_table2.csv
test -f AutoTS/results/v4_table3/results_table3.csv
test -f AutoTS/results/table4_orientation/table4_orientation_best.csv
test -f AutoTS/results/table4_orientation/paper_table4_reference.csv
test -f AutoTS/results/figure7/figure7_sift_heatmaps.png
test -f AutoTS/results/figure8/figure8_qualitative.png
test -f AutoTS/results/map/map_sign_results.json
test -f AutoTS/build_map_sign_results.py
test -f AutoTS/kitti_ts_map.html
test -f AutoTS/kitti_ts_viewer.html
test -f AutoTS/orientation_train.py
```

If all commands exit silently, the expected reproduction artifacts are present.

## 9. Recommended Next Steps

1. Re-run Table IV on CUDA if available and compare against the current MPS run.
2. Investigate why the full orientation feature set underperforms simpler
   variants in the current saved run.
3. Improve the GT-box orientation variant by extracting ROI vectors from GT boxes directly instead of
   matching detector ROI outputs.
4. Revisit Figure 7 sample selection if exact 12-frame cached SIFT sequences are
   required.
5. Add Aalborg only if the project scope expands beyond KITTI-TS.

## 10. Reproduction Status Summary

| Paper item | Current status |
|---|---|
| Table I detector | Strongly reproduced; our metrics are higher. |
| Table II localization | Largely reproduced for KITTI-TS. |
| Table III NSAL ablation | Partially reproduced; DBSCAN challenges the paper claim. |
| Detected-box association fix | Applied; single-point signs down to 3/290; cache rebuilt. |
| Table IV orientation | Improved after retrain; `ours` 77.78/61.19, still below paper; `w/ ROIS` matches paper acc. |
| Figure 7 | Generated, with documented SIFT-row padding. |
| Figure 8 | Regenerated after fix; localization within ~0–0.9 m of paper; Row 1 now correct. |
| KITTI web map | MapLibre map, sign filters, interactive Figure 8 tab, and per-sign result popups. |
| Aalborg | Not reproduced. |
