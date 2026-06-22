# Reproduce Figure 7, Table IV, and Figure 8

## Summary
Continue with a paper-faithful reproduction using scripts plus generated outputs. Add a cached orientation-feature pipeline, run all Table IV ablations, generate Figure 7 heatmaps from position-aware SIFT features, and generate Figure 8 qualitative localization/orientation panels.

## Key Changes
- Add a reusable orientation experiment runner with these variants:
  - `AutoTS w/ ROI`: single central-frame ROI only.
  - `AutoTS w/ ROIS`: ROI sequence only.
  - `AutoTS w/o SIFT`: ROI sequence + mean image, no SIFT.
  - `AutoTS w/o MImg`: ROI sequence + SIFT, no mean image.
  - `AutoTS w/ BiLSTMs`, `AutoTS w/ LSTMs`: full features, alternate sequence encoder.
  - `AutoTS (ours)`: full detected-box pipeline.
  - `AutoTS†`: full pipeline using GT boxes for ROI/SIFT/location extraction.
- Add feature caching before training:
  - Cache image features, ROI features, SIFT features, detected boxes, GT boxes, location points, labels, categories, and image ids once.
  - Reuse cache for every Table IV variant to avoid repeated 10+ minute dataset construction.
- Generate outputs under `AutoTS/results/table4_orientation/`, `AutoTS/results/figure7/`, and `AutoTS/results/figure8/`.

## Interfaces
- New Table IV command:
  ```bash
  cd AutoTS
  python run_table4_orientation.py \
    --cfg ./orientation_config_v2_paper.yaml \
    --epochs 300 \
    --seed 42 \
    --cache-dir ./results/orientation_cache \
    --out-dir ./results/table4_orientation
  ```
- New Figure 7 command:
  ```bash
  python visualize_figure7.py \
    --cache-dir ./results/orientation_cache \
    --backward-id 113 \
    --rightward-id 288 \
    --out ./results/figure7/figure7_sift_heatmaps.png
  ```
- New Figure 8 command:
  ```bash
  python visualize_figure8.py \
    --cache-dir ./results/orientation_cache \
    --orientation-ckpt ./results/table4_orientation/AutoTS_ours_best.pth \
    --out ./results/figure8/figure8_qualitative.png
  ```

## Reproduction Targets
- Table IV should report Accuracy, Left Recall, Back Recall, Right Recall, and computed mRecall for every variant.
- Include a paper-reference CSV with the printed Table IV values:
  - `AutoTS w/ ROI`: 71.43, 12.50, 91.30, 22.22, 42.01
  - `AutoTS w/ ROIS`: 73.02, 37.50, 84.78, 44.44, 55.57
  - `AutoTS w/o SIFT`: 76.19, 37.50, 89.13, 44.44, 62.38
  - `AutoTS w/o MImg`: 80.95, 50.00, 89.13, 66.67, 68.60
  - `AutoTS w/ BiLSTMs`: 82.54, 62.50, 91.30, 55.56, 69.79
  - `AutoTS w/ LSTMs`: 82.54, 50.00, 93.48, 55.56, 67.07
  - `AutoTS (ours)`: 85.71, 50.00, 95.65, 66.67, 70.77
  - `AutoTS†`: 88.89, 75.00, 95.65, 66.67, 79.11
- Flag the paper’s `AutoTS w/o SIFT` mRecall inconsistency: the printed per-class recalls do not average to 62.38.

## Figure Details
- Figure 7:
  - Use cached SIFT features from test sign `113` for backward and `288` for rightward.
  - Use the central 12 frames if the sequence has more than 12 frames.
  - Plot two `12 x 160` heatmaps with the paper-style labels and no x-axis tick clutter.
- Figure 8:
  - Reproduce four qualitative rows.
  - Use exact/nearest paper-caption matches where available: row 1 `train/3`, row 2 `train/274`, row 3 `test/2`.
  - For row 4, choose deterministically: closest paper-caption NoStay candidate if labels fit; otherwise first backward-to-rightward AutoTS failure after Table IV evaluation.
  - Render columns for index, image sequence with boxes, GT GPS map, AutoTS vs ablation location maps, and GT/predicted orientation.
  - Use map tiles when available; fallback to projected-coordinate plots with the fallback noted in the figure metadata.

## Test Plan
- Smoke test feature caching with `--limit 8 --epochs 1`.
- Verify cached feature dimensions match model expectations: ROI `1024`, SIFT `160`, image `2048`.
- Verify Table IV output has all 8 variants and recomputed mRecall equals the mean of per-class recalls.
- Verify Figure 7 output exists, has two heatmaps, and uses 12-frame sequences.
- Verify Figure 8 output exists, has four rows, and each row’s localization error matches the computed CSV values.
- Compare final AutoTS row against the already observed checkpoint target: expected around `85.71%` accuracy and `70.77%` mRecall.

## Assumptions
- Use KITTI-TS only; Aalborg remains out of scope.
- Use seed `42`, batch size `8`, LR `0.0008`, and `300` epochs from `orientation_config_v2_paper.yaml`.
- Report both “best checkpoint by mRecall” and “final epoch,” but use best checkpoint for the reproduction table.
- Keep all paper deviations explicit rather than silently forcing numbers to match.
