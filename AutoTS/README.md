

# Traffic Sign Localization and Orientation Classification for Automated Map Updating


## Introduction

This paper develops an automated traffic sign update system, AutoTS, aimed at extracting the geo-location and orientation of traffic signs.
To facilitate the evaluation and comparison, we construct a traffic sign localization and orientation classification benchmark, KITTI-TS, based on the KITTI dataset.


## Requirements
Set up an environment for the code.
```bash
conda create -n projectname python=3.10
conda activate projectname
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Datasets

The images in our dataset are sourced from the KITTI dataset, and the annotations are stored in the sign_id_GT.json file. Both the images and the annotation file can be accessed via [this link](https://drive.google.com/drive/folders/1tBI-o6KKN_ZdPLeszFe0Muzj_agnrraq?usp=sharing).
## Evaluation and Training

For evaluation, please download our model checkpoint from [this link](https://drive.google.com/drive/folders/1tBI-o6KKN_ZdPLeszFe0Muzj_agnrraq?usp=drive_link).

Evaluation
```bash
python orientation_train.py --eval-only
```
Training Detector
```bash 
python train_net.py
```

Training 
```bash 
python orientation_train.py
```

## Table IV, Figure 7, and Figure 8 Reproduction

The reproduction scripts build a reusable orientation-feature cache, then reuse it
for every Table IV ablation and visualization.

Table IV:
```bash
cd AutoTS
python run_table4_orientation.py \
  --cfg ./orientation_config_v2_paper.yaml \
  --epochs 300 \
  --seed 42 \
  --cache-dir ./results/orientation_cache \
  --out-dir ./results/table4_orientation
```

Figure 7:
```bash
python visualize_figure7.py \
  --cache-dir ./results/orientation_cache \
  --backward-id 113 \
  --rightward-id 288 \
  --out ./results/figure7/figure7_sift_heatmaps.png
```

Figure 8:
```bash
python visualize_figure8.py \
  --cache-dir ./results/orientation_cache \
  --orientation-ckpt ./results/table4_orientation/AutoTS_ours_best.pth \
  --out ./results/figure8/figure8_qualitative.png \
  --web-data-out ./results/figure8/figure8_web_data.json \
  --points-csv-out ./results/figure8/figure8_location_points.csv
```
Open `kitti_ts_map.html#figure8` or `kitti_ts_map.html?view=figure8` to inspect
each frame with side-by-side paper reference and current reproduction values.

Smoke test:
```bash
python run_table4_orientation.py \
  --cfg ./orientation_config_v2_paper.yaml \
  --epochs 1 \
  --seed 42 \
  --cache-dir ./results/orientation_cache \
  --out-dir ./results/table4_orientation_smoke \
  --limit 8
```

Outputs include computed best/final Table IV CSVs, a printed paper-reference CSV,
prediction histories, figure metadata, and notes flagging the printed Table IV
`AutoTS w/o SIFT` mRecall inconsistency.

## Paper and Citing 
If you find this project helps your research, please kindly consider citing our papers in your publications. 

(Under Review)

## Acknowledge

This repository is built on Detectron2.
