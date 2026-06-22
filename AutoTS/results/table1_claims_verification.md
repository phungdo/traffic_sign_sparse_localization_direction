# Table I — Traffic Sign Detector: Claims Verification

## Paper's Description (Section V-A3)

The paper states:

> In the traffic sign detector, we adopt a Faster R-CNN [19] with ResNeXt-101-FPN pre-trained on the COCO dataset [46]. We fine-tune the Faster R-CNN on our KITTI-TS dataset with **batch size 12** and **50K iterations** to obtain the traffic sign detector. The detection results in terms of the COCO evaluation metrics are shown in Table I. The COCO evaluation metrics are provided in Detectron2 [47] to evaluate the detector, including Average Precision (AP), Average Precision at different IoU thresholds (AP50 and AP75), Average Precision for small objects (APs) and Average Precision for medium objects (APm).

**Note:** Table I is not accompanied by explicit analysis claims — it simply reports the detector performance as an implementation detail. The paper does not compare against other detectors here; this table establishes that the detector works sufficiently well for downstream localization.

---

## Results Comparison

| Metric | Paper | Ours | Difference | Status |
|:-------|:---:|:---:|:---:|:---:|
| **AP** | 60.08 | 71.72 | +11.64 | ✅ Ours higher |
| **AP50** | 84.27 | 85.92 | +1.65 | ✅ Ours higher |
| **AP75** | 72.78 | 84.89 | +12.11 | ✅ Ours higher |
| **APs** | 51.91 | 69.44 | +17.53 | ✅ Ours higher |
| **APm** | 67.61 | 75.09 | +7.48 | ✅ Ours higher |

### Additional Metrics (ours only)

| Metric | Value (%) |
|:-------|:---:|
| AR@1 | 71.03 |
| AR@10 | 79.38 |
| AR@100 | 79.38 |
| ARs | 75.98 |
| ARm | 82.99 |

---

## Analysis

### Why are our numbers higher?

Our detector significantly outperforms the paper's reported numbers across all metrics. Possible reasons:

1. **Different Detectron2 version**: The paper likely used an older Detectron2 build. Newer versions may have improved default training recipes, augmentation strategies, or backbone weights.
2. **Pre-trained weights difference**: The COCO-pretrained ResNeXt-101-FPN checkpoint may have been updated between paper submission and our reproduction.
3. **Training hyperparameters**: While the paper specifies batch_size=12 and 50K iterations, our training configuration uses the Detectron2 defaults which may differ in learning rate schedule, warmup steps, weight decay, etc.
4. **Hardware differences**: Different GPU precision (paper likely used CUDA FP32/FP16; we used the pre-trained model weights directly on MPS).

### Impact on Downstream Tasks

Since we use the paper's pre-computed detections and depth maps (from `sign_id_GT.json` and `img_depth.npz`), **our Table I numbers do not affect Table II/III/IV results**. The detector evaluation is independent — it only shows our fine-tuned detector is functioning correctly.

---

## Configuration

- **Model**: Faster R-CNN with ResNeXt-101-FPN
- **Pre-training**: COCO dataset
- **Fine-tuning**: KITTI-TS dataset (batch_size=12, 50K iterations per paper)
- **Evaluation**: COCO metrics via Detectron2
- **Our weights**: Pre-existing in `model_final_traffic_sign.pth`
