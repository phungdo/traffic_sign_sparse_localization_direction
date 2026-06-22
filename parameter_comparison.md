# Paper vs Our Settings — Parameter Comparison

## 🔴 Key Discrepancies Found

| Parameter | **Paper (Section V-A.3)** | **Our Config** | **Match?** |
|-----------|:-------------------------:|:--------------:|:----------:|
| **Learning Rate** | **0.0008** | **0.0005** | ❌ |
| **Training Duration** | **300 steps** | **200 epochs** | ⚠️ Ambiguous |
| **Random Seed** | Not mentioned | **Not set** | ⚠️ |
| **Detector Backbone** | ResNeXt-101-FPN | ResNet-101-FPN | ⚠️ |
| **Detector batch size** | 12 | N/A (pretrained) | — |
| **Detector iterations** | 50K | 50K (ckpt name) | ✅ |

---

## ✅ Matching Parameters

| Parameter | **Paper** | **Our Config** | **Match?** |
|-----------|:---------:|:--------------:|:----------:|
| Optimizer | Adam | Adam | ✅ |
| Batch size (orientation) | 8 | 8 | ✅ |
| Loss function | Weighted Cross-Entropy | Weighted Cross-Entropy | ✅ |
| Class weights | Inverse frequency | Inverse frequency | ✅ |
| Train/Test split | 80%/20% | 227/63 (78%/22%) | ✅ |
| NSAL σ | 2.5 | 2.5 (in code) | ✅ |
| NSAL α | 0.1 | 0.1 (in code) | ✅ |
| Depth model | PlaneDepth | PlaneDepth | ✅ |

---

## ✅ Model Architecture (all match)

| Parameter | **Paper/Config** | **Our Config** |
|-----------|:----------------:|:--------------:|
| ROI_INPUT_SIZE | 1024 | 1024 |
| SIFT_INPUT_SIZE | 160 | 160 |
| ROI_HIDDEN | 96 | 96 |
| SIFT_HIDDEN | 96 | 96 |
| IMG_HIDDEN | 128 | 128 |
| NUM_CLASSES | 3 | 3 |
| NUM_LAYERS (Transformer) | 2 | 2 |
| D_MODEL | 192 | 192 |
| NHEAD | 4 | 4 |

---

## ⚠️ Detailed Notes on Discrepancies

### 1. Learning Rate: 0.0008 (paper) vs 0.0005 (code)
The paper says *"initial learning rate to 8 and 0.0008"* — the code config has `LR: 0.0005`. The author's released code uses 0.0005, which may be the final tuned value.

### 2. Training Duration: "300 steps" (paper) vs "200 epochs" (code)
The paper says *"each training lasts for 300 steps"*. But the code has `MAX_EPOCHS: 200`. With batch_size=8 and 227 train signs → ~28 batches/epoch. So:
- 200 epochs × 28 batches = **5,600 steps**
- Paper says 300 steps which seems too few

This is **likely a typo in the paper** — 300 epochs (not steps) would be more reasonable. The checkpoint name `AutoTS_model_123.pth` suggests epoch 123.

### 3. No Random Seed
Neither the paper nor the code sets a random seed. Results may vary between runs.

### 4. Detector Backbone: ResNeXt-101 vs ResNet-101
The paper says *"Faster R-CNN with ResNeXt-101-FPN"* but the code config `TS-RCNN-FPN.yaml` uses `build_resnet_fpn_backbone` (ResNet-101, not ResNeXt). The pretrained weights match the code config (loads successfully).

---

## Detection Evaluation (Table I)

Our detector eval used the **exact same checkpoint** (`detector_model_0049999.pth`) with:
- `SCORE_THRESH_TEST: 0.05` (low threshold for AP calculation)
- COCO evaluation via `pycocotools`
- 552 test images, 619 annotations

> [!IMPORTANT]
> Our AP numbers (71.72%) are higher than the paper's (60.08%). This may be because:
> 1. Different test split definition
> 2. The paper may have used a stricter evaluation (e.g., different maxDets or area thresholds)
> 3. The checkpoint may be better than what was used for the paper's Table I
