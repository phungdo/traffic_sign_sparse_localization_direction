#!/bin/bash
# =============================================================================
# PlaneDepth Checkpoint Download Instructions
# =============================================================================
#
# The PlaneDepth pretrained weights are hosted on SharePoint and must be
# downloaded manually through a browser.
#
# STEP 1: Open this URL in your browser:
#   https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/wangry3_shanghaitech_edu_cn/EmYCmInpVd5CjJwu8-DCyY4BnJTKQ7IKnRx5GJYqQEVeMg?e=OCRdEl
#
# STEP 2: Download both files:
#   - encoder.pth
#   - depth.pth
#
# STEP 3: Move them to the checkpoint directory:
CHECKPOINT_DIR="$(cd "$(dirname "$0")" && pwd)/PlaneDepth/log/ResNet/exp1_sd/best_models"
echo "Move downloaded files to:"
echo "  $CHECKPOINT_DIR/encoder.pth"
echo "  $CHECKPOINT_DIR/depth.pth"

# STEP 4: Verify
if [ -f "$CHECKPOINT_DIR/encoder.pth" ] && [ -f "$CHECKPOINT_DIR/depth.pth" ]; then
    echo "✅ Both checkpoint files found!"
    ls -lh "$CHECKPOINT_DIR"/*.pth
else
    echo "❌ Missing checkpoint files. Please download from the SharePoint link above."
fi

echo ""
echo "After downloading, run the depth generation script:"
echo "  python generate_depth_maps.py \\"
echo "    --weights_folder $CHECKPOINT_DIR \\"
echo "    --image_dir ./AutoTS/KITTI-TS/train_img \\"
echo "    --val_dir ./AutoTS/KITTI-TS/val2017 \\"
echo "    --output ./AutoTS/data/img_depth.npz \\"
echo "    --device mps"
