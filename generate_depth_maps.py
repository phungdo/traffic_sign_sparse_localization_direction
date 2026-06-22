#!/usr/bin/env python3
"""
Generate img_depth.npz for AutoTS using PlaneDepth inference.

This script loads the PlaneDepth pretrained model (ResNet encoder + depth decoder)
and runs inference on all KITTI-TS images to produce per-image depth maps.
The output is saved as ./AutoTS/data/img_depth.npz with keys = image_id,
values = depth maps of shape (H, W) in meters.

Usage:
    python generate_depth_maps.py \
        --weights_folder ./PlaneDepth/log/ResNet/exp1_sd/best_models \
        --image_dir ./AutoTS/KITTI-TS/train_img \
        --val_dir ./AutoTS/KITTI-TS/val2017 \
        --output ./AutoTS/data/img_depth.npz \
        --device cuda

If --device is 'mps', Apple Silicon GPU will be used.
If --device is 'cpu', CPU will be used (slow but works everywhere).
"""

import os
import sys
import argparse
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Add PlaneDepth to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANEDEPTH_DIR = os.path.join(SCRIPT_DIR, "PlaneDepth")
sys.path.insert(0, PLANEDEPTH_DIR)

import networks


# PlaneDepth stereo scale factor (KITTI baseline = 0.54m, scale = 5.4)
STEREO_SCALE_FACTOR = 5.4
# Inference resolution matching eval.sh
INFER_WIDTH = 1280
INFER_HEIGHT = 384


def load_planedepth_model(weights_folder, device):
    """Load PlaneDepth ResNet encoder + depth decoder."""
    encoder_path = os.path.join(weights_folder, "encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth.pth")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"encoder.pth not found at {encoder_path}\n"
            f"Please download the PlaneDepth self-distillation checkpoint from:\n"
            f"https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/"
            f"wangry3_shanghaitech_edu_cn/EmYCmInpVd5CjJwu8-DCyY4BnJTKQ7IKnRx5GJYqQEVeMg\n"
            f"and place encoder.pth and depth.pth in: {weights_folder}"
        )

    print(f"Loading encoder from {encoder_path}")
    encoder_dict = torch.load(encoder_path, map_location=device)
    encoder = networks.ResnetEncoder(50, False)  # ResNet-50, no pretrained
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    encoder.to(device)
    encoder.eval()

    print(f"Loading depth decoder from {decoder_path}")
    depth_decoder = networks.DepthDecoder(
        encoder.num_ch_enc,
        49,          # disp_levels
        2.,          # disp_min
        300.,        # disp_max
        8,           # num_ep
        pe_type="neural",
        use_denseaspp=True,
        xz_levels=14,
        yz_levels=0,
        use_mixture_loss=True,
        render_probability=False,
        plane_residual=True,
    )
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder


def preprocess_image(image_path, width=INFER_WIDTH, height=INFER_HEIGHT):
    """Load and preprocess a single image for PlaneDepth inference."""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)
    img_resized = img.resize((width, height), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img_resized)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img_tensor, original_size


def disparity_to_depth(disp, width=INFER_WIDTH):
    """Convert PlaneDepth disparity to metric depth.
    
    From evaluate_depth_HR.py line 236:
        pred_depth = 0.1 * 0.58 * opt.width / pred_disp
    
    This uses: baseline=0.1 * focal_length_ratio=0.58 * image_width / disparity
    With STEREO_SCALE_FACTOR=5.4 applied.
    """
    # Avoid division by zero
    disp = np.clip(disp, 1e-6, None)
    depth = STEREO_SCALE_FACTOR * 0.1 * 0.58 * width / disp
    # Clip to reasonable range
    depth = np.clip(depth, 0.1, 100.0)
    return depth


def collect_image_ids(gt_json_paths):
    """Collect all unique image IDs from sign_id_GT.json files."""
    all_ids = set()
    for path in gt_json_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            for sign_data in data.values():
                for img_id in sign_data['images'].keys():
                    all_ids.add(img_id)
    return sorted(all_ids)


def find_all_images(image_dirs):
    """Find all .png images across multiple directories."""
    image_paths = {}
    for img_dir in image_dirs:
        if not os.path.exists(img_dir):
            print(f"Warning: Directory not found: {img_dir}")
            continue
        for path in glob.glob(os.path.join(img_dir, "*.png")):
            img_id = os.path.splitext(os.path.basename(path))[0]
            image_paths[img_id] = path
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Generate img_depth.npz via PlaneDepth")
    parser.add_argument("--weights_folder", type=str, required=True,
                        help="Path to PlaneDepth checkpoint folder (containing encoder.pth + depth.pth)")
    parser.add_argument("--image_dir", type=str, default="./AutoTS/KITTI-TS/train_img",
                        help="Directory containing training images")
    parser.add_argument("--val_dir", type=str, default="./AutoTS/KITTI-TS/val2017",
                        help="Directory containing validation/test images")
    parser.add_argument("--output", type=str, default="./AutoTS/data/img_depth.npz",
                        help="Output path for img_depth.npz")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Find all images
    image_dirs = [args.image_dir, args.val_dir]
    image_paths = find_all_images(image_dirs)
    print(f"Found {len(image_paths)} images across {len(image_dirs)} directories")

    if len(image_paths) == 0:
        print("ERROR: No images found. Check --image_dir and --val_dir paths.")
        sys.exit(1)

    # Also check which IDs are needed from GT files
    base_dir = os.path.dirname(args.image_dir)
    gt_paths = [
        os.path.join(base_dir, "train", "sign_id_GT.json"),
        os.path.join(base_dir, "test", "sign_id_GT.json"),
    ]
    needed_ids = collect_image_ids(gt_paths)
    print(f"Image IDs needed from sign_id_GT.json: {len(needed_ids)}")
    
    missing = [i for i in needed_ids if i not in image_paths]
    if missing:
        print(f"WARNING: {len(missing)} image IDs from GT not found in image directories:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")

    # Load model
    encoder, depth_decoder = load_planedepth_model(args.weights_folder, device)

    # Create grid for PlaneDepth (required by depth decoder)
    grid = torch.meshgrid(
        torch.linspace(-1, 1, INFER_WIDTH),
        torch.linspace(-1, 1, INFER_HEIGHT),
        indexing="xy"
    )
    grid = torch.stack(grid, dim=0).unsqueeze(0).to(device)  # (1, 2, H, W)

    # Run inference
    depth_dict = {}
    img_ids = sorted(image_paths.keys())

    print(f"\nRunning PlaneDepth inference on {len(img_ids)} images...")
    for img_id in tqdm(img_ids, desc="Depth estimation"):
        img_path = image_paths[img_id]
        img_tensor, original_size = preprocess_image(img_path)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = encoder(img_tensor)
            output = depth_decoder(features, grid)

        # Extract disparity map
        pred_disp = output["disp"][0, 0].cpu().numpy()  # (H_infer, W_infer)

        # Convert to metric depth
        pred_depth = disparity_to_depth(pred_disp, INFER_WIDTH)

        # Resize depth to original image size
        orig_w, orig_h = original_size
        if pred_depth.shape != (orig_h, orig_w):
            import cv2
            pred_depth = cv2.resize(pred_depth, (orig_w, orig_h),
                                    interpolation=cv2.INTER_LINEAR)

        depth_dict[img_id] = pred_depth.astype(np.float32)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nSaving {len(depth_dict)} depth maps to {args.output}")
    np.savez_compressed(args.output, **depth_dict)

    # Verify
    verify = np.load(args.output)
    print(f"Verification: {len(verify.files)} arrays in npz")
    sample_key = verify.files[0]
    sample = verify[sample_key]
    print(f"Sample '{sample_key}': shape={sample.shape}, "
          f"min={sample.min():.2f}m, max={sample.max():.2f}m, "
          f"median={np.median(sample):.2f}m")

    print("\n✅ Done! img_depth.npz generated successfully.")
    print(f"Next step: Place this file at ./AutoTS/data/img_depth.npz")


if __name__ == "__main__":
    main()
