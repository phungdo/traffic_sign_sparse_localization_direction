"""
Run COCO-style evaluation of the traffic sign detector on KITTI-TS test set.
Reproduces Table I from Han et al. 2025.
"""
import sys, os, json, csv
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def run_detection_eval():
    # Load detector
    cfg = get_cfg()
    cfg.merge_from_file('./configs/TS-RCNN-FPN.yaml')
    cfg.MODEL.WEIGHTS = './detector_model_0049999.pth'
    cfg.MODEL.DEVICE = 'mps'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Low threshold for AP calculation
    
    print("Loading detector model...")
    predictor = DefaultPredictor(cfg)
    print("✅ Detector loaded on MPS")
    
    # Load COCO annotations
    with open('./KITTI-TS/annotations/instances_test_a.json') as f:
        coco_data = json.load(f)
    
    print(f"Test set: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    
    # Run predictions on all test images
    img_dir = './KITTI-TS/val2017'
    predictions = []
    
    for img_info in tqdm(coco_data['images'], desc="Running detector"):
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue
        
        img = np.array(Image.open(img_path).convert('RGB'))
        outputs, _ = predictor(img)
        instances = outputs[0]['instances'].to('cpu')
        
        for i in range(len(instances)):
            box = instances.pred_boxes[i].tensor.numpy()[0]  # x1,y1,x2,y2
            score = instances.scores[i].item()
            cls = instances.pred_classes[i].item()
            
            # Convert to COCO format: [x, y, w, h]
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            predictions.append({
                'image_id': img_info['id'],
                'category_id': cls + 1,  # COCO categories are 1-indexed
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'score': float(score)
            })
    
    print(f"Total predictions: {len(predictions)}")
    
    # Run COCO evaluation
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    coco_gt = COCO('./KITTI-TS/annotations/instances_test_a.json')
    coco_dt = coco_gt.loadRes(predictions)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    stats = coco_eval.stats
    results = {
        'AP': stats[0] * 100,
        'AP50': stats[1] * 100,
        'AP75': stats[2] * 100,
        'APs': stats[3] * 100,
        'APm': stats[5] * 100,  # stats[4] is APl, stats[5] is... 
    }
    
    # COCO stats order: AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl
    # Actually: stats[3]=APs, stats[4]=APm, stats[5]=APl
    results['APs'] = stats[3] * 100
    results['APm'] = stats[4] * 100
    
    print("\n" + "="*60)
    print("TABLE I: Traffic Sign Detector Performance (COCO Metrics %)")
    print("="*60)
    print(f"{'Method':<25} {'AP':>8} {'AP50':>8} {'AP75':>8} {'APs':>8} {'APm':>8}")
    print("-"*60)
    print(f"{'Ours (reproduced)':<25} {results['AP']:>8.2f} {results['AP50']:>8.2f} {results['AP75']:>8.2f} {results['APs']:>8.2f} {results['APm']:>8.2f}")
    print(f"{'Paper (Han et al.)':<25} {'60.08':>8} {'84.27':>8} {'72.78':>8} {'51.91':>8} {'67.61':>8}")
    print("="*60)
    
    # Save to CSV
    csv_path = './results_table1_detection.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'AP', 'AP50', 'AP75', 'APs', 'APm'])
        writer.writerow(['Traffic sign detector (ours)', f"{results['AP']:.2f}", f"{results['AP50']:.2f}", 
                         f"{results['AP75']:.2f}", f"{results['APs']:.2f}", f"{results['APm']:.2f}"])
        writer.writerow(['Traffic sign detector (paper)', '60.08', '84.27', '72.78', '51.91', '67.61'])
    
    print(f"\n✅ Results saved to {csv_path}")
    
    # Also save full COCO stats
    full_csv = './results_table1_full_coco.csv'
    metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 
                    'AR@1', 'AR@10', 'AR@100', 'ARs', 'ARm', 'ARl']
    with open(full_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value (%)'])
        for name, val in zip(metric_names, stats):
            writer.writerow([name, f"{val*100:.2f}"])
    print(f"✅ Full COCO metrics saved to {full_csv}")

if __name__ == '__main__':
    run_detection_eval()
