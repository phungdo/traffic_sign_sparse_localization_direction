import torch
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import argparse
import numpy as np
import time
import os
from orientation_net import OrientationNet, TS_localization
from orientation_dataset import TSDataset
from yacs.config import CfgNode as CN

def train(cfg, model, train_loader, val_loader):
    device = cfg.MODEL.DEVICE
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    train_start = time.time()
    epoch_times = []

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):

            labels = torch.tensor(batch['direction']).to(device)
            logits = model(batch)

            class_num = np.array(cfg.SOLVER.CLASS_NUM)
            weight = (1.0 / class_num) / np.sum(1.0 / class_num) * 3
            weight = torch.tensor(weight, dtype=torch.float).to(device)
            loss = model.compute_loss(logits, labels, weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"[Train] Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")
        errors = []
        if epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            all_geolocations = []
            all_location_points = []
            all_categories = []

            for batch in train_loader:
                all_geolocations.extend(batch['geolocation'])
                all_location_points.extend(batch['location_point'])
                all_categories.extend(batch['category'])

            geo_data = [
                {
                    'geolocation': geo,
                    'location_point': loc,
                    'category': cat
                }
                for geo, loc, cat in zip(all_geolocations, all_location_points, all_categories)
            ]

            errors, mae, rmse, recall_1m, recall_2m = TS_localization(geo_data)
            print(f"MAE of train: {mae:.2f} m")
            print(f"RMSE        : {rmse:.2f} m")
            print(f"Recall@1m   : {recall_1m * 100:.2f} %")
            print(f"Recall@2m   : {recall_2m * 100:.2f} %")

        evaluate(cfg, model, val_loader, train_loader, epoch, errors)

    # Log training timing summary
    total_train_time = time.time() - train_start
    avg_epoch_time = np.mean(epoch_times)
    print(f"\n{'='*60}")
    print(f"TRAINING TIME SUMMARY")
    print(f"{'='*60}")
    print(f"Total training time:  {total_train_time:.2f}s ({total_train_time/60:.1f} min)")
    print(f"Avg epoch time:       {avg_epoch_time:.2f}s")
    print(f"Number of epochs:     {cfg.SOLVER.MAX_EPOCHS}")
    print(f"Train samples:        {len(train_loader.dataset)}")
    print(f"Batch size:           {cfg.SOLVER.BATCH_SIZE}")
    print(f"{'='*60}")

    return {'total_train_time': total_train_time, 'avg_epoch_time': avg_epoch_time,
            'epoch_times': epoch_times}

def evaluate(cfg, model, test_loader, train_loader, epoch=0, errors_train=[], num_classes=3):
    eval_start = time.time()
    model.eval()
    correct = 0
    total = 0
    correct_per = [0] * num_classes
    total_per = [0] * num_classes
    device = cfg.MODEL.DEVICE
    inference_times = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            t0 = time.time()
            labels = torch.tensor(batch['direction']).to(device)
            logits = model(batch)
            inference_times.append(time.time() - t0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for i in range(num_classes):
                total_per[i] += (labels == i).sum().item()
                correct_per[i] += (preds == labels)[labels == i].sum().item()

    acc = correct / total
    eval_time = time.time() - eval_start
    avg_inference_ms = np.mean(inference_times) * 1000 / cfg.SOLVER.BATCH_SIZE  # per sample
    print(f"[Eval] Accuracy: {acc*100:.2f}% | Eval time: {eval_time:.2f}s | Inference: {avg_inference_ms:.1f}ms/sign")
    if acc > 0.85:
        torch.save(model.state_dict(), 'model_{}.pth'.format(epoch))
        print('ckpt saved')

    mean_accuracy = 0
    for i in range(num_classes):
        accuracy = 100 * correct_per[i] / total_per[i]
        print('Accuracy of class {}: {:.2f}%'.format(i, accuracy))
        mean_accuracy += accuracy
    print('Mean Accuracy of three classes: {:.2f}%'.format(mean_accuracy / num_classes))

    if epoch == cfg.SOLVER.MAX_EPOCHS - 1:
        all_geolocations = []
        all_location_points = []
        all_categories = []

        for batch in test_loader:
            all_geolocations.extend(batch['geolocation'])
            all_location_points.extend(batch['location_point'])
            all_categories.extend(batch['category'])

        geo_data = [
            {
                'geolocation': geo,
                'location_point': loc,
                'category': cat
            }
            for geo, loc, cat in zip(all_geolocations, all_location_points, all_categories)
        ]

        errors, mae, rmse, recall_1m, recall_2m = TS_localization(geo_data)
        print(f"MAE of test: {mae:.2f} m")
        print(f"RMSE       : {rmse:.2f} m")
        print(f"Recall@1m  : {recall_1m * 100:.2f} %")
        print(f"Recall@2m  : {recall_2m * 100:.2f} %")

        if len(errors_train) == 0:
            for batch in train_loader:
                all_geolocations.extend(batch['geolocation'])
                all_location_points.extend(batch['location_point'])
                all_categories.extend(batch['category'])

            geo_data = [
                {
                    'geolocation': geo,
                    'location_point': loc,
                    'category': cat
                }
                for geo, loc, cat in zip(all_geolocations, all_location_points, all_categories)
            ]

            errors_train, mae, rmse, recall_1m, recall_2m = TS_localization(geo_data)

        errors_all = np.concatenate([errors, errors_train])
        mae = np.mean(errors_all)
        rmse = np.sqrt(np.mean(errors_all ** 2))
        recall_1m = np.mean(errors_all < 1.0)
        recall_2m = np.mean(errors_all < 2.0)
        print(f"MAE of all: {mae:.2f} m")
        print(f"RMSE      : {rmse:.2f} m")
        print(f"Recall@1m : {recall_1m * 100:.2f} %")
        print(f"Recall@2m : {recall_2m * 100:.2f} %")


def custom_collate_fn(batch):
    return {
        'image_feature': [item['image_feature'] for item in batch],
        'roi_feature': [item['roi_feature'] for item in batch],
        'sift_feature': [item['sift_feature'] for item in batch],
        'direction': [item['direction'] for item in batch],
        'geolocation': [item['geolocation'] for item in batch],
        'location_point': [item['location_point'] for item in batch],
        'category': [item['category'] for item in batch]
    }

def main():
    total_start = time.time()

    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./orientation_config.yaml")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--resume", type=str, default=None, help="Path to model checkpoint for eval")
    args = parser.parse_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.cfg)

    device = cfg.MODEL.DEVICE

    # Dataset loading with timing
    print("\n⏱️  Loading datasets...")
    data_load_start = time.time()

    train_set = TSDataset(cfg.DATA.TRAIN_JSON, cfg.DATA.TRAIN_IMG_ROOT,
                          cfg.MODEL.DEVICE, cfg.DETECTRON2.CONFIG_PATH, cfg.DETECTRON2.WEIGHT_PATH,
                          training=True)
    train_load_time = time.time() - data_load_start
    print(f"   Train set loaded: {train_load_time:.1f}s ({len(train_set)} signs)")

    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                              collate_fn=custom_collate_fn)

    val_load_start = time.time()
    val_set = TSDataset(cfg.DATA.TEST_JSON, cfg.DATA.TEST_IMG_ROOT,
                        cfg.MODEL.DEVICE, cfg.DETECTRON2.CONFIG_PATH, cfg.DETECTRON2.WEIGHT_PATH)
    val_load_time = time.time() - val_load_start
    print(f"   Val set loaded:   {val_load_time:.1f}s ({len(val_set)} signs)")

    total_data_time = time.time() - data_load_start
    print(f"   Total data load:  {total_data_time:.1f}s")

    val_loader = DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=custom_collate_fn)

    # Model
    model = OrientationNet(
        roi_input_size=cfg.MODEL.ROI_INPUT_SIZE,
        sift_input_size=cfg.MODEL.SIFT_INPUT_SIZE,
        roi_hidden=cfg.MODEL.ROI_HIDDEN,
        sift_hidden=cfg.MODEL.SIFT_HIDDEN,
        img_hidden=cfg.MODEL.IMG_HIDDEN,
        num_classes=cfg.MODEL.NUM_CLASSES,
        num_layers=cfg.MODEL.NUM_LAYERS,
        d_model=cfg.MODEL.D_MODEL,
        nhead=cfg.MODEL.NHEAD,
        device=device
    ).to(device)

    if args.eval_only:
        if args.resume:
            print(f"Loading checkpoint: {args.resume}")
            state_dict = torch.load(args.resume, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Checkpoint loaded")
        evaluate(cfg, model, val_loader, train_loader)
    else:
        timing = train(cfg, model, train_loader, val_loader)

        # Save comprehensive timing report
        total_time = time.time() - total_start
        config_name = os.path.basename(args.cfg).replace('.yaml', '')
        report_path = f'./results/timing_report_{config_name}.txt'
        os.makedirs('./results', exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(f"TIMING REPORT — {config_name}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Hardware:          Apple M-series (MPS)\n")
            f.write(f"Config:            {args.cfg}\n")
            f.write(f"Seed:              {seed}\n\n")
            f.write(f"DATA LOADING\n")
            f.write(f"  Train set:       {train_load_time:.1f}s ({len(train_set)} signs)\n")
            f.write(f"  Val set:         {val_load_time:.1f}s ({len(val_set)} signs)\n")
            f.write(f"  Total:           {total_data_time:.1f}s\n\n")
            if timing:
                f.write(f"TRAINING\n")
                f.write(f"  Total time:      {timing['total_train_time']:.1f}s ({timing['total_train_time']/60:.1f} min)\n")
                f.write(f"  Avg epoch:       {timing['avg_epoch_time']:.2f}s\n")
                f.write(f"  Epochs:          {cfg.SOLVER.MAX_EPOCHS}\n")
                f.write(f"  Batch size:      {cfg.SOLVER.BATCH_SIZE}\n")
                f.write(f"  Train samples:   {len(train_set)}\n")
                f.write(f"  Throughput:      {len(train_set)/timing['avg_epoch_time']:.1f} signs/s\n\n")
            f.write(f"TOTAL PIPELINE\n")
            f.write(f"  Total time:      {total_time:.1f}s ({total_time/60:.1f} min)\n")
        print(f"\n💾 Timing report saved to {report_path}")

if __name__ == '__main__':
    main()