#!/usr/bin/env python3
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from orientation_repro_lib import (
    CachedOrientationDataset,
    OrientationAblationNet,
    PAPER_TABLE_IV,
    build_or_load_cache,
    class_weights,
    compute_metrics,
    ensure_dir,
    load_cfg,
    orientation_collate,
    safe_torch_load,
    set_seed,
    write_csv,
)


VARIANTS = [
    ("AutoTS w/ ROI", "transformer"),
    ("AutoTS w/ ROIS", "transformer"),
    ("AutoTS w/o SIFT", "transformer"),
    ("AutoTS w/o MImg", "transformer"),
    ("AutoTS w/ BiLSTMs", "bilstm"),
    ("AutoTS w/ LSTMs", "lstm"),
    ("AutoTS (ours)", "transformer"),
    ("AutoTS\u2020", "transformer"),
]


def evaluate(model, loader, device):
    model.eval()
    labels_all, preds_all, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = torch.tensor(batch["direction"], dtype=torch.long, device=device)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            labels_all.extend(labels.cpu().numpy().tolist())
            preds_all.extend(preds.cpu().numpy().tolist())
            for sid, split, label, pred in zip(batch["sign_id"], batch["split"],
                                               labels.cpu().numpy().tolist(),
                                               preds.cpu().numpy().tolist()):
                rows.append({
                    "split": split,
                    "sign_id": sid,
                    "label": int(label),
                    "pred": int(pred),
                })
    return compute_metrics(labels_all, preds_all), rows


def run_variant(cfg, variant, encoder, train_samples, test_samples, args):
    train_set = CachedOrientationDataset(train_samples, variant, cfg.MODEL.DEVICE)
    test_set = CachedOrientationDataset(test_samples, variant, cfg.MODEL.DEVICE)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=True,
        collate_fn=orientation_collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=False,
        collate_fn=orientation_collate,
    )
    model = OrientationAblationNet(cfg, encoder=encoder).to(cfg.MODEL.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    weights = class_weights(cfg.SOLVER.CLASS_NUM, cfg.MODEL.DEVICE)

    best_metric = -1.0
    best_state = None
    best_result = None
    final_result = None
    history = []
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"{variant} epoch {epoch+1}/{args.epochs}",
                          leave=False):
            labels = torch.tensor(batch["direction"], dtype=torch.long,
                                  device=cfg.MODEL.DEVICE)
            logits = model(batch)
            loss = F.cross_entropy(logits, labels, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics, pred_rows = evaluate(model, test_loader, cfg.MODEL.DEVICE)
        avg_loss = total_loss / max(len(train_loader), 1)
        history.append({"epoch": epoch + 1, "loss": avg_loss, **metrics})
        final_result = (metrics, pred_rows)
        score = metrics["mrecall"]
        if score > best_metric:
            best_metric = score
            best_result = (metrics, pred_rows)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f"{variant}: epoch {epoch+1:03d} loss={avg_loss:.4f} "
            f"acc={metrics['accuracy']:.2f} mRecall={metrics['mrecall']:.2f}"
        )

    safe_name = variant.replace("/", "").replace(" ", "_").replace("(", "").replace(")", "")
    ckpt_path = os.path.join(args.out_dir, f"{safe_name}_best.pth")
    torch.save(best_state, ckpt_path)

    return {
        "variant": variant,
        "encoder": encoder,
        "checkpoint": ckpt_path,
        "seconds": time.time() - start,
        "best": best_result[0],
        "final": final_result[0],
        "best_predictions": best_result[1],
        "final_predictions": final_result[1],
        "history": history,
    }


def write_outputs(results, args):
    best_rows, final_rows = [], []
    for r in results:
        b = r["best"]
        f = r["final"]
        best_rows.append([
            r["variant"], f"{b['accuracy']:.2f}", f"{b['recall_left']:.2f}",
            f"{b['recall_back']:.2f}", f"{b['recall_right']:.2f}",
            f"{b['mrecall']:.2f}", r["checkpoint"], f"{r['seconds']:.1f}",
        ])
        final_rows.append([
            r["variant"], f"{f['accuracy']:.2f}", f"{f['recall_left']:.2f}",
            f"{f['recall_back']:.2f}", f"{f['recall_right']:.2f}",
            f"{f['mrecall']:.2f}", f"{r['seconds']:.1f}",
        ])
    header = [
        "Method", "Accuracy", "Recall_Left", "Recall_Back", "Recall_Right",
        "mRecall", "Checkpoint", "Seconds",
    ]
    write_csv(os.path.join(args.out_dir, "table4_orientation_best.csv"), best_rows, header)
    write_csv(
        os.path.join(args.out_dir, "table4_orientation_final.csv"),
        final_rows,
        ["Method", "Accuracy", "Recall_Left", "Recall_Back", "Recall_Right",
         "mRecall", "Seconds"],
    )
    paper_rows = [
        [method, f"{acc:.2f}", f"{left:.2f}", f"{back:.2f}",
         f"{right:.2f}", f"{mrecall:.2f}"]
        for method, acc, left, back, right, mrecall in PAPER_TABLE_IV
    ]
    write_csv(
        os.path.join(args.out_dir, "paper_table4_reference.csv"),
        paper_rows,
        ["Method", "Accuracy", "Recall_Left", "Recall_Back", "Recall_Right", "mRecall"],
    )

    pred_path = os.path.join(args.out_dir, "table4_predictions_best.json")
    with open(pred_path, "w") as f:
        json.dump({r["variant"]: r["best_predictions"] for r in results}, f, indent=2)

    history_path = os.path.join(args.out_dir, "table4_history.json")
    with open(history_path, "w") as f:
        json.dump({r["variant"]: r["history"] for r in results}, f, indent=2)

    note = (
        "# Table IV Reproduction Notes\n\n"
        "- Best rows use the checkpoint with highest validation mRecall.\n"
        "- Final rows use the last epoch.\n"
        "- The paper reference is copied from the printed Table IV.\n"
        "- Paper row `AutoTS w/o SIFT` has an mRecall inconsistency: "
        "the printed per-class recalls do not average to the printed 62.38.\n"
        "- `AutoTS\u2020` uses GT boxes for SIFT and localization points. "
        "ROI features use matched detector ROI features because this repo's "
        "DefaultPredictor exposes ROI vectors only for predicted boxes.\n"
    )
    with open(os.path.join(args.out_dir, "table4_notes.md"), "w") as f:
        f.write(note)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table IV orientation ablations.")
    parser.add_argument("--cfg", default="./orientation_config_v2_paper.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./results/orientation_cache")
    parser.add_argument("--out-dir", default="./results/table4_orientation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit signs per split for smoke tests.")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--only-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    if args.epochs is None:
        args.epochs = int(cfg.SOLVER.MAX_EPOCHS)
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    train_payload = build_or_load_cache(
        cfg, args.cache_dir, "train", cfg.DATA.TRAIN_JSON, cfg.DATA.TRAIN_IMG_ROOT,
        limit=args.limit, force=args.force_cache,
    )
    test_payload = build_or_load_cache(
        cfg, args.cache_dir, "test", cfg.DATA.TEST_JSON, cfg.DATA.TEST_IMG_ROOT,
        limit=args.limit, force=args.force_cache,
    )

    if args.only_cache:
        print("Cache built; exiting due to --only-cache.")
        return

    results = []
    for variant, encoder in VARIANTS:
        set_seed(args.seed)
        results.append(run_variant(
            cfg, variant, encoder,
            train_payload["samples"], test_payload["samples"], args,
        ))

    write_outputs(results, args)
    print(f"Saved Table IV outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
