from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader

from src.utils.dataset import LevirCDDataset
from src.models.unet import UNet
from src.utils.train import evaluate
from src.utils.train import DiceBCELoss


def load_best_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def run_test(cfg):
    device = "cuda" if torch.cuda.is_available() and cfg["train"]["use_cuda"] else "cpu"

    test_ds = LevirCDDataset(
        cfg["data"]["root_dir"],
        "test",
        img_size=tuple(cfg["data"]["img_size"]),
        augment=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    model = UNet(
        in_channels=6,
        out_channels=1,
        base=cfg["model"]["base_channels"],
    ).to(device)

    ckpt_path = Path(cfg["train"]["out_dir"]) / "best.pt"
    load_best_checkpoint(model, str(ckpt_path), device)

    if cfg["train"]["loss"] == "dice_bce":
        criterion = DiceBCELoss(bce_weight=cfg["train"]["bce_weight"])
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        thr=cfg["train"]["threshold"],
    )

    print("\n[TEST RESULTS]")
    print(f"loss      : {metrics['loss']:.4f}")
    print(f"iou       : {metrics['iou']:.4f}")
    print(f"f1        : {metrics['f1']:.4f}")
    print(f"precision : {metrics['precision']:.4f}")
    print(f"recall    : {metrics['recall']:.4f}")

    out_csv = Path(cfg["train"]["out_dir"]) / "test_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss", "iou", "f1", "precision", "recall"])
        writer.writerow([
            metrics["loss"],
            metrics["iou"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        ])

    print(f"\n[SAVED] {out_csv}")