import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.utils.dataset import LevirCDDataset
from src.models.unet import UNet
from src.utils.train import train_one_epoch, evaluate, DiceBCELoss, save_checkpoint


def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() and cfg["train"]["use_cuda"] else "cpu"
    print(f"[INFO] device = {device}")

    root = cfg["data"]["root_dir"]
    img_size = tuple(cfg["data"]["img_size"])

    train_ds = LevirCDDataset(root, "train", img_size=img_size, augment=True)
    val_ds = LevirCDDataset(root, "val", img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device == "cuda"),
    )

    model = UNet(in_channels=6, out_channels=1, base=cfg["model"]["base_channels"]).to(device)

    loss_name = cfg["train"]["loss"]
    if loss_name == "dice_bce":
        criterion = DiceBCELoss(bce_weight=cfg["train"]["bce_weight"])
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(out_dir / "best.pt")

    best_val_iou = -1.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = evaluate(model, val_loader, criterion, device, thr=cfg["train"]["threshold"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} | val_loss={val['loss']:.4f} | "
            f"val_iou={val['iou']:.4f} val_f1={val['f1']:.4f} "
            f"val_p={val['precision']:.4f} val_r={val['recall']:.4f}"
        )

        if val["iou"] > best_val_iou:
            best_val_iou = val["iou"]
            save_checkpoint(model, optimizer, epoch, best_val_iou, ckpt_path)
            print(f"[SAVE] best checkpoint -> {ckpt_path} (iou={best_val_iou:.4f})")

    print("[DONE]")


if __name__ == "__main__":
    main()