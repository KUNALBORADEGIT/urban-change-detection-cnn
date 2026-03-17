import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from src.utils.dataset import LevirCDDataset
from src.models.unet import UNet
from src.models.siamese_unet import SiameseUNet
from src.utils.train import train_one_epoch, evaluate, DiceBCELoss, save_checkpoint
from src.utils.test import run_test


def load_config(path: str = "src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg, args):
    """
    Override config values from command-line arguments when provided.
    """
    if args.model is not None:
        cfg["model"]["name"] = args.model

    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    if args.lr is not None:
        cfg["train"]["lr"] = args.lr

    if args.threshold is not None:
        cfg["train"]["threshold"] = args.threshold

    if args.out_dir is not None:
        cfg["train"]["out_dir"] = args.out_dir
    else:
        # auto-create separate output folders per model for clean submission
        model_name = cfg["model"]["name"].lower()
        cfg["train"]["out_dir"] = f"runs/{model_name}"

    return cfg


def build_model(cfg):
    model_name = cfg["model"]["name"].lower()
    base_channels = cfg["model"]["base_channels"]

    if model_name == "unet":
        return UNet(in_channels=6, out_channels=1, base=base_channels)

    if model_name == "siamese_unet":
        return SiameseUNet(in_channels=3, out_channels=1, base=base_channels)

    raise ValueError(
        f"Unsupported model name: {cfg['model']['name']}. "
        f"Use 'unet' or 'siamese_unet'."
    )


def train_main(cfg):
    device = "cuda" if torch.cuda.is_available() and cfg["train"]["use_cuda"] else "cpu"
    print(f"[INFO] device   = {device}")
    print(f"[INFO] model    = {cfg['model']['name']}")
    print(f"[INFO] out_dir  = {cfg['train']['out_dir']}")
    print(f"[INFO] epochs   = {cfg['train']['epochs']}")
    print(f"[INFO] batch    = {cfg['train']['batch_size']}")
    print(f"[INFO] lr       = {cfg['train']['lr']}")
    print(f"[INFO] thr      = {cfg['train']['threshold']}")

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

    model = build_model(cfg).to(device)

    if cfg["train"]["loss"] == "dice_bce":
        criterion = DiceBCELoss(bce_weight=cfg["train"]["bce_weight"])
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(out_dir / "best.pt")

    best_val_iou = -1.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = evaluate(model, val_loader, criterion, device, thr=cfg["train"]["threshold"])

        scheduler.step(val["iou"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={tr_loss:.4f} | val_loss={val['loss']:.4f} | "
            f"val_iou={val['iou']:.4f} val_f1={val['f1']:.4f} "
            f"val_acc={val['accuracy']:.4f} "
            f"val_p={val['precision']:.4f} val_r={val['recall']:.4f}"
        )

        if val["iou"] > best_val_iou:
            best_val_iou = val["iou"]
            save_checkpoint(model, optimizer, epoch, best_val_iou, ckpt_path)
            print(f"[SAVE] best checkpoint -> {ckpt_path} (iou={best_val_iou:.4f})")

    print("[DONE]")


def main():
    parser = argparse.ArgumentParser(
        description="Urban Change Detection using U-Net and Siamese U-Net"
    )

    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--model", choices=["unet", "siamese_unet"], default=None)
    parser.add_argument("--config", default="src/config/config.yaml")

    # optional overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.mode == "train":
        train_main(cfg)
    else:
        run_test(cfg, build_model)


if __name__ == "__main__":
    main()