import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import binarize_from_logits, iou_f1_precision_recall


def save_checkpoint(model, optimizer, epoch: int, best_val_iou: float, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
        },
        path,
    )


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss = 0.0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device: str, thr: float = 0.5) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    agg = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    n = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        pred = binarize_from_logits(logits, thr=thr)
        m = iou_f1_precision_recall(pred, y)
        bs = x.size(0)
        for k in agg:
            agg[k] += m[k] * bs
        n += bs

    out = {k: agg[k] / max(n, 1) for k in agg}
    out["loss"] = total_loss / len(loader.dataset)
    return out


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        dice = (2 * inter + self.eps) / (probs.sum(dim=1) + targets.sum(dim=1) + self.eps)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss