import torch


@torch.no_grad()
def binarize_from_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs >= thr).float()


@torch.no_grad()
def iou_f1_precision_recall(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    """
    pred, target: [B,1,H,W] float {0,1}
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "iou": iou.mean().item(),
    }