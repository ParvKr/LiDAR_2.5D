import torch
import torch.nn.functional as F


def masked_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Computes the Focal Loss to handle extreme class imbalance in LiDAR data.
    """
    if logits.ndim != 4:
        raise ValueError("logits must have shape (B, C, H, W)")
    if target.ndim != 3:
        raise ValueError("target must have shape (B, H, W)")
    if mask.ndim != 3:
        raise ValueError("mask must have shape (B, H, W)")
    if target.shape != mask.shape:
        raise ValueError("target and mask must have the same shape")
    if logits.shape[0] != target.shape[0]:
        raise ValueError("Batch sizes must match")
    if logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("Spatial dimensions must match")
    if not mask.any():
        # If the batch is completely empty, return a zero loss that still tracks gradients
        # to prevent the training script from crashing.
        return (logits * 0).sum()

    # Extract valid cells
    masked_logits = logits.permute(0, 2, 3, 1)[mask]
    masked_target = target[mask]

    # Compute standard Cross Entropy
    ce_loss = F.cross_entropy(masked_logits, masked_target, reduction="none")
    
    # Compute Focal Loss ( modulating factor: (1 - p_t)^gamma )
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    return focal_loss.mean()