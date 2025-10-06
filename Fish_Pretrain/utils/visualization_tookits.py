import torch
import numpy as np
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from Fish_Pretrain.dataset_loading import compute_bbox_from_mask, resize_mask

def _denormalize_image(img: torch.Tensor, processor) -> torch.Tensor:
    """Attempt to denormalize an image tensor (C,H,W) using processor stats if present."""
    if not torch.is_tensor(img):
        return img
    x = img.clone().detach().float()
    mean = getattr(processor, 'image_mean', None)
    std = getattr(processor, 'image_std', None)
    if mean is not None and std is not None:
        if isinstance(mean, (list, tuple)):
            mean_t = torch.tensor(mean).view(-1, 1, 1)
            std_t = torch.tensor(std).view(-1, 1, 1)
            x = x * std_t + mean_t
    x = torch.clamp(x, 0, 1)
    return x

def _maybe_resize_mask_to_image(mask: Optional[torch.Tensor], image: torch.Tensor) -> Optional[torch.Tensor]:
    """Resize a (H,W) mask to image (C,H,W) spatial dims if they differ, using resize_mask."""
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    H, W = mask.shape[-2], mask.shape[-1]
    _, H_i, W_i = image.shape
    if (H, W) != (H_i, W_i):
        mask = resize_mask(mask, size=(H_i, W_i))
    return mask

def _bbox_is_normalized(bbox: torch.Tensor, height: int, width: int) -> bool:
    # Heuristic: if all coords <= 1.5 treat as normalized
    return torch.all(bbox <= 1.5).item()

def _prepare_bbox(bbox: torch.Tensor, height: int, width: int, bbox_mode="corner") -> Tuple[float, float, float, float]:
    """Return bbox in pixel coordinates (x,y,w,h). Accept either normalized or pixel bbox."""
    if bbox.numel() != 4:
        return 0.0, 0.0, 0.0, 0.0
    if bbox_mode == "corner":
        x, y, w, h = bbox.float()
    elif bbox_mode == "center":
        cx, cy, bw, bh = bbox.float()
        x = cx - bw / 2
        y = cy - bh / 2
        w = bw
        h = bh
    else:
        raise ValueError(f"Unsupported bbox_mode: {bbox_mode}, must be 'corner' or 'center'")
    if _bbox_is_normalized(bbox.float(), height, width):
        return x.item() * width, y.item() * height, w.item() * width, h.item() * height
    return x.item(), y.item(), w.item(), h.item()

def _compute_mask_metrics(gt_mask: torch.Tensor, pred_mask: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    if gt_mask is None or pred_mask is None:
        return None, None
    # Ensure same shape
    if gt_mask.shape != pred_mask.shape:
        pred_mask = resize_mask(pred_mask, size=gt_mask.shape)
    gt_bin = (gt_mask > 0).int()
    pred_bin = (pred_mask > 0).int()
    inter = (gt_bin & pred_bin).sum().item()
    union = (gt_bin | pred_bin).sum().item()
    iou = inter / union if union > 0 else None
    dice = (2 * inter) / (gt_bin.sum().item() + pred_bin.sum().item()) if (gt_bin.sum().item() + pred_bin.sum().item()) > 0 else None
    return iou, dice

def visualize_sample_comparison(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_label_name: str,
    pred_label_name: str,
    gt_bbox: Optional[torch.Tensor] = None,
    pred_bbox: Optional[torch.Tensor] = None,
    processor=None,
    save_path: Optional[str] = None,
    dpi: int = 150,
    alpha: float = 0.45,
    cmap_gt: Tuple[float, float, float] = (1.0, 0.0, 0.0),  # red
    cmap_pred: Tuple[float, float, float] = (0.0, 0.7, 1.0),  # cyan/blue
    show: bool = True,
    title_prefix: str = "Sample",
    bbox_mode="corner",
):
    """Create a high-quality side-by-side figure.

    Left: Original image + GT segmentation overlay + GT bbox + label.
    Right: Original image + Pred segmentation overlay + Pred bbox + label.
    """
    img_denorm = _denormalize_image(image, processor)  # (C,H,W) in [0,1]
    H, W = img_denorm.shape[1], img_denorm.shape[2]

    gt_mask = _maybe_resize_mask_to_image(gt_mask, img_denorm)
    pred_mask = _maybe_resize_mask_to_image(pred_mask, img_denorm)

    # Metrics
    iou, dice = _compute_mask_metrics(gt_mask, pred_mask)

    def overlay(base_img: torch.Tensor, mask: torch.Tensor, color, a: float):
        if mask is None:
            return base_img
        mask_bin = (mask > 0).float()
        overlay_img = base_img.clone()
        for c in range(3):
            overlay_img[c] = (1 - mask_bin * a) * overlay_img[c] + (mask_bin * a) * color[c]
        return overlay_img

    img_gt = overlay(img_denorm, gt_mask, cmap_gt, alpha)
    img_pred = overlay(img_denorm, pred_mask, cmap_pred, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)

    # LEFT (Ground Truth)
    axes[0].imshow(np.transpose(img_gt.cpu().numpy(), (1, 2, 0)))
    axes[0].set_title(f"GT: {gt_label_name}")
    if gt_bbox is None and gt_mask is not None:
        # Derive a bbox from mask if not provided
        mbbox = compute_bbox_from_mask(gt_mask.cpu().numpy())
        if mbbox is not None:
            x, y, w_box, h_box = mbbox
            rect = plt.Rectangle((x, y), w_box, h_box, linewidth=2, edgecolor='yellow', facecolor='none')
            axes[0].add_patch(rect)
    elif gt_bbox is not None:
        gx, gy, gw, gh = _prepare_bbox(gt_bbox, H, W, bbox_mode=bbox_mode)
        rect = plt.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='yellow', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].axis('off')

    # RIGHT (Prediction)
    axes[1].imshow(np.transpose(img_pred.cpu().numpy(), (1, 2, 0)))
    metrics_str = []
    if iou is not None:
        metrics_str.append(f"IoU={iou:.3f}")
    if dice is not None:
        metrics_str.append(f"Dice={dice:.3f}")
    metric_line = (" | ".join(metrics_str)) if metrics_str else ""
    axes[1].set_title(f"Pred: {pred_label_name}\n{metric_line}")
    if pred_bbox is not None:
        px, py, pw, ph = _prepare_bbox(pred_bbox, H, W, bbox_mode=bbox_mode)
        rect = plt.Rectangle((px, py), pw, ph, linewidth=2, edgecolor='lime', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].axis('off')

    fig.suptitle(f"{title_prefix}", fontsize=14, fontweight='bold')
    fig.tight_layout()

    # Add simple color legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color='yellow', lw=2, label='GT BBox'),
        Line2D([0], [0], color='lime', lw=2, label='Pred BBox'),
        Line2D([0], [0], marker='s', color='w', label='GT Mask', markerfacecolor=cmap_gt, markersize=12, alpha=alpha),
        Line2D([0], [0], marker='s', color='w', label='Pred Mask', markerfacecolor=cmap_pred, markersize=12, alpha=alpha)
    ]
    axes[1].legend(handles=legend_elems, loc='lower right', framealpha=0.5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
