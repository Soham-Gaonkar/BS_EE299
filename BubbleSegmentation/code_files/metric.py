# metric.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.ndimage import binary_erosion
import warnings
from config import Config

# --- Helper Function for Boundary Extraction ---
def get_boundary_coords(mask):
    if mask.dtype != bool:
        mask = mask.astype(bool)

    if not mask.any():
        return None

    eroded_mask = binary_erosion(mask, border_value=0)
    boundary_mask = mask ^ eroded_mask

    if not boundary_mask.any():
        coords = np.argwhere(mask)
        return coords if coords.size > 0 else None

    coords = np.argwhere(boundary_mask)
    return coords

# --- Main Metric Calculation Function ---
def calculate_all_metrics(predictions, targets, threshold=0.5):
    if predictions.ndim != 4 or targets.ndim != 4:
        raise ValueError("Inputs must be 4D tensors (B, C, H, W)")
    if predictions.shape[1] != 1 or targets.shape[1] != 1:
        raise ValueError("Inputs must be single-channel (B, 1, H, W)")

    preds_prob = torch.sigmoid(predictions)
    preds_prob_np = preds_prob.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy().astype(np.uint8)

    preds_binary_np = (preds_prob_np > threshold).astype(np.uint8)

    targets_flat = targets_np.flatten()
    preds_binary_flat = preds_binary_np.flatten()

    TP = np.sum((preds_binary_flat == 1) & (targets_flat == 1))
    TN = np.sum((preds_binary_flat == 0) & (targets_flat == 0))
    FP = np.sum((preds_binary_flat == 1) & (targets_flat == 0))
    FN = np.sum((preds_binary_flat == 0) & (targets_flat == 1))

    epsilon = 1e-7

    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    dice_coefficient = (2 * TP) / (2 * TP + FP + FN + epsilon)
    iou = TP / (TP + FP + FN + epsilon)
    false_positive_rate = FP / (FP + TN + epsilon)
    false_negative_rate = FN / (FN + TP + epsilon)

    boundary_f1_score = (2 * precision * recall) / (precision + recall + epsilon)

    auc_term1 = TP / (2 * (TP + FN) + epsilon)
    auc_term2 = TN / (2 * (FP + TN) + epsilon)
    auc_specific = auc_term1 + auc_term2

    try:
        targets_roc_flat = targets_np.flatten()
        preds_prob_roc_flat = preds_prob_np.flatten()
        if len(np.unique(targets_roc_flat)) > 1:
            auroc = roc_auc_score(targets_roc_flat, preds_prob_roc_flat)
        else:
            auroc = 0.5
    except ValueError as e:
        warnings.warn(f"AUROC calculation failed: {e}. Setting AUROC to 0.5.")
        auroc = 0.5

    # --- Weighted IoU ---
    weighted_iou_weight = getattr(Config, "WEIGHTED_IOU_WEIGHT", 2.0)
    weighted_iou = (weighted_iou_weight * TP) / (weighted_iou_weight * TP + FP + FN + epsilon)

    # --- Hausdorff Distances ---
    batch_size = preds_binary_np.shape[0]
    hausdorff_means = []
    hausdorff_maxs = []

    for i in range(batch_size):
        pred_slice = preds_binary_np[i, 0]
        target_slice = targets_np[i, 0]

        pred_coords = get_boundary_coords(pred_slice)
        target_coords = get_boundary_coords(target_slice)

        current_mean_hd = np.nan
        current_max_hd = np.nan

        if pred_coords is not None and target_coords is not None and pred_coords.shape[0] > 0 and target_coords.shape[0] > 0:
            try:
                h_pred_to_target, _, _ = directed_hausdorff(pred_coords, target_coords)
                h_target_to_pred, _, _ = directed_hausdorff(target_coords, pred_coords)

                current_max_hd = max(h_pred_to_target, h_target_to_pred)

                dist_pred_to_target = cdist(pred_coords, target_coords).min(axis=1)
                mean_dist_pred = dist_pred_to_target.mean()
                dist_target_to_pred = cdist(target_coords, pred_coords).min(axis=1)
                mean_dist_target = dist_target_to_pred.mean()
                current_mean_hd = (mean_dist_pred + mean_dist_target) / 2.0

            except Exception as e:
                warnings.warn(f"Hausdorff calculation failed for sample {i}: {e}")
                current_max_hd = np.nan
                current_mean_hd = np.nan

        hausdorff_maxs.append(current_max_hd)
        hausdorff_means.append(current_mean_hd)

    mean_hausdorff_batch = np.nanmean(hausdorff_means) if not np.all(np.isnan(hausdorff_means)) else -1.0
    max_hausdorff_batch = np.nanmean(hausdorff_maxs) if not np.all(np.isnan(hausdorff_maxs)) else -1.0

    # Paper metrics
    paper_accuracy = recall
    paper_iou = iou
    paper_bf_score = boundary_f1_score

    results = {
        "Accuracy": accuracy,
        "Dice Coefficient": dice_coefficient,
        "IoU": iou,
        "Weighted IoU": weighted_iou,
        "Boundary F1 Score": boundary_f1_score,
        "AUROC": auroc,
        "AuC": auc_specific,
        "BF Score": boundary_f1_score,
        "Mean Hausdorff": mean_hausdorff_batch,
        "Max Hausdorff": max_hausdorff_batch,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate,
        "Accuracy (Paper)": paper_accuracy,
        "IoU (Paper)": paper_iou,
        "BF Score (Paper)": paper_bf_score,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity
    }

    return results
