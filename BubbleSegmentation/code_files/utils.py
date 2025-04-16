# utils.py
#  Utility functions that don't fit elsewhere, like image saving, etc.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
from glob import glob
import torch.nn as nn

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def freeze_resnet_layers(model, freeze_until="layer2"):
    freeze = True
    for name, child in model.named_children():
        if freeze:
            for param in child.parameters():
                param.requires_grad = False
        if name == freeze_until:
            freeze = False


def plot_metrics_vs_pulses(metrics_csv_path, save_dir, experiment_name):
    """
    Reads metrics from CSV and generates plots similar to research paper figures.
    
    Parameters:
        metrics_csv_path (str): Path to CSV containing individual metrics per sample.
        save_dir (str): Directory where the plot image will be saved.
        experiment_name (str): Name of the experiment (used in plot title and filename).
    """

    metrics_df = pd.read_csv(metrics_csv_path)

    metrics_to_plot = {
        "Accuracy": "Predictive Accuracy (%)",
        "Dice Coefficient": "Dice Similarity Coefficient (%)",
        "Max Hausdorff": "Max Hausdorff Distance (mm)",
        "Mean Hausdorff": "Mean Hausdorff Distance (mm)"
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (metric, ylabel) in enumerate(metrics_to_plot.items()):
        ax = axes[idx]

        if metric not in metrics_df.columns:
            print(f"Warning: Metric '{metric}' not found in DataFrame columns.")
            continue

        grouped = metrics_df.groupby('pulses')[metric].agg(['mean', 'std']).reset_index().sort_values(by='pulses')

        ax.plot(grouped['pulses'], grouped['mean'], 'o-', color='dodgerblue', label='CNN')
        ax.fill_between(grouped['pulses'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        color='dodgerblue', alpha=0.3)

        ax.set_xlabel('Number of Pulses', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.suptitle(f"{experiment_name} Metrics vs. Number of Pulses", fontsize=16, y=1.02)

    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{experiment_name}_metrics_vs_pulses.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Metrics plot saved to {plot_path}")

def plot_ablation_area_comparison(
    mask_folder,
    cnn_metrics_path,
    save_path,
    experiment_name,
    pixel_area_mm2=0.0025,
    filename_pattern=r't3Label(\d+)_(\d+)_(\d+)'
):
    """
    Generates the Ablation Area vs Pulses plot with both Ground Truth and CNN predictions.

    Parameters:
        mask_folder (str): Path to the ground truth mask folder (.png masks).
        cnn_metrics_path (str): Path to CSV with CNN ablation area metrics.
        save_path (str): Directory to save the plot.
        experiment_name (str): Label for the CNN model.
        pixel_area_mm2 (float): Area per pixel in mm².
        filename_pattern (str): Regex pattern to extract pulses from filename.
    """
    # --- Ground Truth Ablation Area from Mask Files ---
    gt_data = []
    mask_files = glob(os.path.join(mask_folder, "*.png"))

    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        match = re.match(filename_pattern, filename)
        if match:
            pulses = int(match.group(1))
            experiment_id = match.group(2)
            dataset_idx = int(match.group(3))

            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            ablation_area = np.sum(binary_mask) * pixel_area_mm2

            gt_data.append({
                'pulses': pulses,
                'experiment_id': experiment_id,
                'dataset_idx': dataset_idx,
                'ablation_area': ablation_area
            })

    gt_df = pd.DataFrame(gt_data)
    gt_grouped = gt_df.groupby('pulses')['ablation_area'].agg(['mean', 'std']).reset_index()
    gt_grouped['pulses'] = gt_grouped['pulses'] * 20  # convert to actual number of pulses

    # --- CNN Prediction Ablation Area from Metrics CSV ---
    cnn_df = pd.read_csv(cnn_metrics_path)
    if 'ablation_area' not in cnn_df.columns:
        # Calculate predicted ablation area from prediction masks
        cnn_df['ablation_area'] = cnn_df['Predicted_Area'] if 'Predicted_Area' in cnn_df.columns else np.nan

    cnn_grouped = cnn_df.groupby('pulses')['ablation_area'].agg(['mean', 'std']).reset_index()
    cnn_grouped = cnn_grouped.sort_values(by='pulses')

    # --- Plot ---
    plt.figure(figsize=(10, 6))

    # Ground Truth
    plt.plot(gt_grouped['pulses'], gt_grouped['mean'], '^-', color='brown', label='Truth')
    plt.fill_between(gt_grouped['pulses'],
                     gt_grouped['mean'] - gt_grouped['std'],
                     gt_grouped['mean'] + gt_grouped['std'],
                     color='orange', alpha=0.5)

    # CNN Prediction
    plt.plot(cnn_grouped['pulses'], cnn_grouped['mean'], 'o', color='dodgerblue', label='CNN')
    plt.fill_between(cnn_grouped['pulses'],
                     cnn_grouped['mean'] - cnn_grouped['std'],
                     cnn_grouped['mean'] + cnn_grouped['std'],
                     color='dodgerblue', alpha=0.3)

    plt.xlabel("Number of Pulses", fontsize=14)
    plt.ylabel("Ablation Area (mm²)", fontsize=14)
    plt.title("Ablation Area vs. Number of Pulses", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    final_path = os.path.join(save_path, f"{experiment_name}_ablation_area_vs_pulses.png")
    plt.savefig(final_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Ablation area comparison plot saved to {final_path}")


import cv2
import numpy as np
import torch

def postprocess_mask(mask_tensor, min_size=100):
    """
    Remove small connected components from binary mask.
    Args:
        mask_tensor (torch.Tensor): Shape [1, H, W] or [H, W]
        min_size (int): Minimum pixel area to keep.
    Returns:
        torch.Tensor: Cleaned mask with small blobs removed.
    """
    mask_np = mask_tensor.cpu().numpy().squeeze().astype(np.uint8)

    num_components, output, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

    cleaned_mask = np.zeros_like(mask_np)
    for i in range(1, num_components):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[output == i] = 1

    return torch.tensor(cleaned_mask).unsqueeze(0).float()


def to_grayscale_numpy(tensor):
    """
    Converts a [1, H, W] or [H, W] tensor to (H, W) NumPy array for imshow.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.squeeze().cpu()
        return tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        return np.squeeze(tensor)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(tensor)}")


# utils/early_stopping.py

import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, monitor='val_iou', mode='max', path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.metric_to_monitor = monitor
        self.delta = delta
        self.path = path

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

        self.score_func = max if self.mode == 'max' else min

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif (
            (self.mode == 'max' and current_score < self.best_score + self.delta) or
            (self.mode == 'min' and current_score > self.best_score - self.delta)
        ):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.metric_to_monitor} did not improve. ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Saved best model checkpoint to {self.path}")
