import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os, glob, re
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import random

# ---- Loss Functions ----
class AsymmetricTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, smooth=1e-6, class_weights=None):
        """
        delta > 0.5 penalizes false negatives more (good for segmentation)
        class_weights: tensor of shape (num_classes,), e.g., [background_weight, foreground_weight]
        """
        super().__init__()
        self.delta = delta
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, preds, targets):
        # preds shape: (batch, num_classes, H, W)
        preds = F.softmax(preds, dim=1)  # probability over classes

        # Assume binary segmentation: background (0), foreground (1)
        foreground_preds = preds[:, 1, :, :]  # shape: (batch, H, W)
        background_preds = preds[:, 0, :, :]  # shape: (batch, H, W)

        foreground_targets = (targets == 1).float()
        background_targets = (targets == 0).float()

        # True Positives, False Negatives, False Positives for foreground
        true_pos_fg  = (foreground_preds * foreground_targets).sum(dim=[1, 2])
        false_neg_fg = (foreground_targets * (1 - foreground_preds)).sum(dim=[1, 2])
        false_pos_fg = ((1 - foreground_targets) * foreground_preds).sum(dim=[1, 2])

        # True Positives, False Negatives, False Positives for background (optional)
        true_pos_bg  = (background_preds * background_targets).sum(dim=[1, 2])
        false_neg_bg = (background_targets * (1 - background_preds)).sum(dim=[1, 2])
        false_pos_bg = ((1 - background_targets) * background_preds).sum(dim=[1, 2])

        # Tversky index for foreground and background
        tversky_fg = (true_pos_fg + self.smooth) / (true_pos_fg + self.delta * false_neg_fg + (1 - self.delta) * false_pos_fg + self.smooth)
        tversky_bg = (true_pos_bg + self.smooth) / (true_pos_bg + self.delta * false_neg_bg + (1 - self.delta) * false_pos_bg + self.smooth)

        if self.class_weights is not None:
            # Weighted sum of background and foreground losses
            loss = (1 - tversky_bg) * self.class_weights[0] + (1 - tversky_fg) * self.class_weights[1]
        else:
            # Just use foreground loss if no class weights given
            loss = 1 - tversky_fg

        return loss  # shape: (batch,)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')  # always 'none' internally

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)  # (batch, H, W)
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        # Apply reduction manually
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:  # 'none' – average over spatial dims per sample
            focal = focal.view(focal.shape[0], -1).mean(dim=1)
            return focal


class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, tversky_weight=0.5, focal_weight=0.5, delta=0.3):
        super().__init__()
        self.tversky = AsymmetricTverskyLoss(delta=delta)
        self.focal = FocalLoss(gamma=2, reduction='none')
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight


    def forward(self, preds, targets):
        # Compute per-sample losses
        tversky_loss = self.tversky(preds, targets)  # shape: (batch,)
        focal_loss = self.focal(preds, targets)      # shape: (batch,)
        # Weighted sum of the two losses
        loss = self.tversky_weight * tversky_loss + self.focal_weight * focal_loss
        return loss.mean()  # Return the average over the batch