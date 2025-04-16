# loss/dice_focal.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, smooth=1e-5):
        super().__init__()
        if dice_weight + focal_weight != 1.0:
            warnings.warn("Dice and Focal weights do not sum to 1. Adjust if needed.")
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Dice Loss (use probabilities)
        inputs_prob = torch.sigmoid(inputs).clamp(min=1e-7, max=1 - 1e-7)
        targets = targets.float()

        inputs_flat = inputs_prob.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        total = inputs_flat.sum() + targets_flat.sum()
        dice_coeff = (2. * intersection + self.smooth) / (total + self.smooth + 1e-6)
        dice_loss = 1. - dice_coeff

        # Focal Loss (use logits)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss
