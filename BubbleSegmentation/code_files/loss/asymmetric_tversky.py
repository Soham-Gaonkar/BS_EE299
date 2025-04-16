import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalTverskyLoss(nn.Module):
    """
    Asymmetric Focal Tversky Loss for binary segmentation.

    Combines the Tversky index with a focal term to handle class imbalance
    and focus on hard negatives. The asymmetry is controlled by alpha and beta.
    Designed to penalize False Positives more aggressively by setting beta > alpha.

    Attributes:
        alpha (float): Weight for False Negatives (FN). Range [0, 1].
        beta (float): Weight for False Positives (FP). Range [0, 1]. Sum alpha + beta = 1.
                      Higher beta penalizes FP more.
        gamma (float): Focusing parameter. Higher gamma focuses more on hard examples.
                       Values typically range from 0.5 to 5. Common starting point: 0.75 or 1.0.
        smooth (float): Smoothing factor to avoid division by zero.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-5):
        super().__init__()
        if not (0 <= alpha <= 1 and 0 <= beta <= 1 and alpha + beta == 1):
            raise ValueError("alpha and beta must be in [0, 1] and sum to 1.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Calculate the Asymmetric Focal Tversky Loss.

        Args:
            inputs (torch.Tensor): Raw model output (logits). Shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth labels (0 or 1). Shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss (scalar).
        """
        # Apply sigmoid to inputs to get probabilities
        inputs_prob = torch.sigmoid(inputs)

        # Flatten input and target tensors
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1).float() # Ensure targets are float for calculations

        # Calculate True Positives, False Positives, and False Negatives
        TP = (inputs_flat * targets_flat).sum()
        FP = (inputs_flat * (1 - targets_flat)).sum()
        FN = ((1 - inputs_flat) * targets_flat).sum()

        # Calculate Tversky Index (TI)
        # TI = TP / (TP + alpha * FN + beta * FP + smooth)
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        # Calculate Focal Tversky Loss
        # Loss = (1 - TI)^gamma
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)

        return focal_tversky_loss