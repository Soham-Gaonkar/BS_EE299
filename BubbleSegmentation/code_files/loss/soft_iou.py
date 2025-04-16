import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftIoULoss(nn.Module):
    """
    Soft IoU Loss for binary segmentation.

    A differentiable approximation of Intersection-over-Union (IoU),
    useful for training segmentation models where the target mask is sparse
    and traditional IoU is non-differentiable.

    Attributes:
        smooth (float): A small constant added to avoid division by zero.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Calculate the Soft IoU Loss.

        Args:
            inputs (torch.Tensor): Raw model output (logits). Shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth labels (0 or 1). Shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss (scalar).
        """
        # Apply sigmoid to convert logits to probabilities
        inputs_prob = torch.sigmoid(inputs)

        # Flatten the predictions and targets
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1).float()  # Ensure float for multiplication

        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum() - intersection

        # Compute IoU and return its loss
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou  # IoU loss
