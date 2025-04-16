import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Dice Loss.

        Args:
            inputs (torch.Tensor): Model output (probabilities or logits).
            targets (torch.Tensor): Ground truth labels (0 or 1).

        Returns:
            torch.Tensor: The Dice Loss.
        """
        # Ensure predictions are probabilities (apply sigmoid if they're logits)
        inputs = torch.sigmoid(inputs)

        # Flatten the tensors
        # inputs = inputs.view(-1)
        # # targets = targets.view(-1)
        # targets = targets.reshape(-1)

        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)



        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        total_area = inputs.sum() + targets.sum()

        # Calculate Dice coefficient
        dice_coefficient = (2 * intersection + self.smooth) / (total_area + self.smooth)

        # Calculate Dice loss
        dice_loss = 1 - dice_coefficient

        return dice_loss