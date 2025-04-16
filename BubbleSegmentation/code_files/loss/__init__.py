# loss/__init__.py
from .dice_focal import DiceFocalLoss
from .dice import DiceLoss
from .asymmetric_tversky import AsymmetricFocalTverskyLoss
from .soft_iou  import SoftIoULoss

# Import other losses here