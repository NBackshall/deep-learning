"""L2 loss."""
import numpy as np

from ..nn.transformation import Transformation


def l2_loss(predictions, targets):
    """L2 Loss."""
    return (predictions - targets) ** 2
