import torch
import torch.nn.functional as F

def mse_loss(recon_x, x):
    """
    Calculates the Mean Squared Error (MSE) loss between reconstructed output and original input.

    Args:
        recon_x (torch.Tensor): Reconstructed output tensor.
        x (torch.Tensor): Original input tensor.

    Returns:
        torch.Tensor: The MSE loss.
    """
    return F.mse_loss(recon_x, x, reduction='mean')
