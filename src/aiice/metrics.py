from typing import Sequence

import pytorch_msssim
import torch

from aiice.constants import DEFAULT_SSIM_KERNEL_WINDOW_SIZE


def _apply_threshold(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (tensor > threshold).to(tensor.dtype)


def _as_tensor(y_true, y_pred, device=None):
    y_true = torch.as_tensor(y_true, dtype=torch.float32, device=device)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device=device)
    return y_true, y_pred


def mae(y_true: Sequence, y_pred: Sequence) -> float:
    """
    MAE (mean absolute error) - determines absolute values range coincidence with real data.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)
    return float(torch.abs(y_true - y_pred).mean())


def mse(y_true: Sequence, y_pred: Sequence) -> float:
    """
    MSE (mean squared error) - similar to MAE but emphasizes larger errors by squaring differences.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)
    return float(torch.mean((y_true - y_pred) ** 2))


def rmse(y_true: Sequence, y_pred: Sequence) -> float:
    """
    RMSE (root mean square error) - determines absolute values range coincidence as MAE
    but making emphasis on spatial error distribution of prediction.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)))


def psnr(y_true: Sequence, y_pred: Sequence) -> float:
    """
    PSNR (peak signal-to-noise ratio) - reflects noise and distortion level on predicted images identifying artifacts.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)

    mse_val = torch.mean((y_true - y_pred) ** 2)
    if mse_val == 0:
        return float("inf")

    max_val = torch.max(y_true)
    return float(20 * torch.log10(max_val) - 10 * torch.log10(mse_val))


def bin_accuracy(y_true: Sequence, y_pred: Sequence, threshold: float = 0.5) -> float:
    """
    Binary accuracy - binarization of ice concentration continuous field with threshold which causing the presence of an ice edge
    gives us possibility to compare binary masks of real ice extent and predicted one.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)

    y_true = _apply_threshold(y_true, threshold)
    y_pred = _apply_threshold(y_pred, threshold)

    return float((y_true == y_pred).float().mean())


def ssim(y_true: Sequence, y_pred: Sequence) -> float:
    """
    SSIM (structural similarity index measure) - determines spatial patterns coincidence on predicted and target images

    Raises
    ------
    ValueError
        If input tensors are not 4D ([N, C, H, W]) or 5D ([N, C, D, H, W]).
        If input spatial/temporal dimensions are smaller than 11 (SSIM kernel window size).
    """
    spatial_dims = y_true.shape[2:]
    if any(dim < DEFAULT_SSIM_KERNEL_WINDOW_SIZE for dim in spatial_dims):
        raise ValueError(
            f"All spatial dimensions {spatial_dims} must be >= win_size={DEFAULT_SSIM_KERNEL_WINDOW_SIZE}"
        )

    y_true, y_pred = _as_tensor(y_true, y_pred)
    return float(pytorch_msssim.ssim(y_true, y_pred, data_range=1.0))
