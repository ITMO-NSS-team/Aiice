from collections.abc import Callable, Sequence
from typing import Sequence

import pytorch_msssim
import torch

from aiice.constants import (
    BIN_ACCURACY_METRIC,
    COUNT_STAT,
    DEFAULT_SSIM_KERNEL_WINDOW_SIZE,
    IOU_METRIC,
    LAST_STAT,
    MAE_METRIC,
    MAX_STAT,
    MEAN_STAT,
    MIN_STAT,
    MSE_METRIC,
    PSNR_METRIC,
    RMSE_METRIC,
    SSIM_METRIC,
)
from aiice.preprocess import apply_threshold


def _as_tensor(y_true: Sequence, y_pred: Sequence, device=None):
    y_true = torch.as_tensor(y_true, dtype=torch.float32, device=device).detach()
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device=device).detach()
    return y_true, y_pred


def mae(y_true: Sequence, y_pred: Sequence) -> float:
    """
    MAE (mean absolute error) - determines absolute values range coincidence with real data.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)
    return torch.abs(y_true - y_pred).mean().item()


def mse(y_true: Sequence, y_pred: Sequence) -> float:
    """
    MSE (mean squared error) - similar to MAE but emphasizes larger errors by squaring differences.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)
    return ((y_true - y_pred) ** 2).mean().item()


def rmse(y_true: Sequence, y_pred: Sequence) -> float:
    """
    RMSE (root mean square error) - determines absolute values range coincidence as MAE
    but making emphasis on spatial error distribution of prediction.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)
    return torch.sqrt(((y_true - y_pred) ** 2).mean()).item()


def psnr(y_true: Sequence, y_pred: Sequence) -> float:
    """
    PSNR (peak signal-to-noise ratio) - reflects noise and distortion level on predicted images identifying artifacts.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)

    mse_val = torch.mean((y_true - y_pred) ** 2)
    if mse_val == 0:
        return float("inf")

    max_val = torch.max(y_true)
    return (20 * torch.log10(max_val) - 10 * torch.log10(mse_val)).item()


def bin_accuracy(y_true: Sequence, y_pred: Sequence, threshold: float = 0.7) -> float:
    """
    Binary accuracy - binarization of ice concentration continuous field with threshold which causing the presence of an ice edge
    gives us possibility to compare binary masks of real ice extent and predicted one.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)

    y_true = apply_threshold(y_true, threshold)
    y_pred = apply_threshold(y_pred, threshold)

    return (y_true == y_pred).float().mean().item()


def ssim(y_true: Sequence, y_pred: Sequence) -> float:
    """
    SSIM (structural similarity index measure) - determines spatial patterns coincidence on predicted and target images

    Raises:
        ValueError:
            - If input tensors are not 4D ([N, C, H, W]) or 5D ([N, C, D, H, W]).
            - If any spatial or temporal dimension is smaller than 11 (minimum SSIM kernel window size)
    """
    spatial_dims = y_true.shape[2:]
    if any(dim < DEFAULT_SSIM_KERNEL_WINDOW_SIZE for dim in spatial_dims):
        raise ValueError(
            f"All spatial dimensions {spatial_dims} must be >= win_size={DEFAULT_SSIM_KERNEL_WINDOW_SIZE}"
        )

    y_true, y_pred = _as_tensor(y_true, y_pred)
    return float(pytorch_msssim.ssim(y_true, y_pred, data_range=1.0))


def iou(y_true: Sequence, y_pred: Sequence, threshold: float = 0.7) -> float:
    """
    IoU (Intersection over Union) - measures overlap between binary masks
    of ground truth and prediction.

    Similar to bin_accuracy but focuses on overlap quality instead of per-pixel equality.
    """
    y_true, y_pred = _as_tensor(y_true, y_pred)

    y_true = apply_threshold(y_true, threshold)
    y_pred = apply_threshold(y_pred, threshold)

    y_true = y_true.view(y_true.size(0), -1)
    y_pred = y_pred.view(y_pred.size(0), -1)

    intersection = (y_true * y_pred).sum(dim=1)
    # union = |A| + |B| - |A ∩ B|
    union = y_true.sum(dim=1) + y_pred.sum(dim=1) - intersection

    eps = 1e-7
    iou = intersection / (union + eps)
    return iou.mean().item()


MetricFn = Callable[[Sequence, Sequence], float]


class Evaluator:
    """
    Compute and aggregate evaluation metrics over multiple evaluation steps.

    Args:
        metrics (dict[str, MetricFn] | list[str] | None, optional):
            Metrics to use. If a list of strings is provided, metrics are resolved
            from the built-in registry. If None, default metrics are used.
        accumulate (bool, optional):
            Whether to accumulate metric values across multiple `eval` calls. Defaults to True.
    """

    _metrics_registry: dict[str, MetricFn] = {
        MAE_METRIC: mae,
        MSE_METRIC: mse,
        RMSE_METRIC: rmse,
        PSNR_METRIC: psnr,
        BIN_ACCURACY_METRIC: bin_accuracy,
        SSIM_METRIC: ssim,
        IOU_METRIC: iou,
    }

    def __init__(
        self,
        metrics: dict[str, MetricFn] | list[str] | None = None,
        accumulate: bool = True,
    ):
        if metrics is None:
            self._metrics = self._metrics_registry
        elif isinstance(metrics, list):
            self._metrics = self._init_metrics(metrics)
        else:
            self._metrics = metrics

        self._accumulate = accumulate
        self._report: dict[str, list[float]] = {k: [] for k in self._metrics}

    def _init_metrics(self, metrics: list[str]) -> dict[str, MetricFn]:
        result: dict[str, MetricFn] = {}
        for name in metrics:
            try:
                result[name] = self._metrics_registry[name]
            except KeyError:
                raise ValueError(
                    f"Unknown metric '{name}', choose from {list(self._metrics_registry.keys())}"
                )
        return result

    @property
    def metrics(self) -> list[str]:
        return list(self._metrics.keys())

    def eval(self, y_true: Sequence, y_pred: Sequence) -> dict[str, float]:
        """
        Evaluate all metrics on a single batch or sample and updates the internal
        report state depending on the ``accumulate`` mode.
        """
        step_result: dict[str, float] = {}

        for name, fn in self._metrics.items():
            value = fn(y_true, y_pred)
            step_result[name] = value

            if self._accumulate:
                self._report[name].append(value)
            else:
                self._report[name] = [value]

        return step_result

    def report(self) -> dict[str, dict[str, float]]:
        """
        Return aggregated statistics for all evaluated metrics.
        """
        summary: dict[str, dict[str, float]] = {}
        for name, values in self._report.items():
            if not values:
                continue

            summary[name] = {
                MEAN_STAT: sum(values) / len(values),
                LAST_STAT: values[-1],
                COUNT_STAT: len(values),
                MIN_STAT: min(values),
                MAX_STAT: max(values),
            }
        return summary
