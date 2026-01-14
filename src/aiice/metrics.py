from collections.abc import Callable, Sequence
from typing import Sequence

import pytorch_msssim
import torch

from aiice.constants import DEFAULT_SSIM_KERNEL_WINDOW_SIZE


def _apply_threshold(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (tensor > threshold).to(tensor.dtype)


def _as_tensor(y_true: Sequence, y_pred: Sequence, device=None):
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


MetricFn = Callable[[Sequence, Sequence], float]


class Evaluator:
    """
    Computes and aggregates evaluation metrics over multiple evaluation steps.

    Parameters
    ----------
    metrics : dict[str, MetricFn] or list[str] or None, optional
        Metrics to use. If a list of strings is provided, metrics are resolved
        from the built-in registry. If None, default metrics are used.

    accumulate : bool, default=True
        Whether to accumulate metric values across multiple ``eval`` calls.
    """

    _default_metrics: list[str] = ["mae", "mse", "rmse", "psnr", "bin_accuracy", "ssim"]

    def __init__(
        self,
        metrics: dict[str, MetricFn] | list[str] | None = None,
        accumulate: bool = True,
    ):
        if metrics is None:
            self._metrics = self._init_metrics(self._default_metrics)
        elif isinstance(metrics, list):
            self._metrics = self._init_metrics(metrics)
        else:
            self._metrics = metrics

        self._accumulate = accumulate
        self._report: dict[str, list[float]] = {k: [] for k in self._metrics}

    def _init_metrics(self, metrics: list[str]) -> dict[str, MetricFn]:
        registry: dict[str, MetricFn] = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "psnr": psnr,
            "bin_accuracy": bin_accuracy,
            "ssim": ssim,
        }

        result = {}
        for name in metrics:
            try:
                result[name] = registry[name]
            except KeyError:
                raise ValueError(
                    f"Unknown metric '{name}', choose from {list(registry)}"
                )
        return result

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
        summary = {}
        for name, values in self._report.items():
            if not values:
                continue

            summary[name] = {
                "mean": sum(values) / len(values),
                "last": values[-1],
                "count": len(values),
                "min": min(values),
                "max": max(values),
            }
        return summary
