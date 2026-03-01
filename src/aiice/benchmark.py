from datetime import date

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiice.loader import Loader
from aiice.metrics import Evaluator, MetricFn
from aiice.preprocess import SlidingWindowDataset


class AIICE:
    """
    High-level interface for loading Arctic ice data, preparing datasets, and benchmarking models.

    This class provides a simple API to:
    1. Load historical ice data within a specified date range (see `aiice.loader.Loader`)
    2. Convert the data into sliding-window datasets (see `aiice.preprocess.SlidingWindowDataset`)
    3. Create a PyTorch DataLoader for batch processing
    4. Benchmark any PyTorch model on the OSI-SAF dataset with specified metrics

    Args:
        pre_history_len (int): Number of past time steps to include in each input sample (X).
        forecast_len (int): Number of future time steps to predict (Y) in each sample.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 16.
        start (date | str | None, optional): Start date of the data to load. If None, defaults to the earliest available data.
        end (date | str | None, optional): End date of the data to load. If None, defaults to the latest available data.
        step (int | None, optional): Step in days between data points. Defaults to 1 if not provided.
        threshold (float | None, optional): Threshold for binarizing the target Y. Values above threshold are set to 1, below or equal set to 0. Defaults to None.
        x_binarize (bool, optional): Whether to apply the same threshold binarization to input X. Defaults to False.
        device (str | None, optional): Device to place tensors on ("cpu", "cuda", etc.). If None, uses PyTorch default device.

    Example:
        >>> aiice = AIICE(pre_history_len=30, forecast_len=7, batch_size=32, start="2022-01-01", end="2022-12-31")
        >>> model = MyModel()
        >>> results = aiice.bench(model, metrics={"mae", "psnr"})
    """

    def __init__(
        self,
        pre_history_len: int,
        forecast_len: int,
        batch_size: int = 16,
        start: date | str | None = None,
        end: date | str | None = None,
        step: int | None = None,
        threshold: float | None = None,
        x_binarize: bool = False,
        device: str | None = None,
    ):
        self._device = device

        raw_data = Loader().get(
            start=start,
            end=end,
            step=step,
            tensor_out=True,
            idx_out=True,
        )

        self._idx_dates = raw_data[0]
        self._matrices = raw_data[1]

        dataset = SlidingWindowDataset(
            data=self._matrices,
            pre_history_len=pre_history_len,
            forecast_len=forecast_len,
            threshold=threshold,
            x_binarize=x_binarize,
            device=self._device,
        )

        self._dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
        )

    def bench(
        self, model: nn.Module, metrics: dict[str, MetricFn] | list[str] | None = None
    ) -> dict[str, list[float]]:
        """
        Run a benchmarking evaluation on the dataset using the provided model,
        accumulates evaluation metrics, and returns the aggregated report.

        Args:
            model (nn.Module):
                PyTorch model used to generate predictions. The model is expected
                to accept batches of inputs `x` with shape `(batch, pre_history_len, ...)` shape
                and return predictions compatible with the provided metrics.
            metrics (dict[str, MetricFn] | list[str] | None, optional):
                Metrics to use. If a list of strings is provided, metrics are resolved
                from the built-in registry. If None, default metrics are used. Defaults to None.
                Check in more details: `aiice.metrics.Evaluator`
        """
        evaluator = Evaluator(metrics=metrics, accumulate=True)
        for x, y in tqdm(self._dataloader):
            x, y = x.to(self._device), y.to(self._device)
            pred = model(x)
            evaluator.eval(y, pred)

        return evaluator.report()
