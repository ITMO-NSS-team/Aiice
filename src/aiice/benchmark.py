import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
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
        sea (str, optional): Name of the sea (e.g., "Barents Sea"). Check `Loader.seas` for available ones.
        threshold (float | None, optional): Threshold for binarizing the target Y. Values above threshold are set to 1, below or equal set to 0. Defaults to None.
        x_binarize (bool, optional): Whether to apply the same threshold binarization to input X. Defaults to False.
        threads (int, optional): Number of parallel download threads. You can reduce this value in case of rate limiting HuggingFace API errors. Defaults to 16.
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
        sea: str | None = None,
        threshold: float | None = None,
        x_binarize: bool = False,
        threads: int = 16,
        device: str | None = None,
    ):
        self._device = device
        self._sea = sea

        raw_data = Loader().get(
            start=start,
            end=end,
            step=step,
            sea=sea,
            threads=threads,
            tensor_out=True,
            idx_out=True,
        )

        indices = raw_data[0]
        matrices = raw_data[1]

        dataset = SlidingWindowDataset(
            data=matrices,
            idx=indices,
            pre_history_len=pre_history_len,
            forecast_len=forecast_len,
            threshold=threshold,
            x_binarize=x_binarize,
            device=self._device,
        )

        self._dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self._default_collate_fn,
        )

    def bench(
        self,
        model: nn.Module,
        metrics: dict[str, MetricFn] | list[str] | None = None,
        path: str | None = None,
        detailed: bool = True,
        plot_workers: int = 4,
        fps: int = 2,
    ) -> dict[str, list[float]]:
        """
        Run benchmarking evaluation of a model on the prepared dataset.

        The method iterates over the internal DataLoader, generates model
        predictions, computes evaluation metrics, and optionally produces
        visualization GIFs comparing ground truth and predicted forecasts.

        When `path` is provided, visualization generation is executed
        asynchronously using a thread pool so that plotting does not block
        model inference.

        Args:
            model (nn.Module):
                PyTorch model used to generate predictions. The model is expected
                to accept inputs `x` with shape `(batch, pre_history_len, ...)`
                and return predictions compatible with the selected metrics.

            metrics (dict[str, MetricFn] | list[str], optional):
                Metrics to compute during evaluation. If a list of metric names is
                provided, the metrics are resolved from the built-in registry.
                If `None`, default metrics are used.
                See `aiice.metrics.Evaluator` for details.

            path (str, optional):
                Directory where forecast visualizations will be saved.
                If provided, each sample in the dataset will produce a GIF
                animation showing the forecast horizon, comparing ground truth
                and model predictions frame by frame.

                The files are named: `<start_forecast_date>_<end_forecast_date>.gif`
                If `None`, visualization generation is skipped.

            detailed (bool, optional):
                If True, returns full statistics for each metric like
                mean, last value, count, min, and max.
                If False, returns only the mean value per metric.

            plot_workers (int, optional):
                Number of worker threads used for asynchronous plot generation.
                Increasing this value can speed up visualization when many samples
                are processed. Defaults to 4.

            fps (int, optional):
                Frames per second of the generated GIF animations. Defaults to 2.

        Returns:
            dict[str, list[float]]:
                Aggregated metric results returned by the evaluator.
        """
        if path is not None:
            os.makedirs(path, exist_ok=True)
            executor = ThreadPoolExecutor(max_workers=plot_workers)
            futures = []

        evaluator = Evaluator(metrics=metrics, accumulate=True)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(self._dataloader, desc="Prediction"):
                dates, x, y = batch
                x, y = x.to(self._device), y.to(self._device)

                pred = model(x)
                evaluator.eval(y, pred)

                if path is None:
                    continue

                futures.append(
                    executor.submit(
                        self._save_batch_plot,
                        sea=self._sea,
                        path=path,
                        dates=dates,
                        y=y.detach().cpu().numpy(),
                        pred=pred.detach().cpu().numpy(),
                        fps=fps,
                    )
                )

        if path is not None:
            for f in tqdm(futures, desc="Saving plots"):
                f.result()
            executor.shutdown(wait=True)

        return evaluator.report(detailed=detailed)

    @staticmethod
    def _save_batch_plot(
        sea: str | None,
        path: str,
        dates: list[list[date]],
        y: np.ndarray,
        pred: np.ndarray,
        fps: int,
    ) -> None:
        """
        Generate GIF visualizations for a batch of forecast samples.

        For each sample in the batch, a GIF animation is created showing
        the temporal evolution of the forecast horizon. Each frame displays
        a side-by-side comparison between the ground truth ice map and the
        model prediction for the corresponding forecast date.

        The resulting GIF file is saved to `path` with the name: `<start_forecast_date>_<end_forecast_date>.gif`
        where the dates correspond to the forecast window of the sample.
        """
        matplotlib.use("Agg")

        batch_size, forecast_len = y.shape[:2]
        for i in range(batch_size):

            start_date = dates[i][-forecast_len].strftime("%d-%m-%Y")
            end_date = dates[i][-1].strftime("%d-%m-%Y")

            save_path = os.path.join(path, f"{start_date}_{end_date}.gif")
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            im_gt = axes[0].imshow(y[i, 0])
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")

            im_pred = axes[1].imshow(pred[i, 0])
            axes[1].set_title("Prediction")
            axes[1].axis("off")

            frames = []
            for j in range(forecast_len):

                im_gt.set_data(y[i, j])
                im_pred.set_data(pred[i, j])

                forecast_date = dates[i][-forecast_len + j]
                if sea is None:
                    fig.suptitle(f"Forecast: {forecast_date.strftime('%d-%m-%Y')}")
                else:
                    fig.suptitle(
                        f"{sea} | Forecast: {forecast_date.strftime('%d-%m-%Y')}"
                    )

                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                frames.append(frame)

            plt.close(fig)
            imageio.mimsave(save_path, frames, duration=1 / fps, loop=0)

    @staticmethod
    def _default_collate_fn(
        batch: list[tuple[list[date], torch.Tensor, torch.Tensor]],
    ) -> tuple[list[list[date]], torch.Tensor, torch.Tensor]:
        """
        Collates SlidingWindow dataset samples into a batch
        input  -> batch of samples
        output -> batched tensors + list of date sequences

        Example:
        ```
        d1 = [date1...date2]
        x1.shape = (T, H, W)
        y1.shape = (H, W)

        batch = [
            (d1, x1, y1),
            (d2, x2, y2)
        ]

        Output:
            dates -> [d1, d2]
            x     -> torch.Tensor (B, T, H, W)
            y     -> torch.Tensor (B, H, W)
        ```
        """
        dates, x, y = zip(*batch)
        return list(dates), torch.stack(x), torch.stack(y)
