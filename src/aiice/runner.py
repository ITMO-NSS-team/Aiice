import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiice.metrics import Evaluator, MetricFn


class Runner:
    """
    Utility class for running inference over a dataset and computing metrics.

    The Runner iterates over a given DataLoader, performs forward passes
    with the provided model, and accumulates evaluation metrics using an Evaluator instance.

    Args:
        model (nn.Module): PyTorch model used to generate predictions. The model is expected
            to accept batches of inputs `x` and return predictions compatible with the provided metrics.
        dataloader (DataLoader): Iterable DataLoader that yields batches of `(x, y)` pairs.
            `x` represents model inputs and `y` the corresponding targets.
        metrics (dict[str, MetricFn] | list[str] | None, optional):
            Metrics to use. If a list of strings is provided, metrics are resolved
            from the built-in registry. If None, default metrics are used. Defaults to None.
        device (str | None, optional): Device on which to place tensors (e.g., "cpu", "cuda"). Defaults to None.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        metrics: dict[str, MetricFn] | list[str] | None = None,
        device: str | None = None,
    ):
        self._model = model
        self._data = dataloader
        self._device = device
        self._evaluator = Evaluator(metrics=metrics, accumulate=True)
        self._last_report: dict[str, list[float]] = {
            k: [] for k in self._evaluator._metrics
        }

    @property
    def last_report(self):
        return self._last_report

    def run(self) -> dict[str, list[float]]:
        """
        Runs inference over the entire dataloader, accumulates evaluation
        metrics, and returns the aggregated report.
        """

        for x, y in tqdm(self._data):
            x, y = x.to(self._device), y.to(self._device)
            pred = self._model(x)
            self._evaluator.eval(y, pred)

        self._last_report = self._evaluator.report()
        return self.last_report
