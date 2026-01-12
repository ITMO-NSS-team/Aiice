from typing import Sequence

import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Turns a time series into (X, Y) pairs using sliding windows.

    X = past window (pre_history_len)
    Y = future window (forecast_len)

    The dataset is generated lazily: windows are sliced on demand from the
    original tensor without materializing the full dataset in memory.

    The time dimension is assumed to be the first axis of the input tensor.

    Parameters
    ----------
    data : Sequence
        Time series data of shape ``[T, ...]`` where ``T`` is the time dimension
        and the remaining dimensions represent features or channels.

    pre_history_len : int
        Number of time steps in each input window (X).

    forecast_len : int
        Number of time steps in each output window (Y).

    threshold : float or None, optional
        If provided, binarizes the target tensor ``Y`` using this threshold.
        Values strictly greater than the threshold are set to 1, and values
        less than or equal to the threshold are set to 0.

    x_binarize : bool, default=False
        If True and ``threshold`` is provided, applies the same binarization
        to the input tensor ``X``.

    device : str or None, optional
        Device on which to place the tensor (e.g. ``"cpu"``, ``"cuda"``).

    dtype : torch.dtype, default=torch.float32
        Data type used to convert the input sequence.
    """

    def __init__(
        self,
        data: Sequence,
        pre_history_len: int,
        forecast_len: int,
        threshold: float | None = None,
        x_binarize: bool = False,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self._data = torch.as_tensor(data, dtype=dtype, device=device)
        self._threshold = threshold
        self._x_binarize = x_binarize

        if self._data.ndim == 1:
            self._data = self._data.unsqueeze(-1)  # [T] -> [T, 1]

        self._pre_history_len = pre_history_len
        self._forecast_len = forecast_len

        self._T = self._data.shape[0]
        self._length = self._T - pre_history_len - forecast_len + 1

        if self._length <= 0:
            raise ValueError(
                f"Not enough data: got {self._T}, need at least {pre_history_len + forecast_len}"
            )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        x = self._data[idx : idx + self._pre_history_len]
        y = self._data[
            idx
            + self._pre_history_len : idx
            + self._pre_history_len
            + self._forecast_len
        ]

        if isinstance(self._threshold, float):
            y = self._apply_threshold(y)
            x = self._apply_threshold(x) if self._x_binarize else x

        return x, y

    def _apply_threshold(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor > self._threshold).to(tensor.dtype)
