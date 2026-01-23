from typing import Sequence

import torch
from torch.utils.data import Dataset


def apply_threshold(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    "Binarize tensor with a threshold"
    return (tensor > threshold).to(tensor.dtype)


def apply_downsample(t: torch.Tensor, i: int) -> torch.Tensor:
    """
    Removes every i-th element along the last axis of the tensor.

    Example:
        i=2 keeps indices [0,2,4,...]
        i=3 keeps indices [0,3,6,...]
    """
    if i <= 0:
        raise ValueError("i must be > 0")

    idx = torch.arange(t.shape[-1], device=t.device)
    keep = idx % i == 0
    return t.index_select(-1, idx[keep])


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

    def __getitem__(self, idx: int):
        if not isinstance(idx, int):
            raise TypeError("index must be int")

        if idx < 0 or idx >= self._length:
            raise IndexError("index out of range")

        x = self._data[idx : idx + self._pre_history_len]
        y = self._data[
            idx
            + self._pre_history_len : idx
            + self._pre_history_len
            + self._forecast_len
        ]

        if isinstance(self._threshold, float):
            y = apply_threshold(y, self._threshold)
            x = apply_threshold(x, self._threshold) if self._x_binarize else x

        return x, y
