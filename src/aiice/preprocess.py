from typing import Sequence

import torch
from torch.utils.data import Dataset


def apply_threshold(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    "Binarize tensor with a threshold"
    return (tensor > threshold).to(tensor.dtype)


def apply_downsample(
    t: torch.Tensor, i: int, axes: tuple[int, ...] = (-1,)
) -> torch.Tensor:
    """
    Downsample a tensor by keeping every i-th element along specified axes.

    Args:
        t (`torch.Tensor`): Input tensor.
        i (`int`): Step for downsampling. Must be greater than 0.
        axes (`tuple[int]`): Axes along which to downsample. Negative axes are supported.
    """
    if i <= 0:
        raise ValueError("i must be > 0")

    out = t
    for axis in axes:
        axis = axis if axis >= 0 else t.dim() + axis

        idx = torch.arange(out.shape[axis], device=out.device)
        keep = idx % i == 0
        out = torch.index_select(out, axis, idx[keep])

    return out


class SlidingWindowDataset(Dataset):
    """
    Convert a time series into (X, Y) pairs using sliding windows.

    X represents past observations of length `pre_history_len`,
    Y represents future observations of length `forecast_len`.

    ![image](../../.doc/media/sliding-window.png)

    The dataset is generated lazily: windows are sliced on demand from the
    original tensor without materializing the full dataset in memory.
    The time dimension is assumed to be the first axis of the input tensor.

    Args:
        data (`Sequence`): Time series data of shape `[T, ...]` where `T` is the time dimension
            and remaining dimensions represent features or channels.
        pre_history_len (`int`): Number of time steps in each input window (X).
        forecast_len (`int`): Number of time steps in each output window (Y).
        idx (`Sequence`, optional): Optional sequence of any indeces corresponding
            to each time step in `data`. Must have the same length as the time dimension `T`.
            If provided, `__getitem__` returns a tuple `(id, X, Y)` containing the
            corresponding timestamps for the selected window, otherwise it returns only `(X, Y)`.
        threshold (`float`, optional): If provided, binarizes the target tensor Y using this threshold.
            Values strictly greater than the threshold are set to 1, and values less than or equal to
            the threshold are set to 0. Defaults to None.
        x_binarize (`bool`, optional): If True and `threshold` is provided, applies the same binarization
            to the input tensor X. Defaults to False.
        device (`str`, optional): Device on which to place the tensors (e.g., "cpu", "cuda"). Defaults to None.
        dtype (torch.dtype, optional): Data type used to convert the input sequence. Defaults to torch.float32.
    """

    def __init__(
        self,
        data: Sequence,
        pre_history_len: int,
        forecast_len: int,
        idx: Sequence | None = None,
        threshold: float | None = None,
        x_binarize: bool = False,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self._data = torch.as_tensor(data, dtype=dtype, device=device)
        self._indices = idx

        self._threshold = threshold
        self._x_binarize = x_binarize

        if self._data.ndim == 1:
            self._data = self._data.unsqueeze(-1)  # [T] -> [T, 1]

        self._pre_history_len = pre_history_len
        self._forecast_len = forecast_len

        self._T = self._data.shape[0]
        if self._indices is not None and self._T != len(self._indices):
            raise ValueError(
                f"Data length (got {self._T}) should be equal to indices length (got {len(self._indices)})"
            )

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

        if self._indices is not None:
            idx_slice = self._indices[
                idx : idx + self._pre_history_len + self._forecast_len
            ]
            return idx_slice, x, y

        return x, y
