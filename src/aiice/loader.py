import csv
import functools
import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime
from io import BytesIO
from typing import TypeAlias

import numpy as np
import torch

from aiice.constants import (
    DATASET_SHAPE,
    MASK_SEA_DATA_MAX_VALUE,
    MASK_SEA_DATA_PATH,
    MASK_SEA_IDX_PATH,
    MASK_SEA_NAME_COLUMN,
    MASK_SEA_NAME_ID,
)
from aiice.core.huggingface import HfDatasetClient
from aiice.core.utils import get_date_from_filename_template

NpWithIdx: TypeAlias = tuple[list[date], np.ndarray]
TorchWithIdx: TypeAlias = tuple[list[date], torch.Tensor]


class Loader:
    """
    Dataset Loader with a Hugging Face dataset client.

    Downloading a large number of files in parallel may lead to
    request timeouts or temporary server-side errors from
    Hugging Face. If this happens, reduce the number of threads
    or split the download into smaller date ranges.
    """

    def __init__(self):
        self._hf = HfDatasetClient()

        sea_csv_reader = csv.DictReader(
            io.StringIO(self._get_raw_file(MASK_SEA_IDX_PATH).decode("utf-8"))
        )
        self._sea_map: dict[str, int] = {
            row[MASK_SEA_NAME_COLUMN]: int(row[MASK_SEA_NAME_ID])
            for row in sea_csv_reader
        }

        self._sea_mask: np.ndarray = self._decode_raw_matrix(
            self._get_raw_file(MASK_SEA_DATA_PATH)
        )
        self._sea_mask[self._sea_mask == MASK_SEA_DATA_MAX_VALUE] = np.nan

    @property
    def seas(self) -> tuple[str, ...]:
        """
        Return available seas.
        """
        return tuple(self._sea_map.keys())

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return shape of a single dataset sample.
        """
        return self._hf.shape

    @property
    def dataset_start(self) -> date:
        """
        Return earliest available date in the dataset.
        """
        return self._hf.dataset_start

    @property
    def dataset_end(self) -> date:
        """
        Return latest available date in the dataset.
        """
        return self._hf.dataset_end

    def info(self, per_year: bool = False) -> dict[str, any]:
        """
        Collect dataset statistics.

        Args:
            per_year (bool): If True, include per-year statistics.
        """
        return self._hf.info(per_year=per_year)

    def download(
        self,
        local_dir: str,
        start: date | str | None = None,
        end: date | str | None = None,
        step: int | None = None,
        threads: int = 16,
    ) -> list[str | None]:
        """
        Download dataset files to a local directory in parallel.
        Raw numpy matrices in the dataset have range values from 0 to 100.

        Args:
            local_dir (str): Directory to save downloaded files.
            start (date | str, optional): Start date for files. Defaults to earliest dataset date.
            end (date | str, optional): End date for files. Defaults to latest dataset date.
            step (int, optional): Step in days between files. Defaults to 1.
            threads (int, optional): Number of parallel download threads. Defaults to 24.
        """
        start = self._convert_date(start)
        end = self._convert_date(end)

        filenames = self._hf.get_filenames(start=start, end=end, step=step)
        with ThreadPoolExecutor(max_workers=threads) as pool:
            return list(
                pool.map(
                    lambda f: self._hf.download_file(filename=f, local_dir=local_dir),
                    filenames,
                )
            )

    def get(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        step: int | None = None,
        sea: str | None = None,
        tensor_out: bool = False,
        idx_out: bool = False,
        threads: int = 16,
        processes: int | None = None,
    ) -> np.ndarray | torch.Tensor | NpWithIdx | TorchWithIdx:
        """
        Load dataset files into memory as numpy arrays or torch tensors.
        Loaded matrices are normalized to float values in the range 0 to 1.

        Args:
            start (date | str, optional): Start date for files. Defaults to earliest dataset date.
            end (date | str, optional): End date for files. Defaults to latest dataset date.
            step (int, optional): Step in days between files. Defaults to 1.
            sea (str, optional): Name of the sea (e.g., "Barents Sea"). Check `Loader.seas` for available ones.
            tensor_out (bool, optional): If True, returns a torch.Tensor instead of numpy array. Defaults to False.
            idx_out (bool, optional): If True, returns a tuple of (date indexes, matrices). Defaults to False.
            threads (int, optional): Number of parallel download threads. Defaults to 18.
            processes (int, optional): Number of worker processes for decoding raw bytes. Defaults to CPU core count.
        """
        if sea is not None and sea not in self._sea_map:
            raise ValueError(f"No such sea. Check available options: {self.seas}")

        start = self._convert_date(start)
        end = self._convert_date(end)

        filenames = self._hf.get_filenames(start=start, end=end, step=step)
        with ThreadPoolExecutor(max_workers=threads) as tpool:
            raw_files = list(tpool.map(self._get_raw_file, filenames))

        with ProcessPoolExecutor(max_workers=processes) as ppool:
            arrays = list(
                ppool.map(functools.partial(self._decode_and_crop, sea=sea), raw_files)
            )

        # numpy matrix values are ints in range 0...100
        result: np.ndarray | torch.Tensor = np.stack(arrays).astype(np.float32) / 100.0

        if tensor_out:
            result = torch.from_numpy(result)

        if idx_out:
            dates = [get_date_from_filename_template(f) for f in filenames]
            return dates, result

        return result

    def _decode_and_crop(self, raw: bytes, sea: str | None):
        matrix = self._decode_raw_matrix(raw)
        if sea is None:
            return matrix
        return self._get_sea_by_name(sea, matrix)

    def _get_sea_by_name(self, sea: str, matrix: np.ndarray) -> np.array:
        sea_id = self._sea_map[sea]
        boolean_mask = self._sea_mask == sea_id

        rows = np.any(boolean_mask, axis=1)
        cols = np.any(boolean_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        cropped_sea = matrix[rmin : rmax + 1, cmin : cmax + 1]
        return cropped_sea

    def _get_raw_file(self, filename: str) -> bytes:
        raw = self._hf.read_file(filename=filename)
        if raw is None:
            raise ValueError(f"Remote file {filename} not found")
        return raw

    def _decode_raw_matrix(self, raw: bytes) -> np.ndarray:
        matrix: np.ndarray = np.load(BytesIO(raw))
        if tuple(matrix.shape) != DATASET_SHAPE:
            raise ValueError(
                f"Matrix shape ({matrix.shape}) is not the same as a default one {DATASET_SHAPE=}"
            )
        return matrix

    def _convert_date(self, d: str | date) -> date:
        if isinstance(d, str):
            return datetime.strptime(d, "%Y-%m-%d").date()
        return d
