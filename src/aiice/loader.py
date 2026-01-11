import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date
from io import BytesIO

import numpy as np

from aiice.core.huggingface import HfDatasetClient


class Loader:
    def __init__(self):
        """
        Dataset Loader with a Hugging Face dataset client.
        """
        self._hf = HfDatasetClient()

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

        Parameters
        ----------
        per_year : bool
            If True, include per-year statistics.
        """
        return self._hf.info(per_year=per_year)

    def download(
        self,
        local_dir: str,
        start: date | None = None,
        end: date | None = None,
        step: date | None = None,
        threads: int = 32,
    ) -> list[str | None]:
        """
        Download dataset files to a local directory in parallel.

        Parameters
        ----------
        local_dir : str
            Directory to save files.
        start : date, optional
            Start date for files.
        end : date, optional
            End date for files.
        step : int, optional
            Step in days between files.
        threads : int
            Number of parallel download threads.
        """
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
        start: date | None = None,
        end: date | None = None,
        step: int | None = None,
        threads: int = 18,
        processes: int | None = None,
    ) -> np.ndarray:
        """
        Load dataset into memory.

        Parameters
        ----------
        start : date, optional
            Start date for files.
        end : date, optional
            End date for files.
        step : int, optional
            Step in days between files.
        threads: int
            Number of parallel download threads.
        processes: int, optional
            Number of worker processes used for decoding bytes into numpy matrices.
            If None, uses as many processes as there are CPU cores.
        """
        filenames = self._hf.get_filenames(start=start, end=end, step=step)
        with ThreadPoolExecutor(max_workers=threads) as tpool:
            raw_files = list(tpool.map(self._get_raw_file, filenames))

        with ProcessPoolExecutor(max_workers=processes) as ppool:
            arrays = list(ppool.map(self._decode_raw_file, raw_files))

        return np.stack(arrays)

    def _get_raw_file(self, filename: str) -> bytes:
        raw = self._hf.read_file(filename=filename)
        if raw is None:
            raise ValueError(f"Remote file {filename} not found")
        return raw

    def _decode_raw_file(self, raw: bytes) -> np.ndarray:
        return np.load(BytesIO(raw))
