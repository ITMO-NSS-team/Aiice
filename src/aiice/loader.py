from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np

from aiice.constants import MAX_SPLIT_FRACTION, MIN_SPLIT_FRACTION
from aiice.core.huggingface import HfDatasetClient


class Loader:
    def __init__(self):
        self._hf = HfDatasetClient()

    def download(
        self,
        local_dir: str,
        start: date | None = None,
        end: date | None = None,
        threads: int = 24,
    ) -> list[str | None]:
        filenames = self._hf.get_filenames(start=start, end=end)
        with ThreadPoolExecutor(max_workers=threads) as pool:
            return list(
                pool.map(
                    lambda filename: self._hf.download_file(
                        filename=filename, local_dir=local_dir
                    ),
                    filenames,
                )
            )

    def get(
        self,
        start: date | None = None,
        end: date | None = None,
        threads: int = 24,
        test_size: float | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        filenames = self._hf.get_filenames(start=start, end=end)
        if test_size is None:
            return self._get_files(filenames=filenames, threads=threads)

        if not MIN_SPLIT_FRACTION <= test_size <= MAX_SPLIT_FRACTION:
            raise ValueError(
                f"Test size should be between {MAX_SPLIT_FRACTION} and {MAX_SPLIT_FRACTION}"
            )

        if test_size == MIN_SPLIT_FRACTION:
            return self._get_files(filenames, threads), np.empty((0,))

        if test_size == MAX_SPLIT_FRACTION:
            return np.empty((0,)), self._get_files(filenames, threads)

        split_index = len(filenames) - int(len(filenames) * test_size)
        train_files = filenames[:split_index]
        test_files = filenames[split_index:]

        return (
            self._get_files(train_files, threads),
            self._get_files(test_files, threads),
        )

    def _get_file(self, filename: str) -> np.ndarray:
        npy = self._hf.read_file(filename=filename)
        if npy is None:
            raise ValueError(f"Remote file {filename} not found")
        return npy

    def _get_files(self, filenames: list[str], threads: int):
        with ThreadPoolExecutor(max_workers=threads) as pool:
            npys = list(pool.map(lambda filename: self._get_file(filename), filenames))
        return np.array(npys)
