from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from functools import lru_cache
from io import BytesIO

import requests
from huggingface_hub import HfApi
from huggingface_hub.constants import DEFAULT_REQUEST_TIMEOUT
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import build_hf_headers

from aiice.constants import (
    BYTES_IN_MB,
    DATASET_SHAPE,
    DEFAULT_BACKOFF,
    DEFAULT_RETRIES,
    HF_BASE_URL,
    HF_DATASET_REPO,
    HF_PACKAGE_NAME,
    HF_REPO_TYPE,
    KEY_DATASET_END,
    KEY_DATASET_START,
    KEY_FILES,
    KEY_PER_YEAR,
    KEY_SHAPE,
    KEY_SIZE_BYTES,
    KEY_SIZE_MB,
    MAX_DATASET_END,
    MIN_DATASET_START,
    YEAR_STATS_CACHE_SIZE,
)
from aiice.core.utils import (
    convert_step_to_delta,
    get_filename_template,
    retry_on_network_errors,
)


class HfDatasetClient:
    """
    Client for accessing the AIICE Hugging Face dataset.
    """

    def __init__(self):
        self._api_base_url = HF_BASE_URL
        self._api = HfApi(endpoint=self._api_base_url, library_name=HF_PACKAGE_NAME)
        self._api_headers = build_hf_headers(library_name=HF_PACKAGE_NAME)

        self._dataset_repo = HF_DATASET_REPO
        self._dataset_repo_type = HF_REPO_TYPE

        self._min_dataset_start, self._max_dataset_end = (
            MIN_DATASET_START,
            MAX_DATASET_END,
        )
        self._shape = DATASET_SHAPE

    @property
    def dataset_start(self) -> date:
        """
        Earliest available date in the dataset.
        """
        return self._min_dataset_start

    @property
    def dataset_end(self) -> date:
        """
        Latest available date in the dataset.
        """
        return self._max_dataset_end

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of a single dataset sample.
        """
        return self._shape

    @retry_on_network_errors(retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF)
    def info(self, per_year: bool = False, threads: int = 24) -> dict[str, any]:
        """
        Collect dataset size statistics.

        Args:
            per_year (`bool`, optional): If True, include per-year file and size statistics. Defaults to False.
            threads (`int`, optional): Number of threads used for parallel HTTP requests. Defaults to 24.
        """
        total_files, total_size = 0, 0
        per_year_result = defaultdict(
            lambda: {
                KEY_FILES: 0,
                KEY_SIZE_BYTES: 0,
                KEY_SIZE_MB: 0.0,
            }
        )

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(self._fetch_year_stats, year)
                for year in range(
                    self.dataset_start.year,
                    self.dataset_end.year + 1,
                )
            ]

            for future in as_completed(futures):
                year, files, size = future.result()

                per_year_result[year][KEY_FILES] = files
                per_year_result[year][KEY_SIZE_BYTES] = size
                per_year_result[year][KEY_SIZE_MB] = round(size / BYTES_IN_MB, 2)

                total_files += files
                total_size += size

        result: dict[str, any] = {
            KEY_DATASET_START: self.dataset_start,
            KEY_DATASET_END: self.dataset_end,
            KEY_SHAPE: self.shape,
            f"total_{KEY_FILES}": total_files,
            f"total_{KEY_SIZE_BYTES}": total_size,
            f"total_{KEY_SIZE_MB}": round(total_size / BYTES_IN_MB, 2),
        }

        if per_year:
            result[KEY_PER_YEAR] = dict(per_year_result)

        return result

    def get_filenames(
        self,
        start: date | None = None,
        end: date | None = None,
        step: int | str | None = None,
    ) -> list[str]:
        """
        Generate dataset filenames for a date range.

        Args:
            start (`date`, optional): Start date (inclusive). Defaults to dataset start.
            end (`date`, optional): End date (inclusive). Defaults to dataset end.
            step (`int` or `str`, optional): Step between files. If `int` - number of days.
                If `str` - format like `"1d"`, `"1w"`, `"1m"`, `"1y"`.
                For month or years steps (`"1m"`, `"2m"`, etc.), the date always lands on the last day
                of the month (e.g., Jan 31 + 1 month = Feb 28/29, then Mar 31).
                Defaults to 1 day.
        """
        start = start or self.dataset_start
        end = end or self.dataset_end

        if start < self.dataset_start:
            raise ValueError(f"date start value should be > {self.dataset_start}")

        if end > self.dataset_end:
            raise ValueError(f"date end value should be < {self.dataset_end}")

        if start > end:
            raise ValueError("start date must be <= date end")

        filenames: list[str] = []
        current = start
        delta = convert_step_to_delta(step=step)

        while current <= end:
            filenames.append(get_filename_template(current))
            current += delta

        return filenames

    @retry_on_network_errors(retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF)
    def read_file(self, filename: str) -> bytes | None:
        """
        Load a dataset file from Hugging Face into memory.

        Args:
            filename (`str`): Relative path to the dataset file.
        """
        url = f"{self._api_base_url}/datasets/{self._dataset_repo}/resolve/main/{filename}"
        buffer = BytesIO()
        try:
            http_get(
                url=url,
                temp_file=buffer,
                displayed_filename=filename,
                headers=self._api_headers,
            )
            return buffer.getvalue()

        # ignore if file isn't found
        except RemoteEntryNotFoundError:
            return None

        except Exception as e:
            raise RuntimeError(f"Failed to get file {filename}") from e

    @retry_on_network_errors(retries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF)
    def download_file(self, filename: str, local_dir: str) -> str | None:
        """
        Download a dataset file to a local directory.

        Args:
            filename (`str`): Dataset file path.
            local_dir (`str`): Target directory for download.
        """
        try:
            return self._api.hf_hub_download(
                repo_id=self._dataset_repo,
                repo_type=self._dataset_repo_type,
                filename=filename,
                local_dir=local_dir,
            )

        # ignore if file isn't found
        except RemoteEntryNotFoundError:
            return None

        except Exception as e:
            raise RuntimeError(f"Failed to download file {filename}") from e

    @lru_cache(maxsize=YEAR_STATS_CACHE_SIZE)
    def _fetch_year_stats(self, year: int) -> tuple[int, int, int]:
        url = f"{self._api_base_url}/api/datasets/{self._dataset_repo}/tree/main/global_series/{year}"

        resp = requests.get(
            url, timeout=DEFAULT_REQUEST_TIMEOUT, headers=self._api_headers
        )
        resp.raise_for_status()

        files, size = 0, 0
        for item in resp.json():
            if item.get("type") != "file":
                continue

            files += 1
            size += item.get("size", 0)

        return year, files, size
