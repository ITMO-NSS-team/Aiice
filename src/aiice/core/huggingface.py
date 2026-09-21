from collections import defaultdict
from collections.abc import Callable
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
    HF_YEAR_PATH,
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

    The defaults point at the published ice dataset. Every one of them is an
    argument so the same client can serve a second repository - predictors, for
    example - without subclassing or patching module constants.

    Args:
        repo (`str`, optional): Dataset repository id. Defaults to the AIICE ice dataset.
        repo_type (`str`, optional): Hugging Face repository type. Defaults to "dataset".
        start (`date`, optional): Earliest date the repository covers. Defaults to the ice dataset start.
        end (`date`, optional): Latest date the repository covers. Defaults to the ice dataset end.
        shape (`tuple[int, int]`, optional): Shape of a single matrix. Defaults to (432, 432).
        filename_template (`Callable[[date], str]`, optional): Maps a date to a path inside
            the repository. This is what makes the client fetch only the dates asked for,
            with no directory listing, so a second repository needs its own mapping.
            Defaults to the ice layout `global_series/<year>/osisaf_<YYYYMMDD>.npy`.
        year_path (`str`, optional): Template of the per-year folder used by `info()`,
            formatted with `year`. Defaults to `"global_series/{year}"`.
    """

    def __init__(
        self,
        repo: str = HF_DATASET_REPO,
        repo_type: str = HF_REPO_TYPE,
        start: date = MIN_DATASET_START,
        end: date = MAX_DATASET_END,
        shape: tuple[int, int] = DATASET_SHAPE,
        filename_template: Callable[[date], str] = get_filename_template,
        year_path: str = HF_YEAR_PATH,
    ):
        self._api_base_url = HF_BASE_URL
        self._api = HfApi(endpoint=self._api_base_url, library_name=HF_PACKAGE_NAME)
        self._api_headers = build_hf_headers(library_name=HF_PACKAGE_NAME)

        self._dataset_repo = repo
        self._dataset_repo_type = repo_type

        self._min_dataset_start, self._max_dataset_end = start, end
        self._shape = shape

        self._filename_template = filename_template
        self._year_path = year_path

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
            filenames.append(self._filename_template(current))
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
    def read_file_range(self, filename: str, offset: int, length: int) -> bytes | None:
        """
        Load a byte range of a dataset file into memory.

        Chunked layouts keep one file per month instead of one per day, which is
        what keeps the repository under the Hugging Face file-count recommendation.
        A ranged request costs the same as fetching a whole small file, so reading
        a single slice out of a chunk is no more expensive than the per-day layout.

        Args:
            filename (`str`): Relative path to the dataset file.
            offset (`int`): First byte to read, counted from the start of the file.
            length (`int`): Number of bytes to read.
        """
        if offset < 0 or length <= 0:
            raise ValueError(f"invalid range: {offset=}, {length=}")

        url = f"{self._api_base_url}/datasets/{self._dataset_repo}/resolve/main/{filename}"
        headers = dict(self._api_headers)
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"

        response = requests.get(url, headers=headers, timeout=DEFAULT_REQUEST_TIMEOUT)

        # ignore if file isn't found
        if response.status_code == 404:
            return None

        response.raise_for_status()

        # a 200 here means the server ignored the range and sent the whole file
        if response.status_code != 206:
            raise RuntimeError(
                f"Range request for {filename} was not honoured: status {response.status_code}"
            )

        if len(response.content) != length:
            raise RuntimeError(
                f"Range request for {filename} returned {len(response.content)} bytes, expected {length}"
            )

        return response.content

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
        year_path = self._year_path.format(year=year)
        url = f"{self._api_base_url}/api/datasets/{self._dataset_repo}/tree/main/{year_path}"

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
