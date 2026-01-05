from datetime import date, timedelta
from io import BytesIO

import numpy as np
import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import RemoteEntryNotFoundError

from aiice.constants import (
    _DATASET_END,
    _DATASET_START,
    _HF_API_READ_TIMEOUT,
    _HF_BASE_URL,
    _HF_DATASET_REPO,
    _HF_REPO_TYPE,
    _PACKAGE_NAME,
)


class HfDatasetClient:
    def __init__(self):
        self._api = HfApi(endpoint=_HF_BASE_URL, library_name=_PACKAGE_NAME)
        self._api_read_timeout = _HF_API_READ_TIMEOUT
        self._dataset_start = _DATASET_START
        self._dataset_end = _DATASET_END

    @property
    def dataset_start(self) -> date:
        return self._dataset_start

    @property
    def dataset_end(self) -> date:
        return self._dataset_end

    def get_filename_template(self, d: date):
        return f"global_series/{d.year}/osisaf_{d.year}{d.month:02d}{d.day:02d}.npy"

    def get_filenames(
        self, start: None | date = None, end: None | date = None
    ) -> list[str]:
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

        while current <= end:
            filenames.append(self.get_filename_template(current))
            current += timedelta(days=1)

        return filenames

    def read_file(self, filename: str) -> np.ndarray | None:
        url = f"{_HF_BASE_URL}/datasets/{_HF_DATASET_REPO}/resolve/main/{filename}"

        try:
            response = requests.get(url, timeout=self._api_read_timeout)

            # ignore if file isn't found
            if response.status_code == 404:
                return None

            response.raise_for_status()
            return np.load(BytesIO(response.content))

        except requests.RequestException as e:
            raise RuntimeError(f"Network error {url}") from e

        except ValueError as e:
            raise RuntimeError(f"Failed to decode npy file {url}") from e
        
    def download_file(self, filename: str, local_dir: str) -> str | None:
        try:
            return self._api.hf_hub_download(
                repo_id=_HF_DATASET_REPO,
                repo_type=_HF_REPO_TYPE,
                filename=filename,
                local_dir=local_dir,
            )

        # ignore if file isn't found
        except RemoteEntryNotFoundError:
            return None

        except Exception as e:
            raise e
