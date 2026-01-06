from datetime import date, timedelta
from io import BytesIO

import numpy as np
import requests
from huggingface_hub import HfApi
from huggingface_hub.errors import RemoteEntryNotFoundError
from huggingface_hub.file_download import http_get

from aiice.constants import (
    HF_BASE_URL,
    HF_DATASET_REPO,
    HF_REPO_TYPE,
    MAX_DATASET_END,
    MIN_DATASET_START,
    PACKAGE_NAME,
)


class HfDatasetClient:
    def __init__(self):
        self._api_base_url: str = HF_BASE_URL
        self._api: HfApi = HfApi(endpoint=self._api_base_url, library_name=PACKAGE_NAME)

        self._dataset_repo: str = HF_DATASET_REPO
        self._dataset_repo_type: str = HF_REPO_TYPE
        self._dataset_start: date = MIN_DATASET_START
        self._dataset_end: date = MAX_DATASET_END

    @property
    def dataset_start(self) -> date:
        return self._dataset_start

    @property
    def dataset_end(self) -> date:
        return self._dataset_end

    def get_filename_template(self, d: date) -> str:
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
        url: str = (
            f"{self._api_base_url}/datasets/{self._dataset_repo}/resolve/main/{filename}"
        )
        buffer = BytesIO()
        try:
            http_get(
                url=url,
                temp_file=buffer,
                displayed_filename=filename,
            )
            buffer.seek(0)
            return np.load(buffer)

        # ignore if file isn't found
        except RemoteEntryNotFoundError:
            return None

        except requests.RequestException as e:
            raise RuntimeError(f"Network error {url}") from e

        except ValueError as e:
            raise RuntimeError(f"Failed to decode npy file {url}") from e

    def download_file(self, filename: str, local_dir: str) -> str | None:
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
