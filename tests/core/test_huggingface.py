from datetime import date, timedelta
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
import pytest
import requests
from huggingface_hub.errors import RemoteEntryNotFoundError

from aiice.constants import (
    _HF_API_READ_TIMEOUT,
    _HF_BASE_URL,
    _HF_DATASET_REPO,
    _HF_REPO_TYPE,
)
from aiice.core.huggingface import HfDatasetClient


class BaseTestHfDatasetClient:
    @pytest.fixture
    def client(self) -> HfDatasetClient:
        return HfDatasetClient()


class TestHfDatasetClient_get_filename_template(BaseTestHfDatasetClient):

    @pytest.mark.parametrize(
        "value, expected",
        [
            (date(2021, 6, 1), "global_series/2021/osisaf_20210601.npy"),
            (date(1991, 12, 12), "global_series/1991/osisaf_19911212.npy"),
        ],
    )
    def test(self, client, value, expected):
        assert client.get_filename_template(value) == expected


class TestHfDatasetClient_get_filenames(BaseTestHfDatasetClient):

    def test_full_range(self, client):
        files = client.get_filenames()
        expected_len = (client.dataset_end - client.dataset_start).days + 1

        assert len(files) == expected_len
        assert files[0] == client.get_filename_template(client.dataset_start)
        assert files[-1] == client.get_filename_template(client.dataset_end)

    def test_start_only(self, client):
        start = client.dataset_start + timedelta(days=10)
        files = client.get_filenames(start=start)

        assert files[0] == client.get_filename_template(start)
        assert files[-1] == client.get_filename_template(client.dataset_end)

    def test_end_only(self, client):
        end = client.dataset_start + timedelta(days=10)
        files = client.get_filenames(end=end)

        assert files[0] == client.get_filename_template(client.dataset_start)
        assert files[-1] == client.get_filename_template(end)

    def test_start_and_end(self, client):
        start, end = date(2020, 1, 1), date(2020, 1, 5)
        files = client.get_filenames(start=start, end=end)

        assert len(files) == 5
        assert files[0] == client.get_filename_template(start)
        assert files[-1] == client.get_filename_template(end)

    def test_single_day_range(self, client):
        day = date(2021, 6, 15)
        files = client.get_filenames(start=day, end=day)

        assert files == [client.get_filename_template(day)]

    def test_start_or_end_equals_defaults(self, client):
        files = client.get_filenames(start=client.dataset_start)
        assert files[0] == client.get_filename_template(client.dataset_start)

        files = client.get_filenames(end=client.dataset_end)
        assert files[-1] == client.get_filename_template(client.dataset_end)

    def test_start_before_dataset_start_raises(self, client):
        with pytest.raises(ValueError):
            client.get_filenames(start=client.dataset_start - timedelta(days=1))

    def test_end_after_dataset_end_raises(self, client):
        with pytest.raises(ValueError):
            client.get_filenames(end=client.dataset_end + timedelta(days=1))

    def test_start_after_end_raises(self, client):
        with pytest.raises(ValueError):
            client.get_filenames(start=date(2022, 1, 10), end=date(2022, 1, 1))


class TestHfDatasetClient_read_file(BaseTestHfDatasetClient):

    @patch("aiice.core.huggingface.requests.get")
    def test_ok(self, mock_get, client):
        arr = np.array([[1, 2], [3, 4]])
        buf = BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = buf.getvalue()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = client.read_file("dummy.npy")

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
        mock_get.assert_called_once_with(
            f"{_HF_BASE_URL}/datasets/{_HF_DATASET_REPO}/resolve/main/dummy.npy",
            timeout=_HF_API_READ_TIMEOUT,
        )

    @patch("aiice.core.huggingface.requests.get")
    def test_file_not_found(self, mock_get, client):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = client.read_file("missing.npy")
        assert result is None

    @patch("aiice.core.huggingface.requests.get")
    def test_network_error(self, mock_get, client):
        mock_get.side_effect = requests.exceptions.RequestException("network down")

        with pytest.raises(RuntimeError) as err:
            client.read_file("dummy.npy")
        assert "Network error" in str(err.value)

    @patch("aiice.core.huggingface.requests.get")
    def test_invalid_npy(self, mock_get, client):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"not a npy file"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError) as err:
            client.read_file("dummy.npy")
        assert "Failed to decode npy file" in str(err.value)


class TestHfDatasetClient_download_file(BaseTestHfDatasetClient):

    @patch("aiice.core.huggingface.HfApi.hf_hub_download")
    def test_ok(self, mock_download, client):
        mock_download.return_value = "/tmp/dummy.npy"
        result = client.download_file("dummy.npy", "/tmp")

        assert result == "/tmp/dummy.npy"
        mock_download.assert_called_once_with(
            repo_id=_HF_DATASET_REPO,
            repo_type=_HF_REPO_TYPE,
            filename="dummy.npy",
            local_dir="/tmp",
        )

    @patch("aiice.core.huggingface.HfApi.hf_hub_download")
    def test_file_not_found(self, mock_download, client):
        mock_download.side_effect = RemoteEntryNotFoundError(
            "not found", response=requests.Response()
        )
        result = client.download_file("missing.npy", "/tmp")

        assert result is None

    @patch("aiice.core.huggingface.HfApi.hf_hub_download")
    def test_other_exception(self, mock_download, client):
        mock_download.side_effect = ValueError("bad value")

        with pytest.raises(ValueError) as err:
            client.download_file("dummy.npy", "/tmp")
        assert "bad value" in str(err.value)
