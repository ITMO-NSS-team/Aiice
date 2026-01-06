from datetime import date, timedelta
from io import BytesIO
from unittest.mock import ANY, patch

import numpy as np
import pytest
import requests
from huggingface_hub.errors import RemoteEntryNotFoundError

from aiice.constants import HF_BASE_URL, HF_DATASET_REPO, HF_REPO_TYPE
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
    def test_ok(self, client: HfDatasetClient, value, expected):
        assert client.get_filename_template(value) == expected


class TestHfDatasetClient_get_filenames(BaseTestHfDatasetClient):

    def test_full_range(self, client: HfDatasetClient):
        files = client.get_filenames()
        expected_len = (client.dataset_end - client.dataset_start).days + 1

        assert len(files) == expected_len
        assert files[0] == client.get_filename_template(client.dataset_start)
        assert files[-1] == client.get_filename_template(client.dataset_end)

    def test_start_only(self, client: HfDatasetClient):
        start = client.dataset_start + timedelta(days=10)
        files = client.get_filenames(start=start)

        assert files[0] == client.get_filename_template(start)
        assert files[-1] == client.get_filename_template(client.dataset_end)

    def test_end_only(self, client: HfDatasetClient):
        end = client.dataset_start + timedelta(days=10)
        files = client.get_filenames(end=end)

        assert files[0] == client.get_filename_template(client.dataset_start)
        assert files[-1] == client.get_filename_template(end)

    def test_start_and_end(self, client: HfDatasetClient):
        start, end = date(2020, 1, 1), date(2020, 1, 5)
        files = client.get_filenames(start=start, end=end)

        assert len(files) == 5
        assert files[0] == client.get_filename_template(start)
        assert files[-1] == client.get_filename_template(end)

    def test_single_day_range(self, client: HfDatasetClient):
        day = date(2021, 6, 15)
        files = client.get_filenames(start=day, end=day)

        assert files == [client.get_filename_template(day)]

    def test_start_or_end_equals_defaults(self, client: HfDatasetClient):
        files = client.get_filenames(start=client.dataset_start)
        assert files[0] == client.get_filename_template(client.dataset_start)

        files = client.get_filenames(end=client.dataset_end)
        assert files[-1] == client.get_filename_template(client.dataset_end)

    def test_start_before_dataset_start_raises(self, client: HfDatasetClient):
        with pytest.raises(ValueError):
            client.get_filenames(start=client.dataset_start - timedelta(days=1))

    def test_end_after_dataset_end_raises(self, client: HfDatasetClient):
        with pytest.raises(ValueError):
            client.get_filenames(end=client.dataset_end + timedelta(days=1))

    def test_start_after_end_raises(self, client: HfDatasetClient):
        with pytest.raises(ValueError):
            client.get_filenames(start=date(2022, 1, 10), end=date(2022, 1, 1))


class TestHfDatasetClient_read_file(BaseTestHfDatasetClient):

    @patch("aiice.core.huggingface.http_get")
    def test_ok(self, mock_http_get, client: HfDatasetClient):
        arr = np.array([[1, 2], [3, 4]])
        buf = BytesIO()
        np.save(buf, arr)
        buf.seek(0)

        def fake_http_get(url, temp_file, **kwargs):
            temp_file.write(buf.getvalue())

        mock_http_get.side_effect = fake_http_get

        result = client.read_file("dummy.npy")

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
        mock_http_get.assert_called_once_with(
            url=f"{HF_BASE_URL}/datasets/{HF_DATASET_REPO}/resolve/main/dummy.npy",
            temp_file=ANY,
            displayed_filename="dummy.npy",
        )

    @patch("aiice.core.huggingface.http_get")
    def test_file_not_found(self, mock_http_get, client: HfDatasetClient):
        mock_http_get.side_effect = RemoteEntryNotFoundError(
            "not found", response=requests.Response()
        )

        result = client.read_file("missing.npy")
        assert result is None

    @patch("aiice.core.huggingface.http_get")
    def test_network_error(self, mock_http_get, client: HfDatasetClient):
        mock_http_get.side_effect = requests.RequestException(
            "network down", response=requests.Response()
        )

        with pytest.raises(RuntimeError) as err:
            client.read_file("dummy.npy")
        assert "Network error" in str(err.value)

    @patch("aiice.core.huggingface.http_get")
    def test_invalid_npy(self, mock_http_get, client: HfDatasetClient):
        def fake_http_get(url, temp_file, **kwargs):
            temp_file.write(b"not a npy file")

        mock_http_get.side_effect = fake_http_get

        with pytest.raises(RuntimeError) as err:
            client.read_file("dummy.npy")
        assert "Failed to decode npy file" in str(err.value)


class TestHfDatasetClient_download_file(BaseTestHfDatasetClient):

    @patch("aiice.core.huggingface.HfApi.hf_hub_download")
    def test_ok(self, mock_download, client: HfDatasetClient):
        mock_download.return_value = "/tmp/dummy.npy"
        result = client.download_file("dummy.npy", "/tmp")

        assert result == "/tmp/dummy.npy"
        mock_download.assert_called_once_with(
            repo_id=HF_DATASET_REPO,
            repo_type=HF_REPO_TYPE,
            filename="dummy.npy",
            local_dir="/tmp",
        )

    @patch("aiice.core.huggingface.HfApi.hf_hub_download")
    def test_file_not_found(self, mock_download, client: HfDatasetClient):
        mock_download.side_effect = RemoteEntryNotFoundError(
            "not found", response=requests.Response()
        )
        result = client.download_file("missing.npy", "/tmp")

        assert result is None

    @patch("aiice.core.huggingface.HfApi.hf_hub_download")
    def test_other_exception(self, mock_download, client: HfDatasetClient):
        mock_download.side_effect = ValueError("bad value")

        with pytest.raises(RuntimeError) as err:
            client.download_file("dummy.npy", "/tmp")
        assert "Failed to download file" in str(err.value)
