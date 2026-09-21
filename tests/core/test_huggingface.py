from datetime import date, timedelta
from io import BytesIO
from unittest.mock import ANY, patch

import pytest
import requests
from huggingface_hub.errors import RemoteEntryNotFoundError

from aiice.constants import (
    BYTES_IN_MB,
    DATASET_SHAPE,
    HF_BASE_URL,
    HF_DATASET_REPO,
    HF_REPO_TYPE,
    KEY_FILES,
    KEY_PER_YEAR,
    KEY_SIZE_BYTES,
    KEY_SIZE_MB,
    MAX_DATASET_END,
    MIN_DATASET_START,
)
from aiice.core.huggingface import HfDatasetClient
from aiice.core.utils import get_filename_template


class BaseTestHfDatasetClient:
    @pytest.fixture
    def client(self) -> HfDatasetClient:
        return HfDatasetClient()


class TestHfDatasetClient_configuration:
    """
    The defaults describe the ice dataset; every one of them is overridable so a
    second repository can be read through the same client.
    """

    def test_defaults_match_constants(self):
        client = HfDatasetClient()

        assert client.dataset_start == MIN_DATASET_START
        assert client.dataset_end == MAX_DATASET_END
        assert client.shape == DATASET_SHAPE
        assert client.get_filenames(start=date(2020, 1, 1), end=date(2020, 1, 1)) == [
            get_filename_template(date(2020, 1, 1))
        ]

    def test_custom_repo_is_used_by_read_file(self):
        client = HfDatasetClient(repo="ITMO-NSS/Aiice-predictors")

        with patch("aiice.core.huggingface.http_get") as mock_http_get:
            mock_http_get.side_effect = (
                lambda url, temp_file, **kwargs: temp_file.write(b"x")
            )
            client.read_file("t2m/2020/t2m_20200101.npy")

            mock_http_get.assert_called_once_with(
                url=(
                    f"{HF_BASE_URL}/datasets/ITMO-NSS/Aiice-predictors"
                    "/resolve/main/t2m/2020/t2m_20200101.npy"
                ),
                temp_file=ANY,
                displayed_filename="t2m/2020/t2m_20200101.npy",
                headers=ANY,
            )

    def test_custom_repo_is_used_by_download_file(self):
        client = HfDatasetClient(repo="ITMO-NSS/Aiice-predictors")

        with patch(
            "aiice.core.huggingface.HfApi.hf_hub_download"
        ) as mock_hf_hub_download:
            mock_hf_hub_download.return_value = "/tmp/x.npy"
            client.download_file("x.npy", "/tmp")

            mock_hf_hub_download.assert_called_once_with(
                repo_id="ITMO-NSS/Aiice-predictors",
                repo_type=HF_REPO_TYPE,
                filename="x.npy",
                local_dir="/tmp",
            )

    def test_custom_filename_template(self):
        client = HfDatasetClient(
            start=date(2020, 1, 1),
            end=date(2020, 1, 3),
            filename_template=lambda d: f"t2m/{d.year}/t2m_{d:%Y%m%d}.npy",
        )

        assert client.get_filenames() == [
            "t2m/2020/t2m_20200101.npy",
            "t2m/2020/t2m_20200102.npy",
            "t2m/2020/t2m_20200103.npy",
        ]

    def test_custom_date_bounds_are_enforced(self):
        client = HfDatasetClient(start=date(2000, 1, 1), end=date(2000, 12, 31))

        assert client.dataset_start == date(2000, 1, 1)
        assert client.dataset_end == date(2000, 12, 31)

        with pytest.raises(ValueError):
            client.get_filenames(start=date(1999, 12, 31))

        with pytest.raises(ValueError):
            client.get_filenames(end=date(2001, 1, 1))

    def test_custom_shape(self):
        client = HfDatasetClient(shape=(64, 64))
        assert client.shape == (64, 64)

    def test_custom_year_path_used_by_stats(self):
        client = HfDatasetClient(
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            year_path="t2m/{year}",
        )

        with patch("aiice.core.huggingface.requests.get") as mock_get:
            mock_get.return_value.json.return_value = [
                {"type": "file", "size": 10},
                {"type": "directory", "size": 999},
            ]
            mock_get.return_value.raise_for_status = lambda: None

            year, files, size = client._fetch_year_stats(2020)

            assert (year, files, size) == (2020, 1, 10)
            called_url = mock_get.call_args.args[0]
            assert called_url.endswith("/tree/main/t2m/2020")

    def test_two_clients_do_not_share_state(self):
        ice = HfDatasetClient()
        other = HfDatasetClient(
            repo="ITMO-NSS/Aiice-predictors",
            start=date(2000, 1, 1),
            end=date(2000, 1, 2),
        )

        assert ice.dataset_start == MIN_DATASET_START
        assert other.dataset_start == date(2000, 1, 1)
        assert len(other.get_filenames()) == 2


class TestHfDatasetClient_get_filenames(BaseTestHfDatasetClient):
    def test_full_range(self, client: HfDatasetClient):
        files = client.get_filenames()
        expected_len = (client.dataset_end - client.dataset_start).days + 1

        assert len(files) == expected_len
        assert files[0] == get_filename_template(client.dataset_start)
        assert files[-1] == get_filename_template(client.dataset_end)

    def test_start_only(self, client: HfDatasetClient):
        start = client.dataset_start + timedelta(days=10)
        files = client.get_filenames(start=start)

        assert files[0] == get_filename_template(start)
        assert files[-1] == get_filename_template(client.dataset_end)

    def test_end_only(self, client: HfDatasetClient):
        end = client.dataset_start + timedelta(days=10)
        files = client.get_filenames(end=end)

        assert files[0] == get_filename_template(client.dataset_start)
        assert files[-1] == get_filename_template(end)

    def test_start_and_end(self, client: HfDatasetClient):
        start, end = date(2020, 1, 1), date(2020, 1, 5)
        files = client.get_filenames(start=start, end=end)

        assert len(files) == 5
        assert files[0] == get_filename_template(start)
        assert files[-1] == get_filename_template(end)

    @pytest.mark.parametrize(
        "start, end, step, expected_dates",
        [
            (
                date(2020, 1, 30),
                date(2020, 2, 5),
                3,
                [date(2020, 1, 30), date(2020, 2, 2), date(2020, 2, 5)],
            ),
            (
                date(2020, 1, 1),
                date(2020, 1, 10),
                "2d",
                [
                    date(2020, 1, 1),
                    date(2020, 1, 3),
                    date(2020, 1, 5),
                    date(2020, 1, 7),
                    date(2020, 1, 9),
                ],
            ),
            (
                date(2020, 1, 1),
                date(2020, 1, 15),
                "1w",
                [date(2020, 1, 1), date(2020, 1, 8), date(2020, 1, 15)],
            ),
            (
                date(2019, 12, 31),
                date(2020, 4, 30),
                "1m",
                [
                    date(2019, 12, 31),
                    date(2020, 1, 31),
                    date(2020, 2, 29),
                    date(2020, 3, 31),
                    date(2020, 4, 30),
                ],
            ),
            (
                date(2019, 12, 30),
                date(2020, 4, 30),
                "2m",
                [date(2019, 12, 30), date(2020, 2, 29), date(2020, 4, 30)],
            ),
            (
                date(2020, 1, 1),
                date(2022, 1, 1),
                "1y",
                [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
            ),
        ],
    )
    def test_step(self, client: HfDatasetClient, start, end, step, expected_dates):
        files = client.get_filenames(start=start, end=end, step=step)

        assert len(files) == len(expected_dates)
        for i, expected_date in enumerate(expected_dates):
            assert files[i] == get_filename_template(expected_date)

    def test_single_day_range(self, client: HfDatasetClient):
        day = date(2021, 6, 15)
        files = client.get_filenames(start=day, end=day)

        assert files == [get_filename_template(day)]

    def test_start_or_end_equals_defaults(self, client: HfDatasetClient):
        files = client.get_filenames(start=client.dataset_start)
        assert files[0] == get_filename_template(client.dataset_start)

        files = client.get_filenames(end=client.dataset_end)
        assert files[-1] == get_filename_template(client.dataset_end)

    def test_start_before_dataset_start_raises(self, client: HfDatasetClient):
        with pytest.raises(ValueError) as err:
            client.get_filenames(start=client.dataset_start - timedelta(days=1))
            assert "date start value should be > " in str(err.value)

    def test_end_after_dataset_end_raises(self, client: HfDatasetClient):
        with pytest.raises(ValueError) as err:
            client.get_filenames(end=client.dataset_end + timedelta(days=1))
            assert "date end value should be <" in str(err.value)

    def test_start_after_end_raises(self, client: HfDatasetClient):
        with pytest.raises(ValueError) as err:
            client.get_filenames(start=date(2022, 1, 10), end=date(2022, 1, 1))
            assert "start date must be <= date end" in str(err.value)


class TestHfDatasetClient_read_file(BaseTestHfDatasetClient):
    def test_ok(self, client: HfDatasetClient):
        with patch("aiice.core.huggingface.http_get") as mock_http_get:
            file_value = b"content doesn't matter"
            buf = BytesIO(file_value)
            mock_http_get.side_effect = (
                lambda url, temp_file, **kwargs: temp_file.write(buf.getvalue())
            )

            result = client.read_file("dummy.npy")

            assert isinstance(result, bytes)
            assert result == file_value
            mock_http_get.assert_called_once_with(
                url=f"{HF_BASE_URL}/datasets/{HF_DATASET_REPO}/resolve/main/dummy.npy",
                temp_file=ANY,
                displayed_filename="dummy.npy",
                headers=ANY,
            )

    def test_file_not_found(self, client: HfDatasetClient):
        with patch("aiice.core.huggingface.http_get") as mock_http_get:
            mock_http_get.side_effect = RemoteEntryNotFoundError(
                "not found", response=requests.Response()
            )

            result = client.read_file("missing.npy")
            assert result is None

    def test_network_error(self, client: HfDatasetClient):
        with patch("aiice.core.huggingface.http_get") as mock_http_get:
            mock_http_get.side_effect = requests.RequestException(
                "network down", response=requests.Response()
            )

            with pytest.raises(RuntimeError) as err:
                client.read_file("dummy.npy")
                assert "Network error" in str(err.value)


class TestHfDatasetClient_download_file(BaseTestHfDatasetClient):
    def test_ok(self, client: HfDatasetClient):
        with patch(
            "aiice.core.huggingface.HfApi.hf_hub_download"
        ) as mock_hf_hub_download:
            mock_hf_hub_download.return_value = "/tmp/dummy.npy"
            result = client.download_file("dummy.npy", "/tmp")

            assert result == "/tmp/dummy.npy"
            mock_hf_hub_download.assert_called_once_with(
                repo_id=HF_DATASET_REPO,
                repo_type=HF_REPO_TYPE,
                filename="dummy.npy",
                local_dir="/tmp",
            )

    def test_file_not_found(self, client: HfDatasetClient):
        with patch(
            "aiice.core.huggingface.HfApi.hf_hub_download"
        ) as mock_hf_hub_download:
            mock_hf_hub_download.side_effect = RemoteEntryNotFoundError(
                "not found", response=requests.Response()
            )
            result = client.download_file("missing.npy", "/tmp")

            assert result is None

    def test_other_exception(self, client: HfDatasetClient):
        with patch(
            "aiice.core.huggingface.HfApi.hf_hub_download"
        ) as mock_hf_hub_download:
            mock_hf_hub_download.side_effect = ValueError("bad value")

            with pytest.raises(RuntimeError) as err:
                client.download_file("dummy.npy", "/tmp")
                assert "Failed to download file" in str(err.value)


class TestHfDatasetClient_info(BaseTestHfDatasetClient):
    def test_info_without_per_year(self, client: HfDatasetClient):
        with patch.object(
            HfDatasetClient, "_fetch_year_stats"
        ) as mock_fetch_year_stats:
            mock_fetch_year_stats.side_effect = lambda year: (
                year,
                year - 1999,
                (year - 1999) * 1000,
            )

            result = client.info(per_year=False, threads=2)

            expected_total_files = sum(
                year - 1999
                for year in range(
                    client.dataset_start.year, client.dataset_end.year + 1
                )
            )
            expected_total_size = sum(
                (year - 1999) * 1000
                for year in range(
                    client.dataset_start.year, client.dataset_end.year + 1
                )
            )

            assert result[f"total_{KEY_FILES}"] == expected_total_files
            assert result[f"total_{KEY_SIZE_BYTES}"] == expected_total_size
            assert result[f"total_{KEY_SIZE_MB}"] == round(
                expected_total_size / BYTES_IN_MB, 2
            )
            assert KEY_PER_YEAR not in result

    def test_info_with_per_year(self, client: HfDatasetClient):
        with patch.object(
            HfDatasetClient, "_fetch_year_stats"
        ) as mock_fetch_year_stats:
            mock_fetch_year_stats.side_effect = lambda year: (
                year,
                year - 1999,
                (year - 1999) * 1000,
            )

            result = client.info(per_year=True, threads=2)
            per_year = result[KEY_PER_YEAR]

            for year in range(client.dataset_start.year, client.dataset_end.year + 1):
                assert per_year[year][KEY_FILES] == year - 1999
                assert per_year[year][KEY_SIZE_BYTES] == (year - 1999) * 1000
                assert per_year[year][KEY_SIZE_MB] == round(
                    (year - 1999) * 1000 / BYTES_IN_MB, 2
                )
