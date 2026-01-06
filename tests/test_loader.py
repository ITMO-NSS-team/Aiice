from datetime import date
from unittest.mock import call, patch

import numpy as np
import pytest

from aiice.loader import Loader


class BaseTestLoader:
    @pytest.fixture
    def loader(self) -> Loader:
        return Loader()


class TestLoader_download(BaseTestLoader):
    test_local_dir = "/tmp"

    @patch("aiice.loader.HfDatasetClient.download_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_ok(self, mock_get_filenames, mock_download, loader: Loader):
        mock_get_filenames.return_value = [
            "a.npy",
            "b.npy",
            "c.npy",
        ]
        mock_download.side_effect = [
            f"{self.test_local_dir}/a.npy",
            f"{self.test_local_dir}/b.npy",
            None,
        ]

        result = loader.download(
            local_dir=self.test_local_dir,
            start=date(2020, 1, 1),
            end=date(2020, 1, 3),
            threads=2,
        )

        assert result == [
            f"{self.test_local_dir}/a.npy",
            f"{self.test_local_dir}/b.npy",
            None,
        ]
        mock_get_filenames.assert_called_once()
        mock_download.assert_has_calls(
            [
                call(filename=f, local_dir=self.test_local_dir)
                for f in mock_get_filenames.return_value
            ],
            any_order=False,
        )
        mock_get_filenames.assert_called_once()

    @patch("aiice.loader.HfDatasetClient.download_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_empty_filenames(self, mock_get_filenames, mock_download, loader: Loader):
        mock_get_filenames.return_value = []

        result = loader.download(local_dir=self.test_local_dir, threads=4)

        assert result == []
        mock_get_filenames.assert_called_once()
        mock_download.assert_not_called()

    @patch("aiice.loader.HfDatasetClient.download_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_single_file(self, mock_get_filenames, mock_download, loader: Loader):
        mock_get_filenames.return_value = ["only.npy"]
        mock_download.return_value = f"{self.test_local_dir}/only.npy"

        result = loader.download(self.test_local_dir)

        assert result == [f"{self.test_local_dir}/only.npy"]
        mock_get_filenames.assert_called_once()
        mock_download.assert_called_once()


class TestLoader_get(BaseTestLoader):
    def setup_method(self):
        self.fake_data = np.array([[1, 2], [3, 4]])

    @patch("aiice.loader.HfDatasetClient.read_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_ok(self, mock_get_filenames, mock_read_file, loader: Loader):
        mock_get_filenames.return_value = ["a.npy", "b.npy", "c.npy"]
        mock_read_file.side_effect = [self.fake_data] * 3

        result = loader.get(threads=2)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2, 2)
        mock_read_file.assert_has_calls(
            [call(filename=f) for f in mock_get_filenames.return_value], any_order=False
        )
        mock_get_filenames.assert_called_once()

    @patch("aiice.loader.HfDatasetClient.read_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_split_train_test(self, mock_get_filenames, mock_read_file, loader: Loader):
        mock_get_filenames.return_value = ["a.npy", "b.npy", "c.npy", "d.npy"]
        mock_read_file.side_effect = [self.fake_data] * 4

        train, test = loader.get(threads=2, test_size=0.25)

        # train = first 3 matrices, test = last one matrix
        assert train.shape == (3, 2, 2)
        assert test.shape == (1, 2, 2)
        mock_read_file.assert_has_calls(
            [call(filename=f) for f in mock_get_filenames.return_value],
            any_order=False,
        )
        mock_get_filenames.assert_called_once()

    @patch("aiice.loader.HfDatasetClient.read_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_test_size_0_or_1(self, mock_get_filenames, mock_read_file, loader: Loader):
        mock_get_filenames.return_value = ["a.npy", "b.npy"]
        mock_read_file.side_effect = lambda filename: self.fake_data

        train, test = loader.get(test_size=0.0)
        assert train.shape == (2, 2, 2)
        assert test.size == 0

        train, test = loader.get(threads=1, test_size=1.0)
        assert train.size == 0
        assert test.shape == (2, 2, 2)

    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_invalid_test_size_raises(self, mock_get_filenames, loader: Loader):
        mock_get_filenames.return_value = ["a.npy", "b.npy"]

        with pytest.raises(ValueError) as err:
            loader.get(test_size=-0.1)
        assert "Test size should be between" in str(err.value)

        with pytest.raises(ValueError) as err:
            loader.get(test_size=1.1)
        assert "Test size should be between" in str(err.value)

    @patch("aiice.loader.HfDatasetClient.read_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_not_found_raises(self, mock_get_filenames, mock_read_file, loader: Loader):
        mock_get_filenames.return_value = ["a.npy", "b.npy"]
        mock_read_file.side_effect = [self.fake_data, None]

        with pytest.raises(ValueError) as err:
            loader.get()
        assert "Remote file" in str(err.value)
