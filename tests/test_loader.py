from datetime import date
from io import BytesIO
from unittest.mock import call, patch

import numpy as np
import pytest
import torch

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

        # shouldn't use callable objects as mock cause of thread race safety
        mock_data = {
            "a.npy": f"{self.test_local_dir}/a.npy",
            "b.npy": f"{self.test_local_dir}/b.npy",
            "c.npy": None,
        }
        mock_download.side_effect = lambda filename, local_dir: mock_data[filename]

        result = loader.download(
            local_dir=self.test_local_dir,
            start=date(2020, 1, 1),
            end=date(2020, 1, 3),
            step=2,
            threads=2,
        )

        assert result == [
            f"{self.test_local_dir}/a.npy",
            f"{self.test_local_dir}/b.npy",
            None,
        ]
        mock_get_filenames.assert_has_calls(
            [call(start=date(2020, 1, 1), end=date(2020, 1, 3), step=2)],
            any_order=False,
        )
        mock_download.assert_has_calls(
            [
                call(filename=f, local_dir=self.test_local_dir)
                for f in mock_get_filenames.return_value
            ],
            any_order=True,  # in higher python versions threadpool order can be not deterministic
        )

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
        buffer = BytesIO()
        np.save(buffer, np.array([[1, 2], [3, 4]]))
        self.fake_bytes = buffer.getvalue()

    @patch("aiice.loader.HfDatasetClient.read_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    @pytest.mark.parametrize(
        "tensor_out, expected_type",
        [
            (False, np.ndarray),
            (True, torch.Tensor),
        ],
    )
    def test_ok(
        self,
        mock_get_filenames,
        mock_read_file,
        tensor_out,
        expected_type,
        loader: Loader,
    ):
        mock_get_filenames.return_value = ["a.npy", "b.npy", "c.npy"]

        def fake_read(filename):
            return self.fake_bytes

        mock_read_file.side_effect = fake_read

        result = loader.get(threads=2, step=3, tensor_out=tensor_out)

        assert isinstance(result, expected_type)
        assert tuple(result.shape) == (3, 2, 2)

        expected_result = np.array([[[1, 2], [3, 4]]] * 3)
        if tensor_out:
            np.testing.assert_array_equal(result.numpy(), expected_result)
        else:
            np.testing.assert_array_equal(result, expected_result)

        mock_get_filenames.assert_has_calls(
            [call(start=None, end=None, step=3)],
            any_order=False,
        )
        mock_read_file.assert_has_calls(
            [call(filename=f) for f in mock_get_filenames.return_value],
            any_order=True,  # in higher python versions processpool order can be not deterministic
        )

    @patch("aiice.loader.HfDatasetClient.read_file")
    @patch("aiice.loader.HfDatasetClient.get_filenames")
    def test_not_found_raises(self, mock_get_filenames, mock_read_file, loader: Loader):
        mock_get_filenames.return_value = ["a.npy", "b.npy"]
        mock_read_file.side_effect = [self.fake_bytes, None]

        with pytest.raises(ValueError) as err:
            loader.get()
        assert "Remote file" in str(err.value)
