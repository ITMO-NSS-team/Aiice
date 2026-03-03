from datetime import date
from io import BytesIO
from unittest.mock import call, patch

import numpy as np
import pytest
import torch

from aiice.constants import (
    DATASET_SHAPE,
    MASK_SEA_DATA_MAX_VALUE,
    MASK_SEA_DATA_PATH,
    MASK_SEA_IDX_PATH,
)
from aiice.loader import Loader


def make_center_sea_mask(
    shape: tuple[int, int],
    sea_id: int,
    sea_size: tuple[int, int],
) -> np.ndarray:
    M, N = shape
    h, w = sea_size

    mask = np.zeros((M, N), dtype=np.float32)

    r_start = (M - h) // 2
    c_start = (N - w) // 2

    mask[r_start : r_start + h, c_start : c_start + w] = sea_id
    return mask


class BaseTestLoader:
    @pytest.fixture
    def loader(self):
        with (
            patch("aiice.loader.HfDatasetClient.read_file") as mock_read_file,
            patch("aiice.loader.Loader._decode_raw_matrix") as mock_decode_raw_matrix,
        ):
            mock_read_file.side_effect = [
                b"name,id\nBarents Sea,1\n",
                b"binary",
            ]

            mask = make_center_sea_mask(
                shape=DATASET_SHAPE,
                sea_id=1,
                sea_size=(DATASET_SHAPE[0] // 2, DATASET_SHAPE[1] // 2),
            )
            mask[0, 0] = MASK_SEA_DATA_MAX_VALUE
            mock_decode_raw_matrix.return_value = mask

            loader = Loader()

            assert mock_read_file.call_count == 2
            mock_read_file.assert_has_calls(
                [
                    call(filename=MASK_SEA_IDX_PATH),
                    call(filename=MASK_SEA_DATA_PATH),
                ]
            )
            return loader


class TestLoader_download(BaseTestLoader):
    test_local_dir = "/tmp"

    def test_ok(self, loader: Loader):
        with (
            patch("aiice.loader.HfDatasetClient.download_file") as mock_download_file,
            patch("aiice.loader.HfDatasetClient.get_filenames") as mock_get_filenames,
        ):
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
            mock_download_file.side_effect = lambda filename, local_dir: mock_data[
                filename
            ]

            result = loader.download(
                local_dir=self.test_local_dir,
                start="2020-01-01",
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
            mock_download_file.assert_has_calls(
                [
                    call(filename=f, local_dir=self.test_local_dir)
                    for f in mock_get_filenames.return_value
                ],
                any_order=True,  # in higher python versions threadpool order can be not deterministic
            )

    def test_empty_filenames(self, loader: Loader):
        with (
            patch("aiice.loader.HfDatasetClient.download_file") as mock_download_file,
            patch("aiice.loader.HfDatasetClient.get_filenames") as mock_get_filenames,
        ):
            mock_get_filenames.return_value = []

            result = loader.download(local_dir=self.test_local_dir, threads=4)

            assert result == []
            mock_get_filenames.assert_called_once()
            mock_download_file.assert_not_called()

    def test_single_file(self, loader: Loader):
        with (
            patch("aiice.loader.HfDatasetClient.download_file") as mock_download_file,
            patch("aiice.loader.HfDatasetClient.get_filenames") as mock_get_filenames,
        ):
            mock_get_filenames.return_value = ["only.npy"]
            mock_download_file.return_value = f"{self.test_local_dir}/only.npy"

            result = loader.download(self.test_local_dir)

            assert result == [f"{self.test_local_dir}/only.npy"]
            mock_get_filenames.assert_called_once()
            mock_download_file.assert_called_once()


class TestLoader_get(BaseTestLoader):
    def setup_method(self):
        buffer = BytesIO()

        # sed seed for CI test
        np.random.seed(42)

        # numpy matrix values are ints in range 0...100
        self.fake_matrix = np.random.randint(low=0, high=100, size=DATASET_SHAPE)

        np.save(buffer, self.fake_matrix)
        self.fake_bytes = buffer.getvalue()

    @pytest.mark.parametrize(
        "tensor_out, idx_out, expected_data_type",
        [
            (False, False, np.ndarray),
            (True, False, torch.Tensor),
            (False, True, np.ndarray),
            (True, True, torch.Tensor),
        ],
    )
    def test_ok(
        self,
        tensor_out,
        idx_out,
        expected_data_type,
        loader: Loader,
    ):
        with (
            patch("aiice.loader.get_date_from_filename_template") as mock_get_date_from,
            patch("aiice.loader.HfDatasetClient.read_file") as mock_read_file,
            patch("aiice.loader.HfDatasetClient.get_filenames") as mock_get_filenames,
        ):
            filenames = ["a.npy", "b.npy", "c.npy"]
            mock_get_filenames.return_value = filenames
            mock_read_file.side_effect = lambda filename: self.fake_bytes

            fake_dates = [date(2020, 1, i + 1) for i in range(len(filenames))]
            mock_get_date_from.side_effect = fake_dates

            result = loader.get(
                start="2020-01-01",
                threads=2,
                step=3,
                tensor_out=tensor_out,
                idx_out=idx_out,
            )

            expected_array = np.stack([self.fake_matrix] * 3).astype(np.float32) / 100.0

            if idx_out:
                dates, data = result
                assert dates == fake_dates
            else:
                data = result

            assert isinstance(data, expected_data_type)
            assert tuple(data.shape) == tuple([3, *DATASET_SHAPE])

            if tensor_out:
                np.testing.assert_array_equal(data.numpy(), expected_array)
            else:
                np.testing.assert_array_equal(data, expected_array)

            mock_get_filenames.assert_called_once_with(
                start=date(2020, 1, 1), end=None, step=3
            )
            mock_read_file.assert_has_calls(
                [call(filename=f) for f in mock_get_filenames.return_value],
                any_order=True,  # in higher python versions threadpool order can be not deterministic
            )

    def test_ok_sea(self, loader: Loader):
        with (
            patch("aiice.loader.get_date_from_filename_template") as mock_get_date_from,
            patch("aiice.loader.HfDatasetClient.read_file") as mock_read_file,
            patch("aiice.loader.HfDatasetClient.get_filenames") as mock_get_filenames,
        ):
            filenames = ["a.npy"]
            mock_get_filenames.return_value = filenames
            mock_read_file.side_effect = lambda filename: self.fake_bytes
            mock_get_date_from.side_effect = [date(2020, 1, 1)]

            result = loader.get(
                sea="Barents Sea",
                start="2020-01-01",
                threads=2,
            )

            assert isinstance(result, np.ndarray)
            assert result.shape == (
                1,
                DATASET_SHAPE[0] // 2,
                DATASET_SHAPE[1] // 2,
            )  # check fixture mocks in BaseTestLoader

            mock_get_filenames.assert_called_once_with(
                start=date(2020, 1, 1),
                end=None,
                step=None,
            )
            mock_read_file.assert_has_calls(
                [call(filename=f) for f in mock_get_filenames.return_value],
            )

    def test_not_found_raises(self, loader: Loader):
        with (
            patch("aiice.loader.HfDatasetClient.read_file") as mock_read_file,
            patch("aiice.loader.HfDatasetClient.get_filenames") as mock_get_filenames,
        ):
            mock_get_filenames.return_value = ["a.npy", "b.npy"]
            mock_read_file.side_effect = [self.fake_bytes, None]

            with pytest.raises(ValueError) as err:
                loader.get()
                assert "Remote file" in str(err.value)

    def test_no_sea_raises(self, loader: Loader):
        with pytest.raises(ValueError) as err:
            loader.get(sea="Dummy Sea")
            assert f"No such sea. Check available options: {loader.seas}" in str(
                err.value
            )
