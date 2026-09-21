import json
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

from aiice.constants import DATASET_SHAPE, NPY_HEADER_BYTES
from aiice.predictors import PredictorLoader

SLICE_BYTES = DATASET_SHAPE[0] * DATASET_SHAPE[1] * 4


def manifest(step="daily", chunks=None):
    return {
        "variable": "t2m",
        "step": step,
        "shape": list(DATASET_SHAPE),
        "dtype": "float32",
        "npy_header_bytes": NPY_HEADER_BYTES,
        "slice_bytes": SLICE_BYTES,
        "chunks": chunks or [],
    }


DAILY_MANIFEST = manifest(
    chunks=[
        {
            "path": "t2m/2025/t2m_202501.npy",
            "period": "2025-01",
            "slices": 31,
            "first": "2025-01-01",
            "last": "2025-01-31",
            "complete_month": True,
        },
        {
            "path": "t2m/2025/t2m_202502.npy",
            "period": "2025-02",
            "slices": 10,  # tail month that stops early
            "first": "2025-02-01",
            "last": "2025-02-10",
            "complete_month": False,
        },
    ]
)

MONTHLY_MANIFEST = manifest(
    step="monthly",
    chunks=[
        {
            "path": "ohc300/ohc300_2025.npy",
            "period": "2025",
            "slices": 12,
            "first": "2025-01",
            "last": "2025-12",
            "complete_year": True,
        }
    ],
)


def frame(value: float) -> np.ndarray:
    return np.full(DATASET_SHAPE, value, dtype=np.float32)


class BaseTestPredictorLoader:
    @pytest.fixture
    def loader(self):
        return PredictorLoader(threads=2)

    @staticmethod
    def _manifest_bytes(m) -> bytes:
        return json.dumps(m).encode("utf-8")


class TestPredictorLoader_manifest(BaseTestPredictorLoader):
    def test_manifest_is_fetched_once(self, loader):
        with patch("aiice.predictors.HfDatasetClient.read_file") as mock_read:
            mock_read.return_value = self._manifest_bytes(DAILY_MANIFEST)

            assert loader.step("t2m") == "daily"
            assert loader.step("t2m") == "daily"
            assert loader.coverage("t2m") == (date(2025, 1, 1), date(2025, 2, 10))

            mock_read.assert_called_once_with("t2m/manifest.json")

    def test_missing_manifest_raises(self, loader):
        with patch("aiice.predictors.HfDatasetClient.read_file") as mock_read:
            mock_read.return_value = None
            with pytest.raises(ValueError) as exc:
                loader.manifest("nope")
            assert "No manifest" in str(exc.value)

    def test_common_coverage_takes_the_narrowest(self, loader):
        short = manifest(
            chunks=[
                {
                    "path": "sit/2025/sit_202501.npy",
                    "period": "2025-01",
                    "slices": 20,
                    "first": "2025-01-01",
                    "last": "2025-01-20",
                    "complete_month": False,
                }
            ]
        )
        with patch("aiice.predictors.HfDatasetClient.read_file") as mock_read:
            mock_read.side_effect = [
                self._manifest_bytes(DAILY_MANIFEST),
                self._manifest_bytes(short),
            ]
            assert loader.common_coverage(["t2m", "sit"]) == (
                date(2025, 1, 1),
                date(2025, 1, 20),
            )


class TestPredictorLoader_get(BaseTestPredictorLoader):
    def test_offsets_and_shape(self, loader):
        wanted = [date(2025, 1, 1), date(2025, 1, 15), date(2025, 2, 3)]
        calls = []

        def fake_range(filename, offset, length):
            calls.append((filename, offset, length))
            index = (offset - NPY_HEADER_BYTES) // SLICE_BYTES
            return frame(float(index)).tobytes()

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.return_value = self._manifest_bytes(DAILY_MANIFEST)
            out = loader.get(wanted, ["t2m"])

        assert out.shape == (3, 1, *DATASET_SHAPE)
        assert out.dtype == np.float32

        # January 1 is index 0, January 15 is index 14, February 3 is index 2
        assert sorted(calls) == sorted(
            [
                ("t2m/2025/t2m_202501.npy", NPY_HEADER_BYTES, SLICE_BYTES),
                (
                    "t2m/2025/t2m_202501.npy",
                    NPY_HEADER_BYTES + 14 * SLICE_BYTES,
                    SLICE_BYTES,
                ),
                (
                    "t2m/2025/t2m_202502.npy",
                    NPY_HEADER_BYTES + 2 * SLICE_BYTES,
                    SLICE_BYTES,
                ),
            ]
        )
        np.testing.assert_array_equal(out[0, 0], frame(0.0))
        np.testing.assert_array_equal(out[1, 0], frame(14.0))
        np.testing.assert_array_equal(out[2, 0], frame(2.0))

    def test_order_follows_the_requested_dates(self, loader):
        wanted = [date(2025, 1, 20), date(2025, 1, 2), date(2025, 1, 11)]

        def fake_range(filename, offset, length):
            return frame(float((offset - NPY_HEADER_BYTES) // SLICE_BYTES)).tobytes()

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.return_value = self._manifest_bytes(DAILY_MANIFEST)
            out = loader.get(wanted, ["t2m"])

        # the loader must not sort: the ice series defines the order
        assert [out[i, 0, 0, 0] for i in range(3)] == [19.0, 1.0, 10.0]

    def test_channels_follow_variable_order(self, loader):
        def fake_range(filename, offset, length):
            base = 100.0 if filename.startswith("sit") else 0.0
            return frame(base + (offset - NPY_HEADER_BYTES) // SLICE_BYTES).tobytes()

        sit = json.loads(json.dumps(DAILY_MANIFEST))
        sit["variable"] = "sit"
        sit["chunks"][0]["path"] = "sit/2025/sit_202501.npy"

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.side_effect = [
                self._manifest_bytes(sit),
                self._manifest_bytes(DAILY_MANIFEST),
            ]
            out = loader.get([date(2025, 1, 6)], ["sit", "t2m"])

        assert out.shape == (1, 2, *DATASET_SHAPE)
        assert out[0, 0, 0, 0] == 105.0  # sit first
        assert out[0, 1, 0, 0] == 5.0  # t2m second

    def test_date_past_the_tail_of_a_chunk_raises(self, loader):
        # February chunk holds 10 slices, so the 11th is not there
        with patch("aiice.predictors.HfDatasetClient.read_file") as mock_read:
            mock_read.return_value = self._manifest_bytes(DAILY_MANIFEST)
            with pytest.raises(ValueError) as exc:
                loader.get([date(2025, 2, 11)], ["t2m"])
        assert "no data for 2025-02-11" in str(exc.value)

    def test_date_outside_coverage_raises(self, loader):
        with patch("aiice.predictors.HfDatasetClient.read_file") as mock_read:
            mock_read.return_value = self._manifest_bytes(DAILY_MANIFEST)
            with pytest.raises(ValueError) as exc:
                loader.get([date(2030, 1, 1)], ["t2m"])
        assert "covered range" in str(exc.value)

    def test_nothing_is_fetched_when_a_date_is_missing(self, loader):
        """A bad date must fail before any bytes move."""
        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch("aiice.predictors.HfDatasetClient.read_file_range") as mock_range,
        ):
            mock_read.return_value = self._manifest_bytes(DAILY_MANIFEST)
            with pytest.raises(ValueError):
                loader.get([date(2025, 1, 1), date(2030, 1, 1)], ["t2m"])
            mock_range.assert_not_called()

    @pytest.mark.parametrize("bad", [[], None])
    def test_empty_variables_raises(self, loader, bad):
        with pytest.raises(ValueError) as exc:
            loader.get([date(2025, 1, 1)], bad or [])
        assert "variables must not be empty" in str(exc.value)

    def test_empty_dates_raises(self, loader):
        with pytest.raises(ValueError) as exc:
            loader.get([], ["t2m"])
        assert "dates must not be empty" in str(exc.value)

    def test_string_dates_are_accepted(self, loader):
        def fake_range(filename, offset, length):
            return frame(float((offset - NPY_HEADER_BYTES) // SLICE_BYTES)).tobytes()

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.return_value = self._manifest_bytes(DAILY_MANIFEST)
            out = loader.get(["2025-01-03"], ["t2m"])

        assert out[0, 0, 0, 0] == 2.0


class TestPredictorLoader_monthly(BaseTestPredictorLoader):
    def test_monthly_step_resolves_to_the_month_slice(self, loader):
        def fake_range(filename, offset, length):
            return frame(float((offset - NPY_HEADER_BYTES) // SLICE_BYTES)).tobytes()

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.return_value = self._manifest_bytes(MONTHLY_MANIFEST)
            out = loader.get([date(2025, 1, 31), date(2025, 3, 31)], ["ohc300"])

        assert out.shape == (2, 1, *DATASET_SHAPE)
        assert out[0, 0, 0, 0] == 0.0  # January is index 0
        assert out[1, 0, 0, 0] == 2.0  # March is index 2

    def test_upsampling_is_refused_by_default(self, loader):
        with patch("aiice.predictors.HfDatasetClient.read_file") as mock_read:
            mock_read.return_value = self._manifest_bytes(MONTHLY_MANIFEST)
            with pytest.raises(ValueError) as exc:
                loader.get([date(2025, 1, 1), date(2025, 1, 2)], ["ohc300"])
        assert "allow_monthly_upsampling" in str(exc.value)

    def test_upsampling_can_be_opted_into(self, loader):
        def fake_range(filename, offset, length):
            return frame(float((offset - NPY_HEADER_BYTES) // SLICE_BYTES)).tobytes()

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.return_value = self._manifest_bytes(MONTHLY_MANIFEST)
            out = loader.get(
                [date(2025, 1, 1), date(2025, 1, 2)],
                ["ohc300"],
                allow_monthly_upsampling=True,
            )

        # both dates land on the same monthly value, no interpolation
        assert out[0, 0, 0, 0] == out[1, 0, 0, 0] == 0.0


class TestPredictorLoader_valid_mask(BaseTestPredictorLoader):
    def _mask_bytes(self) -> bytes:
        from io import BytesIO

        mask = np.zeros(DATASET_SHAPE, dtype=np.int8)
        mask[:10] = 1
        buf = BytesIO()
        np.save(buf, mask)
        return buf.getvalue()

    def test_mask_is_not_applied_by_default(self, loader):
        def fake_range(filename, offset, length):
            return frame(5.0).tobytes()

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.return_value = self._manifest_bytes(DAILY_MANIFEST)
            out = loader.get([date(2025, 1, 1)], ["t2m"])

        assert not np.isnan(out).any()

    def test_mask_turns_filler_into_nan(self, loader):
        def fake_range(filename, offset, length):
            return frame(5.0).tobytes()

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.side_effect = [
                self._manifest_bytes(DAILY_MANIFEST),
                self._mask_bytes(),
            ]
            out = loader.get([date(2025, 1, 1)], ["t2m"], apply_valid_mask=True)

        assert not np.isnan(out[0, 0, :10]).any()
        assert np.isnan(out[0, 0, 10:]).all()

    def test_variable_without_a_mask_is_left_alone(self, loader):
        def fake_range(filename, offset, length):
            return frame(5.0).tobytes()

        with (
            patch("aiice.predictors.HfDatasetClient.read_file") as mock_read,
            patch(
                "aiice.predictors.HfDatasetClient.read_file_range",
                side_effect=fake_range,
            ),
        ):
            mock_read.side_effect = [self._manifest_bytes(DAILY_MANIFEST), None]
            out = loader.get([date(2025, 1, 1)], ["t2m"], apply_valid_mask=True)

        assert not np.isnan(out).any()
