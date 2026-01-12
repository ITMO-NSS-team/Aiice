import math

import pytest
import torch

from aiice.metrics import bin_accuracy, mae, mse, psnr, rmse, ssim


class TestMetrics:

    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            ([1, 2, 3], [1, 2, 3], 0.0),
            ([1, 2, 3], [2, 2, 4], 2 / 3),
            ([0, 0, 0], [1, 1, 1], 1.0),
        ],
    )
    def test_mae_ok(self, y_true, y_pred, expected):
        assert math.isclose(mae(y_true, y_pred), expected, abs_tol=1e-6)

    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            ([1, 2, 3], [1, 2, 3], 0.0),
            ([1, 2, 3], [2, 2, 4], (1 + 0 + 1) / 3),
            ([0, 0, 0], [1, 1, 1], 1.0),
        ],
    )
    def test_mse_ok(self, y_true, y_pred, expected):
        assert math.isclose(mse(y_true, y_pred), expected, abs_tol=1e-6)

    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            ([1, 2, 3], [1, 2, 3], 0.0),
            ([1, 2, 3], [2, 2, 4], math.sqrt((1 + 0 + 1) / 3)),
            ([0, 0, 0], [1, 1, 1], 1.0),
        ],
    )
    def test_rmse_ok(self, y_true, y_pred, expected):
        assert math.isclose(rmse(y_true, y_pred), expected, abs_tol=1e-6)

    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            ([1, 1, 1], [1, 1, 1], float("inf")),
            ([1, 2, 3], [1, 2, 3], float("inf")),
            ([1, 2, 3], [0, 2, 3], 20 * math.log10(3) - 10 * math.log10((1**2) / 3)),
        ],
    )
    def test_psnr_ok(self, y_true, y_pred, expected):
        val = psnr(y_true, y_pred)
        assert math.isclose(val, expected, abs_tol=1e-6)

    @pytest.mark.parametrize(
        "y_true, y_pred, threshold, expected",
        [
            ([0, 0, 1, 1], [0, 1, 1, 0], 0.5, 0.5),
            ([0, 0, 1, 1], [0, 0, 1, 1], 0.5, 1.0),
            ([0.2, 0.7, 0.8], [0.1, 0.6, 0.9], 0.5, 1.0),
        ],
    )
    def test_bin_accuracy_ok(self, y_true, y_pred, threshold, expected):
        val = bin_accuracy(y_true, y_pred, threshold)
        assert math.isclose(val, expected, abs_tol=1e-6)

    @pytest.mark.parametrize(
        "y_true, y_pred, expected_ssim",
        [
            (torch.ones(1, 1, 11, 11), torch.ones(1, 1, 11, 11), 1.0),
            (torch.ones(1, 1, 11, 11), torch.zeros(1, 1, 11, 11), 0.0),
            (torch.rand(2, 1, 11, 11), None, 1.0),
        ],
    )
    def test_ssim_ok(self, y_true, y_pred, expected_ssim):
        if y_pred is None:
            y_pred = y_true.clone()
        val = ssim(y_true, y_pred)
        assert math.isclose(val, expected_ssim, abs_tol=1e-4)

    @pytest.mark.parametrize(
        "y_true, y_pred",
        [
            # not enough dimgs
            (torch.ones(8, 8), torch.ones(8, 8)),
            # different dims
            (torch.ones(1, 8, 8), torch.ones(1, 8)),
            # not enough values in dims for the DEFAULT_SSIM_KERNEL_WINDOW_SIZE
            (torch.ones(1, 1, 5, 11, 11), torch.ones(1, 1, 5, 11, 11)),
        ],
    )
    def test_ssim_raise(self, y_true, y_pred):
        with pytest.raises(ValueError):
            ssim(y_true, y_pred)
