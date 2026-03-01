import math

import pytest
import torch

import aiice.constants as constants
from aiice.metrics import Evaluator, bin_accuracy, mae, mse, psnr, rmse, ssim


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
            # not enough dims
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


class TestEvaluator:
    def test_default_metrics_initialized(self):
        ev = Evaluator()

        assert set(ev.metrics) == {
            constants.MAE_METRIC,
            constants.MSE_METRIC,
            constants.RMSE_METRIC,
            constants.PSNR_METRIC,
            constants.BIN_ACCURACY_METRIC,
            constants.SSIM_METRIC,
        }
        for k in ev._metrics:
            assert ev._report[k] == []

    def test_eval_single_step_accumulate(self):
        ev = Evaluator(metrics=[constants.MAE_METRIC, constants.MSE_METRIC])

        y_true = [1, 2, 3]
        y_pred = [2, 2, 4]

        step = ev.eval(y_true, y_pred)

        assert math.isclose(step[constants.MAE_METRIC], 2 / 3, abs_tol=1e-6)
        assert math.isclose(step[constants.MSE_METRIC], (1 + 0 + 1) / 3, abs_tol=1e-6)

        rep = ev.report()

        assert rep[constants.MAE_METRIC][constants.COUNT_STAT] == 1
        assert rep[constants.MSE_METRIC][constants.COUNT_STAT] == 1
        assert math.isclose(
            rep[constants.MAE_METRIC][constants.MEAN_STAT], 2 / 3, abs_tol=1e-6
        )
        assert math.isclose(
            rep[constants.MAE_METRIC][constants.LAST_STAT], 2 / 3, abs_tol=1e-6
        )

    def test_eval_multiple_steps_accumulate(self):
        ev = Evaluator(metrics=[constants.MAE_METRIC], accumulate=True)

        ev.eval([1, 2], [1, 2])  # mae = 0
        ev.eval([1, 2], [2, 2])  # mae = 0.5
        ev.eval([1, 2], [3, 2])  # mae = 1.0

        rep = ev.report()[constants.MAE_METRIC]

        assert rep[constants.COUNT_STAT] == 3
        assert math.isclose(rep[constants.MEAN_STAT], (0 + 0.5 + 1.0) / 3)
        assert math.isclose(rep[constants.LAST_STAT], 1.0)
        assert math.isclose(rep[constants.MIN_STAT], 0.0)
        assert math.isclose(rep[constants.MAX_STAT], 1.0)

    def test_eval_overwrite_mode(self):
        ev = Evaluator(metrics=[constants.MAE_METRIC], accumulate=False)

        ev.eval([1, 2], [1, 2])  # 0
        ev.eval([1, 2], [2, 2])  # 0.5
        ev.eval([1, 2], [3, 2])  # 1.0

        rep = ev.report()[constants.MAE_METRIC]

        assert rep[constants.COUNT_STAT] == 1
        assert math.isclose(rep[constants.MEAN_STAT], 1.0)
        assert math.isclose(rep[constants.LAST_STAT], 1.0)

    def test_eval_returns_step_metrics(self):
        ev = Evaluator(metrics=[constants.MAE_METRIC, constants.MSE_METRIC])

        step = ev.eval([1, 2, 3], [2, 2, 4])

        assert set(step.keys()) == {constants.MAE_METRIC, constants.MSE_METRIC}
        assert math.isclose(step[constants.MAE_METRIC], 2 / 3, abs_tol=1e-6)
        assert math.isclose(step[constants.MSE_METRIC], (1 + 0 + 1) / 3, abs_tol=1e-6)

    def test_custom_metrics(self):
        ev = Evaluator(metrics={"custom": lambda y_true, y_pred: 1.0})

        ev.eval([1, 2], [100, 200])
        ev.eval([5, 6], [7, 8])

        rep = ev.report()["custom"]

        assert rep[constants.COUNT_STAT] == 2
        assert rep[constants.MEAN_STAT] == 1.0
        assert rep[constants.LAST_STAT] == 1.0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError) as e:
            Evaluator(metrics=[constants.MAE_METRIC, "unknown_metric"])
            assert "Unknown metric" in str(e.value)
