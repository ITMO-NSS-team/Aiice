import math
from datetime import date, timedelta
from unittest.mock import patch

import imageio
import pytest
import torch
import torch.nn as nn

import aiice.constants as constants
from aiice import AIICE


class DummyModel(nn.Module):
    def __init__(self, forecast_len: int):
        super().__init__()
        self.forecast_len = forecast_len

    def forward(self, x):
        # X.shape = [B, pre_history_len, H, W]
        # Y.shape = [B, forecast_len, H, W]
        return x[:, -1:, :, :].repeat(1, self.forecast_len, 1, 1)


class TestAIICE:
    def make_mock_loader_return(self, num_dates: int, height: int, width: int):
        start_date = date(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(num_dates)]
        data = torch.arange(num_dates * height * width, dtype=torch.float32).reshape(
            num_dates, height, width
        )
        data = data / data.max()
        return dates, data

    @pytest.mark.parametrize(
        "pre_history_len, forecast_len, batch_size, num_dates, height, width",
        [
            (5, 1, 2, 50, 30, 30),
            (10, 5, 4, 100, 28, 28),
            (3, 2, 1, 20, 12, 12),
            (7, 3, 3, 60, 32, 32),
            (8, 4, 2, 40, 15, 20),
        ],
    )
    def test_ok(
        self,
        pre_history_len,
        forecast_len,
        batch_size,
        num_dates,
        height,
        width,
    ):
        with patch("aiice.loader.Loader.get") as mock_loader_get:
            mock_loader_get.return_value = self.make_mock_loader_return(
                num_dates=num_dates, height=height, width=width
            )

            aiice = AIICE(
                pre_history_len=pre_history_len,
                forecast_len=forecast_len,
                batch_size=batch_size,
            )
            report = aiice.bench(DummyModel(forecast_len=forecast_len))

            for metric in report.keys():
                assert report[metric][constants.COUNT_STAT] == len(aiice._dataloader)

                assert not math.isnan(report[metric][constants.MEAN_STAT])
                assert not math.isnan(report[metric][constants.LAST_STAT])
                assert not math.isnan(report[metric][constants.MAX_STAT])
                assert not math.isnan(report[metric][constants.MIN_STAT])

                assert report[metric][constants.MEAN_STAT] >= 0
                assert report[metric][constants.LAST_STAT] >= 0
                assert report[metric][constants.MAX_STAT] >= 0
                assert report[metric][constants.MIN_STAT] >= 0

    def test_plot_generation(self, tmp_path):
        forecast_len = 3

        with patch("aiice.loader.Loader.get") as mock_loader_get:
            mock_loader_get.return_value = self.make_mock_loader_return(
                num_dates=30,
                height=15,
                width=15,
            )

            aiice = AIICE(
                pre_history_len=5,
                forecast_len=forecast_len,
                batch_size=2,
            )

            plot_dir = tmp_path / "plots"
            aiice.bench(
                DummyModel(forecast_len=forecast_len),
                path=str(plot_dir),
                plot_workers=1,  # deterministic for tests
                fps=2,
            )

        gifs = list(plot_dir.glob("*.gif"))
        assert len(gifs) > 0

        frames = imageio.mimread(gifs[0])
        assert len(frames) == forecast_len

        # frames not static
        assert not (frames[0] == frames[-1]).all()
