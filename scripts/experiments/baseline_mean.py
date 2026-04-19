import logging
import os

import torch
import torch.nn as nn
import yaml
from config import Config

from aiice import AIICE


class BaseLineMeanModel(nn.Module):
    def __init__(self, forecast_len: int, pre_history_size: int):
        super().__init__()
        self.forecast_len = forecast_len
        self.pre_history_size = pre_history_size

    def forward(self, x):
        # x: [B, T, H, W]
        outputs = []

        history = x[:, -self.pre_history_size :, :, :]

        for _ in range(self.forecast_len):
            mean = history.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            outputs.append(mean)

            history = torch.cat([history, mean], dim=1)
            history = history[:, -self.pre_history_size :, :, :]

        return torch.cat(outputs, dim=1)


def run(
    logger: logging.Logger,
    cfg: Config,
    sea: str,
):
    model = BaseLineMeanModel(
        forecast_len=cfg.aiice.forecast_len,
        pre_history_size=cfg.aiice.pre_history_len,
    ).to(cfg.device)

    experiment_path = f"{cfg.output_path}/baseline_mean/{sea}"
    os.makedirs(experiment_path, exist_ok=True)

    aiice = AIICE(
        pre_history_len=cfg.aiice.pre_history_len,
        forecast_len=cfg.aiice.forecast_len,
        batch_size=cfg.aiice.batch_size,
        start=cfg.aiice.end_date,
        step=cfg.aiice.step,
        sea=sea,
        device=cfg.device,
    )
    report = aiice.bench(
        model=model,
        # path=f"{experiment_path}/gif/",
        plot_workers=8,
    )

    with open(f"{experiment_path}/report.yaml", "w") as f:
        yaml.safe_dump(report, f)

    logger.info("Eval is done!")
