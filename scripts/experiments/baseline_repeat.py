import logging
import os

import torch.nn as nn
import yaml
from config import Config

from aiice import AIICE


class BaselineRepeatModel(nn.Module):
    def __init__(self, forecast_len: int):
        super().__init__()
        self.forecast_len = forecast_len

    def forward(self, x):
        # X.shape = [B, pre_history_len, H, W]
        # Y.shape = [B, forecast_len, H, W]
        return x[:, -1:, :, :].repeat(1, self.forecast_len, 1, 1)


def run(
    logger: logging.Logger,
    cfg: Config,
    sea: str,
):
    model = BaselineRepeatModel(
        forecast_len=cfg.aiice.forecast_len,
    ).to(cfg.device)

    experiment_path = f"{cfg.output_path}/baseline_repeat/{sea}"
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
