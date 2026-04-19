import argparse
import logging

import baseline_mean
import baseline_repeat
import config
import conv2d
import conv3d
import convlstm
import torch
import yaml
from torch.utils.data import DataLoader

from aiice.loader import Loader
from aiice.preprocess import SlidingWindowDataset


def init_logger() -> logging.Logger:
    logger = logging.getLogger("logger")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logger


def init_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def init_config() -> config.Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        data = yaml.safe_load(f)

    return config.Config(**data)


def init_train(
    cfg: config.Aiice,
    device: str,
    sea: str | None,
) -> DataLoader:
    loader = Loader()
    train_data = loader.get(
        start=cfg.start_date,
        end=cfg.end_date,
        sea=sea,
        step=cfg.step,
        tensor_out=True,
        threads=8,
    )
    train_dataset = SlidingWindowDataset(
        data=train_data,
        pre_history_len=cfg.pre_history_len,
        forecast_len=cfg.forecast_len,
        device=device,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    return train_dataloader


def main():
    logger = init_logger()
    cfg = init_config()

    if cfg.device is None:
        cfg.device = init_device()

    seas: list[str | None] = []
    if isinstance(cfg.aiice.sea, list):
        seas = cfg.aiice.sea
    else:
        seas.append(cfg.aiice.sea)

    for sea in seas:
        logger.info(f"=== Running for sea: {sea} ===")

        match cfg.run.model_name:
            case "conv2d":
                train_dataloader = init_train(
                    cfg.aiice,
                    device=cfg.device,
                    sea=sea,
                )

                conv2d.run(
                    logger=logger, cfg=cfg, sea=sea, train_dataloader=train_dataloader
                )
            case "conv3d":
                train_dataloader = init_train(
                    cfg.aiice,
                    device=cfg.device,
                    sea=sea,
                )

                conv3d.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                    train_dataloader=train_dataloader,
                )
            case "convlstm":
                train_dataloader = init_train(
                    cfg.aiice,
                    device=cfg.device,
                    sea=sea,
                )

                convlstm.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                    train_dataloader=train_dataloader,
                )
            case "baseline_mean":
                baseline_mean.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                )
            case "baseline_repeat":
                baseline_repeat.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                )
            case _:
                raise ValueError("unknown experiment run type")


if __name__ == "__main__":
    main()
