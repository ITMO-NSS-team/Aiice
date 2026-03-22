import argparse
import logging

import config
import conv3d
import conv2d
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


def init_dataloaders(
    cfg: config.Aiice,
    device: str,
    sea: str | None,
) -> tuple[DataLoader, DataLoader]:
    loader = Loader()

    train_data = loader.get(
        start=cfg.start_date,
        end=cfg.end_date,
        sea=sea,
        step=cfg.step,
        tensor_out=True,
    )
    val_data = loader.get(
        start=cfg.end_date,
        sea=sea,
        step=cfg.step,
        tensor_out=True,
    )

    train_dataset = SlidingWindowDataset(
        data=train_data,
        pre_history_len=cfg.pre_history_len,
        forecast_len=cfg.forecast_len,
        device=device,
    )
    val_dataset = SlidingWindowDataset(
        data=val_data,
        pre_history_len=cfg.pre_history_len,
        forecast_len=cfg.forecast_len,
        device=device,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.batch_size
    )

    return train_dataloader, val_dataloader


def main():
    logger = init_logger()
    device = init_device()
    cfg = init_config()

    seas: list[str | None] = []
    if isinstance(cfg.aiice.sea, list):
        seas = cfg.aiice.sea
    else:
        seas.append(cfg.aiice.sea)

    for sea in seas:
        logger.info(f"=== Running for sea: {sea} ===")

        train_dataloader, val_dataloader = init_dataloaders(
            cfg.aiice, 
            device=device, 
            sea=sea,
        )

        match cfg.run.model_name:
            case "conv2d":
                conv2d.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    device=device,
                )
            case "conv3d":
                conv3d.run(
                    logger=logger,
                    cfg=cfg,
                    sea=sea,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    device=device,
                )
            case _:
                raise ValueError("unknown experiment run type")


if __name__ == "__main__":
    main()
