import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import utils
import yaml
import math
from config import Config
from torch.utils.data import DataLoader
from torchcnnbuilder.models import ForecasterBase
from tqdm import tqdm


def run(
    logger: logging.Logger,
    cfg: Config,
    sea: str | None,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: str,
):
    experiment_path = f"{cfg.output_path}/conv2d"
    if sea is not None:
        experiment_path = f"{experiment_path}/{sea}"
    os.makedirs(experiment_path, exist_ok=True)

    best_loss_value = math.inf
    best_iteration = 0
    best_model: nn.Module | None = None

    first_batch = next(iter(train_dataloader))

    for i, experiment in enumerate(cfg.run.experiments):

        i_experiment_path = f"{experiment_path}/{i}"
        os.makedirs(i_experiment_path, exist_ok=True)
        
        loss_value, model = train(
            logger=logger,
            train_dataloader=train_dataloader,
            experiment_path=i_experiment_path,
            data_shape=first_batch[0].shape[-2:],
            in_time_points=cfg.aiice.pre_history_len,
            out_time_point=cfg.aiice.forecast_len,
            args=experiment,
            device=device,
        )

        if loss_value < best_loss_value:
            best_iteration = i
            best_model = model

    logger.info(f"Best loss model is here: {experiment_path}/{best_iteration}")

    report = utils.val(model=best_model, val_dataloader=val_dataloader)
    with open(f"{experiment_path}/best-model-{best_iteration}-report.yaml", "w") as f:
        yaml.safe_dump(report, f)

    logger.info("Eval is done!")


def train(
    logger: logging.Logger,
    train_dataloader: DataLoader,
    experiment_path: str,
    data_shape: int,
    in_time_points: int,
    out_time_point: int,
    args: dict[str, any],
    device: str,
) -> tuple[float, nn.Module]:
    forecaster_params = {
        "input_size": data_shape,
        "n_layers": 5,
        "in_time_points": in_time_points,
        "out_time_points": out_time_point,
        "convolve_params": {"kernel_size": args["kernel_size"]},
        "transpose_convolve_params": {"kernel_size": args["kernel_size"]},
        "conv_dim": 2,
    }
    forecaster_model = ForecasterBase(**forecaster_params)
    model = forecaster_model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=args["lr"])
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=args["lr"],
        max_lr=0.005,
        step_size_up=30,
        mode="triangular2",
        cycle_momentum=False,
    )

    criterion = nn.L1Loss()
    loss_history = []
    epochs_no_improve = 0

    for epoch in range(args["max_epoch"]):

        loss = 0
        for x, y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            train_loss = criterion(outputs, y)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss = loss / len(train_dataloader)
        scheduler.step()
        loss_history.append(loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f'-- epoch : {epoch + 1}/{args["max_epoch"]}, {loss=}, {current_lr=}'
        )

        # early stopping if loss do not change
        if epoch != 0:
            relative_change = abs(loss_history[-2] - loss) / max(loss_history[-2], 1e-8)
            if relative_change < args["min_delta"]:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0

        if epochs_no_improve >= args["patience"]:
            logger.warning("EARLY STOPPING TRIGGERED")
            break

    logger.info("- End of training")

    torch.save(model.state_dict(), f"{experiment_path}/model.pt")
    utils.plot_history(loss_history, f"{experiment_path}/loss_history.png", logger)

    logger.info("- All savings are done!")
    return loss, model
