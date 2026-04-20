import logging
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import utils
import yaml
from config import Config
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiice import AIICE

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.w_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.w_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.w_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))

        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
    def forward(self, x, prev_state):
        batch_size, _, height, width = x.size()

        if prev_state is None:
            h_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = prev_state

        i = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev) + self.w_ci * c_prev)

        f = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev) + self.w_cf * c_prev)

        c = f * c_prev + i * torch.tanh(self.Wxc(x) + self.Whc(h_prev))

        o = torch.sigmoid(self.Wxo(x) + self.Who(h_prev) + self.w_co * c)

        h = o * torch.tanh(c)

        return h, (h, c)
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))
    
class ConvLSTMEncoderDecoder(nn.Module):
    """Многошаговая модель Encoder-Decoder"""

    def __init__(self, num_prediction_steps: int):
        super(ConvLSTMEncoderDecoder, self).__init__()

        self.num_prediction_steps = num_prediction_steps

        self.encoder_lstm1 = ConvLSTMCell(1, 32)
        self.encoder_lstm2 = ConvLSTMCell(32, 16)

        self.decoder_lstm1 = ConvLSTMCell(1, 16)
        self.decoder_lstm2 = ConvLSTMCell(16, 32)

        self.conv1 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 1, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, input_sequences):
        x = input_sequences.permute(1, 0, 2, 3)
        x = x.unsqueeze(2)  # [input_seq_len, batch, 1, height, width]

        encoder_state1 = None
        encoder_state2 = None

        for t in range(x.size(0)):
            frame = x[t]

            h1, encoder_state1 = self.encoder_lstm1(frame, encoder_state1)
            h2, encoder_state2 = self.encoder_lstm2(h1, encoder_state2)


        h_enc1, c_enc1 = encoder_state1
        h_enc2, c_enc2 = encoder_state2
        decoder_state1 = (h_enc2, c_enc2)
        decoder_state2 = (h_enc1, c_enc1)

        all_predictions = []
        current_input = x[-1]

        for step in range(self.num_prediction_steps):
            d1, decoder_state1 = self.decoder_lstm1(current_input, decoder_state1)
            d2, decoder_state2 = self.decoder_lstm2(d1, decoder_state2)

            out = self.relu(self.conv1(d2))
            out = self.relu(self.conv2(out))
            out = self.conv3(out)

            prediction = out.squeeze(1)
            all_predictions.append(prediction)
            current_input = out

        return torch.stack(all_predictions, dim=0).permute(1, 0, 2, 3).contiguous()

def run(
    logger: logging.Logger,
    cfg: Config,
    sea: str | None,
    train_dataloader: DataLoader,
):
    experiment_path = f"{cfg.output_path}/convlstm/{sea}"
    os.makedirs(experiment_path, exist_ok=True)

    best_loss_value = math.inf
    best_iteration = 0
    best_model: nn.Module | None = None

    for i, experiment in enumerate(cfg.run.experiments):

        i_experiment_path = f"{experiment_path}/{i}"
        os.makedirs(i_experiment_path, exist_ok=True)
        
        loss_value, model = train(
            logger=logger,
            train_dataloader=train_dataloader,
            experiment_path=i_experiment_path,
            out_time_point=cfg.aiice.forecast_len,
            args=experiment,
            device=cfg.device,
        )

        if loss_value < best_loss_value:
            best_iteration = i
            best_model = model
            best_loss_value = loss_value

    logger.info(f"Best loss model is here: {experiment_path}/{best_iteration}")

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
            model=best_model,
            # path=f"{experiment_path}/gif/",
            plot_workers=8,
        )
    with open(f"{experiment_path}/best-model-{best_iteration}-report.yaml", "w") as f:
        yaml.safe_dump(report, f)

    logger.info("Eval is done!")


def train(
    logger: logging.Logger,
    train_dataloader: DataLoader,
    experiment_path: str,
    out_time_point: int,
    args: dict[str, any],
    device: str,
) -> tuple[float, nn.Module]:
    
    model = ConvLSTMEncoderDecoder(num_prediction_steps = out_time_point).to(device)
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
