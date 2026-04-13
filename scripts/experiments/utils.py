from typing import Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aiice.metrics import Evaluator


def plot_history(history: Sequence[float], save_path: str, show: bool = True) -> None:
    plt.plot(list(range(len(history))), history)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"History per epoch ({save_path})")
    plt.savefig(f"{save_path}")
    
    if show:
        plt.show()
    plt.close()


def val(model: nn.Module, val_dataloader: DataLoader) -> dict[str, dict[str, float]]:
    evaluator = Evaluator()

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(val_dataloader):
            preds = model(x)
            evaluator.eval(y, preds)

    return evaluator.report()
