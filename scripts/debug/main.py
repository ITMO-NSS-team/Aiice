import sys
import time
from datetime import date

from aiice.core.huggingface import HfDatasetClient
from aiice.loader import Loader

import math
import torch
import numpy as np

from aiice.runner import Runner
from aiice.preprocess import SlidingWindowDataset


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x[-1:] + 1.0

def hugging_face():
    hf = HfDatasetClient()
    filename = hf.get_filename_template(date(2024, 1, 1))
    test = hf.read_file(filename)
    print(test.shape)

def loader_get():
    loader = Loader()
    start_date = date(2024, 1, 1)
    end_date = date(2025, 1, 1)
    data = loader.get(start_date, end_date)
    print(data.shape)

def runner_run():
    
    
    data = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)

    dataset = SlidingWindowDataset(
        data=data,
        pre_history_len=2,
        forecast_len=1,
    )

    model = DummyModel()
    runner = Runner(model=model, dataset=dataset, metrics=["mae", "mse"])

    report = runner.run()
    print(report)


def main():
    start = time.perf_counter()

    runner_run()
    print((time.perf_counter() - start) * 1000)


if __name__ == "__main__":
    main()
