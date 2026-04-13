# Aiice

[![uv](https://img.shields.io/badge/uv-F0DB4F?style=flat&logo=uv&logoColor=black)](https://uv.io)
[![Hugging Face](https://img.shields.io/badge/huggingface-FF9900?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/pytorch-CB2C31?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)

---

**AIICE** is an open-source Python framework designed as a standardized benchmark for spatio-temporal forecasting of Arctic sea ice concentration. It provides reproducible pipelines for loading, preprocessing, and evaluating satellite-derived OSI-SAF data, supporting both short- and long-term prediction horizons

## Installation

The simplest way to install framework with `pip`:
```shell
pip install aiice-bench
```

## Quickstart

The AIICE class provides a simple interface for loading Arctic ice data, preparing datasets, and benchmarking PyTorch models:

```python
from aiice import AIICE

# Initialize AIICE with a sliding window 
# of past 30 days and forecast of 7 days
aiice = AIICE(
    pre_history_len=30,
    forecast_len=7,
    batch_size=32,
    start="2022-01-01",
    end="2022-12-31"
)

# Define your PyTorch model
model = MyModel()

# Run benchmarking to compute metrics on the dataset
report = aiice.bench(model)
print(report)
```

Check [package doc](https://itmo-nss-team.github.io/Aiice/aiice.html) and see more [usage examples](https://github.com/ITMO-NSS-team/Aiice/tree/main/scripts/examples)
