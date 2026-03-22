from pydantic import BaseModel
from typing import Any


class Aiice(BaseModel):
    sea: str | None = None
    start_date: str  # YYYY-mm-dd
    end_date: str  # YYYY-mm-dd
    pre_history_len: int
    forecast_len: int
    step: int
    batch_size: int
    data_shape: tuple[int, int]


class Run(BaseModel):
    model_name: str
    experiments: list[dict[str, Any]]


class Config(BaseModel):
    aiice: Aiice
    run: Run
    output_path: str
