import functools
import re
import time
from datetime import date
from pathlib import Path

import httpx
from dateutil.relativedelta import relativedelta

from aiice.constants import DEFAULT_BACKOFF, DEFAULT_RETRIES

RETRY_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.TimeoutException,
)


def retry_on_network_errors(
    retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRY_EXCEPTIONS as e:
                    if attempt < retries:
                        time.sleep(backoff * attempt)
                    else:
                        raise e

        return wrapper

    return decorator


def get_filename_template(d: date) -> str:
    # filename looks like: global_series/1999/osisaf_19991101.npy
    return f"global_series/{d.year}/osisaf_{d.year}{d.month:02d}{d.day:02d}.npy"


def get_date_from_filename_template(f: str) -> date:
    # filename looks like: global_series/1999/osisaf_19991101.npy
    name = Path(f).name
    date_part = name.removeprefix("osisaf_").removesuffix(".npy")

    year = int(date_part[0:4])
    month = int(date_part[4:6])
    day = int(date_part[6:8])
    return date(year, month, day)


def convert_step_to_delta(step: int | str | None) -> relativedelta:
    if step is None:
        return relativedelta(days=1)

    if isinstance(step, int):
        return relativedelta(days=step)

    if isinstance(step, str):
        match = re.match(r"^(\d+)([dwmy])$", step)
        if not match:
            raise ValueError(
                f"Invalid step format: {step}. Expected format: <number><unit>, where unit is [d, w, m, y]"
            )

        value = int(match.group(1))
        unit = match.group(2)

        match unit:
            case "d":
                return relativedelta(days=value)
            case "w":
                return relativedelta(weeks=value)
            case "m":
                return relativedelta(months=value, day=31)
            case "y":
                return relativedelta(years=value, day=366)

    raise ValueError(f"Invalid step type: {type(step)}")
