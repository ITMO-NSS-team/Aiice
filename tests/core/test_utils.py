from datetime import date
from unittest.mock import Mock

import httpx
import pytest
from dateutil.relativedelta import relativedelta

from aiice.core.utils import (
    convert_step_to_delta,
    get_date_from_filename_template,
    get_filename_template,
    retry_on_network_errors,
)


@pytest.fixture
def retry_decorator():
    return retry_on_network_errors(retries=3, backoff=0)


class Test_retry_on_network_errors:
    def test_call_n_times_on_error(self, retry_decorator):
        mock_func = Mock(side_effect=httpx.RemoteProtocolError("fail"))
        decorated = retry_decorator(mock_func)

        with pytest.raises(httpx.RemoteProtocolError):
            decorated()

        assert mock_func.call_count == 3

    def test_stop_on_success(self, retry_decorator):
        state = {"count": 0}

        def func():
            if state["count"] < 2:
                state["count"] += 1
                raise httpx.ConnectError("fail")
            return "ok"

        decorated = retry_decorator(func)
        result = decorated()
        assert result == "ok"
        assert state["count"] == 2


class Test_get_filename_template:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (date(2021, 6, 1), "global_series/2021/osisaf_20210601.npy"),
            (date(1991, 12, 12), "global_series/1991/osisaf_19911212.npy"),
        ],
    )
    def test_ok(self, value, expected):
        assert get_filename_template(value) == expected


class Test_get_date_from_filename_template:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("global_series/2021/osisaf_20210601.npy", date(2021, 6, 1)),
            ("global_series/1991/osisaf_19911212.npy", date(1991, 12, 12)),
        ],
    )
    def test_ok(self, value, expected):
        assert get_date_from_filename_template(value) == expected


class Test_convert_step_to_delta:
    @pytest.mark.parametrize(
        "step, expected",
        [
            (None, relativedelta(days=1)),
            (30, relativedelta(days=30)),
            ("7d", relativedelta(days=7)),
            ("30d", relativedelta(days=30)),
            ("4w", relativedelta(weeks=4)),
            ("3m", relativedelta(months=3, day=31)),
            ("12m", relativedelta(months=12, day=31)),
            ("1y", relativedelta(years=1)),
            ("2y", relativedelta(years=2)),
            ("15d", relativedelta(days=15)),
            ("52w", relativedelta(weeks=52)),
            ("24m", relativedelta(months=24, day=31)),
            ("100y", relativedelta(years=100)),
        ],
    )
    def test_ok(self, step, expected):
        assert convert_step_to_delta(step) == expected

    @pytest.mark.parametrize(
        "step, expected_error, expected_message",
        [
            ("1", ValueError, r"Invalid step format: .+"),
            ("1x", ValueError, r"Invalid step format: .+"),
            ("", ValueError, r"Invalid step format: .+"),
            ("1.5d", ValueError, r"Invalid step format: .+"),
            (3.14, ValueError, r"Invalid step type: .+"),
        ],
    )
    def test_error(self, step, expected_error, expected_message):
        with pytest.raises(expected_error, match=expected_message):
            convert_step_to_delta(step)
