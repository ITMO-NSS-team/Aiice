from unittest.mock import Mock

import httpx
import pytest

from aiice.core.utils import retry_on_network_errors


@pytest.fixture
def retry_decorator():
    return retry_on_network_errors(retries=3, backoff=0)


class TestRetryOnNetworkErrors:

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
