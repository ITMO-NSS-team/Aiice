import functools
import time

import httpx

from aiice.constants import DEFAULT_BACKOFF, DEFAULT_RETRIES

RETRY_EXCEPTIONS = (httpx.RemoteProtocolError, httpx.ConnectError)


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
