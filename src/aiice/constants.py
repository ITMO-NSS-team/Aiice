from datetime import date

# hugging face constants

HF_BASE_URL: str = "https://huggingface.co"
HF_REPO_TYPE: str = "dataset"
HF_DATASET_REPO: str = "ITMO-NSS/Aiice"

# dataset constants

MIN_DATASET_START: date = date(1980, 1, 1)
MAX_DATASET_END: date = date(2025, 7, 1)
DATASET_SHAPE: tuple[int, int] = (432, 432)

BYTES_IN_MB = 1024 * 1024

YEAR_STATS_CACHE_SIZE = 64

KEY_DATASET_START = "start_date"
KEY_DATASET_END = "end_date"
KEY_PER_YEAR = "per_year"
KEY_SHAPE = "shape"
KEY_FILES = "files"
KEY_SIZE_BYTES = "size_bytes"
KEY_SIZE_MB = "size_mb"

# aiice constants

PACKAGE_NAME: str = "aiice"

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 2.0
