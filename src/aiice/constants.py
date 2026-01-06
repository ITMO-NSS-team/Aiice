from datetime import date

# hugging face constants

HF_BASE_URL: str = "https://huggingface.co"
HF_REPO_TYPE: str = "dataset"
HF_DATASET_REPO: str = "ITMO-NSS/Aiice"

# dataset constants

MIN_DATASET_START: date = date(1980, 1, 1)
MAX_DATASET_END: date = date(2025, 7, 1)

# aiice constants

PACKAGE_NAME: str = "aiice"
MIN_SPLIT_FRACTION: float = 0.0
MAX_SPLIT_FRACTION: float = 1.0
