import sys
import time
from datetime import date

from aiice.core.huggingface import HfDatasetClient
from aiice.loader import Loader

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


def main():
    start = time.perf_counter()

    loader_get()
    print((time.perf_counter() - start) * 1000)


if __name__ == "__main__":
    main()
