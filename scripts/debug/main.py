import time
import sys
from datetime import date

from aiice.core.huggingface import HfDatasetClient
from aiice.loader import Loader

start = time.perf_counter()
"""
hf = HfDatasetClient()
filename = hf.get_filename_template(date(2024, 1, 1))
test = hf.read_file(filename)
print(test.shape)
print((time.perf_counter() - start) * 1000)
sys.exit(0)
"""

start_date = date(2024, 1, 1)
end_date = date(2025, 1, 1)

loader = Loader()
test = loader.get(start_date, end_date)
print(test.shape)

print((time.perf_counter() - start) * 1000)
