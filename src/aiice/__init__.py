from aiice.core.huggingface import HfDatasetClient
from datetime import date
import time

start = time.perf_counter()
hf = HfDatasetClient()
image = hf.get_filename_template(date(2024, 1, 1))
# test = hf.download_file(image, "local/test")
test = hf.read_file(image)
print(test)

print((time.perf_counter() - start) * 1000)
