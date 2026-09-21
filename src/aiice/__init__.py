from aiice import core, loader, metrics, predictors, preprocess
from aiice.benchmark import AIICE

# visible modules to pdoc
__all__ = ["AIICE", "core", "loader", "metrics", "predictors", "preprocess"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aiice-bench")
except PackageNotFoundError:
    __version__ = "0.0.0"
