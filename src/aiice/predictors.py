import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Sequence

import numpy as np

from aiice.constants import (
    DATASET_SHAPE,
    HF_PREDICTORS_REPO,
    NPY_HEADER_BYTES,
    PREDICTOR_MANIFEST_PATH,
    PREDICTOR_STEP_DAILY,
    PREDICTOR_STEP_MONTHLY,
    PREDICTOR_VALID_MASK_PATH,
)
from aiice.core.huggingface import HfDatasetClient


class PredictorLoader:
    """
    Read ERA5 and ocean predictors aligned to a date list you already have.

    The loader never decides which dates to read. It is handed the list the ice
    `Loader` produced and must return exactly one frame per date or fail - the
    two series cannot drift apart if only one of them decides what a date is.
    Generating dates here as well would reintroduce exactly that risk: the
    package step `"1m"` lands on the last day of the month, and a second
    implementation would almost certainly land on the first.

    Predictors are stored as chunks: daily variables keep one file per month,
    monthly variables one file per year. A slice is fetched with a ranged
    request, which costs the same as fetching a whole per-day file, so the
    layout saves files without costing time.

    Args:
        client (`HfDatasetClient`, optional): Client to read through. Defaults to
            one pointing at the predictors repository.
        threads (`int`, optional): Parallel range requests. Defaults to 16.

    Example:
        ```python
        ice = Loader()
        dates, ice_frames = ice.get(start="2025-01-01", end="2025-03-31", idx_out=True)

        predictors = PredictorLoader()
        frames = predictors.get(dates, ["t2m", "sit"])   # [T, C, H, W]
        ```
    """

    def __init__(self, client: HfDatasetClient | None = None, threads: int = 16):
        self._hf = client or HfDatasetClient(repo=HF_PREDICTORS_REPO)
        self._threads = threads
        self._manifests: dict[str, dict] = {}
        self._masks: dict[str, np.ndarray | None] = {}

    def manifest(self, variable: str) -> dict:
        """
        Return the chunk manifest of a variable, fetching it once.

        Args:
            variable (`str`): Predictor name, for example "t2m".
        """
        if variable not in self._manifests:
            raw = self._hf.read_file(PREDICTOR_MANIFEST_PATH.format(variable=variable))
            if raw is None:
                raise ValueError(
                    f"No manifest for {variable} in {self._hf._dataset_repo}"
                )
            self._manifests[variable] = json.loads(raw.decode("utf-8"))
        return self._manifests[variable]

    def step(self, variable: str) -> str:
        """
        Return "daily" or "monthly" - the native resolution of a variable.

        Args:
            variable (`str`): Predictor name.
        """
        return self.manifest(variable)["step"]

    def coverage(self, variable: str) -> tuple[date, date]:
        """
        Return the first and last date a variable covers.

        Args:
            variable (`str`): Predictor name.
        """
        chunks = self.manifest(variable)["chunks"]
        first, last = chunks[0]["first"], chunks[-1]["last"]
        return self._as_date(first), self._as_date(last, end_of_month=True)

    def common_coverage(self, variables: Sequence[str]) -> tuple[date, date]:
        """
        Return the range every requested variable covers.

        The end moves with the selection - `sit` stops months before the
        atmospheric variables - so the bound is reported rather than silently
        trimming a date list.

        Args:
            variables (`Sequence[str]`): Predictor names.
        """
        spans = [self.coverage(v) for v in variables]
        return max(s for s, _ in spans), min(e for _, e in spans)

    def valid_mask(self, variable: str) -> np.ndarray | None:
        """
        Return the validity mask of a variable, or None when it has none.

        Matrices are stored dense: cells without data hold a filler value. A
        variable with no gaps ships no mask, and everything is valid.

        Args:
            variable (`str`): Predictor name.
        """
        if variable not in self._masks:
            raw = self._hf.read_file(
                PREDICTOR_VALID_MASK_PATH.format(variable=variable)
            )
            self._masks[variable] = None if raw is None else self._decode(raw)
        return self._masks[variable]

    def get(
        self,
        dates: Sequence[date | str],
        variables: Sequence[str],
        apply_valid_mask: bool = False,
        allow_monthly_upsampling: bool = False,
    ) -> np.ndarray:
        """
        Read predictors for a date list, shaped `[T, C, H, W]`.

        `T` always equals `len(dates)` and `C` follows the order of `variables`,
        so the result concatenates straight onto the ice frames along the
        channel axis.

        Args:
            dates (`Sequence[date]` or `Sequence[str]`): Dates to read, in the order
                the ice loader produced them.
            variables (`Sequence[str]`): Predictor names, in the channel order you want.
            apply_valid_mask (`bool`, optional): Replace filler cells with NaN using the
                variable mask. Off by default, so what you get is what is stored;
                turn it on for statistics, where a filler is not an observation.
            allow_monthly_upsampling (`bool`, optional): Permit several dates inside one
                month to resolve to the same monthly value. Off by default, because the
                ocean reanalysis exists only per month and repeating it across a daily
                window invents variability the source does not have. Defaults to False.
        """
        if not variables:
            raise ValueError("variables must not be empty")

        days = [self._as_date(d) for d in dates]
        if not days:
            raise ValueError("dates must not be empty")

        channels = [
            self._read_variable(v, days, apply_valid_mask, allow_monthly_upsampling)
            for v in variables
        ]

        result = np.stack(channels, axis=1)  # [T, C, H, W]
        if result.shape[0] != len(days):
            raise RuntimeError(
                f"expected {len(days)} frames, produced {result.shape[0]}"
            )
        return result

    def _read_variable(
        self,
        variable: str,
        days: list[date],
        apply_valid_mask: bool,
        allow_monthly_upsampling: bool,
    ) -> np.ndarray:
        manifest = self.manifest(variable)
        step = manifest["step"]

        if step == PREDICTOR_STEP_MONTHLY and not allow_monthly_upsampling:
            months = {(d.year, d.month) for d in days}
            if len(months) != len(days):
                raise ValueError(
                    f"{variable} is monthly and several requested dates fall in the same "
                    f"month, which would repeat one value across a daily window. "
                    f"Use a monthly step, or pass allow_monthly_upsampling=True."
                )

        # every date is resolved before a single byte is fetched, so a date the
        # variable does not cover fails before any work is done
        plan = [self._locate(manifest, variable, d) for d in days]

        slice_bytes = manifest["slice_bytes"]
        header = manifest.get("npy_header_bytes", NPY_HEADER_BYTES)

        def fetch(item):
            path, index = item
            raw = self._hf.read_file_range(
                filename=path,
                offset=header + index * slice_bytes,
                length=slice_bytes,
            )
            if raw is None:
                raise ValueError(f"Chunk {path} not found in the repository")
            return np.frombuffer(raw, dtype="<f4").reshape(tuple(manifest["shape"]))

        with ThreadPoolExecutor(max_workers=self._threads) as pool:
            frames = list(pool.map(fetch, plan))

        stacked = np.stack(frames)

        if apply_valid_mask:
            mask = self.valid_mask(variable)
            if mask is not None:
                stacked = np.where(mask.astype(bool), stacked, np.nan)

        return stacked

    def _locate(self, manifest: dict, variable: str, day: date) -> tuple[str, int]:
        """Find the chunk holding a date and the index of the slice inside it."""
        step = manifest["step"]
        period = (
            f"{day.year}-{day.month:02d}"
            if step == PREDICTOR_STEP_DAILY
            else str(day.year)
        )

        for chunk in manifest["chunks"]:
            if chunk["period"] != period:
                continue

            first = self._as_date(chunk["first"])
            if step == PREDICTOR_STEP_DAILY:
                index = (day - first).days
            else:
                index = (day.year - first.year) * 12 + (day.month - first.month)

            # the manifest states how many slices the chunk really holds, so a
            # tail month that stops early is caught instead of read past the end
            if not 0 <= index < chunk["slices"]:
                raise ValueError(
                    f"{variable} has no data for {day}: chunk {chunk['path']} covers "
                    f"{chunk['first']}..{chunk['last']}"
                )
            return chunk["path"], index

        first, last = self.coverage(variable)
        raise ValueError(
            f"{variable} has no data for {day}: covered range is {first}..{last}"
        )

    @staticmethod
    def _decode(raw: bytes) -> np.ndarray:
        from io import BytesIO

        matrix = np.load(BytesIO(raw))
        if tuple(matrix.shape) != DATASET_SHAPE:
            raise ValueError(f"mask shape {matrix.shape}, expected {DATASET_SHAPE}")
        return matrix

    @staticmethod
    def _as_date(value: date | str, end_of_month: bool = False) -> date:
        if isinstance(value, date):
            return value
        parts = value.split("-")
        if len(parts) == 3:
            return datetime.strptime(value, "%Y-%m-%d").date()
        # a monthly manifest writes "YYYY-MM"
        year, month = int(parts[0]), int(parts[1])
        if not end_of_month:
            return date(year, month, 1)
        import calendar

        return date(year, month, calendar.monthrange(year, month)[1])
