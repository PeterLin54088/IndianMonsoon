from local_package._constants.Strict_Read_Only import StrictReadOnlyMeta
from typing import Final
from datetime import datetime


class ERA5DataManager(metaclass=StrictReadOnlyMeta):
    COORDS: Final[tuple[str]] = ("time", "plev", "lat", "lon")
    RAW_GRID: Final[tuple[int]] = (15695, 8, 360, 576)
    SPARSE_GRID: Final[tuple[int]] = (15695, 8, 90, 144)
    YYMMDD: Final[datetime] = datetime(1979, 1, 1)
