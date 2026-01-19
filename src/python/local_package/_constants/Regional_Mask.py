from local_package._constants.Strict_Read_Only import StrictReadOnlyMeta
from typing import Final


class Regional_Mask(metaclass=StrictReadOnlyMeta):
    PRESSURE_LEVEL: Final[int] = 85000
    LATITUDE_NORTH: Final[float] = 35
    LATITUDE_SOUTH: Final[float] = -5
    LONGITUDE_EAST: Final[float] = 120
    LONGITUDE_WEST: Final[float] = 15
