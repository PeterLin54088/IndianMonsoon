from local_package._constants.Strict_Read_Only import StrictReadOnlyMeta
from typing import Final


class MonsoonIndex_Mask(metaclass=StrictReadOnlyMeta):
    PRESSURE_LEVEL: Final[int] = 85000
    NORTHERN_LATITUDE_NORTH: Final[float] = 30
    NORTHERN_LATITUDE_SOUTH: Final[float] = 20
    NORTHERN_LONGITUDE_EAST: Final[float] = 90
    NORTHERN_LONGITUDE_WEST: Final[float] = 70
    SOUTHERN_LATITUDE_NORTH: Final[float] = 15
    SOUTHERN_LATITUDE_SOUTH: Final[float] = 5
    SOUTHERN_LONGITUDE_EAST: Final[float] = 80
    SOUTHERN_LONGITUDE_WEST: Final[float] = 40
