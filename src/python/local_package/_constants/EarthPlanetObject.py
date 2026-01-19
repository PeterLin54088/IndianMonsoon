import numpy as np
from local_package._constants.Strict_Read_Only import StrictReadOnlyMeta
from typing import Final


class EarthPlanetObject(metaclass=StrictReadOnlyMeta):
    NAME: Final[str] = "Earth"
    RADIUS: Final[float] = 6.371e6
    GRAVITY_ACCELERATION: Final[float] = 9.78
    SOLAR_DAY_TO_SECONDS: Final[float] = 86400.0
    ANGULAR_VELOCITY: Final[float] = 2 * np.pi / 86164.1
