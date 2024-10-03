from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ProjectPathManager:
    NAME: str
    ABSOLUTE_PATH_MAIN: str
    ABSOLUTE_PATH_ERA5: str
    ABSOLUTE_PATH_TEMPEST: str
    ABSOLUTE_PATH_SATELLITE_TEST: str
    ABSOLUTE_PATH_IMAGES: str


ENVIRONMENT_PATH = ProjectPathManager(
    NAME="IndianMonsoon",
    ABSOLUTE_PATH_MAIN="/work/b08209033/DATA/IndianMonsoon",
    ABSOLUTE_PATH_ERA5="/work/b08209033/DATA/IndianMonsoon/ERA5",
    ABSOLUTE_PATH_TEMPEST="/work/b08209033/DATA/IndianMonsoon/TempestExtremes",
    ABSOLUTE_PATH_SATELLITE_TEST="/work/b08209033/DATA/IndianMonsoon/Satellite",
    ABSOLUTE_PATH_IMAGES="/home/b08209033/IndianMonsoon/img",
)


@dataclass(frozen=True)
class PlanetConstants:
    NAME: str
    RADIUS: float
    GRAVITY_ACCELERATION: float
    SOLAR_DAY_TO_SECONDS: float
    ANGULAR_VELOCITY: float


EARTH = PlanetConstants(
    NAME="Earth",
    RADIUS=6.371e6,
    GRAVITY_ACCELERATION=9.78,
    SOLAR_DAY_TO_SECONDS=86400,
    ANGULAR_VELOCITY=2 * np.pi / 86164.1,
)


@dataclass(frozen=True)
class IndianMaskRegion:
    latitude_north: float
    latitude_south: float
    longitude_east: float
    longitude_west: float


REGIONAL_MASK = IndianMaskRegion(
    latitude_north=15,
    latitude_south=-15,
    longitude_east=360,
    longitude_west=0,
)

REGIONAL_MASK_TUPLE = (
    REGIONAL_MASK.latitude_north,
    REGIONAL_MASK.latitude_south,
    REGIONAL_MASK.longitude_east,
    REGIONAL_MASK.longitude_west,
)

WINDOW_SIZE = 28
