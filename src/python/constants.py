from dataclasses import dataclass
import numpy as np

######################
# Absolute Paths
######################


@dataclass(frozen=True)
class ProjectPathManager:
    """
    Manage absolute paths for the project directories.
    """

    NAME: str
    ABSOLUTE_PATH_MAIN: str
    ABSOLUTE_PATH_ERA5_RAW: str
    ABSOLUTE_PATH_ERA5_SPARSE: str
    ABSOLUTE_PATH_TEMPEST: str
    ABSOLUTE_PATH_SATELLITE_TEST: str
    ABSOLUTE_PATH_IMAGES: str


# Define the absolute paths for various directories
ENVIRONMENT_PATH = ProjectPathManager(
    NAME="IndianMonsoon",
    ABSOLUTE_PATH_MAIN="/work/b08209033/DATA/IndianMonsoon",
    ABSOLUTE_PATH_ERA5_RAW="/work/b08209033/DATA/IndianMonsoon/ERA5/raw_grid",
    ABSOLUTE_PATH_ERA5_SPARSE="/work/b08209033/DATA/IndianMonsoon/ERA5/sparse_grid",
    ABSOLUTE_PATH_TEMPEST="/work/b08209033/DATA/IndianMonsoon/TempestExtremes",
    ABSOLUTE_PATH_SATELLITE_TEST="/work/b08209033/DATA/IndianMonsoon/Satellite",
    ABSOLUTE_PATH_IMAGES="/home/b08209033/IndianMonsoon/img",
)


######################
# Earth Parameters
######################


@dataclass(frozen=True)
class PlanetConstants:
    """
    Constants related to Earth's physical properties and parameters.
    """

    NAME: str
    RADIUS: float  # in meters
    GRAVITY_ACCELERATION: float  # in m/s^2
    SOLAR_DAY_TO_SECONDS: float  # length of a solar day in seconds
    ANGULAR_VELOCITY: float  # in radians per second


# Constants related to the planet Earth
EARTH = PlanetConstants(
    NAME="Earth",
    RADIUS=6.371e6,  # Earth's radius in meters
    GRAVITY_ACCELERATION=9.78,  # in m/s^2
    SOLAR_DAY_TO_SECONDS=86400,  # one solar day in seconds
    ANGULAR_VELOCITY=2 * np.pi / 86164.1,  # radians/second (1 sidereal day)
)


######################
# Regional Mask, Indian
######################


@dataclass(frozen=True)
class IndianMonsoonRegionMask:
    """
    Define the latitudinal and longitudinal bounds of the Indian monsoon region.
    """

    LATITUDE_NORTH: float
    LATITUDE_SOUTH: float
    LONGITUDE_EAST: float
    LONGITUDE_WEST: float


# The lat/lon boundaries for the Indian monsoon region
# INDIAN_MASK = IndianMonsoonRegionMask(
#     LATITUDE_NORTH=20,
#     LATITUDE_SOUTH=5,
#     LONGITUDE_EAST=90,
#     LONGITUDE_WEST=45,
# )
INDIAN_MASK = IndianMonsoonRegionMask(
    LATITUDE_NORTH=15,
    LATITUDE_SOUTH=-15,
    LONGITUDE_EAST=360,
    LONGITUDE_WEST=0,
)


######################
# Regional Mask, ZWS indexing
######################


@dataclass(frozen=True)
class ZonalWindShearRegionMask:
    """
    Define the latitudinal and longitudinal bounds of the predefined zonal wind shear region.
    Follow BIN WANG
    """

    NORTHERN_LATITUDE_NORTH: float
    NORTHERN_LATITUDE_SOUTH: float
    NORTHERN_LONGITUDE_EAST: float
    NORTHERN_LONGITUDE_WEST: float
    SOUTHERN_LATITUDE_NORTH: float
    SOUTHERN_LATITUDE_SOUTH: float
    SOUTHERN_LONGITUDE_EAST: float
    SOUTHERN_LONGITUDE_WEST: float


# The lat/lon boundaries for the predefined zonal wind shear region
ZWS_MASK = ZonalWindShearRegionMask(
    NORTHERN_LATITUDE_NORTH=30,
    NORTHERN_LATITUDE_SOUTH=20,
    NORTHERN_LONGITUDE_EAST=90,
    NORTHERN_LONGITUDE_WEST=70,
    SOUTHERN_LATITUDE_NORTH=15,
    SOUTHERN_LATITUDE_SOUTH=5,
    SOUTHERN_LONGITUDE_EAST=80,
    SOUTHERN_LONGITUDE_WEST=40,
)


######################
# Moving Average Parameters
######################


@dataclass(frozen=True)
class MovingAverageConstants:
    """
    Parameters for the moving average window size.
    """

    WINDOW_SIZE: int


# Moving average window size for smoothing operations
MOVING_AVERAGE = MovingAverageConstants(WINDOW_SIZE=28)


######################
# WK99 Parameters
######################


@dataclass(frozen=True)
class WK99Constants:
    """
    Constants for WK99 segmentation, including segmentation length, overlap length, and sample rate.
    """

    SEGMENTATION_LENGTH: int  # Length of each segment
    OVERLAP_LENGTH: int  # Overlap between segments
    SAMPLE_RATE: int  # Sample rate per day


# Constants for the WK99 data segmentation approach
WK99 = WK99Constants(SEGMENTATION_LENGTH=96, OVERLAP_LENGTH=10, SAMPLE_RATE=1)
