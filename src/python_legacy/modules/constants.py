#!/usr/bin/env python
# coding: utf-8

# ## NBconvertApp

if __name__ == "__main__":
    import subprocess, os

    subprocess.run(["bash", "../convert.sh"], check=True)


# ## Dependencies

from dataclasses import dataclass
import numpy as np


# ## Absolute Paths

# **Summary**
# - Manage absolute paths for the project directories.

@dataclass(frozen=True)
class ProjectPathManager:
    NAME: str
    ABSOLUTE_PATH_MAIN: str
    ABSOLUTE_PATH_ERA5_RAW: str
    ABSOLUTE_PATH_ERA5_SPARSE: str
    ABSOLUTE_PATH_TEMPEST: str
    ABSOLUTE_PATH_SATELLITE_TEST: str
    ABSOLUTE_PATH_IMAGES: str
    ABSOLUTE_PATH_IMAGES_PRELIMINARY: str


ENVIRONMENT_PATH = ProjectPathManager(
    NAME="IndianMonsoon",
    ABSOLUTE_PATH_MAIN="/work/b08209033/DATA/IndianMonsoon",
    ABSOLUTE_PATH_ERA5_RAW="/work/b08209033/DATA/IndianMonsoon/ERA5/raw_grid",
    ABSOLUTE_PATH_ERA5_SPARSE="/work/b08209033/DATA/IndianMonsoon/ERA5/sparse_grid",
    ABSOLUTE_PATH_TEMPEST="/work/b08209033/DATA/IndianMonsoon/TempestExtremes",
    ABSOLUTE_PATH_SATELLITE_TEST="/work/b08209033/DATA/IndianMonsoon/Satellite",
    ABSOLUTE_PATH_IMAGES="/home/b08209033/IndianMonsoon/img",
    ABSOLUTE_PATH_IMAGES_PRELIMINARY="/home/b08209033/IndianMonsoon/img/preliminary",
)


# ## Moving Average Parameters

# **Summary**
# - Parameters for the moving average operator.

@dataclass(frozen=True)
class MovingAverageOperator:
    WINDOW_SIZE: int


MOVING_AVERAGE_PARAMETER = MovingAverageOperator(WINDOW_SIZE=28)


# ## Earth Parameters

# **Summary**
# - Constants related to Earth's physical properties.

@dataclass(frozen=True)
class PlanetObject:

    NAME: str
    RADIUS: float  # in meters
    GRAVITY_ACCELERATION: float  # in m/s^2
    SOLAR_DAY_TO_SECONDS: float  # length of a solar day in seconds
    ANGULAR_VELOCITY: float  # in radians per second


EARTH_PARAMETER = PlanetObject(
    NAME="Earth",
    RADIUS=6.371e6,
    GRAVITY_ACCELERATION=9.78,
    SOLAR_DAY_TO_SECONDS=86400,
    ANGULAR_VELOCITY=2 * np.pi / 86164.1,
)


# ## Wheeler-Kiladis Diagram Parameters

# **Summary**
# - Parameters for operations related to Wheeler-Kiladis diagram.
# - Including segmentation length, overlap length.
# - Does not include sample rate (not yet implemented).

@dataclass(frozen=True)
class WheelerKiladisDiagramOperator:

    SEGMENTATION_LENGTH: int  # Length of each segment
    OVERLAP_LENGTH: int  # Overlap between segments


WKD_PARAMETER = WheelerKiladisDiagramOperator(SEGMENTATION_LENGTH=96, OVERLAP_LENGTH=65)


# ## Regional Mask (region of interest)

# **Summary**
# - Parameters to specify **regions of interest** from the whole globe.
# - **Regions of interest** refers to where analysis are conducted and displayed.

@dataclass(frozen=True)
class Default_RegionMaskObject:

    LATITUDE_NORTH: float
    LATITUDE_SOUTH: float
    LONGITUDE_EAST: float
    LONGITUDE_WEST: float


REGION_MASK = Default_RegionMaskObject(
    LATITUDE_NORTH=35,
    LATITUDE_SOUTH=-5,
    LONGITUDE_EAST=120,
    LONGITUDE_WEST=15,
)
# INDIAN_MASK = IndianMonsoonRegionMask(
#     LATITUDE_NORTH=15,
#     LATITUDE_SOUTH=-15,
#     LONGITUDE_EAST=360,
#     LONGITUDE_WEST=0,
# )


# ## Regional Mask (for IMI only)

# **Summary**
# - Parameters to specify **regions given by Bin Wang** from the whole globe.
# - For more details about **regions** in this context, see Bin Wang (2008)

@dataclass(frozen=True)
class BinWang2008_RegionMaskObject:
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
BinWang2008_MASK = BinWang2008_RegionMaskObject(
    NORTHERN_LATITUDE_NORTH=30,
    NORTHERN_LATITUDE_SOUTH=20,
    NORTHERN_LONGITUDE_EAST=90,
    NORTHERN_LONGITUDE_WEST=70,
    SOUTHERN_LATITUDE_NORTH=15,
    SOUTHERN_LATITUDE_SOUTH=5,
    SOUTHERN_LONGITUDE_EAST=80,
    SOUTHERN_LONGITUDE_WEST=40,
)

