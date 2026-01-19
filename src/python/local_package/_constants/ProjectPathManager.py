from local_package._constants.Strict_Read_Only import StrictReadOnlyMeta
from typing import Final


class ProjectPathManager(metaclass=StrictReadOnlyMeta):
    PROJECT_NAME: Final[str] = "IndianMonsoonOnset"
    ABSOLUTE_PATH_MAIN: Final[str] = "/work/b08209033/DATA/IndianMonsoonOnset"
    ABSOLUTE_PATH_ERA5_RAW: Final[str] = (
        "/work/b08209033/DATA/IndianMonsoonOnset/ERA5/raw_grid"
    )
    ABSOLUTE_PATH_ERA5_SPARSE: Final[str] = (
        "/work/b08209033/DATA/IndianMonsoonOnset/ERA5/sparse_grid"
    )
    ABSOLUTE_PATH_IMAGES_PYTHON: Final[str] = (
        "/home/b08209033/Projects/python/Indian_Monsoon_Onset/img/python"
    )
