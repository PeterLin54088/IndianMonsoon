import numpy as np
from netCDF4 import Dataset


def get_monsoon_index(filepath: str) -> np.ndarray:
    from local_package._utils.split_dimension import split_dimension

    def reader(filepath: str) -> np.ndarray:
        from local_package._constants.ERA5DataManager import ERA5DataManager
        from local_package._constants.MonsoonIndex_Mask import MonsoonIndex_Mask

        with Dataset(filepath, mode="r") as dataset:
            varname = "u"
            dimension_names = dataset[varname].dimensions
            if ERA5DataManager.COORDS != dimension_names:
                raise ValueError("Dimension mismatch!")
            elif ERA5DataManager.SPARSE_GRID != np.shape(
                dataset[varname]
            ) and ERA5DataManager.RAW_GRID != np.shape(dataset[varname]):
                raise ValueError("Dimension mismatch!")

            dimensions = {
                coord: dataset[coord][:] for coord in ERA5DataManager.COORDS
            }
            northern_slice = [slice(None)] * len(dimensions)
            southern_slice = [slice(None)] * len(dimensions)
            for idx, coord in enumerate(ERA5DataManager.COORDS):
                if coord == "time":
                    continue
                elif coord == "plev":
                    plev_mask = np.argwhere(
                        dimensions[coord] == MonsoonIndex_Mask.PRESSURE_LEVEL
                    )[-1]
                    dimensions[coord] = dimensions[coord][plev_mask] / 100
                    northern_slice[idx] = southern_slice[idx] = plev_mask
                elif coord == "lat":
                    northern_slice[idx] = (
                        dimensions["lat"]
                        <= MonsoonIndex_Mask.NORTHERN_LATITUDE_NORTH
                    ) & (
                        dimensions["lat"]
                        >= MonsoonIndex_Mask.NORTHERN_LATITUDE_SOUTH
                    )
                    southern_slice[idx] = (
                        dimensions["lat"]
                        <= MonsoonIndex_Mask.SOUTHERN_LATITUDE_NORTH
                    ) & (
                        dimensions["lat"]
                        >= MonsoonIndex_Mask.SOUTHERN_LATITUDE_SOUTH
                    )
                elif coord == "lon":
                    northern_slice[idx] = (
                        dimensions["lon"]
                        <= MonsoonIndex_Mask.NORTHERN_LONGITUDE_EAST
                    ) & (
                        dimensions["lon"]
                        >= MonsoonIndex_Mask.NORTHERN_LONGITUDE_WEST
                    )
                    southern_slice[idx] = (
                        dimensions["lon"]
                        <= MonsoonIndex_Mask.SOUTHERN_LONGITUDE_EAST
                    ) & (
                        dimensions["lon"]
                        >= MonsoonIndex_Mask.SOUTHERN_LONGITUDE_WEST
                    )

            zonal_wind_south = dataset[varname][tuple(southern_slice)]
            zonal_wind_north = dataset[varname][tuple(northern_slice)]

        return zonal_wind_south, zonal_wind_north

    def regional_mean(zonal_wind):
        regional_mean_zonal_wind = zonal_wind.squeeze().mean(axis=(1, 2))
        return regional_mean_zonal_wind

    zonal_wind_south, zonal_wind_north = reader(filepath)
    monsoon_index = regional_mean(zonal_wind_south) - regional_mean(
        zonal_wind_north
    )
    monsoon_index = split_dimension(monsoon_index)
    return monsoon_index
