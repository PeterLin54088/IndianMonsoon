import numpy as np
from netCDF4 import Dataset


def get_meridional_mass_streamfunction(
    filepath: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from local_package._utils.split_dimension import split_dimension

    def reader(filepath: str) -> np.ndarray:
        from local_package._constants.ERA5DataManager import ERA5DataManager
        from local_package._constants.Regional_Mask import Regional_Mask

        with Dataset(filepath, mode="r") as dataset:
            varname = "v"
            dimension_names = dataset[varname].dimensions

            if ERA5DataManager.COORDS != dimension_names:
                raise ValueError("Dimension mismatch!")
            elif ERA5DataManager.SPARSE_GRID != np.shape(
                dataset[varname]
            ) and ERA5DataManager.RAW_GRID != np.shape(dataset[varname]):
                raise ValueError("Dimension mismatch!")

            dimensions = {name: dataset[name][:] for name in dimension_names}
            data_slice = [slice(None)] * len(dimensions)
            for idx, key in enumerate(dimension_names):
                if key == "time":
                    continue
                if key == "plev":
                    data_slice[idx] = slice(None, None, -1)
                    dimensions["plev"] = dimensions["plev"][data_slice[idx]]
                elif key == "lat":
                    data_slice[idx] = (
                        dimensions["lat"] <= Regional_Mask.LATITUDE_NORTH
                    ) & (dimensions["lat"] >= Regional_Mask.LATITUDE_SOUTH)
                    dimensions["lat"] = dimensions["lat"][data_slice[idx]]
                elif key == "lon":
                    data_slice[idx] = (
                        dimensions["lon"] <= Regional_Mask.LONGITUDE_EAST
                    ) & (dimensions["lon"] >= Regional_Mask.LONGITUDE_WEST)
                    dimensions["lon"] = dimensions["lon"][data_slice[idx]]
            divergent_meridional_wind = dataset[varname][tuple(data_slice)]
        return divergent_meridional_wind, dimensions

    def calculate(
        divergent_meridional_wind: np.ndarray,
        dimensions: dict[str : np.ndarray],
    ):
        from local_package._constants.EarthPlanetObject import EarthPlanetObject

        #
        divergent_meridional_wind = np.insert(
            divergent_meridional_wind, 0, 0, axis=1
        )
        pressure_levels = np.insert(dimensions["plev"], 0, 0)
        #
        divergent_meridional_wind = np.mean(divergent_meridional_wind, axis=-1)
        #
        divergent_meridional_wind = (
            divergent_meridional_wind[:, :-1, :]
            + divergent_meridional_wind[:, 1:, :]
        ) / 2  # Riemann sum, middle point method (discrete integration)
        #
        longitudinal_extent = np.deg2rad(
            dimensions["lon"][-1] - dimensions["lon"][0]
        )
        latitudinal_weighting = np.cos(np.deg2rad(dimensions["lat"]))
        weighting_factor = (
            (EarthPlanetObject.RADIUS / EarthPlanetObject.GRAVITY_ACCELERATION)
            * longitudinal_extent
            * latitudinal_weighting
        )
        #
        streamfunction = np.swapaxes(
            divergent_meridional_wind, 1, -1
        ) * np.diff(pressure_levels)
        streamfunction = np.cumsum(streamfunction, axis=-1)
        streamfunction = np.swapaxes(streamfunction, -1, 1) * weighting_factor
        #
        dimensions["plev"] /= 100  # Convert pressure levels to hPa
        return streamfunction, dimensions

    v_D, dimensions = reader(filepath)
    streamfunction, dimensions = calculate(v_D, dimensions)

    #
    streamfunction = np.flip(streamfunction, axis=1)
    dimensions["plev"] = np.flip(dimensions["plev"])

    # streamfunction = moving_average(streamfunction, axis=0)
    streamfunction = split_dimension(streamfunction, axis=0)

    return streamfunction, dimensions
