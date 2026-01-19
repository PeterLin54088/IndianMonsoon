import numpy as np
from netCDF4 import Dataset


def get_vertical_mse_flux(
    mse_filepath: str, w_filepath: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from local_package._utils.split_dimension import split_dimension

    def reader(mse_filepath, w_filepath):
        from local_package._constants.ERA5DataManager import ERA5DataManager
        from local_package._constants.Regional_Mask import Regional_Mask

        with (
            Dataset(mse_filepath, mode="r") as mse_dataset,
            Dataset(w_filepath, mode="r") as w_dataset,
        ):
            if ERA5DataManager.COORDS != mse_dataset["mse"].dimensions:
                raise ValueError("Dimension mismatch!")
            elif ERA5DataManager.SPARSE_GRID != np.shape(
                mse_dataset["mse"]
            ) and ERA5DataManager.RAW_GRID != np.shape(mse_dataset["mse"]):
                raise ValueError("Dimension mismatch!")

            if ERA5DataManager.COORDS != w_dataset["w"].dimensions:
                raise ValueError("Dimension mismatch!")
            elif ERA5DataManager.SPARSE_GRID != np.shape(
                w_dataset["w"]
            ) and ERA5DataManager.RAW_GRID != np.shape(w_dataset["w"]):
                raise ValueError("Dimension mismatch!")

            if np.shape(mse_dataset["mse"]) != np.shape(w_dataset["w"]):
                raise ValueError("Dimension mismatch!")

            dimensions = {
                name: mse_dataset[name][:]
                for name in mse_dataset["mse"].dimensions
            }

            data_slice = [slice(None)] * len(dimensions)
            for idx, key in enumerate(ERA5DataManager.COORDS):
                if key == "time":
                    continue
                elif key == "plev":
                    continue
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
            moist_static_energy = mse_dataset["mse"][tuple(data_slice)]
            pressure_tendency = w_dataset["w"][tuple(data_slice)]
            dimensions["plev"] /= 100
        return moist_static_energy, pressure_tendency, dimensions

    moist_static_energy, pressure_tendency, dimensions = reader(
        mse_filepath, w_filepath
    )
    moist_static_energy_flux = np.mean(
        moist_static_energy * pressure_tendency, axis=-1
    )
    moist_static_energy_flux = split_dimension(moist_static_energy_flux, axis=0)
    return moist_static_energy_flux, dimensions
