import numpy as np
from netCDF4 import Dataset


def get_budget_component(
    u_filepath: str, v_filepath: str, w_filepath: str, mse_filepath: str
):
    from local_package._utils.split_dimension import split_dimension
    from local_package._constants.EarthPlanetObject import EarthPlanetObject

    def reader(varname: str, filepath: str) -> np.ndarray:
        from local_package._constants.ERA5DataManager import ERA5DataManager
        from local_package._constants.Regional_Mask import Regional_Mask

        with Dataset(filepath, mode="r") as dataset:
            dimension_names = dataset[varname].dimensions
            if ERA5DataManager.COORDS != dimension_names:
                raise ValueError("Dimension mismatch!")
            elif ERA5DataManager.SPARSE_GRID != np.shape(
                dataset[varname]
            ) and ERA5DataManager.RAW_GRID != np.shape(dataset[varname]):
                raise ValueError("Dimension mismatch!")

            dimensions = {name: dataset[name][:] for name in dimension_names}
            data_slice = [slice(None)] * len(dimensions)
            for idx, coord in enumerate(dimension_names):
                if coord == "time":
                    continue
                if coord == "plev":
                    continue
                elif coord == "lat":
                    data_slice[idx] = (
                        dimensions["lat"] <= Regional_Mask.LATITUDE_NORTH
                    ) & (dimensions["lat"] >= Regional_Mask.LATITUDE_SOUTH)
                    dimensions["lat"] = dimensions["lat"][data_slice[idx]]
                elif coord == "lon":
                    data_slice[idx] = (
                        dimensions["lon"] <= Regional_Mask.LONGITUDE_EAST
                    ) & (dimensions["lon"] >= Regional_Mask.LONGITUDE_WEST)
                    dimensions["lon"] = dimensions["lon"][data_slice[idx]]
            variable = dataset[varname][tuple(data_slice)]
            variable_mean = np.mean(variable, axis=-1, keepdims=True)
            variable_deviation = variable - variable_mean
        return variable_mean, variable_deviation, dimensions

    def zonal_mean(field):
        zonal_mean = np.mean(field, axis=-1)
        return split_dimension(zonal_mean)

    def gradient_longitude(field, dims):
        shape = [1] * len(field.shape)
        shape[-2] = -1
        tmp = np.gradient(field, np.deg2rad(dims["lon"]), axis=-1, edge_order=2)
        tmp *= 1 / EarthPlanetObject.RADIUS
        tmp *= 1 / np.cos(np.deg2rad(dims["lat"])).reshape(shape)
        return tmp

    def gradient_latitude(field, dims):
        tmp = np.gradient(
            field,
            np.deg2rad(dims["lat"]),
            axis=-2,
            edge_order=2,
        )
        tmp *= 1 / EarthPlanetObject.RADIUS
        return tmp

    def gradient_pressure(field, dims):
        tmp = np.gradient(
            field,
            dims["plev"],
            axis=-3,
            edge_order=2,
        )
        return tmp

    zonal_wind_mean, zonal_wind_deviation, _ = reader(
        varname="u", filepath=u_filepath
    )
    meridional_wind_mean, meridional_wind_deviation, _ = reader(
        varname="v", filepath=v_filepath
    )
    pressure_tendency_mean, pressure_tendency_deviation, _ = reader(
        varname="w", filepath=w_filepath
    )
    mse_mean, mse_deviation, dimensions = reader(
        varname="mse", filepath=mse_filepath
    )

    ####################################################################################################
    uq_x = zonal_mean(
        (zonal_wind_mean + zonal_wind_deviation)
        * gradient_longitude(mse_mean + mse_deviation, dimensions)
    )
    vq_y = zonal_mean(
        (meridional_wind_mean + meridional_wind_deviation)
        * gradient_latitude(mse_mean + mse_deviation, dimensions)
    )
    wq_p = zonal_mean(
        (pressure_tendency_mean + pressure_tendency_deviation)
        * gradient_pressure(mse_mean + mse_deviation, dimensions)
    )

    dq_mean_dy = gradient_latitude(mse_mean, dimensions)
    dq_mean_dp = gradient_pressure(mse_mean, dimensions)
    dq_deviation_dx = gradient_longitude(mse_deviation, dimensions)
    dq_deviation_dy = gradient_latitude(mse_deviation, dimensions)
    dq_deviation_dp = gradient_pressure(mse_deviation, dimensions)

    u_m_q_d = zonal_mean(zonal_wind_mean * dq_deviation_dx)
    u_d_q_d = zonal_mean(zonal_wind_deviation * dq_deviation_dx)

    v_m_q_m = zonal_mean(meridional_wind_mean * dq_mean_dy)
    v_m_q_d = zonal_mean(meridional_wind_mean * dq_deviation_dy)
    v_d_q_d = zonal_mean(meridional_wind_deviation * dq_deviation_dy)

    w_m_q_m = zonal_mean(pressure_tendency_mean * dq_mean_dp)
    w_m_q_d = zonal_mean(pressure_tendency_mean * dq_deviation_dp)
    w_d_q_d = zonal_mean(pressure_tendency_deviation * dq_deviation_dp)

    ###
    terms = {}
    terms["u_m_q_d"] = u_m_q_d
    terms["u_d_q_d"] = u_d_q_d
    terms["v_m_q_m"] = v_m_q_m
    terms["v_m_q_d"] = v_m_q_d
    terms["v_d_q_d"] = v_d_q_d
    terms["w_m_q_m"] = w_m_q_m
    terms["w_m_q_d"] = w_m_q_d
    terms["w_d_q_d"] = w_d_q_d
    ### Verification
    # tmp = u_m_q_d + u_d_q_d
    # print(tmp.min(), tmp.mean(), tmp.max())
    # print(uq_x.min(), uq_x.mean(), uq_x.max())

    # tmp = v_m_q_m + v_m_q_d + v_d_q_d
    # print(tmp.min(), tmp.mean(), tmp.max())
    # print(vq_y.min(), vq_y.mean(), vq_y.max())

    # tmp = w_m_q_m + w_m_q_d + w_d_q_d
    # print(tmp.min(), tmp.mean(), tmp.max())
    # print(wq_p.min(), wq_p.mean(), wq_p.max())
    return uq_x, vq_y, wq_p, terms, dimensions
