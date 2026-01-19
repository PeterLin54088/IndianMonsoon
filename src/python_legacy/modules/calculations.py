#!/usr/bin/env python
# coding: utf-8

# ## NBconvertApp

if __name__ == "__main__":
    import subprocess, os

    subprocess.run(["bash", "../convert.sh"], check=True)


# ## Dependencies

import numpy as np
from netCDF4 import Dataset


# ## Functions

# ### Indian Monsoon Index (Bin Wang)

def Indian_Monsoon_Onset_Bin_Wang(filepath: str) -> np.ndarray:
    """
    Computes the Indian Monsoon Index (IMI) as the difference in zonal wind
    between southern and northern regions defined by geographic masks, using
    data from a netCDF file.

    Parameters:
    ----------
    filepath : str
        Path to the netCDF file containing zonal wind data.

    Returns:
    -------
    np.ndarray
        A 2D array of smoothed IMI values with shape (-1, 365), where each row
        represents a year's daily values.
    """
    from .constants import INDIAN_MONSOON_MASK
    from .utils import moving_average
    import time

    start = time.time()
    with Dataset(filepath, mode="r") as dataset:
        varname = "u"
        dimension_names = dataset[varname].dimensions
        dimensions = {name: dataset[name][:] for name in dimension_names}

        northern_slice = [slice(None)] * len(dimensions)
        southern_slice = [slice(None)] * len(dimensions)
        for idx, key in enumerate(dimension_names):
            if key == "time":
                continue
            elif key == "plev":
                plev_mask = np.argwhere(dimensions[key] == 85000)[-1]
                dimensions[key] = dimensions[key][plev_mask] / 100
                northern_slice[idx] = southern_slice[idx] = plev_mask
            elif key == "lat":
                northern_slice[idx] = (
                    dimensions["lat"] <= INDIAN_MONSOON_MASK.NORTHERN_LATITUDE_NORTH
                ) & (dimensions["lat"] >= INDIAN_MONSOON_MASK.NORTHERN_LATITUDE_SOUTH)
                southern_slice[idx] = (
                    dimensions["lat"] <= INDIAN_MONSOON_MASK.SOUTHERN_LATITUDE_NORTH
                ) & (dimensions["lat"] >= INDIAN_MONSOON_MASK.SOUTHERN_LATITUDE_SOUTH)
            elif key == "lon":
                northern_slice[idx] = (
                    dimensions["lon"] <= INDIAN_MONSOON_MASK.NORTHERN_LONGITUDE_EAST
                ) & (dimensions["lon"] >= INDIAN_MONSOON_MASK.NORTHERN_LONGITUDE_WEST)
                southern_slice[idx] = (
                    dimensions["lon"] <= INDIAN_MONSOON_MASK.SOUTHERN_LONGITUDE_EAST
                ) & (dimensions["lon"] >= INDIAN_MONSOON_MASK.SOUTHERN_LONGITUDE_WEST)
        zonal_wind_north = (
            dataset[varname][tuple(northern_slice)].squeeze().mean(axis=(1, 2))
        )
        zonal_wind_south = (
            dataset[varname][tuple(southern_slice)].squeeze().mean(axis=(1, 2))
        )
    indian_monsoon_index = zonal_wind_south - zonal_wind_north
    indian_monsoon_index_smoothed = moving_average(indian_monsoon_index).reshape(
        -1, 365
    )
    return indian_monsoon_index_smoothed


def streamfunction_Schwendike(
    filepath: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute the mass streamfunction using the Schwendike et al. method.

    Parameters:
        filepath (str): Path to the dataset file containing meridional wind data.

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]:
            - Smoothed mass streamfunction (3D array).
            - Dimensions dictionary containing pressure levels, latitude, and longitude arrays.
    """
    from .utils import moving_average, split_dimension
    from .constants import INDIAN_MASK
    from .constants import EARTH

    with Dataset(filepath, mode="r") as dataset:
        varname = "v"
        dimension_names = dataset[varname].dimensions
        dimensions = {name: dataset[name][:] for name in dimension_names}

        data_slice = [slice(None)] * len(dimensions)
        for idx, key in enumerate(dimension_names):
            if key == "time":
                continue
            if key == "plev":
                data_slice[idx] = slice(None, None, -1)
                dimensions["plev"] = dimensions["plev"][data_slice[idx]]
            elif key == "lat":
                data_slice[idx] = (dimensions["lat"] <= INDIAN_MASK.LATITUDE_NORTH) & (
                    dimensions["lat"] >= INDIAN_MASK.LATITUDE_SOUTH
                )
                dimensions["lat"] = dimensions["lat"][data_slice[idx]]
            elif key == "lon":
                data_slice[idx] = (dimensions["lon"] <= INDIAN_MASK.LONGITUDE_EAST) & (
                    dimensions["lon"] >= INDIAN_MASK.LONGITUDE_WEST
                )
                dimensions["lon"] = dimensions["lon"][data_slice[idx]]

        divergent_meridional_wind = dataset[varname][tuple(data_slice)]

    divergent_meridional_wind = np.insert(divergent_meridional_wind, 0, 0, axis=1)
    pressure_levels = np.insert(dimensions["plev"], 0, 0)
    divergent_meridional_wind = np.mean(divergent_meridional_wind, axis=-1)
    divergent_meridional_wind = (
        divergent_meridional_wind[:, :-1, :] + divergent_meridional_wind[:, 1:, :]
    ) / 2

    longitudinal_extent = np.deg2rad(dimensions["lon"][-1] - dimensions["lon"][0])
    latitudinal_weighting = np.cos(np.deg2rad(dimensions["lat"]))
    weighting_factor = (
        (EARTH.RADIUS / EARTH.GRAVITY_ACCELERATION)
        * longitudinal_extent
        * latitudinal_weighting
    )

    streamfunction = np.swapaxes(divergent_meridional_wind, 1, -1) * np.diff(
        pressure_levels
    )
    streamfunction = np.cumsum(streamfunction, axis=-1)
    streamfunction = np.swapaxes(streamfunction, -1, 1) * weighting_factor

    streamfunction_smoothed = moving_average(streamfunction, axis=0)
    streamfunction_smoothed = split_dimension(streamfunction_smoothed, axis=0)
    dimensions["plev"] /= 100  # Convert pressure levels to hPa

    return streamfunction_smoothed, dimensions


def calculate_MSE_vertical_flux(
    mse_filepath: str, w_filepath: str
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Calculate the mean and smoothed vertical flux of moist static energy (MSE)
    within a specific geographical mask.

    Args:
        mse_filepath (str): Filepath to the NetCDF file containing moist static energy data.
        w_filepath (str): Filepath to the NetCDF file containing vertical velocity data.

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]]:
            - Smoothed MSE vertical flux data as a NumPy array.
            - Dimensions dictionary with processed coordinate arrays.
    """
    from .utils import moving_average, split_dimension
    from .constants import INDIAN_MASK

    with Dataset(mse_filepath, mode="r") as mse_dataset, Dataset(
        w_filepath, mode="r"
    ) as w_dataset:
        varname = "mse"
        dimension_names = mse_dataset[varname].dimensions
        dimensions = {name: mse_dataset[name][:] for name in dimension_names}

        data_slice = [slice(None)] * len(dimensions)
        for idx, key in enumerate(mse_dataset[varname].dimensions):
            if key == "time":
                continue
            elif key == "plev":
                continue
            elif key == "lat":
                data_slice[idx] = (dimensions["lat"] <= INDIAN_MASK.LATITUDE_NORTH) & (
                    dimensions["lat"] >= INDIAN_MASK.LATITUDE_SOUTH
                )
                dimensions["lat"] = dimensions["lat"][data_slice[idx]]
            elif key == "lon":
                data_slice[idx] = (dimensions["lon"] <= INDIAN_MASK.LONGITUDE_EAST) & (
                    dimensions["lon"] >= INDIAN_MASK.LONGITUDE_WEST
                )
                dimensions["lon"] = dimensions["lon"][data_slice[idx]]
        moist_static_energy = mse_dataset["mse"][tuple(data_slice)]
        pressure_tendency = w_dataset["w"][tuple(data_slice)]

    moist_static_energy_flux = np.mean(moist_static_energy * pressure_tendency, axis=-1)
    moist_static_energy_flux_smoothed = moving_average(moist_static_energy_flux, axis=0)
    moist_static_energy_flux_smoothed = split_dimension(
        moist_static_energy_flux_smoothed, axis=0
    )

    dimensions["plev"] /= 100
    return moist_static_energy_flux_smoothed, dimensions


def power_spectrum_Wheeler_Kiladis(
    variable: np.ndarray, dimensions: dict[str, np.ndarray], **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    from scipy.signal import detrend as scipy_linear_detrend
    import time

    def initialize_kwargs(kwargs: dict) -> dict:
        """
        Initializes the provided `kwargs` dictionary with default values.

        Parameters:
            kwargs (dict): A dictionary of optional parameters to initialize or update.

        Returns:
            dict: The updated dictionary containing default values for missing keys.
        """
        from .constants import INDIAN_MASK, WK99

        boundary_defaults = {
            "east_boundary": INDIAN_MASK.LONGITUDE_EAST,
            "west_boundary": INDIAN_MASK.LONGITUDE_WEST,
            "north_boundary": INDIAN_MASK.LATITUDE_NORTH,
            "south_boundary": INDIAN_MASK.LATITUDE_SOUTH,
        }

        segmentation_defaults = {
            "segment_length": WK99.SEGMENTATION_LENGTH,
            "overlap_length": WK99.OVERLAP_LENGTH,
        }

        convolution_defaults = {"kernel": np.array([1 / 4, 1 / 2, 1 / 4])}
        convolution_defaults.update(
            {
                "smoother": lambda m: np.convolve(
                    m, convolution_defaults["kernel"], mode="valid"
                )
            }
        )

        other_defaults = {
            "cutoff_frequency": 1 / WK99.SEGMENTATION_LENGTH,
            "sampling_rate": 1,
        }

        defaults = {}
        defaults.update(boundary_defaults)
        defaults.update(segmentation_defaults)
        defaults.update(convolution_defaults)
        defaults.update(other_defaults)

        for key, value in defaults.items():
            kwargs.setdefault(key, value)

        return kwargs

    kwargs = initialize_kwargs(kwargs)

    def __high_pass_filter(
        signal: np.ndarray,
        axis: int,
        cutoff_frequency: float = kwargs["cutoff_frequency"],
    ) -> np.ndarray:
        """
        Applies a high-pass filter to the input signal by removing frequency components
        below the specified cutoff frequency.

        Parameters:
            signal (np.ndarray): The input signal array to filter.
            axis (int): The axis along which to apply the Fourier transform and filtering.
            cutoff_frequency (float): The cutoff frequency for the high-pass filter.

        Returns:
            np.ndarray: The filtered signal after removing low-frequency components.
        """
        signal_length = signal.shape[axis]

        fourier_component = np.fft.rfft(signal, axis=axis)
        positive_frequencies = np.fft.rfftfreq(signal_length)

        filter_condition = [slice(None)] * signal.ndim
        filter_condition[axis] = positive_frequencies < cutoff_frequency
        fourier_component[tuple(filter_condition)] = 0.0

        filtered_signal = np.fft.irfft(fourier_component, n=signal_length, axis=axis)

        return filtered_signal

    def __decompose_symmetric_antisymmetric(
        variable: np.ndarray, axis: int
    ) -> np.ndarray:
        """
        Decomposes the input array into its symmetric and antisymmetric components
        along equator.

        Parameters:
            variable (np.ndarray): The input array to decompose.
            axis (int): The axis along which the decomposition is performed, should be latitude.

        Returns:
            np.ndarray: A two-element array containing the symmetric component
            at index 0 and the antisymmetric component at index 1.
        """
        flipped_variable = np.flip(variable, axis=axis)
        symmetric_component = (variable + flipped_variable) / 2
        antisymmetric_component = (variable - flipped_variable) / 2

        return np.array([symmetric_component, antisymmetric_component])

    def __latitude_masking(
        variable: np.ndarray, dimensions: dict[str, np.ndarray], axis: int
    ):
        """
        Applies a latitude mask to filter a variable and its corresponding latitude dimension.

        Parameters:
            variable (np.ndarray): The input array to be masked.
            dimensions (dict[str, np.ndarray]): A dictionary containing dimension arrays,
                including "lat" for latitude.
            axis (int): The axis in the `variable` array corresponding to the latitude dimension.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The masked variable array.
                - dict[str, np.ndarray]: The updated dimensions dictionary with the masked "lat".
        """
        data_slice = [slice(None)] * variable.ndim
        lat_mask = (dimensions["lat"] <= kwargs["north_boundary"]) & (
            dimensions["lat"] >= kwargs["south_boundary"]
        )
        data_slice[axis] = lat_mask
        variable = variable[tuple(data_slice)]
        dimensions["lat"] = dimensions["lat"][lat_mask]
        return variable, dimensions

    def __smoothing_filter(
        signal: np.ndarray,
        axis: int,
        iterations: int,
    ) -> np.ndarray:
        """
        Applies a smoothing filter to the input signal along a specified axis for a given
        number of iterations.

        Parameters:
            signal (np.ndarray): The input signal array to be smoothed.
            axis (int): The axis along which the smoothing filter is applied.
            iterations (int): The number of smoothing iterations to perform.

        Returns:
            np.ndarray: The smoothed signal array.
        """

        def __duplicate_boundaries(signal: np.ndarray, axis: int) -> np.ndarray:
            boundary_slice = [slice(None)] * signal.ndim
            boundary_slice[axis] = slice(None, 1)
            left_boundary = signal[tuple(boundary_slice)]

            boundary_slice = [slice(None)] * signal.ndim
            boundary_slice[axis] = slice(-1, None)
            right_boundary = signal[tuple(boundary_slice)]

            return np.concatenate(
                (left_boundary, signal, right_boundary),
                axis=axis,
            )

        def __Gaussian_blur(signal: np.ndarray, axis: int):
            return np.apply_along_axis(
                kwargs["smoother"],
                axis=axis,
                arr=signal,
            )

        for _ in range(iterations):
            signal = __duplicate_boundaries(signal, axis=axis)
            signal = __Gaussian_blur(signal, axis=axis)
        return signal

    def __temporal_taper(signal: np.ndarray, portion: float, axis: int) -> np.ndarray:
        """
        Applies a temporal tapering window to the signal along the specified axis.

        Parameters:
            signal (np.ndarray): The input signal array to which the taper will be applied.
            portion (float): The fraction of the signal's length (per axis) to apply the taper.
                Must be between 0 and 1.
            axis (int): The axis along which the taper is applied.

        Returns:
            np.ndarray: A tapering window that can be applied to the signal.
        """

        def __get_taper():
            taper_width = int(signal.shape[axis] * (portion / 2))
            taper = 0.5 * (
                1 - np.cos(2 * np.pi * np.arange(taper_width) / (2 * taper_width))
            )

            taper_window = np.ones(signal.shape[axis])
            taper_window[:taper_width] = taper
            taper_window[-taper_width:] = taper[::-1]

            broadcast_shape = [1] * signal.ndim
            broadcast_shape[axis] = signal.shape[axis]
            taper_window = taper_window.reshape(broadcast_shape)
            return taper_window

        taper = __get_taper()

        return taper

    def __spatial_taper(
        signal: np.ndarray,
        dimensions: dict[str, np.ndarray],
        axis: int,
    ) -> np.ndarray:
        """
        Applies a spatial tapering window to the signal along a longitude axis.

        Parameters:
            signal (np.ndarray): The input signal array to which the taper will be applied.
            dimensions (dict[str, np.ndarray]): A dictionary containing dimension arrays,
                including "lon" for longitude.
            axis (int): The axis corresponding to the longitude dimension in the signal.

        Returns:
            np.ndarray: A spatial tapering window that can be applied to the signal.
        """

        def __get_taper():
            longitude = dimensions["lon"]
            global_flag = (
                int(abs(kwargs["east_boundary"] - kwargs["west_boundary"])) == 360
            )

            if global_flag:
                taper_window = np.ones_like(longitude)
            else:
                taper_window = np.zeros_like(longitude, dtype=float)
                regional_mask = (longitude >= kwargs["west_boundary"]) & (
                    longitude <= kwargs["east_boundary"]
                )
                regional_taper_window = taper_window[regional_mask]
                taper_width = int(len(regional_taper_window) * 0.1)
                taper = 0.5 * (
                    1 - np.cos(2 * np.pi * np.arange(taper_width) / (2 * taper_width))
                )
                regional_taper_window[:taper_width] = taper
                regional_taper_window[taper_width:-taper_width] = 1
                regional_taper_window[-taper_width:] = taper[::-1]
                taper_window[regional_mask] = regional_taper_window

            broadcast_shape = [1] * signal.ndim
            broadcast_shape[axis] = len(taper_window)
            taper_window = taper_window.reshape(broadcast_shape)
            return taper_window

        taper = __get_taper()
        return taper

    def __process_segment(
        variable: np.ndarray,
        dimensions: dict[str, np.ndarray],
        axis: int,
    ):
        """
        Processes a multidimensional array by dividing it into segments, applying tapers,
        detrending, and performing spectral analysis.

        Parameters:
            variable (np.ndarray): The input data array to be processed.
            dimensions (dict[str, np.ndarray]): Dictionary of dimension arrays, such as "lon" for longitude.
            axis (int): The axis along which the segmentation and processing are applied.

        Returns:
            np.ndarray: The computed symmetric-antisymmetric power spectrum.
        """

        def __segment_data(
            data: np.ndarray,
            axis: int,
        ):
            step = kwargs["segment_length"] - kwargs["overlap_length"]
            num_iterations = (data.shape[axis] - kwargs["overlap_length"]) // step

            def __generator():
                for counter in range(num_iterations):
                    start = counter * step
                    end = start + kwargs["segment_length"]
                    segment_slice = [
                        slice(None)
                    ] * data.ndim  # Create slices for all dimensions
                    segment_slice[axis] = slice(
                        start, end
                    )  # Set slice for the target axis
                    yield data[tuple(segment_slice)]

            return num_iterations, __generator()

        powerspec_shape = list(np.shape(variable))
        del powerspec_shape[-2]
        powerspec_shape[axis] = kwargs["segment_length"]
        sym_asym_powerspec = np.zeros(shape=tuple(powerspec_shape), dtype=float)

        __segment_data = kwargs.get("segment_method", __segment_data)
        num_segments, segment_iterator = __segment_data(
            data=variable,
            axis=axis,
        )

        for idx, segment in enumerate(segment_iterator):
            if idx == 0:
                temporal_taper = __temporal_taper(segment, portion=0.2, axis=axis)
                spatial_taper = __spatial_taper(segment, dimensions=dimensions, axis=-1)

            segment = scipy_linear_detrend(segment, axis=axis)
            segment *= temporal_taper
            segment *= spatial_taper
            segment = np.fft.fft(segment, axis=-1, norm="ortho")
            segment = np.fft.ifft(segment, axis=1, norm="ortho")

            segment = np.sum(np.abs(segment) ** 2, axis=-2)
            sym_asym_powerspec += segment

        sym_asym_powerspec /= num_segments
        sym_asym_powerspec = np.fft.fftshift(sym_asym_powerspec, axes=(axis, -1))

        return sym_asym_powerspec

    def __update_dimensions(dimensions: dict[str, np.ndarray]):
        """
        Updates the dimensions dictionary by calculating and adding zonal wavenumbers
        and segment frequencies in cycles per day (CPD).

        Parameters:
            dimensions (dict[str, np.ndarray]): A dictionary containing dimension arrays,
                including "lon" for longitude.

        Returns:
            dict[str, np.ndarray]: The updated dimensions dictionary with added keys:
                - "segment_frequency": Segment frequencies in CPD.
                - "zonal_wavenumber": Wavenumbers corresponding to the longitude dimension.
        """
        ordinary_wavenumber = np.fft.fftshift(
            np.fft.fftfreq(len(dimensions["lon"]), 1 / len(dimensions["lon"]))
        )
        ordinary_frequency = np.fft.fftshift(
            np.fft.fftfreq(kwargs["segment_length"], 1 / kwargs["segment_length"])
        )
        CPD_frequency = (
            ordinary_frequency / len(ordinary_frequency) * kwargs["sampling_rate"]
        )  # Convert frequency to cycles per day (CPD)
        dimensions.update(
            {
                "segment_frequency": CPD_frequency,
                "zonal_wavenumber": ordinary_wavenumber,
            }
        )
        return dimensions

    def __process_powerspec(
        powerspec: np.ndarray, dimensions: dict[str, np.ndarray], axis: int
    ):
        """
        Processes the power spectrum by applying smoothing filters to positive and negative
        frequency components and setting zero-frequency components to NaN.

        Parameters:
            powerspec (np.ndarray): The input power spectrum to be processed.
            dimensions (dict[str, np.ndarray]): A dictionary containing dimension arrays,
                including "segment_frequency" for frequency values.
            axis (int): The axis corresponding to the frequency dimension in the power spectrum.

        Returns:
            np.ndarray: The processed power spectrum with smoothed positive and negative
            frequencies and zero-frequency components set to NaN.
        """
        positive_data_slice = [slice(None)] * powerspec.ndim
        positive_data_slice[axis] = dimensions["segment_frequency"] > 0
        negative_data_slice = [slice(None)] * powerspec.ndim
        negative_data_slice[axis] = dimensions["segment_frequency"] < 0
        zero_data_slice = [slice(None)] * powerspec.ndim
        zero_data_slice[axis] = dimensions["segment_frequency"] == 0

        powerspec[tuple(positive_data_slice)] = __smoothing_filter(
            powerspec[tuple(positive_data_slice)], axis=axis, iterations=1
        )
        powerspec[tuple(negative_data_slice)] = __smoothing_filter(
            powerspec[tuple(negative_data_slice)], axis=axis, iterations=1
        )
        powerspec[tuple(zero_data_slice)] = np.nan

        return powerspec

    def __process_background_powerspec(
        powerspec: np.ndarray, dimensions: dict[str, np.ndarray], axis: int
    ):
        """
        Processes the background power spectrum by applying frequency-dependent smoothing filters
        and handling positive, negative, and zero-frequency components.

        Parameters:
            powerspec (np.ndarray): The input background power spectrum to be processed.
            dimensions (dict[str, np.ndarray]): A dictionary containing dimension arrays,
                including "segment_frequency" for frequency values.
            axis (int): The axis corresponding to the frequency dimension in the power spectrum.

        Returns:
            np.ndarray: The processed background power spectrum.
        """
        powerspec = np.mean(powerspec, axis=axis)
        for i, freq in enumerate(dimensions["segment_frequency"]):
            data_slice = [slice(None)] * powerspec.ndim
            data_slice[axis] = i
            if abs(freq) == 0.0:
                continue
            elif abs(freq) <= 0.1:
                powerspec[tuple(data_slice)] = __smoothing_filter(
                    powerspec[tuple(data_slice)], axis=axis, iterations=5
                )
            elif abs(freq) <= 0.2:
                powerspec[tuple(data_slice)] = __smoothing_filter(
                    powerspec[tuple(data_slice)], axis=axis, iterations=10
                )
            elif abs(freq) <= 0.3:
                powerspec[tuple(data_slice)] = __smoothing_filter(
                    powerspec[tuple(data_slice)], axis=axis, iterations=20
                )
            else:
                powerspec[tuple(data_slice)] = __smoothing_filter(
                    powerspec[tuple(data_slice)], axis=axis, iterations=40
                )
        positive_data_slice = [slice(None)] * powerspec.ndim
        positive_data_slice[axis] = dimensions["segment_frequency"] > 0
        negative_data_slice = [slice(None)] * powerspec.ndim
        negative_data_slice[axis] = dimensions["segment_frequency"] < 0
        zero_data_slice = [slice(None)] * powerspec.ndim
        zero_data_slice[axis] = dimensions["segment_frequency"] == 0

        powerspec[tuple(positive_data_slice)] = __smoothing_filter(
            powerspec[tuple(positive_data_slice)], axis=axis, iterations=10
        )
        powerspec[tuple(negative_data_slice)] = __smoothing_filter(
            powerspec[tuple(negative_data_slice)], axis=axis, iterations=10
        )
        powerspec[tuple(zero_data_slice)] = np.nan
        return powerspec

    variable = scipy_linear_detrend(variable, axis=0)
    variable = __high_pass_filter(variable, axis=0)
    variable = __decompose_symmetric_antisymmetric(variable, axis=1)
    variable, dimensions = __latitude_masking(variable, dimensions, axis=2)
    powerspec = __process_segment(variable, dimensions, axis=1)
    dimensions = __update_dimensions(dimensions)
    sym_asym_powerspec = __process_powerspec(powerspec, dimensions, axis=1)
    background_powerspec = __process_background_powerspec(powerspec, dimensions, axis=0)

    return (
        sym_asym_powerspec[0],
        sym_asym_powerspec[1],
        background_powerspec,
        dimensions,
    )


def calculate_filtered_signal(
    file_path: str,
    zonal_wavenumber_limit: np.ndarray,
    segmentation_frequency_limit: np.ndarray,
    variable_name: str = "Undefined",
    pressure_level: int = -1,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Filter out signal by decomposing into symmetric and antisymmetric components, FFT and mask out coefficient
    , and inverse FFT to derive filtered signal.

    Parameters:
    - file_path (str): Path to the NetCDF dataset.
    - zonal_wavenumber_limit (np.ndarray): Bounds for the zonal wavenumber filter (min, max).
    - segmentation_frequency_limit (np.ndarray): Bounds for frequency filter (min, max).
    - variable_name (str): Name of the variable to process in the dataset.
    - pressure_level (int): Pressure level index to slice data on if it contains a "plev" dimension.

    Returns:
        tuple: A tuple containing:
            - Symmetric components of the PSD.
            - Antisymmetric components of the PSD.
            - Dictionary of relevant dimensions.
    """
    from gc import collect as free_memory
    from .utils import decompose_symmetric_antisymmetric
    from .constants import INDIAN_MASK

    # Load data and dimensions from the dataset
    with Dataset(file_path) as dataset:
        dims = {dim: dataset[dim][:] for dim in dataset[variable_name].dimensions}
        data_slices = [slice(None)] * len(dataset[variable_name].dimensions)

        for idx, dim in enumerate(dataset[variable_name].dimensions):
            if dim == "time" or dim == "lat" or dim == "lon":
                continue
            elif dim == "plev":
                data_slices[idx] = pressure_level
                dims[dim] = dims[dim][data_slices[idx]]

        # Extract the data for the variable using the slices
        data = dataset[variable_name][tuple(data_slices)]

    # Mask latitudes based on geographic boundaries defined in INDIAN_MASK
    lat_mask = (dims["lat"] <= INDIAN_MASK.LATITUDE_NORTH) & (
        dims["lat"] >= INDIAN_MASK.LATITUDE_SOUTH
    )

    # Decompose data into symmetric and antisymmetric components
    symmetric_components, antisymmetric_components = decompose_symmetric_antisymmetric(
        data, axis=1
    )
    del data
    free_memory()

    # Mask the symmetric and antisymmetric components by latitude
    symmetric_components = symmetric_components[:, lat_mask, :]
    antisymmetric_components = antisymmetric_components[:, lat_mask, :]
    dims["lat"] = dims["lat"][lat_mask]

    # Perform FFT in both wavenumber (lon) and frequency (time) directions
    symmetric_components = np.fft.fft(symmetric_components, axis=-1, norm="ortho")
    symmetric_components = np.fft.ifft(symmetric_components, axis=0, norm="ortho")
    antisymmetric_components = np.fft.fft(
        antisymmetric_components, axis=-1, norm="ortho"
    )
    antisymmetric_components = np.fft.ifft(
        antisymmetric_components, axis=0, norm="ortho"
    )

    # Compute the wavenumber and frequency
    ordinary_wavenumber = np.fft.fftfreq(
        symmetric_components.shape[-1], 1 / symmetric_components.shape[-1]
    )
    CPD_frequency = (
        np.fft.fftfreq(symmetric_components.shape[0], 1 / symmetric_components.shape[0])
        / symmetric_components.shape[0]
    )

    # Create a filter mask based on zonal wavenumber and frequency limits
    mask = np.zeros_like(symmetric_components, dtype=bool)

    # Positive wavenumber and frequency filtering
    t_mask, y_mask, x_mask = np.meshgrid(
        (
            (CPD_frequency >= segmentation_frequency_limit[0])
            & (CPD_frequency <= segmentation_frequency_limit[1])
        ),
        np.ones(len(dims["lat"]), dtype=bool),
        (ordinary_wavenumber >= zonal_wavenumber_limit[0])
        & (ordinary_wavenumber <= zonal_wavenumber_limit[1]),
        indexing="ij",
    )
    mask = np.logical_or(mask, (t_mask & x_mask))

    # Negative wavenumber and frequency filtering
    t_mask, y_mask, x_mask = np.meshgrid(
        (
            (CPD_frequency >= -segmentation_frequency_limit[1])
            & (CPD_frequency <= -segmentation_frequency_limit[0])
        ),
        np.ones(len(dims["lat"]), dtype=bool),
        (ordinary_wavenumber >= -zonal_wavenumber_limit[1])
        & (ordinary_wavenumber <= -zonal_wavenumber_limit[0]),
        indexing="ij",
    )
    mask = np.logical_or(mask, (t_mask & x_mask))

    # Apply mask to the symmetric and antisymmetric components
    symmetric_components *= mask
    antisymmetric_components *= mask

    # Perform inverse FFT to bring the data back to the spatial and time domain
    symmetric_components = np.fft.fft(symmetric_components, axis=0, norm="ortho")
    symmetric_components = np.fft.ifft(symmetric_components, axis=-1, norm="ortho")
    antisymmetric_components = np.fft.fft(
        antisymmetric_components, axis=0, norm="ortho"
    )
    antisymmetric_components = np.fft.ifft(
        antisymmetric_components, axis=-1, norm="ortho"
    )
    return symmetric_components, antisymmetric_components, dims

