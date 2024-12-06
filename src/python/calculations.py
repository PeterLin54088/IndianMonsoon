import numpy as np
from netCDF4 import Dataset


def calculate_zonal_wind_shear(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Zonal Wind Shear (ZWS) between two predefined atmospheric regions from a NetCDF dataset.

    The function computes the ZWS at the 850 hPa pressure level by calculating the difference in mean zonal wind
    between a southern region (5°N-15°N, 40°E-80°E) and a northern region (20°N-30°N, 70°E-90°E).

    Parameters
    ----------
    filepath : str
        Path to the NetCDF dataset file.

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray)
        - zws_raw: The raw ZWS values.
        - zws_smoothed: ZWS values after applying a moving average.
        - zws_smoothed_grad: Gradient of the smoothed ZWS values.

    Notes
    -----
    - Assumes 43 years of data with 365 days per year.
    """
    from constants import ZWS_MASK
    from utils import moving_average, split_dimension

    with Dataset(filepath) as dataset:
        # Extract dimensions from the dataset
        dims = {dim: dataset[dim][:] for dim in dataset["u"].dimensions}

        # Prepare slicing objects for northern and southern regions
        northern_slice = [slice(None)] * len(dims)
        southern_slice = [slice(None)] * len(dims)

        # Iterate through dimensions and create region-specific slices
        for idx, dim in enumerate(dataset["u"].dimensions):
            if dim == "time":
                # Keep time dimension unchanged
                continue
            elif dim == "plev":
                # Handle pressure level: find and select 850 hPa level
                plev_mask = np.argwhere(dims[dim] == 85000)[-1]  # 85000 Pa = 850 hPa
                dims[dim] = dims[dim][plev_mask] / 100  # Convert Pa to hPa
                northern_slice[idx] = plev_mask
                southern_slice[idx] = plev_mask
            elif dim == "lat":
                # Define latitude bounds for northern and southern regions
                northern_slice[idx] = (
                    dims[dim] <= ZWS_MASK.NORTHERN_LATITUDE_NORTH
                ) & (dims[dim] >= ZWS_MASK.NORTHERN_LATITUDE_SOUTH)
                southern_slice[idx] = (
                    dims[dim] <= ZWS_MASK.SOUTHERN_LATITUDE_NORTH
                ) & (dims[dim] >= ZWS_MASK.SOUTHERN_LATITUDE_SOUTH)
            elif dim == "lon":
                # Define longitude bounds for northern and southern regions
                northern_slice[idx] = (
                    dims[dim] <= ZWS_MASK.NORTHERN_LONGITUDE_EAST
                ) & (dims[dim] >= ZWS_MASK.NORTHERN_LONGITUDE_WEST)
                southern_slice[idx] = (
                    dims[dim] <= ZWS_MASK.SOUTHERN_LONGITUDE_EAST
                ) & (dims[dim] >= ZWS_MASK.SOUTHERN_LONGITUDE_WEST)
        # Extract zonal wind data for both regions and squeeze unnecessary dimensions
        zonal_wind_north = dataset["u"][tuple(northern_slice)].squeeze()
        zonal_wind_south = dataset["u"][tuple(southern_slice)].squeeze()
    # Calculate mean zonal wind over latitude and longitude axes for both regions
    mean_wind_south = zonal_wind_south.mean(axis=(1, 2))
    mean_wind_north = zonal_wind_north.mean(axis=(1, 2))

    # Calculate raw zonal wind shear as the difference between the two regions
    zws_raw = mean_wind_south - mean_wind_north

    # Apply moving average smoothing
    zws_smoothed = moving_average(zws_raw, axis=0)

    # Compute gradient of the smoothed ZWS
    zws_smoothed_grad = np.gradient(zws_smoothed)

    # Split dimensions for easier processing
    zws_raw = split_dimension(zws_raw, axis=0)
    zws_smoothed = split_dimension(zws_smoothed, axis=0)
    zws_smoothed_grad = split_dimension(zws_smoothed_grad, axis=0)
    return zws_raw, zws_smoothed, zws_smoothed_grad


def calculate_occurrence(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate raw and smoothed occurrences of low-pressure systems from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the occurrence data.

    Returns
    -------
    tuple : (np.ndarray, np.ndarray)
        - occurrence_raw: Raw occurrence data.
        - occurrence_smoothed: Smoothed occurrence data using a moving average.

    Notes
    -----
    - Assumes 43 years of data, each with 365 days.
    """
    from utils import moving_average, split_dimension
    from pandas import read_csv

    # Read the CSV file
    dataframe = read_csv(filepath, sep="\t", on_bad_lines="skip", header=None)

    # Assign appropriate column names and ensure integer type
    dataframe.columns = ["Year", "Month", "Day", "Occurrence", "Time"]
    dataframe = dataframe.astype(int)

    # Extract occurrence data as a NumPy array
    occurrence_raw = dataframe["Occurrence"].to_numpy()

    # Apply moving average to smooth the occurrence data
    occurrence_smoothed = moving_average(occurrence_raw, axis=0)

    # Split dimensions for easier processing
    occurrence_raw = split_dimension(occurrence_raw, axis=0)
    occurrence_smoothed = split_dimension(occurrence_smoothed, axis=0)

    return occurrence_raw, occurrence_smoothed


def calculate_streamfunction(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    r"""
    Calculate the mass streamfunction on the Y-Z plane by integrating the meridional divergent wind.

    Reference:
    - https://derekyuntao.github.io/jekyll-clean-dark/2021/02/mass-stream-func/

    The streamfunction is computed using the following formula:
    - $$\psi_{\vartheta} = \frac{R}{g}\Delta{\lambda} \cos{\vartheta}\int^{p}_{0}[v_D]dp$$
    where:
        - $\psi_{\vartheta}$ is the zonal mean meridional mass stream function.
        - $R$ is Earth's radius.
        - $g$ is the gravitational acceleration.
        - $\Delta \lambda$ is the longitudinal extent.
        - $\cos{\vartheta}$ is the latitudinal area weighting.
        - $[v_D]$ is the zonal mean meridional divergent wind.

    Assumptions:
    - The pressure levels (plev) in the original netCDF file are in **descending** order.
    - The dataset contains 43 years of daily data (365 days/year).

    Parameters:
    ----------
    filepath: str
        Path to the NetCDF file containing wind and dimensional data.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        - streamfunction: np.ndarray
            The calculated mass streamfunction.
        - streamfunction_smoothed: np.ndarray
            Smoothed version of the mass streamfunction.
        - dims: dict[str, np.ndarray]
            Dictionary containing dimension data (time, lat, lon, plev).
    """

    from utils import moving_average, split_dimension
    from constants import INDIAN_MASK
    from constants import EARTH

    # Open the dataset and extract dimensions and data slices
    with Dataset(filepath) as dataset:
        dims = {dim: dataset[dim][:] for dim in dataset["v"].dimensions}
        data_slice = [slice(None)] * len(dims)  # Initialize data slice for indexing

        # Apply data slicing based on dimension name
        for idx, dim in enumerate(dataset["v"].dimensions):
            if dim == "time":
                continue  # Skip time dimension processing
            elif dim == "plev":
                # Reverse pressure levels (ascending order for calculation)
                data_slice[idx] = slice(None, None, -1)
                dims[dim] = dims[dim][data_slice[idx]]
            elif dim == "lat":
                # Slice latitude based on the INDIAN_MASK region
                data_slice[idx] = (dims[dim] <= INDIAN_MASK.LATITUDE_NORTH) & (
                    dims[dim] >= INDIAN_MASK.LATITUDE_SOUTH
                )
                dims[dim] = dims[dim][data_slice[idx]]
            elif dim == "lon":
                # Slice longitude based on the INDIAN_MASK region
                data_slice[idx] = (dims[dim] <= INDIAN_MASK.LONGITUDE_EAST) & (
                    dims[dim] >= INDIAN_MASK.LONGITUDE_WEST
                )
                dims[dim] = dims[dim][data_slice[idx]]
        # Extract the meridional divergent wind component
        divergent_meridional_wind = dataset["v"][tuple(data_slice)]
    # Insert boundary conditions for pressure levels (prepend 0 at the first level)
    divergent_meridional_wind = np.insert(divergent_meridional_wind, 0, 0, axis=1)
    pressure_levels = np.insert(dims["plev"], 0, 0)

    # Compute zonal mean (average over longitudes)
    divergent_meridional_wind = np.mean(divergent_meridional_wind, axis=-1)

    # Midpoint Riemann sum for integration
    divergent_meridional_wind = (
        divergent_meridional_wind[:, :-1, :] + divergent_meridional_wind[:, 1:, :]
    ) / 2

    # Calculate longitudinal extent and latitudinal weighting
    longitudinal_extent = np.deg2rad(dims["lon"][-1] - dims["lon"][0])
    latitudinal_weighting = np.cos(np.deg2rad(dims["lat"]))
    weighting_factor = (
        (EARTH.RADIUS / EARTH.GRAVITY_ACCELERATION)
        * longitudinal_extent
        * latitudinal_weighting
    )

    # Calculate streamfunction by integrating over pressure levels
    streamfunction = np.swapaxes(divergent_meridional_wind, 1, -1) * np.diff(
        pressure_levels
    )
    streamfunction = np.cumsum(streamfunction, axis=-1)
    streamfunction = np.swapaxes(streamfunction, -1, 1) * weighting_factor

    # Apply moving average for smoothing the streamfunction
    streamfunction_smoothed = moving_average(streamfunction, axis=0)

    # Split the streamfunction and smoothed version for further processing (time splitting)
    streamfunction = split_dimension(streamfunction, axis=0)
    streamfunction_smoothed = split_dimension(streamfunction_smoothed, axis=0)

    # Convert pressure levels from Pa to hPa
    dims["plev"] /= 100

    # Return the streamfunction, smoothed version, and dimensions
    return streamfunction, streamfunction_smoothed, dims


def calculate_potential_temperature(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Calculate the potential temperature for a specific geographical region using latitude and longitude boundaries.

    This function slices data to focus on a region defined by `INDIAN_MASK` and calculates the potential temperature.
    Assumes 43 years of data with 365 days per year.

    Parameters:
    ----------
    filepath: str
        Path to the NetCDF file containing potential temperature and dimensional data.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        - potential_temperature: np.ndarray
            The extracted and processed potential temperature data.
        - potential_temperature_smoothed: np.ndarray
            Smoothed version of the potential temperature.
        - dims: dict[str, np.ndarray]
            Dictionary containing dimension data (time, lat, lon, plev).
    """

    # Local imports
    from utils import moving_average, split_dimension
    from constants import INDIAN_MASK

    # Open the dataset and extract dimensions
    with Dataset(filepath) as dataset:
        dims = {dim: dataset[dim][:] for dim in dataset["pt"].dimensions}
        data_slice = [slice(None)] * len(dims)  # Initialize data slice for indexing

        # Slice latitude and longitude based on the INDIAN_MASK region
        for idx, dim in enumerate(dataset["pt"].dimensions):
            if dim == "time" or dim == "plev":
                continue  # Skip time and pressure dimensions
            elif dim == "lat":
                # Slice latitude based on the INDIAN_MASK region
                data_slice[idx] = (dims[dim] <= INDIAN_MASK.LATITUDE_NORTH) & (
                    dims[dim] >= INDIAN_MASK.LATITUDE_SOUTH
                )
                dims[dim] = dims[dim][data_slice[idx]]
            elif dim == "lon":
                # Slice longitude based on the INDIAN_MASK region
                data_slice[idx] = (dims[dim] <= INDIAN_MASK.LONGITUDE_EAST) & (
                    dims[dim] >= INDIAN_MASK.LONGITUDE_WEST
                )
                dims[dim] = dims[dim][data_slice[idx]]

        # Extract the potential temperature data with the applied slices
        potential_temperature = dataset["pt"][tuple(data_slice)]
    # Average over the longitude axis (axis 3)
    potential_temperature = np.mean(potential_temperature, axis=3)

    # Apply moving average smoothing over the time axis (axis 0)
    potential_temperature_smoothed = moving_average(potential_temperature, axis=0)

    # Split the potential temperature and smoothed data over time (axis 0)
    potential_temperature = split_dimension(potential_temperature, axis=0)
    potential_temperature_smoothed = split_dimension(
        potential_temperature_smoothed, axis=0
    )

    # Convert pressure levels from Pa to hPa
    dims["plev"] /= 100

    # Return the potential temperature, its smoothed version, and the dimension data
    return (
        potential_temperature,
        potential_temperature_smoothed,
        dims,
    )


def calculate_equivalent_potential_temperature(
    filepath: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Calculate the equivalent potential temperature for a specific geographical region using latitude and longitude boundaries.

    This function slices data to focus on a region defined by `INDIAN_MASK` and calculates the equivalent potential temperature.
    Assumes 43 years of data with 365 days per year.

    Parameters:
    ----------
    filepath: str
        Path to the NetCDF file containing equivalent potential temperature and dimensional data.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        - equivalent_potential_temperature: np.ndarray
            The extracted and processed equivalent potential temperature data.
        - equivalent_potential_temperature_smoothed: np.ndarray
            Smoothed version of the equivalent potential temperature.
        - dims: dict[str, np.ndarray]
            Dictionary containing dimension data (time, lat, lon, plev).
    """

    # Local imports
    from utils import moving_average, split_dimension
    from constants import INDIAN_MASK

    # Open the dataset and extract dimensions
    with Dataset(filepath) as dataset:
        dims = {dim: dataset[dim][:] for dim in dataset["ept"].dimensions}
        data_slice = [slice(None)] * len(dims)  # Initialize data slice for indexing

        # Slice latitude and longitude based on the INDIAN_MASK region
        for idx, dim in enumerate(dataset["ept"].dimensions):
            if dim == "time" or dim == "plev":
                continue  # Skip time and pressure dimensions
            elif dim == "lat":
                # Slice latitude based on the INDIAN_MASK region
                data_slice[idx] = (dims[dim] <= INDIAN_MASK.LATITUDE_NORTH) & (
                    dims[dim] >= INDIAN_MASK.LATITUDE_SOUTH
                )
                dims[dim] = dims[dim][data_slice[idx]]
            elif dim == "lon":
                # Slice longitude based on the INDIAN_MASK region
                data_slice[idx] = (dims[dim] <= INDIAN_MASK.LONGITUDE_EAST) & (
                    dims[dim] >= INDIAN_MASK.LONGITUDE_WEST
                )
                dims[dim] = dims[dim][data_slice[idx]]

        # Extract the equivalent potential temperature data with the applied slices
        equivalent_potential_temperature = dataset["ept"][tuple(data_slice)]

    # Average over the longitude axis (axis 3)
    equivalent_potential_temperature = np.mean(equivalent_potential_temperature, axis=3)

    # Apply moving average smoothing over the time axis (axis 0)
    equivalent_potential_temperature_smoothed = moving_average(
        equivalent_potential_temperature, axis=0
    )

    # Split the equivalent potential temperature and smoothed data over time (axis 0)
    equivalent_potential_temperature = split_dimension(
        equivalent_potential_temperature, axis=0
    )
    equivalent_potential_temperature_smoothed = split_dimension(
        equivalent_potential_temperature_smoothed, axis=0
    )

    # Convert pressure levels from Pa to hPa
    dims["plev"] /= 100

    # Return the equivalent potential temperature, its smoothed version, and the dimension data
    return (
        equivalent_potential_temperature,
        equivalent_potential_temperature_smoothed,
        dims,
    )


def calculate_MSE_vertical_flux(
    mse_filepath: str, w_filepath: str
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Calculate raw and smoothed moist static energy (MSE) flux on the Y-Z plane.

    The function calculates the vertical flux of moist static energy by multiplying the MSE with vertical velocity (w). Latitudinal and longitudinal
    Masking is applied using the `INDIAN_MASK` region.

    Assumptions:
    - The data represents 43 years, with 365 days per year.

    Parameters:
    ----------
    mse_filepath : str
        Path to the NetCDF file containing MSE data.
    w_filepath : str
        Path to the NetCDF file containing vertical velocity (w) data.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        - moist_static_energy_flux: np.ndarray
            The calculated MSE flux.
        - moist_static_energy_flux_smoothed: np.ndarray
            Smoothed version of the calculated MSE flux.
        - dims: dict[str, np.ndarray]
            Dictionary of dimension data (time, lat, lon, plev).

    Notes:
    ------
    - `w` represents pressure tendency (vertical velocity in pressure coordinates).
    """
    from utils import moving_average, split_dimension
    from constants import INDIAN_MASK

    # Open both datasets and extract dimensions
    with Dataset(mse_filepath) as mse_dataset, Dataset(w_filepath) as w_dataset:
        mse_dims = {dim: mse_dataset[dim][:] for dim in mse_dataset["mse"].dimensions}
        w_dims = {dim: w_dataset[dim][:] for dim in w_dataset["w"].dimensions}

        # Initialize slicing indices for MSE and w
        mse_slice = [slice(None)] * len(mse_dims)
        w_slice = [slice(None)] * len(w_dims)

        # Process each dimension to apply slicing and matching
        for idx, dim in enumerate(mse_dataset["mse"].dimensions):
            if dim == "time":
                # Skip time slicing, use full range
                continue
            elif dim == "plev":
                # Find mutual pressure levels between MSE and w datasets
                pressure_levels_mutual = np.intersect1d(mse_dims[dim], w_dims[dim])

                # Get indices for the common pressure levels
                mse_slice[idx] = np.isin(mse_dims[dim], pressure_levels_mutual)
                w_slice[idx] = np.isin(w_dims[dim], pressure_levels_mutual)

                # Update the pressure dimension in both datasets to mutual levels
                mse_dims[dim] = mse_dims[dim][mse_slice[idx]]
                w_dims[dim] = w_dims[dim][w_slice[idx]]
            elif dim == "lat":
                # Apply latitude mask using INDIAN_MASK for both MSE and w
                mse_slice[idx] = (mse_dims[dim] <= INDIAN_MASK.LATITUDE_NORTH) & (
                    mse_dims[dim] >= INDIAN_MASK.LATITUDE_SOUTH
                )
                w_slice[idx] = (w_dims[dim] <= INDIAN_MASK.LATITUDE_NORTH) & (
                    w_dims[dim] >= INDIAN_MASK.LATITUDE_SOUTH
                )

                # Update the latitude dimension after applying the mask
                mse_dims[dim] = mse_dims[dim][mse_slice[idx]]
                w_dims[dim] = w_dims[dim][w_slice[idx]]
            elif dim == "lon":
                # Apply longitude mask using INDIAN_MASK for both MSE and w
                mse_slice[idx] = (mse_dims[dim] <= INDIAN_MASK.LONGITUDE_EAST) & (
                    mse_dims[dim] >= INDIAN_MASK.LONGITUDE_WEST
                )
                w_slice[idx] = (w_dims[dim] <= INDIAN_MASK.LONGITUDE_EAST) & (
                    w_dims[dim] >= INDIAN_MASK.LONGITUDE_WEST
                )

                # Update the longitude dimension after applying the mask
                mse_dims[dim] = mse_dims[dim][mse_slice[idx]]
                w_dims[dim] = w_dims[dim][w_slice[idx]]
        # Extract MSE and vertical velocity (w) data based on the computed slices
        moist_static_energy = mse_dataset["mse"][tuple(mse_slice)]
        pressure_tendency = w_dataset["w"][tuple(w_slice)]
    # Calculate the vertical flux of moist static energy
    moist_static_energy_flux = np.mean(moist_static_energy * pressure_tendency, axis=-1)

    # Apply smoothing over the time dimension
    moist_static_energy_flux_smoothed = moving_average(moist_static_energy_flux, axis=0)

    # Split the time dimension for easier time-based analysis
    moist_static_energy_flux = split_dimension(moist_static_energy_flux, axis=0)
    moist_static_energy_flux_smoothed = split_dimension(
        moist_static_energy_flux_smoothed, axis=0
    )

    # Convert pressure levels from Pa to hPa for easier interpretation
    mse_dims["plev"] /= 100
    w_dims["plev"] /= 100

    # Return the calculated flux, smoothed flux, and the updated dimension data
    return (
        moist_static_energy_flux,
        moist_static_energy_flux_smoothed,
        mse_dims,
    )


def calculate_SPSD(
    variable: np.ndarray, dimensions: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Performs stochastic wavenumber-frequency spectral analysis based on Wheeler-Kiladis (1999).

    This function processes geophysical data to analyze its spectral characteristics in both
    zonal wavenumber and temporal frequency domains.

    Args:
        variable (np.ndarray): Raw multi-dimensional variable
        dimensions (dict[str, np.ndarray]): dimensions corresponding to variable in dict.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Symmetric component of power spectrum of the spectral data.
            - np.ndarray: Antisymmetric component of power spectrum of the spectral data.
            - np.ndarray: Smoothed background components of power spectrum.
            - dict: A dictionary of updated dimensions, including:
                - 'segmentation_frequency': Frequencies in cycles per day.
                - 'zonal_wavenumber': Wavenumber axis.

    Steps:
        1. Preprocess the data:
            - Remove large-scale signals through detrending and high-pass filtering.
            - Decompose into symmetric and antisymmetric components along the latitude axis.
            - Mask latitude and segment the data for spectral analysis.
        2. Apply temporal and spatial tapering to the data.
        3. Perform Fourier Transforms and compute spectral density.
        4. Smooth background components based on frequency and wavenumber, respectively.
        5. Generate and return symmetric, antisymmetric, and background components along with updated metadata.
    """
    from scipy.signal import detrend as scipy_linear_detrend
    from constants import INDIAN_MASK, WK99

    def high_pass_filter(
        signal: np.ndarray,
        axis: int,
        cutoff_frequency: float = 1 / WK99.SEGMENTATION_LENGTH,
    ) -> np.ndarray:
        """
        Applies a high-pass filter to remove low-frequency components from a signal.

        Args:
            signal (np.ndarray): Input signal as a multi-dimensional array.
            cutoff_frequency (float): Frequency threshold for the filter. Frequencies below this
                value are removed. Defaults to 1/SEGMENTATION_LENGTH.
            axis (int): Axis along which the filter is applied.

        Returns:
            np.ndarray: Signal with low-frequency components removed, maintaining the
            same shape as the input.
        """
        signal_length = signal.shape[axis]

        fourier_component = np.fft.rfft(signal, axis=axis)
        positive_frequencies = np.fft.rfftfreq(signal_length)

        filter_condition = [slice(None)] * signal.ndim
        filter_condition[axis] = positive_frequencies < cutoff_frequency
        fourier_component[tuple(filter_condition)] = 0.0

        filtered_signal = np.fft.irfft(fourier_component, n=signal_length, axis=axis)

        return filtered_signal

    def decompose_symmetric_antisymmetric(
        variable: np.ndarray, axis: int
    ) -> np.ndarray:
        """
        Decomposes a variable into symmetric and antisymmetric components along a specified axis.

        Args:
            variable (np.ndarray): Input array to decompose.
            axis (int): Axis along which to perform the decomposition, typically the latitude axis.

        Returns:
            np.ndarray: An array with an added first dimension of size 2:
                - Index 0: Symmetric component.
                - Index 1: Antisymmetric component.
        """
        flipped_variable = np.flip(variable, axis=axis)
        symmetric_component = (variable + flipped_variable) / 2
        antisymmetric_component = (variable - flipped_variable) / 2

        return np.array([symmetric_component, antisymmetric_component])

    def segment_data(
        variable: np.ndarray,
        axis: int,
        segment_length: int = WK99.SEGMENTATION_LENGTH,
        overlap_length: int = WK99.OVERLAP_LENGTH,
    ) -> np.ndarray:
        """
        Segments an array along a specified axis with overlap between segments.

        Args:
            variable (np.ndarray): Input array to be segmented, with any number of dimensions.
            axis (int): Axis along which to perform segmentation, typically the time axis.
            segment_length (int): Length of each segment. Defaults to WK99.SEGMENTATION_LENGTH.
            overlap_length (int): Overlap length between consecutive segments. Defaults to WK99.OVERLAP_LENGTH.

        Returns:
            np.ndarray: Segmented array with an additional dimension for the number of segments.
        """
        step = segment_length - overlap_length
        num_segments = (variable.shape[axis] - overlap_length) // step

        segmented_shape = list(variable.shape)
        segmented_shape.pop(axis)
        segmented_shape.insert(axis, num_segments)
        segmented_shape.insert(axis + 1, segment_length)

        segmented_variable = np.empty(shape=tuple(segmented_shape), dtype=float)
        for i in range(num_segments):

            segment_index_slices = [slice(None)] * segmented_variable.ndim
            segment_index_slices[axis] = i
            segment_index_slices = tuple(segment_index_slices)

            variable_window_slices = [slice(None)] * variable.ndim
            variable_window_slices[axis] = slice(i * step, i * step + segment_length)
            variable_window_slices = tuple(variable_window_slices)

            segmented_variable[segment_index_slices] = variable[variable_window_slices]

        return segmented_variable

    def apply_121_filter(signal: np.ndarray, axis: int, iterations: int) -> np.ndarray:
        """
        Applies a 1-2-1 smoothing filter iteratively along a specified axis.

        Args:
            signal (np.ndarray): Input array to smooth.
            axis (int): Axis along which the filter is applied.
            iterations (int): Number of times to apply the filter.

        Returns:
            np.ndarray: Smoothed array after applying the filter iteratively.
        """

        def extend_boundaries(signal: np.ndarray, axis: int) -> np.ndarray:
            """
            Extends the boundaries of the array along the specified axis by duplicating edge values.
            """
            first_slice = [slice(None)] * signal.ndim
            last_slice = [slice(None)] * signal.ndim
            first_slice[axis] = slice(0, 1)
            last_slice[axis] = slice(-1, None)
            return np.concatenate(
                [signal[tuple(first_slice)], signal, signal[tuple(last_slice)]],
                axis=axis,
            )

        def convolve_along_axis(
            signal: np.ndarray,
            axis: int,
            kernel: np.ndarray = np.array([1 / 4, 1 / 2, 1 / 4]),
        ) -> np.ndarray:
            """
            Convolves the signal along the specified axis with a given kernel.
            """
            extended_signal = extend_boundaries(signal, axis=axis)
            return np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode="valid"),
                axis=axis,
                arr=extended_signal,
            )

        filtered_signal = np.copy(signal)
        for _ in range(iterations):
            filtered_signal = convolve_along_axis(filtered_signal, axis)

        return filtered_signal

    def temporal_taper(signal: np.ndarray, portion: float, axis: int) -> np.ndarray:
        """
        Applies a cosine taper to the beginning and end of a signal along a specified axis.

        Args:
            signal (np.ndarray): Input array to be tapered.
            portion (float): Fraction of the signal's length to taper at both ends. Must be in [0, 1].
            axis (int): Axis along which the taper is applied, typically the time axis.

        Returns:
            np.ndarray: Signal with the taper applied along the specified axis.
        """
        if not (0.0 <= portion <= 1.0):
            raise ValueError("Portion must be between 0 and 1.")

        axis = axis if axis >= 0 else signal.ndim + axis
        if axis < 0 or axis >= signal.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for an array with {signal.ndim} dimensions."
            )

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

        return signal * taper_window

    def spatial_taper(
        signal: np.ndarray,
        dimensions: dict[str, np.ndarray],
        axis: int,
        lon_min: float,
        lon_max: float,
    ) -> np.ndarray:
        """
        Applies a spatial taper along a specified longitudinal axis.

        Args:
            signal (np.ndarray): Input array to be tapered.
            dimensions (dict[str, np.ndarray]): Dictionary containing dimension values, including 'lon' for longitude.
            axis (int): Axis along which the tapering is applied, typically the longitude axis.
            lon_min (float): Minimum longitude to retain without tapering.
            lon_max (float): Maximum longitude to retain without tapering.

        Returns:
            np.ndarray: Signal with the spatial taper applied along the specified axis.
        """
        if "lon" not in dimensions:
            raise ValueError(
                "Longitude values must be provided in dimensions under the key 'lon'."
            )
        lon = dimensions["lon"]

        axis = axis if axis >= 0 else signal.ndim + axis
        if axis < 0 or axis >= signal.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for an array with {signal.ndim} dimensions."
            )

        regional_mask = (lon >= lon_min) & (lon <= lon_max)
        taper_window = np.zeros_like(lon, dtype=float)

        if np.all(regional_mask):
            taper_window = np.ones_like(taper_window)
        else:
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

        return signal * taper_window

    # Remove large-scale signals through linear detrending and high-pass filtering
    variable = scipy_linear_detrend(variable, axis=0)
    variable = high_pass_filter(variable, axis=0)

    # Decompose the variable into symmetric and antisymmetric components along the latitude axis
    variable = decompose_symmetric_antisymmetric(variable, axis=1)

    # Apply a latitude mask to limit the data to the specified latitude range
    lat_mask = (dimensions["lat"] <= INDIAN_MASK.LATITUDE_NORTH) & (
        dimensions["lat"] >= INDIAN_MASK.LATITUDE_SOUTH
    )
    variable = variable[:, :, lat_mask, :]
    dimensions["lat"] = dimensions["lat"][lat_mask]

    # Segment the data along the time axis for spectral analysis
    variable = segment_data(
        variable,
        axis=1,
        segment_length=WK99.SEGMENTATION_LENGTH,
        overlap_length=WK99.OVERLAP_LENGTH,
    )

    # Remove trends in the time axis
    variable = scipy_linear_detrend(variable, axis=2)

    # Apply a temporal taper to reduce spectral leaking along the time axis
    variable = temporal_taper(variable, portion=0.2, axis=2)

    # Apply a spatial taper to smoothly limit data to the specified longitudinal range
    variable = spatial_taper(
        variable,
        dimensions=dimensions,
        axis=-1,
        lon_min=INDIAN_MASK.LONGITUDE_WEST - 30,
        lon_max=INDIAN_MASK.LONGITUDE_EAST + 30,
    )

    # Perform Fourier Transform along longitude (zonal wavenumber) and time (frequency)
    variable = np.fft.fft(variable, axis=-1, norm="ortho")
    variable = np.fft.ifft(variable, axis=2, norm="ortho")

    # Compute the stochastic power spectral density (PSD) by averaging over segments
    variable = np.mean(np.abs(variable) ** 2, axis=1)

    # Compute the stochastic  power spectral density (PSD) by summing up over latitudes
    variable = np.sum(variable, axis=2)

    # Center the zero-frequency and zero-wavenumber components for visualization
    variable = np.fft.fftshift(variable, axes=(1, -1))

    # Compute background components using a smoothing filter
    background_components = np.mean(variable, axis=0)

    # Generate wavenumber and frequency axes for output
    ordinary_wavenumber = np.fft.fftshift(
        np.fft.fftfreq(len(dimensions["lon"]), 1 / len(dimensions["lon"]))
    )
    ordinary_frequency = np.fft.fftshift(
        np.fft.fftfreq(WK99.SEGMENTATION_LENGTH, 1 / WK99.SEGMENTATION_LENGTH)
    )
    CPD_frequency = (
        ordinary_frequency / len(ordinary_frequency) * WK99.SAMPLE_RATE
    )  # Convert frequency to cycles per day (CPD)

    # Smooth symmetric and antisymmetric components over frequencies
    variable[:, CPD_frequency > 0] = apply_121_filter(
        variable[:, CPD_frequency > 0], axis=1, iterations=1
    )
    variable[:, CPD_frequency < 0] = apply_121_filter(
        variable[:, CPD_frequency < 0], axis=1, iterations=1
    )

    # Smooth background components over wavenumbers
    for i, freq in enumerate(CPD_frequency):
        if abs(freq) == 0.0:
            continue
        elif abs(freq) <= 0.1:
            background_components[i] = apply_121_filter(
                background_components[i], axis=0, iterations=5
            )
        elif abs(freq) <= 0.2:
            background_components[i] = apply_121_filter(
                background_components[i], axis=0, iterations=10
            )
        elif abs(freq) <= 0.3:
            background_components[i] = apply_121_filter(
                background_components[i], axis=0, iterations=20
            )
        else:
            background_components[i] = apply_121_filter(
                background_components[i], axis=0, iterations=40
            )

    # Smooth background components over frequencies
    background_components[CPD_frequency > 0] = apply_121_filter(
        background_components[CPD_frequency > 0], axis=0, iterations=10
    )
    background_components[CPD_frequency < 0] = apply_121_filter(
        background_components[CPD_frequency < 0], axis=0, iterations=10
    )

    # Fill NaN in mean
    variable[:, CPD_frequency == 0] = np.nan
    background_components[CPD_frequency == 0] = np.nan

    # Update the dimensions dictionary with computed frequency and wavenumber axes
    dimensions.update(
        {
            "segmentation_frequency": CPD_frequency,
            "zonal_wavenumber": ordinary_wavenumber,
        }
    )

    return variable[0], variable[1], background_components, dimensions


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
    from utils import decompose_symmetric_antisymmetric
    from constants import INDIAN_MASK

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
