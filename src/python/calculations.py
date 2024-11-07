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
        dims = {dim: dataset[dim][:] for dim in dataset["theta"].dimensions}
        data_slice = [slice(None)] * len(dims)  # Initialize data slice for indexing

        # Slice latitude and longitude based on the INDIAN_MASK region
        for idx, dim in enumerate(dataset["theta"].dimensions):
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
        potential_temperature = dataset["theta"][tuple(data_slice)]
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
        dims = {dim: dataset[dim][:] for dim in dataset["equiv_theta"].dimensions}
        data_slice = [slice(None)] * len(dims)  # Initialize data slice for indexing

        # Slice latitude and longitude based on the INDIAN_MASK region
        for idx, dim in enumerate(dataset["equiv_theta"].dimensions):
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
        equivalent_potential_temperature = dataset["equiv_theta"][tuple(data_slice)]

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
        mse_dims = {dim: mse_dataset[dim][:] for dim in mse_dataset["MSE"].dimensions}
        w_dims = {dim: w_dataset[dim][:] for dim in w_dataset["w"].dimensions}

        # Initialize slicing indices for MSE and w
        mse_slice = [slice(None)] * len(mse_dims)
        w_slice = [slice(None)] * len(w_dims)

        # Process each dimension to apply slicing and matching
        for idx, dim in enumerate(mse_dataset["MSE"].dimensions):
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
        moist_static_energy = mse_dataset["MSE"][tuple(mse_slice)]
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
    file_path: str, variable_name: str = "Undefined", pressure_level: int = -1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Calculate the stochastic power spectral density (SPSD) for a given variable using a modified method
    based on Wheeler and Kiladis (1999) for sectional and global analysis.

    The function processes 3D (time, lat, lon) or 4D (time, pressure, lat, lon) datasets,
    performing wavenumber-frequency analysis. It returns the symmetric and antisymmetric components
    of the power spectral density (PSD) and the background PSD.

    Args:
        file_path (str): Path to the input dataset file (NetCDF format).
        variable_name (str): Name of the variable to analyze. Defaults to "Undefined".
        pressure_level (int): Specific pressure level to extract. Defaults to -1 (no specific level).

    Returns:
        tuple: A tuple containing:
            - Symmetric components of the PSD.
            - Antisymmetric components of the PSD.
            - Background PSD components.
            - Dictionary of relevant dimensions, including latitude, frequency, and zonal wavenumber.
    """
    from gc import collect as free_memory
    from scipy.signal import detrend as scipy_linear_detrend
    from utils import decompose_symmetric_antisymmetric, segment_data, apply_121_filter
    from constants import INDIAN_MASK, WK99

    def high_pass_filter(
        array: np.ndarray,
        cutoff: float = (1 / WK99.SEGMENTATION_LENGTH),
        axis: int = -1,
    ):
        array_fft = np.fft.fft(array, axis=axis)
        frequency = np.fft.fftfreq(array_fft.shape[axis], 1)
        array_fft[abs(frequency) < cutoff] = 0.0
        array_filtered = np.fft.ifft(array_fft, axis=axis)
        return np.real(array_filtered)

    def temporal_taper(
        array: np.ndarray, portion: float = 0.1, axis: int = -1
    ) -> np.ndarray:
        """
        Applies a cosine taper along the specified time axis of the input array.

        Parameters:
        array (np.ndarray): Input array to be tapered, which can be multi-dimensional.
        portion (float): Fraction of the array length to taper at each end, between 0 and 1.
        axis (int): The axis corresponding to time along which the taper is applied.

        Returns:
        np.ndarray: Array with the taper applied along the specified time axis.

        Note:
        The `axis` parameter should be set to the index representing the time dimension of `array`.
        This function reshapes the taper window to broadcast across higher dimensions,
        enabling consistent application along the specified time axis.
        """
        # Ensure portion is within [0, 1]
        if not (0 <= portion <= 1):
            raise ValueError("Portion must be between 0 and 1.")

        # Convert negative axis indexing to positive if needed
        axis = axis if axis >= 0 else array.ndim + axis
        if axis < 0 or axis >= array.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for an array with {array.ndim} dimensions."
            )

        # Calculate taper length
        taper_len = int(array.shape[axis] * (portion / 2))

        # Generate the cosine taper
        taper = 0.5 * (1 - np.cos(2 * np.pi * np.arange(taper_len) / (2 * taper_len)))

        # Create the full taper window and reshape for broadcasting
        temp_window = np.ones(array.shape[axis])
        temp_window[:taper_len] = taper
        temp_window[-taper_len:] = taper[::-1]

        # Reshape temp_window to match the array dimensions for broadcasting
        reshape_shape = [1] * array.ndim  # Start with a shape of all ones
        reshape_shape[axis] = array.shape[axis]  # Set the size of the time axis
        temp_window = temp_window.reshape(reshape_shape)  # Reshape for broadcasting

        # Apply the taper window along the specified time axis
        return array * temp_window

    def spatial_taper(
        array: np.ndarray,
        lon_min: float,
        lon_max: float,
        dims: dict[str, np.ndarray],
        axis: int = -1,
    ) -> np.ndarray:
        """
        Applies a spatial taper along the specified longitudinal axis of the input array.

        Parameters:
        array (np.ndarray): Input array to be tapered, which can be multi-dimensional.
        lon_min (float): Minimum longitude of the region to leave intact.
        lon_max (float): Maximum longitude of the region to leave intact.
        dims (dict): Dictionary containing axis values, with "lon" as the key for longitude values.
        axis (int): Axis corresponding to the longitude dimension along which the taper is applied.

        Returns:
        np.ndarray: Array with the taper applied along the specified longitudinal axis.

        Note:
        This function applies a Hanning-like tapering function outside the specified longitudinal range
        (lon_min, lon_max), smoothly transitioning values to zero outside this region.
        """
        # Retrieve longitude values from dims
        if "lon" not in dims:
            raise ValueError(
                "Longitude values must be provided in dims under the key 'lon'."
            )
        lon = dims["lon"]

        # Convert negative axis indexing to positive if needed
        axis = axis if axis >= 0 else array.ndim + axis
        if axis < 0 or axis >= array.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for an array with {array.ndim} dimensions."
            )

        # Create a boolean mask for the target longitudinal range
        lon_mask = (lon >= lon_min) & (lon <= lon_max)

        # Initialize the tapering window for longitude
        lon_window = np.zeros_like(lon, dtype=float)

        # Set up the tapered region
        lon_region = lon_window[lon_mask]
        taper_len = (
            len(lon_region) * 0.1
        )  # Define taper length as 1/10 of the region size

        # Apply tapering conditions
        if np.all(lon_mask):
            lon_window = np.ones_like(
                lon_window
            )  # No tapering if the full region is within the mask
        else:
            # Generate the cosine taper
            taper = 0.5 * (
                1 - np.cos(2 * np.pi * np.arange(taper_len) / (2 * taper_len))
            )

            # Apply the taper at the beginning and end of the masked region
            lon_region[:taper_len] = taper
            lon_region[taper_len:-taper_len] = 1
            lon_region[-taper_len:] = taper[::-1]

            # Update the taper window with the tapered region
            lon_window[lon_mask] = lon_region

        # Reshape lon_window to match the array dimensions for broadcasting
        reshape_shape = [1] * array.ndim
        reshape_shape[axis] = len(lon)
        lon_window = lon_window.reshape(reshape_shape)

        # Apply the spatial taper along the specified longitudinal axis
        return array * lon_window

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

    # Remove dominant signals
    data = scipy_linear_detrend(data, axis=0)
    data = high_pass_filter(data, axis=0)

    # Decompose into Sym/Asym
    data = decompose_symmetric_antisymmetric(data, axis=1)

    # Mask the symmetric and antisymmetric components by latitude
    lat_mask = (dims["lat"] <= INDIAN_MASK.LATITUDE_NORTH) & (
        dims["lat"] >= INDIAN_MASK.LATITUDE_SOUTH
    )
    data = data[:, :, lat_mask]
    dims["lat"] = dims["lat"][lat_mask]

    # Segment the components for analysis
    data = segment_data(np.copy(data), WK99.SEGMENTATION_LENGTH, WK99.OVERLAP_LENGTH)

    # Apply linear detrending
    data = scipy_linear_detrend(data, axis=2)

    # Apply temporal window
    data = temporal_taper(data, portion=0.2, axis=2)
    # Apply spatial window
    data = spatial_taper(
        data,
        lon_min=INDIAN_MASK.LONGITUDE_WEST,
        lon_max=INDIAN_MASK.LONGITUDE_EAST,
        dims=dims,
        axis=-1,
    )

    # Perform FFT in both zonal wavenumber (lon) and frequency (time) directions
    data = np.fft.fft(data, axis=-1, norm="ortho")
    data = np.fft.ifft(data, axis=2, norm="ortho")

    # Compute power spectral densities (PSD) by averaging over segments
    data = np.mean(np.abs(data) ** 2, axis=1)

    # Shift zero-frequency and zero-wavenumber components to center for visualization
    data = np.fft.fftshift(data, axes=(1, -1))

    # Generate frequency and wavenumber axes for output
    ordinary_wavenumber = np.fft.fftshift(
        np.fft.fftfreq(len(dims["lon"]), 1 / len(dims["lon"]))
    )
    ordinary_frequency = np.fft.fftshift(
        np.fft.fftfreq(WK99.SEGMENTATION_LENGTH, 1 / WK99.SEGMENTATION_LENGTH)
    )
    CPD_frequency = (
        ordinary_frequency / len(ordinary_frequency) * WK99.SAMPLE_RATE
    )  # Cycles per day (Assumed daily data)

    # Calculate background components using a smoothing filter
    background_components = np.mean(data, axis=0)
    for i, freq in enumerate(CPD_frequency):
        if abs(freq) <= 0.1:
            background_components[i] = apply_121_filter(
                background_components[i], iterations=5
            )
        elif abs(freq) <= 0.2:
            background_components[i] = apply_121_filter(
                background_components[i], iterations=10
            )
        elif abs(freq) <= 0.3:
            background_components[i] = apply_121_filter(
                background_components[i], iterations=20
            )
        else:
            background_components[i] = apply_121_filter(
                background_components[i], iterations=40
            )
    background_components[CPD_frequency > 0] = apply_121_filter(
        background_components[CPD_frequency > 0], axis=0, iterations=10
    )

    # Update dimension metadata for the output
    dims.update(
        {
            "segmentation_frequency": CPD_frequency,
            "zonal_wavenumber": ordinary_wavenumber,
        }
    )

    return (data[0], data[1], background_components), dims


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
