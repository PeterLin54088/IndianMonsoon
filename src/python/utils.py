import numpy as np
from scipy.signal import detrend as scipy_linear_detrend

# Constants
SIDEREAL_DAY_SECONDS = 86164.1  # Duration of sidereal day in seconds
SOLAR_DAY_SECONDS = 86400  # Duration of solar day in seconds
EARTH_ANGULAR_VELOCITY = (
    2 * np.pi / SIDEREAL_DAY_SECONDS
)  # Angular velocity of Earth in radians/second
EARTH_RADIUS_METERS = 6.371e6  # Radius of Earth in meters
EARTH_GRAVITY_ACCELERATION = (
    9.78  # Gravitational acceleration on Earth in meters/second^2
)
# Size of the moving average window (number of elements)
WINDOW_SIZE = 28


def moving_average(data, axis=0, WINDOW_SIZE=28):
    """
    Compute the moving average of the input data along a specified axis.

    This function smooths the input NumPy array by applying a moving average
    with a predefined window size. The moving average is calculated using a
    convolution with a uniform window. Boundary regions affected by convolution
    are masked with `np.nan` to indicate incomplete averaging.

    Parameters
    ----------
    data : np.ndarray
        The input array to be smoothed. Can be of any dimensionality.
    axis : int, optional
        The axis along which to compute the moving average. Default is 0.
    WINDOW_SIZE : int, optional
        The size that determines the number of elements over which the average is computed.

    Returns
    -------
    np.ndarray
        The smoothed array after applying the moving average. The output array
        has the same shape as the input array, with boundary regions masked as `np.nan`.

    Notes
    -----
    - The function uses 'same' mode in convolution to ensure the output array
      has the same length as the input array along the specified axis.
    - Boundary regions where the moving average is incomplete due to convolution
      are masked with `np.nan`.
    """
    window = np.ones(WINDOW_SIZE) / WINDOW_SIZE
    smoothed_data = np.apply_along_axis(
        lambda values: np.convolve(values, window, mode="same"),
        axis=axis,
        arr=data,
    )

    # Calculate the number of boundary elements to mask
    pad = WINDOW_SIZE // 2

    # Move the target axis to the first axis for easy indexing
    smoothed_data = np.moveaxis(smoothed_data, axis, 0)

    # Mask the boundary regions with np.nan
    smoothed_data[:pad, ...] = np.nan
    smoothed_data[-pad:, ...] = np.nan

    # Move the axis back to its original position
    smoothed_data = np.moveaxis(smoothed_data, 0, axis)

    return smoothed_data


def split_dimension(array, axis, factors):
    """
    Split a specified dimension of a NumPy array into multiple dimensions.

    Parameters:
    - array (np.ndarray): The input array.
    - axis (int): The index of the dimension to split (0-based).
    - factors (tuple of int): The new dimensions to split into.

    Returns:
    - np.ndarray: The reshaped array with the specified dimension split.

    Raises:
    - TypeError: If the input is not a NumPy array.
    - ValueError: If the dimension index is out of range or the product of factors does not match the size of the dimension.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    if axis < 0 or axis >= array.ndim:
        raise ValueError(
            f"Dimension index {axis} is out of range for array with {array.ndim} dimensions."
        )

    original_size = array.shape[axis]
    new_size = np.prod(factors)
    if original_size != new_size:
        raise ValueError(
            f"Cannot split dimension {axis} of size {original_size} into factors {factors} "
            f"with product {new_size}."
        )

    # Compute the new shape by replacing the specified dimension with the new factors
    new_shape = list(array.shape[:axis]) + list(factors) + list(array.shape[axis + 1 :])

    # Reshape the array to the new shape
    reshaped_array = array.reshape(new_shape)

    return reshaped_array


def decompose_symmetric_antisymmetric(data, axis):
    """
    Decompose data into its symmetric and antisymmetric components along a specified axis.

    Parameters:
    data (ndarray): Input data to decompose.
    axis (int): Axis along which to perform the decomposition.

    Returns:
    tuple: A tuple containing:
        - symmetric_component (ndarray): Symmetric part of the data.
        - antisymmetric_component (ndarray): Antisymmetric part of the data.
    """
    symmetric_component = (data + np.flip(data, axis=axis)) / 2
    antisymmetric_component = (data - np.flip(data, axis=axis)) / 2
    return symmetric_component, antisymmetric_component


def compute_segmented_PSD(
    data, data_grid, segment_length=96, overlap_length=65, taper=None
):
    """
    Calculate the stochastic Power Spectral Density (PSD) from time series data segmented with overlap.

    Parameters:
    data : ndarray
        Input data array with dimensions (time, latitude, longitude).
    data_grid : tuple
        Tuple containing (time, latitude, longitude) arrays.
    segment_length : int, optional
        Length of each time segment for FFT analysis. Default is 96.
    overlap_length : int, optional
        Number of overlapping points between segments. Default is 65.
    taper : ndarray, optional
        Custom tapering window to apply to each segment. If None, a default Hanning-like taper is applied.

    Returns:
    PSD : ndarray
        The computed power spectral density.
    (wavemode_time, wavemode_longitude) : tuple of ndarrays
        Wavemodes (nondimensional ordinary wavenumber) in time and longitude.

    Notes:
    - All operations are performed on data points, meaning the units of the data are not considered in the calculations.
      Users should ensure that units and the actual spatial and temporal spacing are handled appropriately outside this function.
    - The resulting `wavemode_time` and `wavemode_longitude` represent the number of wave cycles or modes that fit in the total time
      and over the spatial dimension (longitude), respectively, not actual physical frequencies or wavenumbers.
    """

    def segment_data(array, segment_length, overlap_length):
        """Segments input array into overlapping subarrays."""
        step = segment_length - overlap_length
        num_segments = (len(array) - overlap_length) // step
        segmented_array = np.array(
            [array[i : i + segment_length] for i in range(0, num_segments * step, step)]
        )
        return segmented_array

    def default_taper():
        """Generate default Hanning-like taper over first and last third of the segment."""
        taper_window = np.ones(segment_length)
        third_len = 30
        taper_cosine = 0.5 * (
            1 - np.cos(2 * np.pi * np.arange(third_len) / (2 * third_len))
        )
        taper_window[:third_len] = taper_cosine
        taper_window[-third_len:] = taper_cosine[::-1]
        return taper_window

    time, latitudes, longitudes = data_grid
    if data.shape != (len(time), len(latitudes), len(longitudes)):
        raise ValueError("Input data dimensions must match grid dimensions.")

    if taper is None:
        taper_window = default_taper().reshape(1, -1, 1, 1)
    else:
        taper_window = taper

    # segmentation
    tmp = segment_data(np.copy(data), segment_length, overlap_length)
    # detrend
    tmp = scipy_linear_detrend(tmp, axis=1)
    # tapering
    tmp = np.multiply(tmp, taper_window)
    # fft over longitude
    tmp = np.fft.fft(tmp, axis=3, norm="ortho")
    # fft over time
    tmp = np.fft.ifft(tmp, axis=1, norm="ortho")
    # power spectrum
    tmp = np.abs(tmp) ** 2
    # mean over segment number
    tmp = np.mean(tmp, axis=0, keepdims=True)
    # sum over latitude
    tmp = np.sum(tmp, axis=2, keepdims=True)
    # zero-center PSD diagram
    PSD = np.fft.fftshift(tmp.squeeze())

    # Compute nondimensional ordinary wavenumber for longitude and time
    wavemode_longitude = np.fft.fftshift(
        np.fft.fftfreq(len(longitudes), 1 / len(longitudes))
    )
    wavemode_time = np.fft.fftshift(np.fft.fftfreq(segment_length, 1 / segment_length))

    return PSD, (wavemode_time, wavemode_longitude)


def extract_positive_PSD(PSD, spectral_grid, axis=0):
    """
    Extract the positive half of a symmetric Power Spectral Density (PSD) along the specified axis,
    while preserving Parseval's identity. Assumes the input signal is real-valued, leading to Hermitian
    symmetry in the spectral domain.

    Parameters:
    PSD : ndarray
        The 2D Power Spectral Density array.
    spectral_grid : tuple of ndarrays
        Tuple containing the frequency/wavenumber grids for the corresponding axes.
    axis : int, optional
        The axis along which to extract the positive half of the PSD. Default is 0 (time axis).

    Returns:
    positive_psd : ndarray
        The positive half of the PSD with the specified axis halved.
    (positive_frequencies, wavenumbers) : tuple of ndarrays
        The corresponding positive frequencies (or wavenumbers) and the unmodified spectral grid.
    """
    frequencies = spectral_grid[axis]
    midpoint = len(frequencies) // 2

    # Roll the specified axis to the front for easier manipulation
    PSD = np.moveaxis(PSD, axis, 0)

    if len(frequencies) % 2 == 0:
        # Even case: Include the highest frequency without doubling
        positive_psd = np.concatenate([PSD[midpoint:], PSD[[0]]], axis=0)
        positive_psd[1:-1] *= 2  # Double the power for positive frequencies
        positive_frequencies = np.flip(np.abs(frequencies[: midpoint + 1]))
    else:
        # Odd case: Handle the midpoint and avoid double counting
        positive_psd = np.copy(PSD[midpoint:])
        positive_psd[1:] *= 2  # Double the power for positive frequencies
        positive_frequencies = frequencies[midpoint:]

    # Move the axis back to its original position
    positive_psd = np.moveaxis(positive_psd, 0, axis)

    # Modify the spectral grid for the specified axis, leaving others untouched
    modified_spectral_grid = list(spectral_grid)
    modified_spectral_grid[axis] = positive_frequencies

    return positive_psd, tuple(modified_spectral_grid)


def apply_121_filter(array, axis, iterations):
    """
    Apply a 1-2-1 Gaussian filter along the specified axis, ensuring energy conservation at boundaries.
    """

    def extend_boundaries(arr, ax):
        """Extend boundaries by duplicating the first and last elements along the specified axis."""
        if ax < 0 or ax >= arr.ndim:
            raise ValueError(
                f"Axis {ax} is out of bounds for array of dimension {arr.ndim}."
            )

        # Create slices for the first and last elements along the axis
        first_slice = [slice(None)] * arr.ndim
        last_slice = [slice(None)] * arr.ndim
        first_slice[ax] = slice(0, 1)
        last_slice[ax] = slice(-1, None)

        # Concatenate the boundaries
        extended_arr = np.concatenate(
            [arr[tuple(first_slice)], arr, arr[tuple(last_slice)]], axis=ax
        )
        return extended_arr

    def convolve_along_axis(data, axis, kernel):
        """Extend boundaries and convolve along the specified axis."""
        extended_data = extend_boundaries(data, axis)
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=axis, arr=extended_data
        )

    kernel = np.array([1 / 4, 1 / 2, 1 / 4])

    result = np.copy(array)
    for _ in range(iterations):
        result = convolve_along_axis(result, axis, kernel)

    return result


def get_f_beta(latitude=0.0):
    """
    Calculate the Coriolis frequency (f) and Rossby parameter (beta) at a given latitude.

    Parameters:
    latitude : float
        Latitude in radians.

    Returns:
    coriolis_frequency : float
        Coriolis frequency in 1/second.
    rossby_parameter : float
        Rossby parameter (beta) in 1/meter/second.
    """
    coriolis_frequency = (
        2 * EARTH_ANGULAR_VELOCITY * np.sin(latitude)
    )  # Coriolis frequency
    rossby_parameter = (
        2 * EARTH_ANGULAR_VELOCITY * np.cos(latitude) / EARTH_RADIUS_METERS
    )  # Rossby parameter (beta)
    return coriolis_frequency, rossby_parameter


def get_gravity_wave_speed(equivalent_depth):
    """
    Calculate the gravity wave speed for a given equivalent depth.

    Parameters:
    equivalent_depth : float
        Equivalent depth in meters.

    Returns:
    c : float
        Gravity wave speed in meters/second.
    """
    c = np.sqrt(EARTH_GRAVITY_ACCELERATION * equivalent_depth)  # Gravity wave speed
    return c


def rescale_to_days_and_ordinary_frequency(frequency):
    # Convert from (1/second) to (1/day) and from angular to ordinary frequency
    return frequency * SOLAR_DAY_SECONDS / (2 * np.pi)


def dispersion_kelvin(k):
    """
    Dispersion relation for Kelvin waves.

    Parameters:
    k : array-like
        Nondimensional wavenumber values.

    Returns:
    omega_positive : array-like
        Nondimensional positive frequency (omega) values, NaN for negative values.

    Notes:
    The original function is real-valued and exhibits Hermitian symmetry in the wavenumber-frequency space.
    By convention, scientists retain only the positive frequencies, and this function follows that tradition.
    """
    dispersion_relation = lambda k: k
    omega = dispersion_relation(k)
    omega_positive = np.where(omega >= 0, omega, np.nan)
    return omega_positive


def dispersion_mrg(k):
    """
    Dispersion relation for Mixed Rossby-Gravity (MRG) waves.

    Parameters:
    k : array-like
        Nondimensional wavenumber values.

    Returns:
    omega_positive : array-like
        Nondimensional positive frequency (omega) values, NaN for negative values.

    Notes:
    The original function is real-valued and exhibits Hermitian symmetry in the wavenumber-frequency space.
    By convention, scientists retain only the positive frequencies, and this function follows that tradition.
    """
    dispersion_relation = lambda k: k / 2 + np.sqrt(1 + k**2 / 4)
    omega = dispersion_relation(k)
    omega_positive = np.where(omega >= 0, omega, np.nan)
    return omega_positive


def dispersion_poincare(k, m=1, first_guess=np.inf, niter=50):
    """
    Dispersion relation for Poincaré waves.

    Parameters:
    k : array-like
        Nondimensional wavenumber values.
    m : int, optional
        Meridional mode number, default is 1.
    first_guess : float, optional
        Initial guess for omega, default is infinity.
    niter : int, optional
        Number of iterations for refinement, default is 50.

    Returns:
    omega_positive : array-like
        Nondimensional positive frequency (omega) values, NaN for negative values.

    Notes:
    The dispersion relation for Poincaré waves is a cubic function, making the analytical solution complex.
    An iterative approach is used to approximate the solution.
    The original function is real-valued and exhibits Hermitian symmetry in the wavenumber-frequency space.
    By tradition, only the positive frequencies are retained.
    """
    dispersion_relation = lambda omega: np.sqrt(2 * m + 1 + k**2 + k / omega)

    omega_approx = dispersion_relation(first_guess)
    for _ in range(niter):
        omega_approx = dispersion_relation(omega_approx)

    omega_positive = np.where(omega_approx >= 0, omega_approx, np.nan)
    return omega_positive


def dispersion_rossby(k, m=1, first_guess=0.0, niter=50):
    """
    Dispersion relation for Rossby waves.

    Parameters:
    k : array-like
        Nondimensional wavenumber values.
    m : int, optional
        Meridional mode number, default is 1.
    first_guess : float, optional
        Initial guess for omega, default is 0.
    niter : int, optional
        Number of iterations for refinement, default is 50.

    Returns:
    omega_positive : array-like
        Nondimensional positive frequency (omega) values, NaN for negative values.

    Notes:
    The dispersion relation for Rossby waves is a cubic function, making the analytical solution complex.
    An iterative approach is used to approximate the solution.
    The original function is real-valued and exhibits Hermitian symmetry in the wavenumber-frequency space.
    By tradition, only the positive frequencies are retained.
    """
    dispersion_relation = lambda omega: -(k + omega**3) / (2 * m + 1 + k**2)

    omega_approx = dispersion_relation(first_guess)
    for _ in range(niter):
        omega_approx = dispersion_relation(omega_approx)

    omega_positive = np.where(omega_approx >= 0, omega_approx, np.nan)
    return omega_positive
