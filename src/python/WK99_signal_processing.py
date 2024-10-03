import numpy as np
from scipy.signal import detrend as scipy_linear_detrend


def decompose_symmetric_antisymmetric(data, axis):
    """
    Decompose data into symmetric and antisymmetric components along the specified axis.

    For latitude-dependent variables (e.g., f(y)), the decomposition is:
      - Symmetric: (f(y) + f(-y)) / 2
      - Antisymmetric: (f(y) - f(-y)) / 2
      - Total: f(y) = Symmetric + Antisymmetric

    Parameters:
    data (ndarray): Input array to decompose.
    axis (int): Axis along which the decomposition is applied.

    Returns:
    tuple: (symmetric_component, antisymmetric_component)

    Note:
    Ensure the data includes the full latitude range (e.g., from -L to L) for proper decomposition,
    as symmetry is defined relative to the equator.
    """
    # Pre-allocate memory for symmetric and antisymmetric components
    symmetric_component = np.empty_like(data)
    antisymmetric_component = np.empty_like(data)
    # Perform in-place operations to avoid temporary arrays
    np.add(data, np.flip(data, axis=axis), out=symmetric_component)
    symmetric_component /= 2
    np.subtract(data, np.flip(data, axis=axis), out=antisymmetric_component)
    antisymmetric_component /= 2
    return symmetric_component, antisymmetric_component


def compute_stochastic_PSD(
    data, data_grid, segment_length=96, overlap_length=65, taper=None
):
    """
    Compute the stochastic Power Spectral Density (PSD) from segmented time series data.

    Parameters:
    -----------
    data : ndarray
        Input data array where the first axis represents time and the last axis represents longitude.
        Intermediate dimensions (if any) are retained without modification.
    data_grid : tuple of ndarrays
        Tuple containing arrays for the data grid, where the first axis corresponds to time and the last
        axis corresponds to longitude. Other dimensions can vary based on the data.
    segment_length : int, optional
        Length of each segment for FFT, default is 96.
    overlap_length : int, optional
        Number of overlapping points between segments, default is 65.
    taper : ndarray, optional
        Tapering window applied to segments. If None, a default Hanning-like taper is used.

    Returns:
    --------
    PSD : ndarray
        Stochastic power spectral density.
    modified_data_grid : tuple of ndarrays
        Tuple of modified time and longitude wavenumbers for the PSD, with other grid dimensions unchanged.

    Notes:
    ------
    - The input data can have any shape, but the first axis must represent time and the last axis must represent longitude.
    - The wavenumbers returned are nondimensional ordinary wavenumbers, not angular wavenumbers.
    """

    def segment_data(array, segment_length, overlap_length):
        """Segment the input array into overlapping subarrays."""
        step = segment_length - overlap_length
        num_segments = (len(array) - overlap_length) // step
        return np.array(
            [array[i : i + segment_length] for i in range(0, num_segments * step, step)]
        )

    def default_taper():
        """Generate a default Hanning-like taper for the first and last third of the segment."""
        taper_window = np.ones(segment_length)
        third_len = len(taper_window) // 3
        taper_cosine = 0.5 * (
            1 - np.cos(2 * np.pi * np.arange(third_len) / (2 * third_len))
        )
        taper_window[:third_len] = taper_cosine
        taper_window[-third_len:] = taper_cosine[::-1]
        return taper_window

    # Use provided taper window or generate the default one
    taper_window = default_taper() if taper is None else taper

    # Segment, detrend, taper, and perform FFT over time and longitude
    tmp = segment_data(np.copy(data), segment_length, overlap_length)
    tmp = scipy_linear_detrend(tmp, axis=1)

    # Apply taper across the time axis
    tmp = np.swapaxes(tmp, axis1=1, axis2=-1)
    tmp *= taper_window
    tmp = np.swapaxes(tmp, axis1=1, axis2=-1)

    # FFT along longitude and time axes
    tmp = np.fft.fft(tmp, axis=-1, norm="ortho")
    tmp = np.fft.ifft(tmp, axis=1, norm="ortho")

    # Compute the power spectrum and average across segments
    tmp = np.abs(tmp) ** 2
    tmp = np.mean(tmp, axis=0)

    # Zero-center the PSD for visualization
    PSD = np.fft.fftshift(tmp, axes=(0, -1))

    # Compute nondimensional wavenumbers for time and longitude
    wavemode_longitude = np.fft.fftshift(
        np.fft.fftfreq(PSD.shape[-1], 1 / PSD.shape[-1])
    )
    wavemode_time = np.fft.fftshift(np.fft.fftfreq(segment_length, 1 / segment_length))

    # Update the data grid with modified time and longitude wavenumbers
    modified_data_grid = list(data_grid)
    modified_data_grid[0] = wavemode_time
    modified_data_grid[-1] = wavemode_longitude
    modified_data_grid = tuple(modified_data_grid)

    return PSD, modified_data_grid


def extract_positive_PSD(PSD, spectral_grid, axis=0):
    """
    Extract the positive half of a symmetric Power Spectral Density (PSD) along a specified axis,
    preserving Parseval's identity. Assumes real-valued input, resulting in Hermitian symmetry.

    Parameters:
    PSD : ndarray
        N-D Power Spectral Density array.
    spectral_grid : tuple of ndarrays
        Frequency/wavenumber grids for each axis.
    axis : int, optional
        Axis along which to extract the positive half, default is 0 (time axis).

    Returns:
    positive_psd : ndarray
        Positive half of the PSD with doubled power for positive frequencies.
    (positive_frequencies, wavenumbers) : tuple of ndarrays
        Corresponding positive frequencies/wavenumbers and the unchanged spectral grid.
    """
    frequencies = spectral_grid[axis]
    midpoint = len(frequencies) // 2

    # Move the specified axis to the front for processing
    PSD = np.moveaxis(PSD, axis, 0)

    if len(frequencies) % 2 == 0:
        # Even case: include the highest frequency, double positive power
        positive_psd = np.concatenate([PSD[midpoint:], PSD[[0]]], axis=0)
        positive_psd[1:-1] *= 2
        positive_frequencies = np.flip(np.abs(frequencies[: midpoint + 1]))
    else:
        # Odd case: avoid double counting at the midpoint
        positive_psd = np.copy(PSD[midpoint:])
        positive_psd[1:] *= 2
        positive_frequencies = frequencies[midpoint:]

    # Restore original axis position
    positive_psd = np.moveaxis(positive_psd, 0, axis)

    # Update the spectral grid with positive frequencies
    modified_spectral_grid = list(spectral_grid)
    modified_spectral_grid[axis] = positive_frequencies

    return positive_psd, tuple(modified_spectral_grid)


def apply_121_filter(array, axis, iterations):
    """
    Apply a simple 1-2-1 Gaussian filter along a specified axis using convolution.
    Boundary extension is used to handle Parsevel's identity.

    Parameters:
    array : ndarray
        The input data array to be filtered.
    axis : int
        The axis along which the filter is applied.
    iterations : int
        Number of times the filter is applied.

    Returns:
    result : ndarray
        The filtered array with the same shape as the input array.
    """

    def extend_boundaries(arr, ax):
        """Extend boundaries by duplicating the first and last elements along the given axis."""
        if ax < 0 or ax >= arr.ndim:
            raise ValueError(
                f"Axis {ax} is out of bounds for array with {arr.ndim} dimensions."
            )

        # Create slices for first and last elements along the axis
        first_slice = [slice(None)] * arr.ndim
        last_slice = [slice(None)] * arr.ndim
        first_slice[ax] = slice(0, 1)
        last_slice[ax] = slice(-1, None)

        # Extend array by adding duplicated boundaries
        return np.concatenate(
            [arr[tuple(first_slice)], arr, arr[tuple(last_slice)]], axis=ax
        )

    def convolve_along_axis(data, axis, kernel):
        """Extend boundaries and convolve along the specified axis."""
        extended_data = extend_boundaries(data, axis)
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=axis, arr=extended_data
        )

    # Simple 1-2-1 Gaussian kernel
    kernel = np.array([1 / 4, 1 / 2, 1 / 4])

    result = np.copy(array)
    for _ in range(iterations):
        result = convolve_along_axis(result, axis, kernel)

    return result
