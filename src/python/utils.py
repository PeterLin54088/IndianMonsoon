import numpy as np


def moving_average(
    data: np.ndarray, axis: int = -1, window_size: int = None
) -> np.ndarray:
    """
    Apply a moving average to the specified axis of a NumPy array.

    Parameters
    ----------
    data : np.ndarray
        Input data array to smooth.
    axis : int, optional
        Axis along which the moving average is computed (default is 0).
    window_size : int, optional
        Size of the moving window for averaging. If not provided, a default value
        from the constants module is used.

    Returns
    -------
    np.ndarray
        The smoothed data with the same shape as the input array. Boundary regions
        where the moving average cannot be computed are filled with `np.nan`.

    Notes
    -----
    - Convolution is performed using 'same' mode to ensure the output shape matches
      the input along the specified axis.
    - Boundary effects are handled by padding the result with `np.nan` where the
      window is incomplete.
    """
    # Set the window size to a default if not provided
    if window_size is None:
        from constants import MOVING_AVERAGE

        window_size = MOVING_AVERAGE.WINDOW_SIZE

    # Create a uniform window for averaging
    window = np.ones(window_size) / window_size

    # Apply convolution along the specified axis using the defined window
    smoothed_data = np.apply_along_axis(
        lambda values: np.convolve(values, window, mode="same"), axis=axis, arr=data
    )

    # Compute the padding size for boundary regions
    pad = window_size // 2

    # Handle boundary regions by padding with np.nan
    smoothed_data = np.moveaxis(
        smoothed_data, axis, 0
    )  # Temporarily move axis for easier indexing
    smoothed_data[:pad, ...] = np.nan  # Pad the beginning of the data
    smoothed_data[-pad:, ...] = np.nan  # Pad the end of the data
    smoothed_data = np.moveaxis(
        smoothed_data, 0, axis
    )  # Move axis back to original position

    return smoothed_data


def split_dimension(
    array: np.ndarray, axis: int, factors: tuple = (43, 365)
) -> np.ndarray:
    """
    Split a specified dimension of an array into multiple factors.

    Parameters
    ----------
    array : np.ndarray
        Input array whose specified dimension will be split.
    axis : int
        The axis of the array to split.
    factors : tuple
        Factors to split the axis dimension into. The product of the factors must
        match the size of the axis being split.

    Returns
    -------
    np.ndarray
        Reshaped array with the specified dimension split into the provided factors.

    Raises
    ------
    ValueError
        If the dimension index is out of range or the product of factors does not
        match the size of the specified dimension.
    """
    # Get the original size of the dimension to split
    original_size = array.shape[axis]

    # Calculate the product of the factors
    new_size = np.prod(factors)

    # Ensure the product of factors matches the original dimension size
    if original_size != new_size:
        raise ValueError(
            f"Cannot split dimension {axis} of size {original_size} into factors {factors} "
            f"with product {new_size}."
        )

    # Construct the new shape by inserting the factors at the specified axis
    new_shape = list(array.shape[:axis]) + list(factors) + list(array.shape[axis + 1 :])

    # Reshape the array into the new shape
    reshaped_array = array.reshape(new_shape)

    return reshaped_array


def decompose_symmetric_antisymmetric(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Decompose a dataset into its symmetric and antisymmetric components along a specified axis.

    Parameters
    ----------
    data : np.ndarray
        The input array representing data that should be decomposed.
    axis : int
        The axis along which to decompose the data into symmetric and antisymmetric components.

    Returns
    -------
    np.ndarray
        A np.ndarray with first dimension contains:
        - symmetric_component : The symmetric part of the data.
        - antisymmetric_component : The antisymmetric part of the data.

    Notes
    -----
    - This decomposition assumes that the input data spans a full range relative to the symmetry axis
      (e.g., latitude from 90S to 90N for geographic data).
    """

    # Prepare empty arrays for the symmetric and antisymmetric components
    symmetric_component = np.empty_like(data)
    antisymmetric_component = np.empty_like(data)

    # Compute the symmetric component: (data + flipped_data) / 2
    flipped_data = np.flip(data, axis=axis)
    np.add(data, flipped_data, out=symmetric_component)
    symmetric_component /= 2

    # Compute the antisymmetric component: (data - flipped_data) / 2
    np.subtract(data, flipped_data, out=antisymmetric_component)
    antisymmetric_component /= 2

    return np.array([symmetric_component, antisymmetric_component])


def segment_data(
    array: np.ndarray, segment_length: int = None, overlap_length: int = None
) -> np.ndarray:
    """
    Segment an array into overlapping sections of a specified length, returning an array
    with one additional dimension compared to the input.

    Parameters
    ----------
    array : np.ndarray
        The input array to be segmented. Can be of any dimensionality.
    segment_length : int, optional
        The length of each segment. If not provided, a default value from the WK99 constants is used.
    overlap_length : int, optional
        The number of overlapping elements between consecutive segments. If not provided,
        a default value from the WK99 constants is used.

    Returns
    -------
    np.ndarray
        An array with one additional dimension compared to the input. The added dimension
        corresponds to the segmented portions of the input array, each of length `segment_length`.

    Notes
    -----
    - The input array can have any number of dimensions, and the output will have one additional
      dimension compared to the input.
    - Segmentation occurs along the second axis of the array by default.
    """

    from constants import WK99  # Import constants for default values if not provided

    # Use default values if segment_length or overlap_length are not provided
    if segment_length is None:
        segment_length = WK99.SEGMENTATION_LENGTH
    if overlap_length is None:
        overlap_length = WK99.OVERLAP_LENGTH

    # Calculate the step size based on segment length and overlap
    step = segment_length - overlap_length

    # Determine the number of full segments that can be generated from the array
    num_segments = (array.shape[1] - overlap_length) // step

    # Create an array of slices (each of length `segment_length`) along the first axis
    segmented_shape = list(array.shape)
    segmented_shape.pop(1)
    segmented_shape.insert(1, num_segments)
    segmented_shape.insert(2, segment_length)
    segments = np.empty(shape=tuple(segmented_shape), dtype=float)
    for i in range(0, num_segments):
        start = i * step
        end = i * step + segment_length
        segments[:, i, :, :, :] = array[:, start:end, :, :]

    # Move the new segmentation dimension to the front and maintain other dimensions
    return segments


def apply_121_filter(
    data: np.ndarray, axis: int = -1, iterations: int = 10
) -> np.ndarray:
    """
    Apply a 1-2-1 filter along a specified axis using convolution.
    The filter is applied iteratively, with boundary extension to keep Parsevel's identity.

    Parameters
    ----------
    data : np.ndarray
        Input array to which the filter will be applied.
    axis : int
        The axis along which the filter will be applied.
    iterations : int, optional
        Number of times to apply the filter (default is 10).

    Returns
    -------
    np.ndarray
        The filtered array after the specified number of iterations.

    Notes
    -----
    Boundary extension is performed by duplicating the first and last elements
    along the given axis to ensure that Parsevel's identity is handled properly.
    """

    def extend_boundaries(arr: np.ndarray, ax: int) -> np.ndarray:
        """
        Extend the boundaries of an array by duplicating the first and last elements
        along a specified axis.

        Parameters
        ----------
        arr : np.ndarray
            Input array to extend boundaries on.
        ax : int
            Axis along which to extend the boundaries.

        Returns
        -------
        np.ndarray
            Array with extended boundaries.
        """
        first_slice = [slice(None)] * arr.ndim
        last_slice = [slice(None)] * arr.ndim
        first_slice[ax] = slice(0, 1)
        last_slice[ax] = slice(-1, None)
        # Extend array by adding duplicated boundaries
        return np.concatenate(
            [arr[tuple(first_slice)], arr, arr[tuple(last_slice)]], axis=ax
        )

    def convolve_along_axis(
        arr: np.ndarray, ax: int, kernel: np.ndarray = np.array([1 / 4, 1 / 2, 1 / 4])
    ) -> np.ndarray:
        """
        Convolve an array along a specified axis using the provided kernel, after extending boundaries.

        Parameters
        ----------
        arr : np.ndarray
            Input array to convolve.
        ax : int
            Axis along which to apply the convolution.
        kernel : np.ndarray
            1D convolution kernel.

        Returns
        -------
        np.ndarray
            The convolved array.
        """
        extended_arr = extend_boundaries(arr, ax=ax)
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=ax, arr=extended_arr
        )

    # Create a copy of the input data to avoid modifying the original array
    filtered_data = np.copy(data)

    # Apply the filter for the specified number of iterations
    for _ in range(iterations):
        filtered_data = convolve_along_axis(filtered_data, axis)

    return filtered_data


def get_f_beta(latitude: float = 0.0) -> tuple[float, float]:
    """
    Calculate the Coriolis frequency (f) and Rossby parameter (beta) at a given latitude.

    Parameters
    ----------
    latitude : float, optional
        Latitude in degrees, where the calculation is performed. Default is 0.0 (equator).

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - coriolis_frequency (float): The Coriolis frequency (f) at the specified latitude.
        - rossby_parameter (float): The Rossby parameter (beta) at the specified latitude.
    """
    from constants import EARTH

    # Convert latitude from degrees to radians
    latitude_rad = np.deg2rad(latitude)

    # Calculate the Coriolis frequency (f0)
    coriolis_frequency = 2 * EARTH.ANGULAR_VELOCITY * np.sin(latitude_rad)

    # Calculate the Rossby parameter (beta)
    rossby_parameter = 2 * EARTH.ANGULAR_VELOCITY * np.cos(latitude_rad) / EARTH.RADIUS

    return coriolis_frequency, rossby_parameter


def get_gravity_wave_speed(equivalent_depth: np.ndarray) -> np.ndarray:
    """
    Calculate the gravity wave speed based on the equivalent depth.

    Parameters
    ----------
    equivalent_depth : np.ndarray
        The equivalent depth.

    Returns
    -------
    np.ndarray
        Gravity wave speed.
    """
    from constants import EARTH

    # Calculate gravity wave speed: c = sqrt(g * H)
    gravity_wave_speed = np.sqrt(EARTH.GRAVITY_ACCELERATION * equivalent_depth)

    return gravity_wave_speed


def rescale_to_days_and_ordinary_frequency(angular_frequency: np.ndarray) -> np.ndarray:
    """
    Rescale frequency from angular frequency in radians per second to ordinary frequency in cycles per day.

    Parameters
    ----------
    angular_frequency : np.ndarray
        Input array representing angular frequency in radians per second.

    Returns
    -------
    np.ndarray
        Rescaled frequency in cycles per day (1/day).

    Notes
    -----
    This function converts the input angular frequency to ordinary frequency by:
    - Converting from seconds to days using the Earth's solar day duration (in seconds).
    - Dividing by `2 * pi` to change from angular frequency (radians per second) to ordinary frequency (cycles per second).
    """
    from constants import EARTH

    # Convert from angular frequency (rad/s) to ordinary frequency (cycles/day)
    ordinary_frequency_per_day = (
        angular_frequency * EARTH.SOLAR_DAY_TO_SECONDS / (2 * np.pi)
    )

    return ordinary_frequency_per_day
