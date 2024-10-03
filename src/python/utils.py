import numpy as np
from constants import WINDOW_SIZE
from constants import EARTH

def moving_average(data, axis=0, WINDOW_SIZE=WINDOW_SIZE):
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
        2 * EARTH.ANGULAR_VELOCITY * np.sin(latitude)
    )  # Coriolis frequency
    rossby_parameter = (
        2 * EARTH.ANGULAR_VELOCITY * np.cos(latitude) / EARTH.RADIUS
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
    c = np.sqrt(EARTH.GRAVITY_ACCELERATION * equivalent_depth)  # Gravity wave speed
    return c


def rescale_to_days_and_ordinary_frequency(frequency):
    # Convert from (1/second) to (1/day) and from angular to ordinary frequency
    return frequency * EARTH.SOLAR_DAY_TO_SECONDS / (2 * np.pi)