#!/usr/bin/env python
# coding: utf-8

# ## NBconvertApp

if __name__ == "__main__":
    import subprocess, os

    subprocess.run(["bash", "../convert.sh"], check=True)


# ## Dependencies

import numpy as np


# ## Definitions

# ### Moving Average

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
        from .constants import MOVING_AVERAGE

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


# ### Split Dimension

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


# ### $f_0$ and $\beta$

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
    from .constants import EARTH

    # Convert latitude from degrees to radians
    latitude_rad = np.deg2rad(latitude)

    # Calculate the Coriolis frequency (f0)
    coriolis_frequency = 2 * EARTH.ANGULAR_VELOCITY * np.sin(latitude_rad)

    # Calculate the Rossby parameter (beta)
    rossby_parameter = 2 * EARTH.ANGULAR_VELOCITY * np.cos(latitude_rad) / EARTH.RADIUS

    return coriolis_frequency, rossby_parameter


# ### Gravity Wave Speed

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
    from .constants import EARTH

    # Calculate gravity wave speed: c = sqrt(g * H)
    gravity_wave_speed = np.sqrt(EARTH.GRAVITY_ACCELERATION * equivalent_depth)

    return gravity_wave_speed


# ### Time Rescaling (Legacy)

# def rescale_to_days_and_ordinary_frequency(angular_frequency: np.ndarray) -> np.ndarray:
#     """
#     Rescale frequency from angular frequency in radians per second to ordinary frequency in cycles per day.

#     Parameters
#     ----------
#     angular_frequency : np.ndarray
#         Input array representing angular frequency in radians per second.

#     Returns
#     -------
#     np.ndarray
#         Rescaled frequency in cycles per day (1/day).

#     Notes
#     -----
#     This function converts the input angular frequency to ordinary frequency by:
#     - Converting from seconds to days using the Earth's solar day duration (in seconds).
#     - Dividing by `2 * pi` to change from angular frequency (radians per second) to ordinary frequency (cycles per second).
#     """
#     from constants import EARTH

#     # Convert from angular frequency (rad/s) to ordinary frequency (cycles/day)
#     ordinary_frequency_per_day = (
#         angular_frequency * EARTH.SOLAR_DAY_TO_SECONDS / (2 * np.pi)
#     )

#     return ordinary_frequency_per_day

