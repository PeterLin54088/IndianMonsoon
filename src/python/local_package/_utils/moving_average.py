import numpy as np


def moving_average(
    data: np.ndarray,
    axis: int = -1,
    window_size: int = None,
    masked: bool = True,
) -> np.ndarray:
    if window_size is None:
        from local_package._constants.MovingAverageOperator import (
            MovingAverageOperator,
        )

        window_size = MovingAverageOperator.WINDOW_SIZE
    window = np.ones(window_size) / window_size
    smoothed_data = np.apply_along_axis(
        lambda values: np.convolve(values, window, mode="same"),
        axis=axis,
        arr=data,
    )
    if masked:
        pad = window_size // 2
        smoothed_data = np.moveaxis(smoothed_data, axis, 0)
        smoothed_data[:pad, ...] = np.nan
        smoothed_data[-pad:, ...] = np.nan
        smoothed_data = np.moveaxis(smoothed_data, 0, axis)

    return smoothed_data
