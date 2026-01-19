import numpy as np


def get_monsoon_onset_time(monsoon_index: np.ndarray) -> np.ndarray:
    from local_package._utils.moving_average import moving_average
    from local_package._utils.split_dimension import split_dimension

    monsoon_index_smoothed = split_dimension(
        moving_average(monsoon_index.flatten())
    )
    years = 1979 + np.arange(np.shape(monsoon_index_smoothed)[0])
    onset_time_indices = np.argmax(monsoon_index_smoothed > 0, axis=1)
    monsoon_onset = list(zip(years, onset_time_indices))
    return monsoon_onset
