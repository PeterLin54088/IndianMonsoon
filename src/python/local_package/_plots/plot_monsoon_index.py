import numpy as np
import matplotlib.pyplot as plt


def plot_monsoon_index(monsoon_index: np.ndarray, **kwargs):
    """
    Plot the evolution of the Indian Monsoon Index (IMI) over a year.

    Parameters:
        indian_monsoon_index (np.ndarray):
            A 2D array where each column represents the IMI values for
            a specific day of the year across multiple years.

    Returns:
        plt.Figure: A matplotlib Figure containing the plot.
    """
    from local_package._utils.moving_average import moving_average
    from local_package._utils.split_dimension import split_dimension

    kwargs.setdefault("output_path", "")

    monsoon_index_smoothed = split_dimension(
        moving_average(monsoon_index.flatten())
    )
    climatology_monsoon_index = np.nanmean(monsoon_index_smoothed, axis=0)

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=160)
    ax.plot(
        np.arange(1, 366),
        climatology_monsoon_index,
        c="red",
        linestyle="-",
        linewidth=4,
        zorder=10,
        label="Climatological",
    )
    ax.plot(
        1 + np.arange(365),
        monsoon_index_smoothed.T,
        c="grey",
        linestyle="-",
        linewidth=0.8,
    )
    ax.axhline(y=0, c="black", linestyle="--", linewidth=2.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Zonal Wind Shear (m/s)")
    ax.set_xticks(
        np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]),
        labels=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        fontsize=16,
    )
    ax.set_xlim(1, 365)
    ax.set_ylim(-17, 17)
    ax.set_title("Monsoon Index")
    ax.legend(loc="upper right")

    fig.savefig(kwargs["output_path"])
    plt.close(fig)

    return None
