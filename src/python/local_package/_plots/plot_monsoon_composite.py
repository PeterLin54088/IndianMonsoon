import os
import numpy as np
import matplotlib.pyplot as plt


def plot_monsoon_composite(
    shading: np.ndarray,
    contour: np.ndarray,
    grids: dict[str, np.ndarray],
    calendar_index: int = 0,
    **kwargs,
) -> plt.Figure:
    from datetime import timedelta
    from matplotlib.colors import ListedColormap
    from local_package._constants.ERA5DataManager import ERA5DataManager
    from local_package._utils.moving_average import moving_average

    kwargs.setdefault("cmap", "RdBu_r")
    kwargs.setdefault("shading_levels", np.linspace(-1, 1, 11, endpoint=True))
    kwargs.setdefault("contour_levels", np.linspace(-1, 1, 11, endpoint=True))
    kwargs.setdefault("colorbar_levels", np.linspace(-1, 1, 5, endpoint=True))
    kwargs.setdefault("plt_title", "")
    kwargs.setdefault("plt_label", "")
    kwargs.setdefault("output_path", "")
    kwargs.setdefault("filename", "")
    refdate = ERA5DataManager.YYMMDD

    colormap = plt.get_cmap(kwargs["cmap"], 128)
    colormap = colormap(np.linspace(0, 1, 128))
    colormap[64 - 4 : 64 + 4, :] = np.array([1, 1, 1, 1])
    colormap = ListedColormap(colormap)

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(16, 4.5), dpi=160)

    img = ax.contourf(
        grids["lat"],
        grids["plev"],
        moving_average(
            moving_average(shading, axis=0, window_size=2, masked=False),
            axis=1,
            window_size=3,
            masked=False,
        ),
        levels=kwargs["shading_levels"],
        extend="both",
        cmap=colormap,
    )
    ax.contour(
        grids["lat"],
        grids["plev"],
        moving_average(
            moving_average(contour, axis=0, window_size=2, masked=False),
            axis=1,
            window_size=3,
            masked=False,
        ),
        levels=kwargs["contour_levels"],
        colors="k",
    )
    ax.set_xlabel("Latitude")
    ax.set_yticks([1000, 925, 850, 700, 500, 250, 200, 100])
    ax.invert_yaxis()
    ax.set_title("Monsoon Climatology")

    cbar = plt.colorbar(img)
    cbar.set_ticks(kwargs["colorbar_levels"])
    cbar.ax.set_title(kwargs["plt_label"], fontsize=16)

    date = refdate + timedelta(days=int(calendar_index))
    fig.suptitle(kwargs["plt_title"] + f"Date: {date.strftime('%m/%d')}")
    filename = (
        "_".join([kwargs["filename"], str(date.strftime("%m%d"))]) + ".png"
    )
    filepath = os.path.join(kwargs["output_path"], filename)
    fig.savefig(filepath)
    plt.close(fig)

    return None


def plot_monsoon_early_late_composite(
    shading_early: np.ndarray,
    shading_late: np.ndarray,
    contour_early: np.ndarray,
    contour_late: np.ndarray,
    grids: dict[str, np.ndarray],
    calendar_index: int = 0,
    **kwargs,
) -> plt.Figure:
    from datetime import timedelta
    from matplotlib.colors import ListedColormap
    from local_package._constants.ERA5DataManager import ERA5DataManager
    from local_package._utils.moving_average import moving_average

    kwargs.setdefault("cmap", "RdBu_r")
    kwargs.setdefault("shading_levels", np.linspace(-1, 1, 11, endpoint=True))
    kwargs.setdefault("contour_levels", np.linspace(-1, 1, 11, endpoint=True))
    kwargs.setdefault("colorbar_levels", np.linspace(-1, 1, 5, endpoint=True))
    kwargs.setdefault("plt_title", "")
    kwargs.setdefault("plt_label", "")
    kwargs.setdefault("output_path", "")
    kwargs.setdefault("filename", "")
    refdate = ERA5DataManager.YYMMDD

    colormap = plt.get_cmap(kwargs["cmap"], 128)
    colormap = colormap(np.linspace(0, 1, 128))
    colormap[64 - 4 : 64 + 4, :] = np.array([1, 1, 1, 1])
    colormap = ListedColormap(colormap)

    plt.rcParams.update({"font.size": 20})
    fig = plt.figure(figsize=(16, 9), dpi=160)
    spec = fig.add_gridspec(32, 32)
    ax0 = fig.add_subplot(spec[:15, :30])
    ax1 = fig.add_subplot(spec[17:, :30])
    ax2 = fig.add_subplot(spec[:, -1])
    axes = (ax0, ax1, ax2)

    axes[0].contourf(
        grids["lat"],
        grids["plev"],
        moving_average(
            moving_average(shading_early, axis=0, window_size=2, masked=False),
            axis=1,
            window_size=3,
            masked=False,
        ),
        levels=kwargs["shading_levels"],
        extend="both",
        cmap=colormap,
    )
    axes[0].contour(
        grids["lat"],
        grids["plev"],
        moving_average(
            moving_average(contour_early, axis=0, window_size=2, masked=False),
            axis=1,
            window_size=3,
            masked=False,
        ),
        levels=kwargs["contour_levels"],
        colors="k",
    )
    axes[0].set_xticks([])
    axes[0].set_ylabel("Pressure (hPa)", y=-0.1)
    axes[0].set_yticks([1000, 925, 850, 700, 500, 250, 200, 100])
    axes[0].invert_yaxis()
    axes[0].set_title("Early Onset")

    img = axes[1].contourf(
        grids["lat"],
        grids["plev"],
        moving_average(
            moving_average(shading_late, axis=0, window_size=2, masked=False),
            axis=1,
            window_size=3,
            masked=False,
        ),
        levels=kwargs["shading_levels"],
        extend="both",
        cmap=colormap,
    )
    axes[1].contour(
        grids["lat"],
        grids["plev"],
        moving_average(
            moving_average(contour_late, axis=0, window_size=2, masked=False),
            axis=1,
            window_size=3,
            masked=False,
        ),
        levels=kwargs["contour_levels"],
        colors="k",
    )
    axes[1].set_xlabel("Latitude")
    axes[1].set_yticks([1000, 925, 850, 700, 500, 250, 200, 100])
    axes[1].invert_yaxis()
    axes[1].set_title("Late Onset")

    cbar = plt.colorbar(img, cax=axes[2])
    cbar.set_ticks(kwargs["colorbar_levels"])
    cbar.ax.set_title(kwargs["plt_label"], fontsize=16)

    date = refdate + timedelta(days=int(calendar_index))
    fig.suptitle(kwargs["plt_title"] + f"Date: {date.strftime('%m/%d')}")

    filename = (
        "_".join([kwargs["filename"], str(date.strftime("%m%d"))]) + ".png"
    )
    filepath = os.path.join(kwargs["output_path"], filename)
    fig.savefig(filepath)
    plt.close(fig)

    return None
