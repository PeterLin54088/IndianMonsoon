#!/usr/bin/env python
# coding: utf-8

# ## NBconvertApp

if __name__ == "__main__":
    import subprocess, os

    subprocess.run(["bash", "../convert.sh"], check=True)


# ## Dependencies

import sys, os
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
from matplotlib.patches import Rectangle


# ## IMI Evolution

def display_IMI_evolution(indian_monsoon_index: np.ndarray, **kwargs) -> plt.Figure:
    """
    Plot the evolution of the Indian Monsoon Index (IMI) over a year.

    Parameters:
        indian_monsoon_index (np.ndarray):
            A 2D array where each column represents the IMI values for
            a specific day of the year across multiple years.

    Returns:
        plt.Figure: A matplotlib Figure containing the plot.
    """
    kwargs.setdefault("output_path", "")

    climatology_IMI = np.mean(
        np.where(
            np.isnan(indian_monsoon_index),
            np.nanmean(indian_monsoon_index, axis=0),
            indian_monsoon_index,
        ),
        axis=0,
    )
    yearly_IMI = indian_monsoon_index.T

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=160)

    ax.plot(
        np.arange(1, 366),
        climatology_IMI,
        c="red",
        linestyle="-",
        linewidth=4,
        zorder=10,
        label="Climatological",
    )
    ax.plot(
        np.arange(1, 366),
        yearly_IMI,
        c="grey",
        linestyle="-",
        linewidth=0.8,
    )
    ax.axhline(y=0, c="black", linestyle="--", linewidth=2.5)
    ax.scatter([138], [0], s=100, c="blue", zorder=20)

    ax.set_xlabel("Date")
    ax.set_ylabel("Zonal Wind Shear (m/s)")
    ax.set_xticks(
        [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
        labels=[
            "01/01",
            "02/01",
            "03/01",
            "04/01",
            "05/01",
            "06/01",
            "07/01",
            "08/01",
            "09/01",
            "10/01",
            "11/01",
            "12/01",
        ],
        fontsize=16,
    )
    ax.set_xlim(1, 365)
    ax.set_ylim(-15, 15)
    ax.set_title("Indian Monsoon Index\n 05/18")
    ax.legend(loc="upper right")

    fig.savefig(kwargs["output_path"])
    plt.close(fig)

    return None


# ## Observation Profile Gif

def display_early_late_composite(
    shading_early: np.ndarray,
    shading_late: np.ndarray,
    contour_early: np.ndarray,
    contour_late: np.ndarray,
    grids: dict[str, np.ndarray],
    calendar_index: int = 0,
    **kwargs,
) -> plt.Figure:
    """
    Visualize early and late onset streamfunctions with a colormap and contours.

    Parameters:
    - streamfunction_early (np.ndarray): 2D array of early onset streamfunction values.
    - streamfunction_late (np.ndarray): 2D array of late onset streamfunction values.
    - grids (dict[str, np.ndarray]): Dictionary containing latitude (`lat`) and pressure levels (`plev`).
    - calendar_index (int, optional): Day index since 01/01 for date display. Default is 0.

    Returns:
    - plt.Figure: The generated matplotlib figure object containing the visualizations.
    """

    from datetime import datetime, timedelta
    from matplotlib.colors import ListedColormap

    kwargs.setdefault("cmap", "RdBu_r")
    kwargs.setdefault("shading_levels", np.linspace(-1, 1, 11, endpoint=True))
    kwargs.setdefault("contour_levels", np.linspace(-1, 1, 11, endpoint=True))
    kwargs.setdefault("colorbar_levels", np.linspace(-1, 1, 5, endpoint=True))
    kwargs.setdefault("plt_title", "")
    kwargs.setdefault("plt_label", "")
    kwargs.setdefault("output_path", "")
    kwargs.setdefault("filename", "")
    refdate = datetime(1979, 1, 1)

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
        shading_early,
        levels=kwargs["shading_levels"],
        extend="both",
        cmap=colormap,
    )
    axes[0].contour(
        grids["lat"],
        grids["plev"],
        contour_early,
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
        shading_late,
        levels=kwargs["shading_levels"],
        extend="both",
        cmap=colormap,
    )
    axes[1].contour(
        grids["lat"],
        grids["plev"],
        contour_late,
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
    fig.suptitle(kwargs["plt_title"] + f"Date: {date.strftime("%m/%d")}")

    filename = "_".join([kwargs["filename"], str(date.strftime("%m%d"))]) + ".png"
    filepath = os.path.join(kwargs["output_path"], filename)
    fig.savefig(filepath)
    plt.close(fig)

    return None


# ## Wn_Fr Diagram

def display_wavenumber_frequency_diagram(
    symmetric_PSD: np.ndarray,
    antisymmetric_PSD: np.ndarray,
    background_PSD: np.ndarray,
    dimensions: dict[str, np.ndarray],
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Notes:
    -------
    - **Zonal Wavenumber (k)**: In the wave expression e^{i k x}, where x ∈ [0, 2πR], the zonal wavenumber k = N/R defines the spatial frequency of the wave in the zonal direction, specifying how many oscillations occur over the distance 2πR.
    - **Zonal Wavemode (N)**: In the expression e^{i N/R x}, where x ∈ [0, 2πR], N is a positive integer that quantifies the number of wave oscillations around the full zonal circumference 2πR. Larger N values correspond to more oscillations over the same distance.
    """

    # import

    from matplotlib.style import context
    from constants import ENVIRONMENT_PATH
    from LinearShallowEquatorialWave import (
        dispersion_kelvin,
        dispersion_poincare,
        dispersion_mrg,
        dispersion_rossby,
    )
    from utils import (
        get_f_beta,
        get_gravity_wave_speed,
    )

    def initialize_kwargs(kwargs: dict) -> dict:
        """
        Initializes the provided `kwargs` dictionary with default values.

        Parameters:
            kwargs (dict): A dictionary of optional parameters to initialize or update.

        Returns:
            dict: The updated dictionary containing default values for missing keys.
        """
        from matplotlib.colors import ListedColormap

        tropical_wave_defaults = {
            "zonal_wavemodes": np.linspace(-15, 15, 121, endpoint=True),
            "reference_latitude": 0.0,
            "equivalent_depths": np.array([12.5, 25.0, 50.0]),
        }

        Greys = plt.get_cmap("Greys", 150)
        StepGreys = Greys(np.linspace(0, 1, 150))
        StepGreys[: 15 * 4, :] = np.array([1, 1, 1, 1])
        StepGreys = ListedColormap(StepGreys)
        plot_defaults = {
            "variable_name": "Undefined",
            "dispersion_line_order": "typical",
            "cmap": StepGreys,
            "wavenumber_indices": slice(
                np.argmax(
                    dimensions["zonal_wavenumber"]
                    >= tropical_wave_defaults["zonal_wavemodes"][0]
                ),
                np.argmax(
                    dimensions["zonal_wavenumber"]
                    >= tropical_wave_defaults["zonal_wavemodes"][-1]
                )
                + 1,
            ),
            "frequency_indices": slice(
                np.argmax(dimensions["segment_frequency"] > 0),
                np.argmax(dimensions["segment_frequency"]) + 1,
            ),
        }
        plot_additional = {"wk_filter": None}

        defaults = {}
        defaults.update(tropical_wave_defaults)
        defaults.update(plot_defaults)
        defaults.update(plot_additional)

        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        return kwargs

    kwargs = initialize_kwargs(kwargs)

    def linear_shallow_wave():
        def compute_frequency(
            dispersion_function,
            meridional_mode_number: int = None,
        ):
            """Compute the CPD (cycles per day) frequency using a given dispersion function."""
            from constants import EARTH

            zonal_wavenumbers = kwargs["zonal_wavemodes"] / EARTH.RADIUS
            f_coriolis, rossby_parameter = get_f_beta(kwargs["reference_latitude"])
            gravity_wave_speeds = get_gravity_wave_speed(kwargs["equivalent_depths"])
            characteristic_length = np.sqrt(gravity_wave_speeds / rossby_parameter)
            characteristic_time = 1 / np.sqrt(gravity_wave_speeds * rossby_parameter)

            # Non-dimensionalize wavenumbers and compute frequency
            nondimensional_wavenumbers = (
                zonal_wavenumbers[:, np.newaxis] * characteristic_length
            )
            nondimensional_frequency = (
                dispersion_function(
                    nondimensional_wavenumbers,
                    meridional_mode_number=meridional_mode_number,
                )
                if meridional_mode_number
                else dispersion_function(nondimensional_wavenumbers)
            )
            dimensional_frequency = nondimensional_frequency * (1 / characteristic_time)
            CPD_frequency = (
                dimensional_frequency * EARTH.SOLAR_DAY_TO_SECONDS / (2 * np.pi)
            )
            return CPD_frequency

        # Calculate theoretical dispersion line
        frequency_kelvin = compute_frequency(dispersion_kelvin)
        frequency_mrg = compute_frequency(dispersion_mrg)
        frequency_poincare_m1 = compute_frequency(
            dispersion_poincare, meridional_mode_number=1
        )
        frequency_poincare_m2 = compute_frequency(
            dispersion_poincare, meridional_mode_number=2
        )
        frequency_rossby_m1 = compute_frequency(
            dispersion_rossby, meridional_mode_number=1
        )
        frequency_rossby_m2 = compute_frequency(
            dispersion_rossby, meridional_mode_number=2
        )

        return np.array(
            [frequency_kelvin, frequency_poincare_m1, frequency_rossby_m1]
        ), np.array([frequency_mrg, frequency_poincare_m2, frequency_rossby_m2])

    def processing(var):
        x = dimensions["zonal_wavenumber"][kwargs["wavenumber_indices"]]
        y = dimensions["segment_frequency"][kwargs["frequency_indices"]]
        z = var[kwargs["frequency_indices"], kwargs["wavenumber_indices"]]
        return x, y, z

    def shading(ax, x, y, z):
        img = ax.contourf(
            x,
            y,
            z,
            cmap=kwargs["cmap"],
            levels=np.linspace(0.6, 2.0, 15, endpoint=True),
            extend="both",
            zorder=-10,
        )
        plt.colorbar(img, ax=ax)
        return None

    def contour(ax, x, y, z):
        ax.contour(
            x,
            y,
            z,
            colors="black",
            levels=np.linspace(0.6, 2.0, 15, endpoint=True),
            linewidths=0.4,
            zorder=-9,
        )
        return None

    def reference_line(ax):
        ax.plot([0, 0], [0, 0.5], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 30, 1 / 30], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 6, 1 / 6], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 3, 1 / 3], "k--", lw=1, zorder=-2)
        return None

    def dispersion_line(ax, frequency):
        for freq in frequency:
            ax.plot(kwargs["zonal_wavemodes"], freq, "k-", zorder=-2)
        return None

    def padding(ax):
        #
        xy = (kwargs["zonal_wavemodes"][0], 0)
        width = kwargs["zonal_wavemodes"][-1] - kwargs["zonal_wavemodes"][0]
        height = dimensions["segment_frequency"][1] - dimensions["segment_frequency"][0]
        tmp_patch = Rectangle(
            xy, width, height, facecolor="white", edgecolor=(1, 1, 1, 0), zorder=-1
        )
        ax.add_patch(tmp_patch)
        #
        xy = (kwargs["zonal_wavemodes"][0], dimensions["segment_frequency"][-1])
        width = kwargs["zonal_wavemodes"][-1] - kwargs["zonal_wavemodes"][0]
        height = dimensions["segment_frequency"][1] - dimensions["segment_frequency"][0]
        tmp_patch = Rectangle(
            xy, width, height, facecolor="white", edgecolor=(1, 1, 1, 0), zorder=-1
        )
        ax.add_patch(tmp_patch)
        return None

    def highlight_box(ax):
        #
        if kwargs["wk_filter"]:
            tmp_patch = Rectangle(
                xy=(kwargs["wk_filter"][0][0], kwargs["wk_filter"][0][1]),
                width=kwargs["wk_filter"][1],
                height=kwargs["wk_filter"][2],
                edgecolor="red",
                linewidth=1,
                fill=False,
                zorder=10,
            )
            ax.add_patch(tmp_patch)
        return None

    # Fig.1 Symmetric
    with context("default"):
        x, y, z = processing(symmetric_PSD / background_PSD)

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(16, 9), dpi=160, layout="constrained"
        )
        plt.rcParams.update({"font.size": 24})

        shading(ax, x, y, z)
        contour(ax, x, y, z)
        reference_line(ax)
        if kwargs["dispersion_line_order"] == "typical":
            dispersion_line(ax, linear_shallow_wave()[0])
        elif kwargs["dispersion_line_order"] == "atypical":
            dispersion_line(ax, linear_shallow_wave()[1])
        else:
            raise ValueError("Invalid order specified. Use 'typical' or 'atypical'.")
        padding(ax)

        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Zonal Wavenumbers (cycle per unit)", fontsize=24)
        ax.set_ylabel("Frequencies (cyles per day)", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_title(
            f"{kwargs["variable_name"]}, Symmetric PSD ratio",
            fontsize=28,
        )
        fig.savefig(
            os.path.join(
                ENVIRONMENT_PATH.ABSOLUTE_PATH_IMAGES,
                f"{kwargs["variable_name"]}_symmetric_WK99_diagram.png",
            )
        )
        plt.close(fig)

    # Fig.2 Antisymmetric
    with context("default"):
        x, y, z = processing(antisymmetric_PSD / background_PSD)

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(16, 9), dpi=160, layout="constrained"
        )
        plt.rcParams.update({"font.size": 24})

        shading(ax, x, y, z)
        contour(ax, x, y, z)
        reference_line(ax)
        if kwargs["dispersion_line_order"] == "typical":
            dispersion_line(ax, linear_shallow_wave()[1])
        elif kwargs["dispersion_line_order"] == "atypical":
            dispersion_line(ax, linear_shallow_wave()[0])
        else:
            raise ValueError("Invalid order specified. Use 'typical' or 'atypical'.")
        padding(ax)

        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Zonal Wavenumbers (cycle per unit)", fontsize=24)
        ax.set_ylabel("Frequencies (cyles per day)", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_title(
            f"{kwargs["variable_name"]}, Antisymmetric PSD ratio",
            fontsize=28,
        )
        fig.savefig(
            os.path.join(
                ENVIRONMENT_PATH.ABSOLUTE_PATH_IMAGES,
                f"{kwargs["variable_name"]}_antisymmetric_WK99_diagram.png",
            )
        )
        plt.close(fig)
    return None

