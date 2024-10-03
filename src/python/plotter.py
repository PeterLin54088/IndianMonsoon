import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from constants import WINDOW_SIZE


def display_ZWS_evolution(packed_array, WINDOW_SIZE=WINDOW_SIZE):
    """ """
    #
    zws_raw, zws_smoothed, zws_smoothed_grad, occurrence_smoothed = packed_array
    #
    plt.ioff()
    # Set figure parameters
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=160, sharex=True)

    # Upper Left: Raw ZWS
    for i in range(43):
        axes[0, 0].plot(zws_raw[i], color="gray", lw=0.6)
    axes[0, 0].plot(np.mean(zws_raw, axis=0), color="red")
    axes[0, 0].plot([0, 365], [0, 0], "k--")
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].set_title("ZWS (Raw)")

    # Upper Right: Smoothed ZWS
    for i in range(43):
        axes[0, 1].plot(zws_smoothed[i], color="gray", lw=0.6)
    axes[0, 1].plot(np.mean(zws_smoothed, axis=0), color="red")
    axes[0, 1].plot([0, 365], [0, 0], "k--")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].set_title("ZWS (Smoothed)")

    # Lower Left: Gradient of Smoothed ZWS
    for i in range(43):
        axes[1, 0].plot(zws_smoothed_grad[i], color="gray", lw=0.6)
    axes[1, 0].plot(np.mean(zws_smoothed_grad, axis=0), color="blue", lw=1.2)
    axes[1, 0].plot([0, 365], [0, 0], "k--")
    axes[1, 0].set_xlabel("Days since 01/01")
    axes[1, 0].set_ylabel("ZWS Gradient (m/s/day)")
    axes[1, 0].set_title("ZWS Gradient (Smoothed)")

    # Lower Right: Smoothed Occurrence of Low Pressure Systems
    for i in range(43):
        axes[1, 1].plot(occurrence_smoothed[i], color="gray", lw=0.6)
    axes[1, 1].plot(np.mean(occurrence_smoothed, axis=0), color="red", lw=1.2)
    axes[1, 1].set_xlabel("Days since 01/01")
    axes[1, 1].set_ylabel("Occurrence")
    axes[1, 1].set_title("Low Pressure Occurrence (Smoothed)")

    # Add a common title
    fig.suptitle(f"Variable Charts\nRunning Window = {WINDOW_SIZE} (days)")

    return fig


def display_streamfunction_composites_evolution(
    latitudes, pressure_levels, sf, start, step
):
    """
    Generate an animation of streamfunction composites for early and late monsoon onsets.

    This function creates an animation with two subplots:
    - Left: Composite of early monsoon onset years.
    - Right: Composite of late monsoon onset years.

    The composites are averaged over specified years and smoothed using a moving average.

    Parameters
    ----------
    streamfunction : np.ndarray
        Streamfunction data with dimensions (years, days, lat, pressure).
    latitudes : np.ndarray
        Array of latitude values.
    pressure_levels : np.ndarray
        Array of pressure levels in Pascals.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object for the streamfunction composites.
    """
    # Configure plot aesthetics
    plt.ioff()
    plt.rcParams.update({"font.size": 28})
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation."""
        return []

    def update_frame(days):
        """Update the animation frame by frame."""
        # Define contour levels
        contour_levels = np.linspace(-2e10, 2e10, 16)

        # Plot Early Onset Composite
        axes[0].cla()
        axes[0].invert_yaxis()
        axes[0].set_title("Early Onset")
        axes[0].set_ylabel("Pressure (hPa)")
        axes[0].set_xlabel("Latitude")
        cf1 = axes[0].contourf(
            latitudes,
            pressure_levels,
            sf[0][start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            latitudes,
            pressure_levels,
            sf[0][start + days],
            levels=contour_levels,
            colors="k",
        )

        # Plot Late Onset Composite
        axes[1].cla()
        axes[1].set_title("Late Onset")
        axes[1].set_xlabel("Latitude")
        cf2 = axes[1].contourf(
            latitudes,
            pressure_levels,
            sf[1][start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            latitudes,
            pressure_levels,
            sf[1][start + days],
            levels=contour_levels,
            colors="k",
        )

        # Update the figure's super title
        figure.suptitle(f"Streamfunction\nCalendar Date = {start+days}")
        plt.tight_layout()

        return []

    # Create the animation
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_equiv_theta_composites_evolution(
    latitudes, pressure_levels, equiv_theta, start, step
):
    """
    Generate an animation of equivalent potential temperature (equiv_theta) composites
    for early and late monsoon onsets.

    This function creates an animation with two subplots:
    - Left: Composite of early monsoon onset years.
    - Right: Composite of late monsoon onset years.

    The composites are averaged over specified years and visualized using contour plots.

    Parameters
    ----------
    equiv_theta : np.ndarray
        Equivalent potential temperature data with dimensions (years, days, lat, pressure).
    latitudes : np.ndarray
        Array of latitude values.
    pressure_levels : np.ndarray
        Array of pressure levels in Pascals.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object for the equiv_theta composites.
    """

    plt.ioff()
    # Configure plot aesthetics
    plt.rcParams.update({"font.size": 28})
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation."""
        return []

    def update_frame(days):
        """Update the animation frame by frame."""
        # Define contour levels
        contour_levels = np.linspace(320, 350, 16)

        # Plot Early Onset Composite
        axes[0].cla()
        axes[0].invert_yaxis()
        axes[0].set_title("Early Onset")
        axes[0].set_ylabel("Pressure (hPa)")
        axes[0].set_xlabel("Latitude")
        cf1 = axes[0].contourf(
            latitudes,
            pressure_levels,
            equiv_theta[0][start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            latitudes,
            pressure_levels,
            equiv_theta[0][start + days],
            levels=contour_levels,
            colors="k",
        )

        # Plot Late Onset Composite
        axes[1].cla()
        axes[1].set_title("Late Onset")
        axes[1].set_xlabel("Latitude")
        cf2 = axes[1].contourf(
            latitudes,
            pressure_levels,
            equiv_theta[1][start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            latitudes,
            pressure_levels,
            equiv_theta[1][start + days],
            levels=contour_levels,
            colors="k",
        )

        # Update the figure's super title
        figure.suptitle(f"Equiv_theta\nCalendar Date = {start + days}")
        plt.tight_layout()

        return []

    # Create the animation
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_sf_equiv_theta_composite_evolution(
    latitudes_sf,
    pressure_levels_sf,
    sf,
    latitudes_equiv_theta,
    pressure_levels_equiv_theta,
    equiv_theta,
    start,
    step,
):
    """
    Generate an animation of equivalent potential temperature (equiv_theta) and streamfunction composites
    for early monsoon onsets.

    The animation consists of contour plots where:
    - Shaded regions represent the equiv_theta composites.
    - Contour lines represent the streamfunction composites.

    Parameters
    ----------
    equiv_theta : np.ndarray
        Equivalent potential temperature data with dimensions (years, days, lat, pressure).
    streamfunction : np.ndarray
        Streamfunction data with dimensions (years, days, lat, pressure).
    theta_grids : tuple of np.ndarray
        Grids for equiv_theta, typically (pressure_levels, latitudes).
    stream_grids : tuple of np.ndarray
        Grids for streamfunction, typically (pressure_levels, latitudes).
    monsoon_onset_sorted : dict
        Dictionary mapping years to monsoon onset days, sorted by onset date.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object for the composite plots.
    """
    plt.ioff()

    plt.rcParams.update({"font.size": 28})
    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation."""
        return []

    def update_frame(days):
        """Update the animation for each frame."""
        ax.cla()
        ax.invert_yaxis()

        # Define contour levels
        theta_levels = np.linspace(320, 350, 16)
        stream_levels = np.linspace(-3e10, 3e10, 16)

        # Plot equiv_theta as filled contours
        ax.contourf(
            latitudes_equiv_theta,
            pressure_levels_equiv_theta,
            equiv_theta[start + days],
            levels=theta_levels,
            extend="both",
            cmap="RdBu_r",
        )

        # Overlay streamfunction contours
        ax.contour(
            latitudes_sf,
            pressure_levels_sf,
            sf[start + days],
            levels=stream_levels,
            colors="k",
        )

        # Set labels and title
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Pressure (hPa)")
        figure.suptitle(
            f"Contour - Streamfunction\nShading - Equiv_theta\nCalendar Date = {start+days}"
        )
        plt.tight_layout()

        return []

    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_mse_flux_composites_evolution(
    latitudes, pressure_levels, mse_flux, start, step
):
    """
    Generate an animation of Moist Static Energy (MSE) flux composites for early and late monsoon onsets.

    The animation consists of two subplots:
    - Left: Composite of early monsoon onset years.
    - Right: Composite of late monsoon onset years.

    Parameters
    ----------
    mse_flux : np.ndarray
        MSE flux data with dimensions (years, days, lat, pressure).
    latitudes : np.ndarray
        Array of latitude values.
    pressure_levels : np.ndarray
        Array of pressure levels in Pascals.
    monsoon_onset_sorted : dict
        Dictionary mapping years to monsoon onset days, sorted by onset date.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object for the MSE flux composites.
    """
    plt.ioff()
    # Configure plot aesthetics
    plt.rcParams.update({"font.size": 28})
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation."""
        return []

    def update_frame(days):
        """Update the animation frame by frame."""
        # Define contour levels
        contour_levels = np.linspace(-1e4, 1e4, 16)

        # Plot Early Onset Composite
        axes[0].cla()
        axes[0].invert_yaxis()
        axes[0].set_title("Early Onset")
        cf1 = axes[0].contourf(
            latitudes,
            pressure_levels,
            mse_flux[0][start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            latitudes,
            pressure_levels,
            mse_flux[0][start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[0].set_xlabel("Latitude")
        axes[0].set_ylabel("Pressure (hPa)")

        # Plot Late Onset Composite
        axes[1].cla()
        axes[1].set_title("Late Onset")
        cf2 = axes[1].contourf(
            latitudes,
            pressure_levels,
            mse_flux[1][start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            latitudes,
            pressure_levels,
            mse_flux[1][start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[1].set_xlabel("Latitude")

        # Update the figure's super title
        figure.suptitle(f"Calendar Date = {start + days}")
        plt.tight_layout()

        return []

    # Create the animation
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_mse_flux_divergence_composites_evolution(
    latitudes, pressure_levels, mse_flux_divergence, start, step
):
    """
    Generate an animation of Moist Static Energy (MSE) flux composites for early and late monsoon onsets.

    The animation consists of two subplots:
    - Left: Composite of early monsoon onset years.
    - Right: Composite of late monsoon onset years.

    Parameters
    ----------
    mse_flux : np.ndarray
        MSE flux data with dimensions (years, days, lat, pressure).
    latitudes : np.ndarray
        Array of latitude values.
    pressure_levels : np.ndarray
        Array of pressure levels in Pascals.
    monsoon_onset_sorted : dict
        Dictionary mapping years to monsoon onset days, sorted by onset date.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object for the MSE flux composites.
    """
    plt.ioff()
    # Configure plot aesthetics
    plt.rcParams.update({"font.size": 28})
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation."""
        return []

    def update_frame(days):
        """Update the animation frame by frame."""
        # Define contour levels
        contour_levels = np.linspace(-1, 1, 16)

        # Plot Early Onset Composite
        axes[0].cla()
        axes[0].invert_yaxis()
        axes[0].set_title("Early Onset")
        cf1 = axes[0].contourf(
            latitudes,
            pressure_levels,
            mse_flux_divergence[0][start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            latitudes,
            pressure_levels,
            mse_flux_divergence[0][start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[0].set_xlabel("Latitude")
        axes[0].set_ylabel("Pressure (hPa)")

        # Plot Late Onset Composite
        axes[1].cla()
        axes[1].set_title("Late Onset")
        cf2 = axes[1].contourf(
            latitudes,
            pressure_levels,
            mse_flux_divergence[1][start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            latitudes,
            pressure_levels,
            mse_flux_divergence[1][start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[1].set_xlabel("Latitude")

        # Update the figure's super title
        figure.suptitle(f"Calendar Date = {start + days}")
        plt.tight_layout()

        return []

    # Create the animation
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_mse_streamfunction_composite_evolution(
    latitudes_sf,
    pressure_levels_sf,
    sf,
    latitudes_mse_flux,
    pressure_levels_mse_flux,
    mse_flux,
    start,
    step,
):
    """
    Create an animation of MSE flux and streamfunction composites for early monsoon onsets.

    The animation displays:
    - Shaded regions representing the MSE flux composites.
    - Contour lines representing the streamfunction composites.

    Parameters
    ----------
    mse_flux : np.ndarray
        Moist Static Energy (MSE) flux data with dimensions (years, days, lat, pressure).
    streamfunction : np.ndarray
        Streamfunction data with dimensions (years, days, lat, pressure).
    mse_grids : tuple of np.ndarray
        Grids for MSE flux, typically (pressure_levels, latitudes).
    stream_grids : tuple of np.ndarray
        Grids for streamfunction, typically (pressure_levels, latitudes).
    monsoon_onset_sorted : dict
        Dictionary mapping years to monsoon onset days, sorted by onset date.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation object for the MSE flux and streamfunction composites.
    """
    plt.ioff()
    # Configure plot aesthetics
    plt.rcParams.update({"font.size": 28})
    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation."""
        return []

    def update_frame(days):
        """Update the animation frame by frame."""
        ax.cla()
        ax.invert_yaxis()

        # Define contour levels
        mse_levels = np.linspace(-25000, 25000, 16)
        stream_levels = np.linspace(-3e10, 3e10, 16)

        # Plot MSE_flux as filled contours
        ax.contourf(
            latitudes_mse_flux,  # Latitude
            pressure_levels_mse_flux,  # Pressure
            mse_flux[start + days],
            levels=mse_levels,
            extend="both",
            cmap="RdBu",
        )

        # Overlay streamfunction contours
        ax.contour(
            latitudes_sf,  # Latitude
            pressure_levels_sf,  # Pressure
            sf[start + days],
            levels=stream_levels,
            colors="k",
        )

        # Set labels and title
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Pressure (Pa)")
        figure.suptitle(
            f"Contour - Streamfunction\nShading - MSE Vertical Flux\nCalendar Date = {start + days}"
        )
        plt.tight_layout()

        return []

    # Create the animation
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj
