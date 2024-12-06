import os
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from constants import MOVING_AVERAGE


def display_ZWS_evolution(
    packed_array: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    window_size: int = MOVING_AVERAGE.WINDOW_SIZE,
) -> matplotlib.figure.Figure:
    """
    Display the evolution of zonal wind speed (ZWS) and related metrics using multiple subplots.

    Parameters:
    ----------
    packed_array : tuple
        A tuple containing:
        - zws_raw : ndarray of raw zonal wind speed data.
        - zws_smoothed : ndarray of smoothed zonal wind speed data.
        - zws_smoothed_grad : ndarray of the gradient of the smoothed ZWS.
        - occurrence_smoothed : ndarray of smoothed low-pressure occurrence data.
    window_size : int, optional
        The window size used for smoothing, default is the window size defined in MOVING_AVERAGE.

    Returns:
    -------
    matplotlib.figure.Figure
        A figure object containing the plots.
    """
    zws_raw, zws_smoothed, zws_smoothed_grad, occurrence_smoothed = packed_array

    # Turn off interactive mode for plotting and set font size globally
    plt.ioff()
    plt.rcParams.update({"font.size": 14})

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), dpi=160, sharex=True)

    # Plot raw ZWS data (top-left)
    years = zws_raw.shape[0]
    for i in range(years):
        axes[0, 0].plot(zws_raw[i], color="gray", lw=0.6)
    axes[0, 0].plot(np.mean(zws_raw, axis=0), color="red")  # Mean plot in red
    axes[0, 0].plot([0, 365], [0, 0], "k--")  # Zero line for reference
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].set_title("ZWS (Raw)")

    # Plot smoothed ZWS data (top-right)
    for i in range(years):
        axes[0, 1].plot(zws_smoothed[i], color="gray", lw=0.6)
    axes[0, 1].plot(np.mean(zws_smoothed, axis=0), color="red")  # Mean plot in red
    axes[0, 1].plot([0, 365], [0, 0], "k--")  # Zero line for reference
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].set_title("ZWS (Smoothed)")

    # Plot smoothed ZWS gradient (bottom-left)
    for i in range(years):
        axes[1, 0].plot(zws_smoothed_grad[i], color="gray", lw=0.6)
    axes[1, 0].plot(
        np.mean(zws_smoothed_grad, axis=0), color="blue", lw=1.2
    )  # Mean plot in blue
    axes[1, 0].plot([0, 365], [0, 0], "k--")  # Zero line for reference
    axes[1, 0].set_xlabel("Days since 01/01")
    axes[1, 0].set_ylabel("ZWS Gradient (m/s/day)")
    axes[1, 0].set_title("ZWS Gradient (Smoothed)")

    # Plot smoothed occurrence of low pressure (bottom-right)
    for i in range(years):
        axes[1, 1].plot(occurrence_smoothed[i], color="gray", lw=0.6)
    axes[1, 1].plot(
        np.mean(occurrence_smoothed, axis=0), color="red", lw=1.2
    )  # Mean plot in red
    axes[1, 1].set_xlabel("Days since 01/01")
    axes[1, 1].set_ylabel("Occurrence")
    axes[1, 1].set_title("Low Pressure Occurrence (Smoothed)")

    # Set the overall title of the figure
    fig.suptitle(f"Variable Charts\nRunning Window = {window_size} (days)")
    return fig


def display_streamfunction_composites_evolution(
    streamfunction_composite: tuple[np.ndarray, np.ndarray],
    grids: dict[str, np.ndarray],
    start: int = 1,
    step: int = 1,
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of streamfunction composites as an animation.

    Parameters:
    ----------
    streamfunction_composite : tuple of np.ndarray
        A tuple containing two arrays:
        - early_composite: the streamfunction data for early onset.
        - late_composite: the streamfunction data for late onset.
    grids : dict of np.ndarray
        A dictionary containing grid data with keys "lat" (latitude) and "plev" (pressure level).
    start : int, optional
        The starting index for the animation frames (default is 1).
    step : int, optional
        The step size for frames in the animation (default is 1).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        An animation object showing the evolution of streamfunction composites.
    """
    from datetime import datetime, timedelta

    refdate = datetime(1979, 1, 1)

    early_composite, late_composite = streamfunction_composite

    # Turn off interactive mode for plotting and set font size globally
    plt.rcParams.update({"font.size": 24})
    contour_levels = np.linspace(
        -2e10, 2e10, 11, endpoint=True
    )  # Define contour levels

    # Create a figure with two subplots (early and late onset)
    fig = plt.figure(figsize=(16, 9), dpi=160, layout="constrained")
    spec = fig.add_gridspec(1, 31)
    ax0 = fig.add_subplot(spec[:15])
    ax1 = fig.add_subplot(spec[15:30])
    ax2 = fig.add_subplot(spec[-1])
    axes = (ax0, ax1, ax2)

    def init_animation():
        """Initialize the animation with an empty list (no content initially)."""
        # Early Onset Composite Plot
        axes[0].contourf(
            grids["lat"],
            grids["plev"],
            early_composite[start],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            grids["lat"],
            grids["plev"],
            early_composite[start],
            levels=contour_levels,
            colors="k",
        )

        axes[0].set_xlabel("Latitude")
        axes[0].set_ylabel("Pressure (hPa)")
        axes[0].invert_yaxis()  # Invert the y-axis (pressure levels)
        axes[0].set_title("Early Onset")

        # Late Onset Composite Plot
        img = axes[1].contourf(
            grids["lat"],
            grids["plev"],
            late_composite[start],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            grids["lat"],
            grids["plev"],
            late_composite[start],
            levels=contour_levels,
            colors="k",
        )
        axes[1].set_xlabel("Latitude")
        axes[1].set_title("Late Onset")
        axes[1].invert_yaxis()  # Invert the y-axis (pressure levels)
        # Colorbar
        cbar = plt.colorbar(img, cax=axes[2])
        cbar.set_ticks(contour_levels)
        # Update the figure's overall title with the calendar day
        date = (refdate + timedelta(days=start)).strftime("%m/%d")
        fig.suptitle(f"Variable: $\phi$\nDate: {date}")
        return []

    def update_frame(days):
        """
        Update the animation frame by frame with early and late onset data.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        # Early Onset Composite Plot
        axes[0].cla()  # Clear the previous plot
        axes[0].contourf(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[0].set_title("Early Onset")
        axes[0].set_xlabel("Latitude")
        axes[0].set_xticks([5, 10, 15, 20])
        axes[0].set_ylabel("Pressure (hPa)")
        axes[0].set_yticks([1000, 925, 850, 700, 500, 250, 200, 100])
        axes[0].invert_yaxis()  # Invert the y-axis (pressure levels)
        # Late Onset Composite Plot
        axes[1].cla()  # Clear the previous plot
        img = axes[1].contourf(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[1].set_title("Late Onset")
        axes[1].set_xlabel("Latitude")
        axes[1].set_xticks([5, 10, 15, 20])
        axes[1].set_yticks([1000, 925, 850, 700, 500, 250, 200, 100])
        axes[1].invert_yaxis()  # Invert the y-axis (pressure levels)
        date = (refdate + timedelta(days=start + days)).strftime("%m/%d")
        fig.suptitle(f"Variable: $\Psi$, Date: {date}")
        return []

    # Create the animation object with the initialized and updated frames
    animation_obj = animation.FuncAnimation(
        fig,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_potential_temperature_composites_evolution(
    potential_temperature_composite: tuple[np.ndarray, np.ndarray],
    grids: dict[str, np.ndarray],
    start: int = 1,
    step: int = 1,
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of potential temperature composites as an animation.

    Parameters:
    ----------
    potential_temperature_composite : tuple of np.ndarray
        A tuple containing two arrays:
        - early_composite: the potential temperature data for early onset.
        - late_composite: the potential temperature data for late onset.
    grids : dict of np.ndarray
        A dictionary containing grid data with keys "lat" (latitude) and "plev" (pressure level).
    start : int, optional
        The starting index for the animation frames (default is 1).
    step : int, optional
        The step size for frames in the animation (default is 1).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        An animation object showing the evolution of potential temperature composites.
    """
    # Unpack the composites
    early_composite, late_composite = potential_temperature_composite

    # Turn off interactive mode and set global font size for plots
    plt.ioff()
    plt.rcParams.update({"font.size": 28})

    # Create a figure with two subplots (early and late onset)
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation with an empty list (no initial content)."""
        return []

    def update_frame(days):
        """
        Update the animation frame by frame with early and late onset data.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        # Define contour levels for potential temperature (theta)
        contour_levels = np.linspace(290, 350, 31)

        # Early Onset Composite Plot
        axes[0].cla()  # Clear previous content
        axes[0].invert_yaxis()  # Invert y-axis to have pressure levels top-down
        axes[0].set_title("Early Onset")
        axes[0].set_ylabel("Pressure (hPa)")
        axes[0].set_xlabel("Latitude")
        axes[0].contourf(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            colors="k",
        )

        # Late Onset Composite Plot
        axes[1].cla()  # Clear previous content
        axes[1].set_title("Late Onset")
        axes[1].set_xlabel("Latitude")
        axes[1].contourf(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            colors="k",
        )

        # Update the figure's overall title to reflect the current day
        figure.suptitle(f"Theta\nCalendar Date = {start + days}")
        plt.tight_layout()

        return []

    # Create the animation object with initialized frames
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_equivalent_potential_temperature_composites_evolution(
    equivalent_potential_temperature_composite: tuple[np.ndarray, np.ndarray],
    grids: dict[str, np.ndarray],
    start: int = 1,
    step: int = 1,
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of potential temperature composites as an animation.

    Parameters:
    ----------
    potential_temperature_composite : tuple of np.ndarray
        A tuple containing two arrays:
        - early_composite: the potential temperature data for early onset.
        - late_composite: the potential temperature data for late onset.
    grids : dict of np.ndarray
        A dictionary containing grid data with keys "lat" (latitude) and "plev" (pressure level).
    start : int, optional
        The starting index for the animation frames (default is 1).
    step : int, optional
        The step size for frames in the animation (default is 1).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        An animation object showing the evolution of potential temperature composites.
    """
    # Unpack the composites
    early_composite, late_composite = equivalent_potential_temperature_composite

    # Turn off interactive mode and set global font size for plots
    plt.ioff()
    plt.rcParams.update({"font.size": 28})

    # Create a figure with two subplots (early and late onset)
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation with an empty list (no initial content)."""
        return []

    def update_frame(days):
        """
        Update the animation frame by frame with early and late onset data.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        # Define contour levels for potential temperature (theta)
        contour_levels = np.linspace(300, 360, 31)

        # Early Onset Composite Plot
        axes[0].cla()  # Clear previous content
        axes[0].invert_yaxis()  # Invert y-axis to have pressure levels top-down
        axes[0].set_title("Early Onset")
        axes[0].set_ylabel("Pressure (hPa)")
        axes[0].set_xlabel("Latitude")
        axes[0].contourf(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            colors="k",
        )

        # Late Onset Composite Plot
        axes[1].cla()  # Clear previous content
        axes[1].set_title("Late Onset")
        axes[1].set_xlabel("Latitude")
        axes[1].contourf(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            colors="k",
        )

        # Update the figure's overall title to reflect the current day
        figure.suptitle(f"Theta\nCalendar Date = {start + days}")
        plt.tight_layout()

        return []

    # Create the animation object with initialized frames
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_streamfunction_and_potential_temperature_composite_evolution(
    streamfunction_early_composite: np.ndarray,
    streamfunction_grids: dict[str, np.ndarray],
    potential_temperature_early_composite: np.ndarray,
    potential_temperature_grids: dict[str, np.ndarray],
    start: int = 1,
    step: int = 1,
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of streamfunction and potential temperature composites
    as an animation.

    Parameters:
    ----------
    streamfunction_early_composite : np.ndarray
        The streamfunction data for early onset.
    streamfunction_grids : dict of np.ndarray
        Grid data for the streamfunction with keys "lat" (latitude) and "plev" (pressure levels).
    potential_temperature_early_composite : np.ndarray
        The potential temperature data for early onset.
    potential_temperature_grids : dict of np.ndarray
        Grid data for potential temperature with keys "lat" (latitude) and "plev" (pressure levels).
    start : int, optional
        The starting index for the animation frames (default is 1).
    step : int, optional
        The step size for frames in the animation (default is 1).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        An animation object showing the evolution of streamfunction and equivalent potential temperature.
    """
    # Turn off interactive mode for plotting and set font size
    plt.ioff()
    plt.rcParams.update({"font.size": 26})

    # Create a single subplot for the animation
    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation (no initial content)."""
        return []

    def update_frame(days):
        """
        Update the animation for each frame by plotting streamfunction and equivalent potential temperature.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        # Clear previous plot
        ax.cla()
        ax.invert_yaxis()  # Invert y-axis to have pressure levels top-down

        # Define contour levels for potential temperature (theta) and streamfunction
        theta_levels = np.linspace(300, 330, 16)
        stream_levels = np.linspace(-2e10, 2e10, 16)

        # Plot potential temperature as filled contours
        ax.contourf(
            potential_temperature_grids["lat"],
            potential_temperature_grids["plev"],
            potential_temperature_early_composite[start + days],
            levels=theta_levels,
            extend="both",
            cmap="RdBu_r",
        )

        # Overlay streamfunction contours
        ax.contour(
            streamfunction_grids["lat"],
            streamfunction_grids["plev"],
            streamfunction_early_composite[start + days],
            levels=stream_levels,
            colors="k",
        )

        # Set labels for axes and title
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Pressure (hPa)")
        figure.suptitle(
            f"Contour - Streamfunction\nShading - Potential Temperature\nCalendar Date = {start + days}"
        )
        plt.tight_layout()

        return []

    # Create the animation object
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_streamfunction_and_equivalent_potential_temperature_composite_evolution(
    streamfunction_early_composite: np.ndarray,
    streamfunction_grids: dict[str, np.ndarray],
    equivalent_potential_temperature_early_composite: np.ndarray,
    equivalent_potential_temperature_grids: dict[str, np.ndarray],
    start: int = 1,
    step: int = 1,
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of streamfunction and equivalent potential temperature composites
    as an animation.

    Parameters:
    ----------
    streamfunction_early_composite : np.ndarray
        The streamfunction data for early onset.
    streamfunction_grids : dict of np.ndarray
        Grid data for the streamfunction with keys "lat" (latitude) and "plev" (pressure levels).
    equivalent_potential_temperature_early_composite : np.ndarray
        The equivalent potential temperature data for early onset.
    equivalent_potential_temperature_grids : dict of np.ndarray
        Grid data for equivalent potential temperature with keys "lat" (latitude) and "plev" (pressure levels).
    start : int, optional
        The starting index for the animation frames (default is 1).
    step : int, optional
        The step size for frames in the animation (default is 1).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        An animation object showing the evolution of streamfunction and equivalent potential temperature.
    """
    # Turn off interactive mode for plotting and set font size
    plt.ioff()
    plt.rcParams.update({"font.size": 26})

    # Create a single subplot for the animation
    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation (no initial content)."""
        return []

    def update_frame(days):
        """
        Update the animation for each frame by plotting streamfunction and equivalent potential temperature.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        # Clear previous plot
        ax.cla()
        ax.invert_yaxis()  # Invert y-axis to have pressure levels top-down

        # Define contour levels for equivalent potential temperature (theta) and streamfunction
        theta_levels = np.linspace(320, 350, 16)
        stream_levels = np.linspace(-2e10, 2e10, 16)

        # Plot equivalent potential temperature as filled contours
        ax.contourf(
            equivalent_potential_temperature_grids["lat"],
            equivalent_potential_temperature_grids["plev"],
            equivalent_potential_temperature_early_composite[start + days],
            levels=theta_levels,
            extend="both",
            cmap="RdBu_r",
        )

        # Overlay streamfunction contours
        ax.contour(
            streamfunction_grids["lat"],
            streamfunction_grids["plev"],
            streamfunction_early_composite[start + days],
            levels=stream_levels,
            colors="k",
        )

        # Set labels for axes and title
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Pressure (hPa)")
        figure.suptitle(
            f"Contour - Streamfunction\nShading - Equivalent Potential Temperature\nCalendar Date = {start + days}"
        )
        plt.tight_layout()

        return []

    # Create the animation object
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_mse_flux_composites_evolution(
    mse_flux_composite: tuple[np.ndarray, np.ndarray],
    grids: dict[str, np.ndarray],
    start: int = 1,
    step: int = 1,
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of MSE flux composites as an animation.

    Parameters:
    ----------
    mse_flux_composite : tuple of np.ndarray
        A tuple containing two arrays:
        - early_composite: the MSE flux data for early onset.
        - late_composite: the MSE flux data for late onset.
    grids : dict of np.ndarray
        Grid data for MSE flux with keys "lat" (latitude) and "plev" (pressure levels).
    start : int, optional
        The starting index for the animation frames (default is 1).
    step : int, optional
        The step size for frames in the animation (default is 1).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        An animation object showing the evolution of MSE flux composites.
    """
    early_composite, late_composite = mse_flux_composite

    # Turn off interactive mode and set global font size for plots
    plt.ioff()
    plt.rcParams.update({"font.size": 28})

    # Create a figure with two subplots for early and late onset composites
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation (no initial content)."""
        return []

    def update_frame(days):
        """
        Update the animation frame by frame, plotting the MSE flux composites.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        # Define contour levels for MSE flux
        contour_levels = np.linspace(-1e4, 1e4, 16)

        # Plot Early Onset Composite (left subplot)
        axes[0].cla()  # Clear the previous content
        axes[0].invert_yaxis()  # Invert y-axis to have pressure levels top-down
        axes[0].set_title("Early Onset")
        axes[0].contourf(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[0].set_xlabel("Latitude")
        axes[0].set_ylabel("Pressure (hPa)")

        # Plot Late Onset Composite (right subplot)
        axes[1].cla()  # Clear the previous content
        axes[1].set_title("Late Onset")
        axes[1].contourf(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[1].set_xlabel("Latitude")

        # Update the figure's overall title with the current day
        figure.suptitle(f"MSE Flux\nCalendar Date = {start + days}")
        plt.tight_layout()

        return []

    # Create the animation object
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_mse_flux_divergence_composites_evolution(
    mse_flux_divergence_composite: tuple[np.ndarray, np.ndarray],
    grids: dict[str, np.ndarray],
    start: int = 1,
    step: int = 1,
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of MSE flux divergence composites as an animation.

    Parameters:
    ----------
    mse_flux_composite : tuple of np.ndarray
        A tuple containing two arrays:
        - early_composite: the MSE flux divergence data for early onset.
        - late_composite: the MSE flux divergence data for late onset.
    grids : dict of np.ndarray
        Grid data for MSE flux divergence with keys "lat" (latitude) and "plev" (pressure levels).
    start : int, optional
        The starting index for the animation frames (default is 1).
    step : int, optional
        The step size for frames in the animation (default is 1).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        An animation object showing the evolution of MSE flux composites.
    """
    early_composite, late_composite = mse_flux_divergence_composite

    # Turn off interactive mode and set global font size for plots
    plt.ioff()
    plt.rcParams.update({"font.size": 28})

    # Create a figure with two subplots for early and late onset composites
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation (no initial content)."""
        return []

    def update_frame(days):
        """
        Update the animation frame by frame, plotting the MSE flux composites.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        # Define contour levels for MSE flux
        contour_levels = np.linspace(-1, 1, 16)

        # Plot Early Onset Composite (left subplot)
        axes[0].cla()  # Clear the previous content
        axes[0].invert_yaxis()  # Invert y-axis to have pressure levels top-down
        axes[0].set_title("Early Onset")
        axes[0].contourf(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[0].contour(
            grids["lat"],
            grids["plev"],
            early_composite[start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[0].set_xlabel("Latitude")
        axes[0].set_ylabel("Pressure (hPa)")

        # Plot Late Onset Composite (right subplot)
        axes[1].cla()  # Clear the previous content
        axes[1].set_title("Late Onset")
        axes[1].contourf(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            extend="both",
            cmap="RdBu_r",
        )
        axes[1].contour(
            grids["lat"],
            grids["plev"],
            late_composite[start + days],
            levels=contour_levels,
            colors="k",
        )
        axes[1].set_xlabel("Latitude")

        # Update the figure's overall title with the current day
        figure.suptitle(f"MSE Flux Vertical Divergence\nCalendar Date = {start + days}")
        plt.tight_layout()

        return []

    # Create the animation object
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_streamfunction_and_mse_flux_composite_evolution(
    streamfunction_early_composite: np.ndarray,
    streamfunction_grids: dict[str, np.ndarray],
    mse_flux_early_composite: np.ndarray,
    mse_flux_grids: dict[str, np.ndarray],
    start: int = 1,
    step: int = 1,
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of streamfunction and MSE flux composites as an animation.

    Parameters:
    ----------
    streamfunction_early_composite : np.ndarray
        The streamfunction data for early onset.
    streamfunction_grids : dict of np.ndarray
        Grid data for streamfunction with keys "lat" (latitude) and "plev" (pressure levels).
    mse_flux_early_composite : np.ndarray
        The MSE flux data for early onset.
    mse_flux_grids : dict of np.ndarray
        Grid data for MSE flux with keys "lat" (latitude) and "plev" (pressure levels).
    start : int, optional
        The starting index for the animation frames (default is 1).
    step : int, optional
        The step size for frames in the animation (default is 1).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        An animation object showing the evolution of streamfunction and MSE flux composites.
    """
    from datetime import datetime, timedelta

    refdate = datetime(1979, 1, 1)

    # Turn off interactive mode for plotting and set font size
    plt.ioff()
    plt.rcParams.update({"font.size": 24})

    # Create a figure with a single axis for the animation
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(16, 9),
        dpi=160,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    # Define contour levels for MSE flux and streamfunction
    mse_levels = np.linspace(-2e4, 2e4, 11)
    stream_levels = np.linspace(-2e10, 2e10, 11)

    def init_animation():
        """Initialize the animation with no initial content."""
        ax.invert_yaxis()
        img = ax.contourf(
            mse_flux_grids["lat"],
            mse_flux_grids["plev"],
            mse_flux_early_composite[start],
            levels=mse_levels,
            extend="both",
            cmap="RdBu",
        )

        # Overlay streamfunction contours
        ax.contour(
            streamfunction_grids["lat"],
            streamfunction_grids["plev"],
            streamfunction_early_composite[start],
            levels=stream_levels,
            colors="k",
        )

        # Set labels for axes and title
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Pressure (hPa)")
        cbar = plt.colorbar(img)
        cbar.set_ticks(mse_levels)
        date = (refdate + timedelta(days=start)).strftime("%m/%d")
        figure.suptitle(r"Variable: $\omega*MSE$, " + f"Date: {date}")

        return []

    def update_frame(days):
        """
        Update the animation for each frame by plotting streamfunction and MSE flux.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        # Clear the axis to prepare for the new frame
        ax.cla()
        ax.invert_yaxis()  # Invert y-axis to display pressure levels top-down

        # Plot MSE flux as filled contours
        img = ax.contourf(
            mse_flux_grids["lat"],
            mse_flux_grids["plev"],
            mse_flux_early_composite[start + days],
            levels=mse_levels,
            extend="both",
            cmap="RdBu",
        )

        # Overlay streamfunction contours
        ax.contour(
            streamfunction_grids["lat"],
            streamfunction_grids["plev"],
            streamfunction_early_composite[start + days],
            levels=stream_levels,
            colors="k",
        )

        # Set labels for axes and title
        ax.set_xlabel("Latitude")
        ax.set_xticks([5, 10, 15, 20])
        ax.set_ylabel("Pressure (hPa)")
        ax.set_yticks([1000, 925, 850, 700, 500, 250, 200, 100])
        date = (refdate + timedelta(days=start + days)).strftime("%m/%d")
        figure.suptitle(r"Variable: $\omega*MSE$, " + f"Date: {date}")
        return []

    # Create the animation object
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=100,
        init_func=init_animation,
    )

    return animation_obj


def display_wavenumber_frequency_diagram(
    symmetric_PSD: np.ndarray,
    antisymmetric_PSD: np.ndarray,
    background_PSD: np.ndarray,
    dimensions: dict[str, np.ndarray],
    **settings,
) -> tuple[matplotlib.figure.Figure, matplotlib.artist.Artist.axes]:
    """
    Notes:
    -------
    - **Zonal Wavenumber (k)**: In the wave expression e^{i k x}, where x ∈ [0, 2πR], the zonal wavenumber k = N/R defines the spatial frequency of the wave in the zonal direction, specifying how many oscillations occur over the distance 2πR.
    - **Zonal Wavemode (N)**: In the expression e^{i N/R x}, where x ∈ [0, 2πR], N is a positive integer that quantifies the number of wave oscillations around the full zonal circumference 2πR. Larger N values correspond to more oscillations over the same distance.
    """

    # import
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    from matplotlib.style import context
    from constants import EARTH, ENVIRONMENT_PATH
    from Vallis_dispersion_relation import (
        dispersion_kelvin,
        dispersion_poincare,
        dispersion_mrg,
        dispersion_rossby,
    )
    from utils import (
        get_f_beta,
        get_gravity_wave_speed,
        rescale_to_days_and_ordinary_frequency,
    )

    # kwargs
    settings.setdefault("wk_filter", None)
    settings.setdefault("variable_name", "Undefined")
    settings.setdefault("zonal_wavemodes", np.linspace(-15, 15, 121, endpoint=True))
    settings.setdefault("reference_latitude", 0.0)  # degrees
    settings.setdefault("equivalent_depths", np.array([12.5, 25.0, 50.0]))  # m/s
    settings.setdefault("dispersion_order", None)
    settings.setdefault("cmap_type", "default")

    # range
    wavenumber_indices = slice(
        np.argmax(dimensions["zonal_wavenumber"] >= settings["zonal_wavemodes"][0]),
        np.argmax(dimensions["zonal_wavenumber"] >= settings["zonal_wavemodes"][-1])
        + 1,
    )
    frequency_indices = slice(
        np.argmax(dimensions["segmentation_frequency"] > 0),
        np.argmax(dimensions["segmentation_frequency"]) + 1,
    )

    # cmap
    Greys = cm.get_cmap("Greys", 150)
    StepGreys = Greys(np.linspace(0, 1, 150))
    StepGreys[: 15 * 4, :] = np.array([1, 1, 1, 1])
    StepGreys = ListedColormap(StepGreys)

    # dispersion relation
    def compute_frequency(
        dispersion_function,
        meridional_mode_number: int = None,
    ):
        """Compute the CPD (cycles per day) frequency using a given dispersion function."""
        zonal_wavenumbers = settings["zonal_wavemodes"] / EARTH.RADIUS
        f_coriolis, rossby_parameter = get_f_beta(settings["reference_latitude"])
        gravity_wave_speeds = get_gravity_wave_speed(settings["equivalent_depths"])
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
        CPD_frequency = rescale_to_days_and_ordinary_frequency(dimensional_frequency)
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
    frequency_rossby_m1 = compute_frequency(dispersion_rossby, meridional_mode_number=1)
    frequency_rossby_m2 = compute_frequency(dispersion_rossby, meridional_mode_number=2)

    # dispersion line order
    dispersion_order_list = {
        "original": {
            "symmetric": [
                frequency_kelvin,
                frequency_poincare_m1,
                frequency_rossby_m1,
            ],
            "antisymmetric": [
                frequency_mrg,
                frequency_poincare_m2,
                frequency_rossby_m2,
            ],
        },
        "reverse": {
            "symmetric": [
                frequency_mrg,
                frequency_poincare_m2,
                frequency_rossby_m2,
            ],
            "antisymmetric": [
                frequency_kelvin,
                frequency_poincare_m1,
                frequency_rossby_m1,
            ],
        },
    }
    if settings["dispersion_order"] not in dispersion_order_list:
        raise ValueError("Invalid dispersion_order. Choose 'original' or 'reverse'.")
    cmap_type_list = ["default", "tradition"]
    if settings["cmap_type"] not in cmap_type_list:
        raise ValueError("Invalid cmap_type. Choose 'default' or 'tradition'.")

    # Fig.1 Symmetric
    with context("default"):
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(16, 9), dpi=160, layout="constrained"
        )
        plt.rcParams.update({"font.size": 24})
        x = dimensions["zonal_wavenumber"][wavenumber_indices]
        y = dimensions["segmentation_frequency"][frequency_indices]
        z = (symmetric_PSD / background_PSD)[frequency_indices, wavenumber_indices]
        if settings["cmap_type"] == "default":
            img = ax.contourf(
                x,
                y,
                z,
                cmap="jet",
                levels=np.linspace(0.6, 2.0, 15, endpoint=True),
                extend="both",
                zorder=-10,
            )
        else:
            img = ax.contourf(
                x,
                y,
                z,
                cmap=StepGreys,
                levels=np.linspace(0.6, 2.0, 15, endpoint=True),
                extend="both",
                zorder=-10,
            )
        plt.colorbar(img, ax=ax)
        ax.contour(
            x,
            y,
            z,
            colors="black",
            levels=np.linspace(0.6, 2.0, 15, endpoint=True),
            linewidths=0.4,
            zorder=-9,
        )
        ax.plot([0, 0], [0, 0.5], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 30, 1 / 30], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 6, 1 / 6], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 3, 1 / 3], "k--", lw=1, zorder=-2)

        # Add dispersion curves
        for freq in dispersion_order_list[settings["dispersion_order"]]["symmetric"]:
            ax.plot(settings["zonal_wavemodes"], freq, "k-", zorder=-2)

        # Padding
        xy = (settings["zonal_wavemodes"][0], 0)
        width = settings["zonal_wavemodes"][-1] - settings["zonal_wavemodes"][0]
        height = (
            dimensions["segmentation_frequency"][1]
            - dimensions["segmentation_frequency"][0]
        )
        tmp_patch = Rectangle(
            xy, width, height, facecolor="white", edgecolor=(1, 1, 1, 0), zorder=-1
        )
        ax.add_patch(tmp_patch)
        xy = (settings["zonal_wavemodes"][0], dimensions["segmentation_frequency"][-1])
        width = settings["zonal_wavemodes"][-1] - settings["zonal_wavemodes"][0]
        height = (
            dimensions["segmentation_frequency"][1]
            - dimensions["segmentation_frequency"][0]
        )
        tmp_patch = Rectangle(
            xy, width, height, facecolor="white", edgecolor=(1, 1, 1, 0), zorder=-1
        )
        ax.add_patch(tmp_patch)
        #
        if settings["wk_filter"]:
            tmp_patch = Rectangle(
                xy=(settings["wk_filter"][0][0], settings["wk_filter"][0][1]),
                width=settings["wk_filter"][1],
                height=settings["wk_filter"][2],
                edgecolor="red",
                linewidth=1,
                fill=False,
                zorder=10,
            )
            ax.add_patch(tmp_patch)
        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Zonal Wavenumbers (cycle per unit)", fontsize=24)
        ax.set_ylabel("Frequencies (cyles per day)", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_title(
            f"{settings["variable_name"]}, Symmetric PSD ratio",
            fontsize=28,
        )
        fig.savefig(
            os.path.join(
                ENVIRONMENT_PATH.ABSOLUTE_PATH_IMAGES,
                f"{settings["variable_name"]}_symmetric_WK99_diagram.png",
            )
        )

    # Fig.2 Antisymmetric
    with context("default"):
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(16, 9), dpi=160, layout="constrained"
        )
        plt.rcParams.update({"font.size": 24})
        x = dimensions["zonal_wavenumber"][wavenumber_indices]
        y = dimensions["segmentation_frequency"][frequency_indices]
        z = (antisymmetric_PSD / background_PSD)[frequency_indices, wavenumber_indices]
        if settings["cmap_type"] == "default":
            img = ax.contourf(
                x,
                y,
                z,
                cmap="jet",
                levels=np.linspace(0.6, 2.0, 15, endpoint=True),
                extend="both",
                zorder=-10,
            )
        else:
            img = ax.contourf(
                x,
                y,
                z,
                cmap=StepGreys,
                levels=np.linspace(0.6, 2.0, 15, endpoint=True),
                extend="both",
                zorder=-10,
            )
        plt.colorbar(img, ax=ax)
        ax.contour(
            x,
            y,
            z,
            colors="black",
            levels=np.linspace(0.6, 2.0, 15, endpoint=True),
            linewidths=0.4,
            zorder=-9,
        )
        ax.plot([0, 0], [0, 0.5], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 30, 1 / 30], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 6, 1 / 6], "k--", lw=1, zorder=-2)
        ax.plot([-15, 15], [1 / 3, 1 / 3], "k--", lw=1, zorder=-2)

        # Add dispersion curves
        for freq in dispersion_order_list[settings["dispersion_order"]][
            "antisymmetric"
        ]:
            ax.plot(settings["zonal_wavemodes"], freq, "k-", zorder=-2)

        # Padding
        xy = (settings["zonal_wavemodes"][0], 0)
        width = settings["zonal_wavemodes"][-1] - settings["zonal_wavemodes"][0]
        height = (
            dimensions["segmentation_frequency"][1]
            - dimensions["segmentation_frequency"][0]
        )
        tmp_patch = Rectangle(
            xy, width, height, facecolor="white", edgecolor=(1, 1, 1, 0), zorder=-1
        )
        ax.add_patch(tmp_patch)
        xy = (settings["zonal_wavemodes"][0], dimensions["segmentation_frequency"][-1])
        width = settings["zonal_wavemodes"][-1] - settings["zonal_wavemodes"][0]
        height = (
            dimensions["segmentation_frequency"][1]
            - dimensions["segmentation_frequency"][0]
        )
        tmp_patch = Rectangle(
            xy, width, height, facecolor="white", edgecolor=(1, 1, 1, 0), zorder=-1
        )
        ax.add_patch(tmp_patch)

        #
        if settings["wk_filter"]:
            tmp_patch = Rectangle(
                xy=(settings["wk_filter"][0][0], settings["wk_filter"][0][1]),
                width=settings["wk_filter"][1],
                height=settings["wk_filter"][2],
                edgecolor="red",
                linewidth=1,
                fill=False,
                zorder=10,
            )
            ax.add_patch(tmp_patch)
        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("Zonal Wavenumbers (cycle per unit)", fontsize=24)
        ax.set_ylabel("Frequencies (cyles per day)", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.set_title(
            f"{settings["variable_name"]}, Antisymmetric PSD ratio",
            fontsize=28,
        )
        fig.savefig(
            os.path.join(
                ENVIRONMENT_PATH.ABSOLUTE_PATH_IMAGES,
                f"{settings["variable_name"]}_antisymmetric_WK99_diagram.png",
            )
        )
    return None
