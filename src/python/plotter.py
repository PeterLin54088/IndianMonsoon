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
    early_composite, late_composite = streamfunction_composite

    # Turn off interactive mode for plotting and set font size globally
    plt.ioff()
    plt.rcParams.update({"font.size": 28})

    # Create a figure with two subplots (early and late onset)
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation with an empty list (no content initially)."""
        return []

    def update_frame(days):
        """
        Update the animation frame by frame with early and late onset data.

        Parameters:
        ----------
        days : int
            The current day index for updating the frame.
        """
        contour_levels = np.linspace(-2e10, 2e10, 16)  # Define contour levels

        # Early Onset Composite Plot
        axes[0].cla()  # Clear the previous plot
        axes[0].invert_yaxis()  # Invert the y-axis (pressure levels)
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
        axes[1].cla()  # Clear the previous plot
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

        # Update the figure's overall title with the calendar day
        figure.suptitle(f"Streamfunction\nCalendar Date = {start + days}")
        plt.tight_layout()

        return []

    # Create the animation object with the initialized and updated frames
    animation_obj = animation.FuncAnimation(
        figure,
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
    # Turn off interactive mode for plotting and set font size
    plt.ioff()
    plt.rcParams.update({"font.size": 26})

    # Create a figure with a single axis for the animation
    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def init_animation():
        """Initialize the animation with no initial content."""
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

        # Define contour levels for MSE flux and streamfunction
        mse_levels = np.linspace(-2.5e4, 2.5e4, 16)
        stream_levels = np.linspace(-2e10, 2e10, 16)

        # Plot MSE flux as filled contours
        ax.contourf(
            mse_flux_grids["lat"],
            mse_flux_grids["plev"],
            mse_flux_early_composite[start + days],
            levels=mse_levels,
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
            f"Contour - Streamfunction\nShading - MSE flux\nCalendar Date = {start + days}"
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


def display_wavenumber_frequency_diagram(
    symmetric_PSD: np.ndarray,
    antisymmetric_PSD: np.ndarray,
    background_PSD: np.ndarray,
    grids: dict[str, np.ndarray],
    box: list[tuple[float, float], float, float] = None,
    variable_name: str = "Undefined",
    zonal_wavemodes: np.ndarray = np.linspace(-15, 15, 121, endpoint=True),
    reference_latitude: float = 0.0,
    equivalent_depths: np.ndarray = np.array([12.5, 25.0, 50.0]),
) -> tuple[matplotlib.figure.Figure, matplotlib.artist.Artist.axes]:
    """
    Display a wavenumber-frequency diagram for symmetric and antisymmetric Power Spectral Density (PSD),
    and plot theoretical dispersion relations (Kelvin, Rossby, Poincare, MRG).

    Parameters:
    ----------
    symmetric_PSD : np.ndarray
        The symmetric power spectral density data.
    antisymmetric_PSD : np.ndarray
        The antisymmetric power spectral density data.
    background_PSD : np.ndarray
        The background power spectral density for normalization.
    grids : dict of np.ndarray
        A dictionary containing the grid data for "zonal_wavenumber", "segmentation_frequency", and "lat".
    box : list, optional
        A list defining a bounding box on the plot, specified as [(x_start, y_start), width, height].
    variable_name : str, optional
        The name of the variable being plotted (default is "Undefined").
    zonal_wavemodes : np.ndarray, optional
        Array of zonal wavemodes to plot (default is np.linspace(-15, 15, 121)).
    reference_latitude : float, optional
        Latitude used for Coriolis parameter calculations (default is 0.0).
    equivalent_depths : np.ndarray, optional
        Array of equivalent depths for calculating gravity wave speeds (default is [12.5, 25.0, 50.0]).

    Returns:
    -------
    tuple
        A tuple containing the figure and axes used for plotting.

    Notes:
    -------
    - **Zonal Wavenumber (k)**: In the wave expression e^{i k x}, where x ∈ [0, 2πR], the zonal wavenumber k = N/R defines the spatial frequency of the wave in the zonal direction, specifying how many oscillations occur over the distance 2πR.
    - **Zonal Wavemode (N)**: In the expression e^{i N/R x}, where x ∈ [0, 2πR], N is a positive integer that quantifies the number of wave oscillations around the full zonal circumference 2πR. Larger N values correspond to more oscillations over the same distance.
    """
    from constants import EARTH
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

    def compute_frequency(
        dispersion_function,
        meridional_mode_number=None,
    ):
        """Compute the CPD (cycles per day) frequency using a given dispersion function."""
        zonal_wavenumbers = zonal_wavemodes / EARTH.RADIUS
        f_coriolis, rossby_parameter = get_f_beta(reference_latitude)
        gravity_wave_speeds = get_gravity_wave_speed(equivalent_depths)
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
    frequency_rossby = compute_frequency(dispersion_rossby)

    # Slicing indices for wavenumbers and frequencies
    wavenum_indices = slice(
        np.argmax(grids["zonal_wavenumber"] >= zonal_wavemodes[0]),
        np.argmax(grids["zonal_wavenumber"] >= zonal_wavemodes[-1]) + 1,
    )
    freq_indices = slice(
        np.argmax(grids["segmentation_frequency"] >= 0) + 1,
        np.argmax(grids["segmentation_frequency"]),
    )

    # Plotting setup
    plt.ioff()
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )
    fig.suptitle(f"{variable_name}", fontsize=24)

    # Extract grid dimensions for plotting
    _x = grids["zonal_wavenumber"][wavenum_indices]
    _y = grids["segmentation_frequency"][freq_indices]
    lat_weighting = np.cos(np.deg2rad(grids["lat"]))  # Latitude weighting for averaging

    # Symmetric PSD [0, 0]
    _z = np.average(symmetric_PSD, axis=1, weights=lat_weighting)
    _z = np.log10(_z)[freq_indices, wavenum_indices]
    axes[0, 0].contourf(
        _x,
        _y,
        _z,
        cmap="Greys",
        levels=np.linspace(np.min(_z), np.max(_z), 8, endpoint=True),
    )
    axes[0, 0].contour(
        _x,
        _y,
        _z,
        colors="black",
        levels=np.linspace(np.min(_z), np.max(_z), 15, endpoint=True),
    )
    axes[0, 0].plot([0, 0], [0, 0.5], "k--", lw=1)
    axes[0, 0].set_ylabel("Frequencies (cyles per day)")
    axes[0, 0].set_title("Symmetric PSD")

    # Antisymmetric PSD [0, 1]
    _z = np.average(antisymmetric_PSD, axis=1, weights=lat_weighting)
    _z = np.log10(_z)[freq_indices, wavenum_indices]
    axes[0, 1].contourf(
        _x,
        _y,
        _z,
        cmap="Greys",
        levels=np.linspace(np.min(_z), np.max(_z), 8, endpoint=True),
    )
    axes[0, 1].contour(
        _x,
        _y,
        _z,
        colors="black",
        levels=np.linspace(np.min(_z), np.max(_z), 15, endpoint=True),
    )
    axes[0, 1].plot([0, 0], [0, 0.5], "k--", lw=1)
    axes[0, 1].set_title("Antisymmetric PSD")

    # Symmetric PSD Ratio [1, 0]
    tmp_1 = np.average(symmetric_PSD, axis=1, weights=lat_weighting)
    tmp_2 = np.average(background_PSD, axis=1, weights=lat_weighting)
    _z = (tmp_1 / tmp_2)[freq_indices, wavenum_indices]
    axes[1, 0].contourf(
        _x,
        _y,
        _z,
        cmap="Greys",
        levels=np.linspace(1.1, 2.9, 10, endpoint=True),
        extend="max",
    )
    axes[1, 0].contour(
        _x,
        _y,
        _z,
        colors="black",
        levels=np.linspace(0.7, 2.9, 23, endpoint=True),
        linewidths=0.5,
    )
    axes[1, 0].plot(zonal_wavemodes, frequency_kelvin[:, :], "k-")
    axes[1, 0].plot(zonal_wavemodes, frequency_poincare_m1[:, :], "k-")
    axes[1, 0].plot(zonal_wavemodes, frequency_rossby[:, :], "k-")
    axes[1, 0].plot([0, 0], [0, 0.5], "k--", lw=1)
    if box:
        tmp_patch = Rectangle(
            xy=(box[0][0], box[0][1]),
            width=box[1],
            height=box[2],
            edgecolor="red",
            linewidth=2,
            fill=False,
        )

        axes[1, 0].add_patch(tmp_patch)
    else:
        pass
    axes[1, 0].set_xlabel("Wavenumbers (cycle per unit)")
    axes[1, 0].set_ylabel("Frequencies (cyles per day)")
    axes[1, 0].set_title("Symmetric PSD ratio")

    # Antisymmetric PSD Ratio [1, 1]
    tmp_1 = np.average(antisymmetric_PSD, axis=1, weights=lat_weighting)
    tmp_2 = np.average(background_PSD, axis=1, weights=lat_weighting)
    _z = (tmp_1 / tmp_2)[freq_indices, wavenum_indices]
    axes[1, 1].contourf(
        _x,
        _y,
        _z,
        cmap="Greys",
        levels=np.linspace(1.0, 3, 11),
        extend="max",
    )
    axes[1, 1].contour(
        _x,
        _y,
        _z,
        colors="black",
        levels=np.linspace(1.0, 3, 11),
        linewidths=0.5,
    )
    axes[1, 1].plot(zonal_wavemodes, frequency_mrg[:, :], "k-")
    axes[1, 1].plot(zonal_wavemodes, frequency_poincare_m2[:, :], "k-")
    axes[1, 1].plot([0, 0], [0, 0.5], "k--", lw=1)
    if box:
        tmp_patch = Rectangle(
            xy=(box[0][0], box[0][1]),
            width=box[1],
            height=box[2],
            edgecolor="red",
            linewidth=2,
            fill=False,
        )

        axes[1, 1].add_patch(tmp_patch)
    else:
        pass
    axes[1, 1].set_xlabel("Wavenumbers (cycle per unit)")
    axes[1, 1].set_title("Antisymmetric PSD ratio")
    axes[1, 1].set_ylim(0, 0.5)

    plt.tight_layout()
    return fig, axes


def display_wavenumber_frequency_diagram_evolution(
    symmetric_PSD: np.ndarray,
    antisymmetric_PSD: np.ndarray,
    background_PSD: np.ndarray,
    grids: dict[str, np.ndarray],
    box: list[tuple[float, float], float, float] = None,
    variable_name: str = "Undefined",
    zonal_wavemodes: np.ndarray = np.linspace(-15, 15, 121, endpoint=True),
    reference_latitude: float = 0.0,
    equivalent_depths: np.ndarray = np.array([12.5, 25.0, 50.0]),
) -> matplotlib.animation.FuncAnimation:
    """
    Display the evolution of wavenumber-frequency diagrams as an animation for both
    symmetric and antisymmetric PSDs (Power Spectral Densities), over a range of latitudes.

    Parameters:
    ----------
    symmetric_PSD : np.ndarray
        The symmetric power spectral density data (3D: [frequencies, latitudes, wavenumbers]).
    antisymmetric_PSD : np.ndarray
        The antisymmetric power spectral density data (3D: [frequencies, latitudes, wavenumbers]).
    background_PSD : np.ndarray
        The background power spectral density data used for normalization.
    grids : dict of np.ndarray
        A dictionary containing grid data with keys:
        - "lat" (latitude),
        - "zonal_wavenumber" (zonal wavenumbers),
        - "segmentation_frequency" (frequencies).
    box : list, optional
        A list defining a bounding box on the plot, specified as [(x_start, y_start), width, height].
    variable_name : str, optional
        The name of the variable being plotted (default is "Undefined").
    zonal_wavemodes : np.ndarray, optional
        Array of zonal wavemodes to plot (default is np.linspace(-15, 15, 121)).
    reference_latitude : float, optional
        The latitude used for Coriolis parameter calculations (default is 0.0).
    equivalent_depths : np.ndarray, optional
        Array of equivalent depths for calculating gravity wave speeds (default is [12.5, 25.0, 50.0]).

    Returns:
    -------
    matplotlib.animation.FuncAnimation
        Animation object for the evolution of symmetric and antisymmetric PSD diagrams.
    """
    from constants import EARTH
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

    # Create figure and axes for the symmetric and antisymmetric plots
    plt.ioff()  # Turn off interactive plotting
    plt.rcParams.update({"font.size": 18})  # Set font size
    figure, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 9), dpi=160, sharex=True, sharey=True
    )

    def compute_frequency(
        dispersion_function,
        meridional_mode_number=None,
    ):
        """Compute the CPD (cycles per day) frequency using a given dispersion function."""
        zonal_wavenumbers = zonal_wavemodes / EARTH.RADIUS
        f_coriolis, rossby_parameter = get_f_beta(reference_latitude)
        gravity_wave_speeds = get_gravity_wave_speed(equivalent_depths)
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
    frequency_rossby = compute_frequency(dispersion_rossby)

    # Slicing indices for wavenumbers and frequencies
    wavenum_indices = slice(
        np.argmax(grids["zonal_wavenumber"] >= zonal_wavemodes[0]),
        np.argmax(grids["zonal_wavenumber"] >= zonal_wavemodes[-1]) + 1,
    )
    freq_indices = slice(
        np.argmax(grids["segmentation_frequency"] >= 0) + 1,
        np.argmax(grids["segmentation_frequency"]),
    )

    # Extract grid dimensions for plotting
    _x = grids["zonal_wavenumber"][wavenum_indices]
    _y = grids["segmentation_frequency"][freq_indices]
    step = symmetric_PSD.shape[1]

    def init_animation():
        """Initialize the animation."""
        return []

    def update_frame(ilat):
        """Update the animation frame by frame."""
        axes[0].cla()
        axes[1].cla()

        # Symmetric PSD ratio
        _z = (symmetric_PSD / background_PSD)[freq_indices, ilat, wavenum_indices]
        axes[0].contourf(
            _x,
            _y,
            _z,
            levels=np.linspace(1.1, 2.9, 10, endpoint=True),
            extend="max",
            cmap="Greys",
        )
        axes[0].contour(
            _x,
            _y,
            _z,
            colors="black",
            levels=np.linspace(0.7, 2.9, 23, endpoint=True),
            linewidths=0.5,
        )
        axes[0].plot(zonal_wavemodes, frequency_kelvin[:, :], "k-")
        axes[0].plot(zonal_wavemodes, frequency_poincare_m1[:, :], "k-")
        axes[0].plot(zonal_wavemodes, frequency_rossby[:, :], "k-")
        if box:
            tmp_patch = Rectangle(
                xy=(box[0][0], box[0][1]),
                width=box[1],
                height=box[2],
                edgecolor="red",
                linewidth=2,
                fill=False,
            )

            axes[0].add_patch(tmp_patch)
        else:
            pass

        # Antisymmetric PSD ratio
        _z = (antisymmetric_PSD / background_PSD)[freq_indices, ilat, wavenum_indices]
        axes[1].contourf(
            _x,
            _y,
            _z,
            levels=np.linspace(1.1, 2.9, 10, endpoint=True),
            extend="max",
            cmap="Greys",
        )
        axes[1].contour(
            _x,
            _y,
            _z,
            colors="black",
            levels=np.linspace(0.7, 2.9, 23, endpoint=True),
            linewidths=0.5,
        )
        axes[1].plot(zonal_wavemodes, frequency_mrg[:, :], "k-")
        axes[1].plot(zonal_wavemodes, frequency_poincare_m2[:, :], "k-")
        if box:
            tmp_patch = Rectangle(
                xy=(box[0][0], box[0][1]),
                width=box[1],
                height=box[2],
                edgecolor="red",
                linewidth=2,
                fill=False,
            )

            axes[1].add_patch(tmp_patch)
        else:
            pass

        # Set labels and titles
        axes[0].set_xlabel("Wavenumbers (cycle per unit)")
        axes[0].set_ylabel("Frequencies (cyles per day)")
        axes[0].set_title("Symmetric PSD ratio")
        axes[1].set_xlabel("Wavenumbers (cycle per unit)")
        axes[1].set_ylabel("Frequencies (cyles per day)")
        axes[1].set_title("Antisymmetric PSD ratio")
        axes[1].set_ylim(0, 0.5)

        # Set the figure's title for the current latitude
        clat = grids["lat"][ilat]
        figure.suptitle(f"{variable_name}, latitude : {clat}")
        plt.tight_layout()

        return []

    # Create the animation
    animation_obj = animation.FuncAnimation(
        figure,
        update_frame,
        frames=step,
        interval=200,
        init_func=init_animation,
    )

    return animation_obj
