import numpy as np


def dispersion_kelvin(zonal_wavenumbers: np.ndarray) -> np.ndarray:
    """
    Calculate the nondimensional dispersion relation for Kelvin waves.

    Parameters
    ----------
    zonal_wavenumbers : np.ndarray
        Nondimensional zonal wavenumber values.

    Returns
    -------
    np.ndarray
        Nondimensional positive frequency values. Negative frequency values are replaced with NaN.

    Notes
    -----
    - This function calculates the dispersion relation for Kelvin waves, where the nondimensional frequency
      is equal to the nondimensional zonal wavenumber.
    - Only positive frequency values are retained by convention, with negative values replaced by NaN.
    """

    # Define the dispersion relation for Kelvin waves: frequency = zonal_wavenumbers
    def dispersion_relation(zonal_wavenumbers: np.ndarray) -> np.ndarray:
        return zonal_wavenumbers

    # Calculate the frequencies using the dispersion relation
    frequencies = dispersion_relation(zonal_wavenumbers)

    # Retain only positive frequency values, replace negatives with NaN
    positive_frequencies = np.where(frequencies >= 0, frequencies, np.nan)

    return positive_frequencies


def dispersion_mrg(zonal_wavenumbers: np.ndarray) -> np.ndarray:
    """
    Calculate the nondimensional dispersion relation for Mixed Rossby-Gravity (MRG) waves.

    Parameters
    ----------
    zonal_wavenumbers : np.ndarray
        Nondimensional zonal wavenumber values.

    Returns
    -------
    np.ndarray
        Nondimensional positive frequency values. Negative frequency values are replaced with NaN.

    Notes
    -----
    - This function calculates the dispersion relation for MRG waves, where the nondimensional frequency
      is derived from the zonal wavenumbers.
    - Only positive frequency values are retained by convention, with negative values replaced by NaN.
    """

    # Define the dispersion relation for MRG waves
    def dispersion_relation(zonal_wavenumbers: np.ndarray) -> np.ndarray:
        return zonal_wavenumbers / 2 + np.sqrt(1 + (zonal_wavenumbers**2) / 4)

    # Calculate the frequencies using the dispersion relation
    frequencies = dispersion_relation(zonal_wavenumbers)

    # Retain only positive frequency values, replace negatives with NaN
    positive_frequencies = np.where(frequencies >= 0, frequencies, np.nan)

    return positive_frequencies


def dispersion_poincare(
    zonal_wavenumbers: np.ndarray, meridional_mode_number: int = 1
) -> np.ndarray:
    """
    Calculate the nondimensional dispersion relation for Poincare waves using a numerical iterative approach.

    Parameters
    ----------
    zonal_wavenumbers : np.ndarray
        Nondimensional zonal wavenumber values.
    meridional_mode_number : int, optional
        Meridional mode number (default is 1).

    Returns
    -------
    np.ndarray
        Nondimensional positive frequency values. Negative frequency values are replaced with NaN.

    Notes
    -----
    - This function uses an iterative approach to solve for the nondimensional frequency of Poincare waves
      since there is no analytical solution to the dispersion relation.
    - Only positive frequency values are retained by convention, with negative values replaced by NaN.
    """

    # Define the dispersion relation for Poincare waves
    def dispersion_relation(frequency: np.ndarray) -> np.ndarray:
        return np.sqrt(
            2 * meridional_mode_number
            + 1
            + zonal_wavenumbers**2
            + zonal_wavenumbers / frequency
        )

    # Initial guess for frequency and number of iterations
    initial_guess = np.inf
    num_iterations = 50

    # Start with the initial guess for frequency
    frequencies_approximation = dispersion_relation(initial_guess)

    # Perform iterative refinement of the frequency
    for _ in range(num_iterations):
        frequencies_approximation = dispersion_relation(frequencies_approximation)

    # Retain only positive frequency values, replace negatives with NaN
    positive_frequencies = np.where(
        frequencies_approximation >= 0, frequencies_approximation, np.nan
    )

    return positive_frequencies


def dispersion_rossby(
    zonal_wavenumbers: np.ndarray, meridional_mode_number: int = 1
) -> np.ndarray:
    """
    Calculate the nondimensional dispersion relation for Rossby waves using a numerical iterative approach.

    Parameters
    ----------
    zonal_wavenumbers : np.ndarray
        Nondimensional zonal wavenumber values.
    meridional_mode_number : int, optional
        Meridional mode number (default is 1).

    Returns
    -------
    np.ndarray
        Nondimensional positive frequency values. Negative frequency values are replaced with NaN.

    Notes
    -----
    - This function uses an iterative approach to solve for the nondimensional frequency of Rossby waves
      since there is no analytical solution to the dispersion relation.
    - Only positive frequency values are retained by convention, with negative values replaced by NaN.
    """

    # Define the dispersion relation for Rossby waves
    def dispersion_relation(frequency: np.ndarray) -> np.ndarray:
        return -(zonal_wavenumbers + frequency**3) / (
            2 * meridional_mode_number + 1 + zonal_wavenumbers**2
        )

    # Initial guess for frequency and number of iterations
    initial_guess = 0.0
    num_iterations = 50

    # Start with the initial guess for frequency
    frequencies_approximation = dispersion_relation(initial_guess)

    # Perform iterative refinement of the frequency
    for _ in range(num_iterations):
        frequencies_approximation = dispersion_relation(frequencies_approximation)

    # Retain only positive frequency values, replace negatives with NaN
    positive_frequencies = np.where(
        frequencies_approximation >= 0, frequencies_approximation, np.nan
    )

    return positive_frequencies
