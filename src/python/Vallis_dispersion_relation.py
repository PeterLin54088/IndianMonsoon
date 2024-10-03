import numpy as np


def dispersion_kelvin(k):
    """
    Dispersion relation for Kelvin waves.

    Parameters:
    k : array-like
        Nondimensional wavenumber values.

    Returns:
    omega_positive : array-like
        Nondimensional positive frequency (omega) values, NaN for negative values.

    Notes:
    The original function is real-valued and exhibits Hermitian symmetry in the wavenumber-frequency space.
    By convention, scientists retain only the positive frequencies, and this function follows that tradition.
    """
    dispersion_relation = lambda k: k
    omega = dispersion_relation(k)
    omega_positive = np.where(omega >= 0, omega, np.nan)
    return omega_positive


def dispersion_mrg(k):
    """
    Dispersion relation for Mixed Rossby-Gravity (MRG) waves.

    Parameters:
    k : array-like
        Nondimensional wavenumber values.

    Returns:
    omega_positive : array-like
        Nondimensional positive frequency (omega) values, NaN for negative values.

    Notes:
    The original function is real-valued and exhibits Hermitian symmetry in the wavenumber-frequency space.
    By convention, scientists retain only the positive frequencies, and this function follows that tradition.
    """
    dispersion_relation = lambda k: k / 2 + np.sqrt(1 + k**2 / 4)
    omega = dispersion_relation(k)
    omega_positive = np.where(omega >= 0, omega, np.nan)
    return omega_positive


def dispersion_poincare(k, m=1, first_guess=np.inf, niter=50):
    """
    Dispersion relation for Poincaré waves.

    Parameters:
    k : array-like
        Nondimensional wavenumber values.
    m : int, optional
        Meridional mode number, default is 1.
    first_guess : float, optional
        Initial guess for omega, default is infinity.
    niter : int, optional
        Number of iterations for refinement, default is 50.

    Returns:
    omega_positive : array-like
        Nondimensional positive frequency (omega) values, NaN for negative values.

    Notes:
    The dispersion relation for Poincaré waves is a cubic function, making the analytical solution complex.
    An iterative approach is used to approximate the solution.
    The original function is real-valued and exhibits Hermitian symmetry in the wavenumber-frequency space.
    By tradition, only the positive frequencies are retained.
    """
    dispersion_relation = lambda omega: np.sqrt(2 * m + 1 + k**2 + k / omega)

    omega_approx = dispersion_relation(first_guess)
    for _ in range(niter):
        omega_approx = dispersion_relation(omega_approx)

    omega_positive = np.where(omega_approx >= 0, omega_approx, np.nan)
    return omega_positive


def dispersion_rossby(k, m=1, first_guess=0.0, niter=50):
    """
    Dispersion relation for Rossby waves.

    Parameters:
    k : array-like
        Nondimensional wavenumber values.
    m : int, optional
        Meridional mode number, default is 1.
    first_guess : float, optional
        Initial guess for omega, default is 0.
    niter : int, optional
        Number of iterations for refinement, default is 50.

    Returns:
    omega_positive : array-like
        Nondimensional positive frequency (omega) values, NaN for negative values.

    Notes:
    The dispersion relation for Rossby waves is a cubic function, making the analytical solution complex.
    An iterative approach is used to approximate the solution.
    The original function is real-valued and exhibits Hermitian symmetry in the wavenumber-frequency space.
    By tradition, only the positive frequencies are retained.
    """
    dispersion_relation = lambda omega: -(k + omega**3) / (2 * m + 1 + k**2)

    omega_approx = dispersion_relation(first_guess)
    for _ in range(niter):
        omega_approx = dispersion_relation(omega_approx)

    omega_positive = np.where(omega_approx >= 0, omega_approx, np.nan)
    return omega_positive
