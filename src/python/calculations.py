import numpy as np
from netCDF4 import Dataset
from pandas import read_csv
from constants import EARTH
from constants import REGIONAL_MASK_TUPLE
import utils as utils


def calculate_zonal_wind_shear(filepath):
    """
    Calculate the Zonal Wind Shear (ZWS) between two predefined atmospheric regions from a NetCDF dataset.

    The function computes the ZWS at the 850 hPa level as the difference between the mean zonal wind
    of a southern region (5°N-15°N, 40°E-80°E) and a northern region (20°N-30°N, 70°E-90°E).

    Parameters:
    -----------
    filepath : str
        Path to the NetCDF file containing the zonal wind ('u') data, as well as 'lon', 'lat', and 'plev'.

    Returns:
    --------
    tuple of numpy.ndarray
        zws_raw : Raw Zonal Wind Shear values (south minus north region).
        zws_smoothed : Smoothed ZWS using a moving average.
        zws_smoothed_grad : Gradient of the smoothed ZWS.

    Notes:
    ------
    - Assumes 43 years of data with 365 days per year.
    - External utilities `moving_average` and `split_dimension` are used for smoothing and reshaping.
    """
    # Load dataset and extract relevant variables
    dataset = Dataset(filepath)
    longitudes = dataset["lon"][:]
    latitudes = dataset["lat"][:]
    pressure_levels = dataset["plev"][:] / 100  # Convert Pa to hPa

    # Find the index of the 850 hPa pressure level
    pressure_mask = (pressure_levels < 850 + 1) & (pressure_levels > 850 - 1)

    # Define southern and northern regions
    southern_lat_mask = (latitudes > 5) & (latitudes < 15)
    southern_lon_mask = (longitudes > 40) & (longitudes < 80)
    northern_lat_mask = (latitudes > 20) & (latitudes < 30)
    northern_lon_mask = (longitudes > 70) & (longitudes < 90)

    # Extract zonal wind data for each region
    zonal_wind_south = dataset["u"][
        :, pressure_mask, southern_lat_mask, southern_lon_mask
    ].squeeze()
    zonal_wind_north = dataset["u"][
        :, pressure_mask, northern_lat_mask, northern_lon_mask
    ].squeeze()

    # Calculate mean zonal wind for each region
    mean_wind_south = zonal_wind_south.mean(axis=(1, 2))
    mean_wind_north = zonal_wind_north.mean(axis=(1, 2))

    # Compute raw ZWS
    zws_raw = mean_wind_south - mean_wind_north

    # Apply moving average to smooth ZWS
    zws_smoothed = utils.moving_average(zws_raw, axis=0)

    # Compute gradient of the smoothed ZWS
    zws_smoothed_grad = np.gradient(zws_smoothed)

    # Reshape to (years, days) format
    factors = (43, 365)
    zws_raw = utils.split_dimension(zws_raw, axis=0, factors=factors)
    zws_smoothed = utils.split_dimension(zws_smoothed, axis=0, factors=factors)
    zws_smoothed_grad = utils.split_dimension(
        zws_smoothed_grad, axis=0, factors=factors
    )

    return zws_raw, zws_smoothed, zws_smoothed_grad


def calculate_occurrence(filepath):
    """
    Calculate raw and smoothed occurrences of low-pressure systems from a CSV file.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing low-pressure system occurrence data with columns:
        'Year', 'Month', 'Day', 'Occurrence', and 'Time'.

    Returns:
    --------
    tuple
        - df : pandas.DataFrame
            DataFrame with the raw data.
        - occurrence_raw : numpy.ndarray
            Raw occurrence counts for each timestep.
        - occurrence_smoothed : numpy.ndarray
            Smoothed occurrence counts using a moving average.

    Notes:
    ------
    - Assumes 43 years of data, each with 365 days.
    - External utilities `moving_average` and `split_dimension` are used for smoothing and reshaping.
    """
    # Load CSV data into DataFrame, skipping invalid lines
    df = read_csv(filepath, sep="\t", on_bad_lines="skip", header=None)

    # Assign column names and convert to integers
    df.columns = ["Year", "Month", "Day", "Occurrence", "Time"]
    df = df.astype(int)

    # Extract occurrence data and apply moving average
    occurrence_raw = df["Occurrence"].to_numpy()
    occurrence_smoothed = utils.moving_average(occurrence_raw, axis=0)

    # Reshape raw and smoothed data into (years, days) format
    factors = (43, 365)
    occurrence_raw = utils.split_dimension(occurrence_raw, axis=0, factors=factors)
    occurrence_smoothed = utils.split_dimension(
        occurrence_smoothed, axis=0, factors=factors
    )

    return df, occurrence_raw, occurrence_smoothed


def calculate_streamfunction(filepath):
    """
    Calculate the mass streamfunction on the Y-Z plane by integrating meridional divergent wind.

    Parameters:
    -----------
    filepath : str
        Path to the NetCDF file containing 'plev', 'lon', 'lat', and 'v' variables.

    Returns:
    --------
    tuple
        - streamfunction : numpy.ndarray
            Raw mass streamfunction, organized by year and day.
        - streamfunction_smoothed : numpy.ndarray
            Smoothed mass streamfunction using a moving average.
        - (pressure_levels, latitudes) : tuple of numpy.ndarray
            Pressure levels (hPa) and latitudes (degrees).

    Notes:
    ------
    - Assumes 43 years of data with 365 days per year.
    - Uses `moving_average` and `split_dimension` for smoothing and reshaping.
    - https://derekyuntao.github.io/jekyll-clean-dark/2021/02/mass-stream-func/
    """
    # Load dataset and extract variables
    data = Dataset(filepath)
    pressure_levels = data["plev"][::-1]  # Invert pressure levels
    longitudes = data["lon"][:]
    latitudes = data["lat"][:]

    # Define regional mask for Indian region
    latitude_mask = (latitudes > REGIONAL_MASK_TUPLE[1]) & (
        latitudes < REGIONAL_MASK_TUPLE[0]
    )
    longitude_mask = (longitudes > REGIONAL_MASK_TUPLE[3]) & (
        longitudes < REGIONAL_MASK_TUPLE[2]
    )

    # Apply mask and extract wind data
    latitudes = latitudes[latitude_mask]
    longitudes = longitudes[longitude_mask]
    divergent_meridional_wind = data["v"][:, ::-1, latitude_mask, longitude_mask]

    # Compute pressure thickness between layers
    pressure_thickness = np.diff(np.insert(pressure_levels, 0, 0))

    # Calculate mean meridional wind across longitudes
    mean_divergent_wind = np.mean(divergent_meridional_wind, axis=-1)
    padded_mean_divergent_wind = np.insert(mean_divergent_wind, 0, 0, axis=1)
    interpolated_mean_wind = (
        padded_mean_divergent_wind[:, :-1, :] + padded_mean_divergent_wind[:, 1:, :]
    ) / 2

    # Compute latitude-based weighting factor
    longitude_extent = np.deg2rad(longitudes[-1] - longitudes[0])
    cosine_weighting = np.cos(np.deg2rad(latitudes))
    weighting_factor = (
        longitude_extent * EARTH.RADIUS * cosine_weighting / EARTH.GRAVITY_ACCELERATION
    )

    # Calculate streamfunction by integrating wind and applying thickness
    streamfunction = np.swapaxes(interpolated_mean_wind, 1, -1) * pressure_thickness
    streamfunction = np.cumsum(streamfunction, axis=-1)
    streamfunction = np.swapaxes(streamfunction, -1, 1) * weighting_factor

    # Apply moving average for smoothing
    streamfunction_smoothed = utils.moving_average(streamfunction, axis=0)

    # Reshape for yearly organization (43 years, 365 days per year)
    factors = (43, 365)
    streamfunction = utils.split_dimension(streamfunction, axis=0, factors=factors)
    streamfunction_smoothed = utils.split_dimension(
        streamfunction_smoothed, axis=0, factors=factors
    )

    return streamfunction, streamfunction_smoothed, (pressure_levels / 100, latitudes)


def calculate_equiv_theta(filepath):
    """
    Calculate the (smoothed) equivalent potential temperature on the Y-Z Plane.

    This function computes the zonal mean (average over all longitudes) of the
    equivalent potential temperature from a NetCDF dataset. Additionally, it applies
    a moving average to smooth the zonal mean data.

    Parameters
    ----------
    filename : str
        The name of the NetCDF file containing the equivalent potential temperature data. The file is expected
        to reside in the directory specified by `ABSOLUTE_PATH_ERA5` and should include the following variables:
            - 'lon' : Longitudes (in degrees)
            - 'lat' : Latitudes (in degrees)
            - 'plev' : Pressure levels (in Pascals)
            - 'equiv_theta' : Equivalent potential temperature (in Kelvin)

    Returns
    -------
    tuple
        A tuple containing:
            - equivalent_potential_temperature : numpy.ndarray
                The raw zonal mean equivalent potential temperature organized by year and day. Shape:
                (years, days, pressure_levels, latitudes)
            - equivalent_potential_temperature_smoothed : numpy.ndarray
                The smoothed zonal mean equivalent potential temperature obtained by applying a moving average
                to the raw data. Shape:
                (years, days, pressure_levels, latitudes)
            - (pressure_levels, latitudes) : tuple of numpy.ndarray
                A tuple containing the pressure levels (in hectoPascals) and latitudes (in degrees) used in the calculation.

    Notes
    -----
    - **Assumptions**:
        - The dataset spans 43 years with 365 days each year.
        - The 'equiv_theta' variable is a 4D array with dimensions corresponding to time, pressure level, latitude, and longitude.
    - **Dependencies**:
        - External helper functions `moving_average` and `split_dimension` are used for smoothing and reshaping
          the data, respectively. Ensure these functions are defined and accessible in the scope.
    - **Data Processing**:
        - The zonal mean is calculated by averaging the equivalent potential temperature over the longitude dimension.
        - A moving average is applied along the time axis (axis=0) to smooth the zonal mean data.
    """
    data = Dataset(filepath)
    longitudes = data["lon"][:]
    latitudes = data["lat"][:]
    pressure_levels = data["plev"][:] / 100

    # Define indian region
    latitude_mask = (latitudes > REGIONAL_MASK_TUPLE[1]) & (
        latitudes < REGIONAL_MASK_TUPLE[0]
    )
    longitude_mask = (longitudes > REGIONAL_MASK_TUPLE[3]) & (
        longitudes < REGIONAL_MASK_TUPLE[2]
    )

    # Masked variable
    latitudes = latitudes[latitude_mask]
    longitudes = longitudes[longitude_mask]

    equivalent_potential_temperature = data["equiv_theta"][
        :, :, latitude_mask, longitude_mask
    ]

    # Compute the zonal mean of theta (average over longitude)
    equivalent_potential_temperature = np.mean(equivalent_potential_temperature, axis=3)

    # Apply moving average to smooth the zonal mean
    equivalent_potential_temperature_smoothed = utils.moving_average(
        equivalent_potential_temperature, axis=0
    )

    # Reshape the data to organize by year and day (assuming 365 days per year)
    factors = (43, 365)  # (years, days)
    equivalent_potential_temperature = utils.split_dimension(
        equivalent_potential_temperature, axis=0, factors=factors
    )
    equivalent_potential_temperature_smoothed = utils.split_dimension(
        equivalent_potential_temperature_smoothed, axis=0, factors=factors
    )

    return (
        equivalent_potential_temperature,
        equivalent_potential_temperature_smoothed,
        (pressure_levels, latitudes),
    )


def calculate_equiv_theta(filepath):
    """
    Calculate raw and smoothed equivalent potential temperature (θe) on the Y-Z plane.

    Parameters:
    -----------
    filepath : str
        Path to the NetCDF file containing 'lon', 'lat', 'plev', and 'equiv_theta'.

    Returns:
    --------
    tuple
        - equivalent_potential_temperature : numpy.ndarray
            Zonal mean of equivalent potential temperature, organized by year and day.
        - equivalent_potential_temperature_smoothed : numpy.ndarray
            Smoothed zonal mean equivalent potential temperature.
        - (pressure_levels, latitudes) : tuple of numpy.ndarray
            Pressure levels (hPa) and latitudes (degrees).

    Notes:
    ------
    - Assumes 43 years of data with 365 days per year.
    - Uses `moving_average` and `split_dimension` for smoothing and reshaping.
    """
    # Load dataset and extract dimensions
    data = Dataset(filepath)
    longitudes = data["lon"][:]
    latitudes = data["lat"][:]
    pressure_levels = data["plev"][:] / 100  # Convert to hPa

    # Apply regional mask if defined
    latitude_mask = (latitudes > REGIONAL_MASK_TUPLE[1]) & (
        latitudes < REGIONAL_MASK_TUPLE[0]
    )
    longitude_mask = (longitudes > REGIONAL_MASK_TUPLE[3]) & (
        longitudes < REGIONAL_MASK_TUPLE[2]
    )

    # Mask latitudes and longitudes
    latitudes = latitudes[latitude_mask]
    longitudes = longitudes[longitude_mask]

    # Extract and mask the equivalent potential temperature (θe)
    equivalent_potential_temperature = data["equiv_theta"][
        :, :, latitude_mask, longitude_mask
    ]

    # Calculate zonal mean by averaging over longitude
    equivalent_potential_temperature = np.mean(equivalent_potential_temperature, axis=3)

    # Apply moving average along the time axis
    equivalent_potential_temperature_smoothed = utils.moving_average(
        equivalent_potential_temperature, axis=0
    )

    # Reshape the data for yearly organization (43 years, 365 days)
    factors = (43, 365)
    equivalent_potential_temperature = utils.split_dimension(
        equivalent_potential_temperature, axis=0, factors=factors
    )
    equivalent_potential_temperature_smoothed = utils.split_dimension(
        equivalent_potential_temperature_smoothed, axis=0, factors=factors
    )

    return (
        equivalent_potential_temperature,
        equivalent_potential_temperature_smoothed,
        (pressure_levels, latitudes),
    )
