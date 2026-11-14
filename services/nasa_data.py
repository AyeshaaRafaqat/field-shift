"""NASA POWER Daily API client.

This module only downloads and cleans data. All climate calculations live in
`analysis/climate_analysis.py`.

Data source
-----------
NASA POWER (Prediction Of Worldwide Energy Resources), Daily temporal API,
point endpoint, `AG` (agroclimatology) community. The values are gridded
reanalysis / satellite-derived estimates, NOT weather-station measurements,
and the grid is coarse (tens of kilometres). Daily values are in local solar
time (the API default), not UTC.

Columns in the returned DataFrame (index = `date`, one row per day)
------------------------------------------------------------------
precip_mm_day      PRECTOTCORR  Corrected precipitation            mm/day
tmax_c             T2M_MAX      Maximum temperature at 2 m         degrees C
tmin_c             T2M_MIN      Minimum temperature at 2 m         degrees C
soil_wetness_root  GWETROOT     Root-zone soil wetness (MERRA-2)   fraction, 0 (dry) to 1 (saturated)
                                (only present if the API returned usable values)

NASA POWER marks missing values with a fill value (-999). Those are converted
to NaN here, so "no data" is never mistaken for a real measurement.

Metadata is attached to `df.attrs` (units, request details, warnings,
whether soil wetness was available).
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from utils.helpers import DateLike, validate_coordinates, validate_date_range

POWER_DAILY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DEFAULT_COMMUNITY = "AG"
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_FILL_VALUE = -999.0

# (connect timeout, read timeout) in seconds.
REQUEST_TIMEOUT = (10, 90)
MAX_RETRIES = 2  # retries after the first attempt, for timeouts / connection errors / 429 / 5xx
RETRY_BACKOFF_SECONDS = 2
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# NASA parameter name -> column name used in our DataFrame.
CORE_PARAMETERS: Dict[str, str] = {
    "PRECTOTCORR": "precip_mm_day",
    "T2M_MAX": "tmax_c",
    "T2M_MIN": "tmin_c",
}
# Requested separately so that a problem with it cannot break the core data.
SOIL_MOISTURE_PARAMETER: Dict[str, str] = {"GWETROOT": "soil_wetness_root"}

# Documented units, used if the API response does not include its own.
PARAMETER_UNITS: Dict[str, str] = {
    "PRECTOTCORR": "mm/day",
    "T2M_MAX": "degrees C",
    "T2M_MIN": "degrees C",
    "GWETROOT": "fraction (0 = dry, 1 = saturated)",
}


class NasaPowerError(Exception):
    """Raised when NASA POWER cannot be reached or returns unusable data."""


def _error_message_from_response(response: requests.Response) -> str:
    """Extract a readable message from an error response."""
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:300] or "no details provided"

    parts = []
    if isinstance(body, dict):
        for key in ("messages", "message", "errors", "detail", "error"):
            value = body.get(key)
            if value:
                parts.append(str(value))
    return "; ".join(parts)[:500] or str(body)[:300]


def _request_power(
    latitude: float,
    longitude: float,
    start,
    end,
    parameters: List[str],
    community: str,
    timeout,
) -> dict:
    """Call the NASA POWER daily point endpoint and return the parsed JSON.

    Retries on timeouts, connection errors and HTTP 429/5xx. Other HTTP errors
    (for example 422 for an unsupported parameter) are not retried.
    """
    query = {
        "parameters": ",".join(parameters),
        "community": community,
        "longitude": longitude,
        "latitude": latitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }

    last_problem = "unknown error"
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(POWER_DAILY_POINT_URL, params=query, timeout=timeout)
        except requests.exceptions.Timeout:
            last_problem = "the request timed out"
        except requests.exceptions.RequestException as exc:
            last_problem = f"could not connect ({type(exc).__name__}: {exc})"
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise NasaPowerError("NASA POWER returned a response that is not valid JSON.") from None

            detail = _error_message_from_response(response)
            if response.status_code not in RETRYABLE_STATUS_CODES:
                raise NasaPowerError(
                    f"NASA POWER rejected the request (HTTP {response.status_code}): {detail}"
                )
            last_problem = f"HTTP {response.status_code}: {detail}"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise NasaPowerError(
        f"NASA POWER request failed after {MAX_RETRIES + 1} attempts ({last_problem})."
    )


def _parse_parameters(
    payload: dict, name_map: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Turn a POWER JSON payload into a DataFrame.

    Returns (dataframe, units_by_column, parameters_missing_from_response).
    Fill values become NaN.
    """
    try:
        by_parameter = payload["properties"]["parameter"]
    except (KeyError, TypeError):
        raise NasaPowerError(
            "Unexpected NASA POWER response structure ('properties.parameter' not found)."
        ) from None
    if not isinstance(by_parameter, dict):
        raise NasaPowerError("Unexpected NASA POWER response structure ('parameter' is not a mapping).")

    header = payload.get("header") if isinstance(payload.get("header"), dict) else {}
    try:
        fill_value = float(header.get("fill_value", DEFAULT_FILL_VALUE))
    except (TypeError, ValueError):
        fill_value = DEFAULT_FILL_VALUE

    api_units = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}

    columns = {}
    units: Dict[str, str] = {}
    absent: List[str] = []
    for nasa_name, column in name_map.items():
        daily = by_parameter.get(nasa_name)
        if daily is None:
            absent.append(nasa_name)
            continue
        columns[column] = pd.to_numeric(pd.Series(daily, dtype="object"), errors="coerce")
        reported = api_units.get(nasa_name, {}) if isinstance(api_units.get(nasa_name), dict) else {}
        units[column] = reported.get("units") or PARAMETER_UNITS[nasa_name]

    df = pd.DataFrame(columns)
    # Keys look like "20240131".
    df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
    df = df[df.index.notna()].sort_index()
    df.index.name = "date"
    df = df.replace([fill_value, DEFAULT_FILL_VALUE], np.nan)
    return df, units, absent


def fetch_nasa_power_data(
    latitude: float,
    longitude: float,
    start_date: DateLike = DEFAULT_START_DATE,
    end_date: DateLike = DEFAULT_END_DATE,
    include_soil_moisture: bool = True,
    community: str = DEFAULT_COMMUNITY,
    timeout=REQUEST_TIMEOUT,
) -> pd.DataFrame:
    """Download daily climate data for one point from NASA POWER.

    Parameters
    ----------
    latitude, longitude : decimal degrees (WGS84).
    start_date, end_date : 'YYYY-MM-DD' strings or date objects. Default is
        2016-01-01 to 2025-12-31.
    include_soil_moisture : also try to fetch root-zone soil wetness (GWETROOT).
        If that request fails or has no usable values, the core data is still
        returned and a warning is recorded in `df.attrs["warnings"]`.

    Returns
    -------
    DataFrame indexed by date (see module docstring for columns and units).
    `df.attrs` contains: units, source, latitude, longitude, start_date,
    end_date, soil_moisture_available, warnings.

    Raises
    ------
    ValueError       for invalid coordinates or dates (no request is made).
    NasaPowerError   if NASA POWER cannot be reached, rejects the request, or
                     returns no usable data.
    """
    lat, lon = validate_coordinates(latitude, longitude)
    start, end = validate_date_range(start_date, end_date)

    # 1) Core variables (required).
    payload = _request_power(lat, lon, start, end, list(CORE_PARAMETERS), community, timeout)
    df, units, absent = _parse_parameters(payload, CORE_PARAMETERS)
    if absent:
        raise NasaPowerError(f"NASA POWER response did not include: {', '.join(absent)}.")
    if df.empty or df.isna().all().all():
        raise NasaPowerError(
            "NASA POWER returned no valid values for this location and period "
            "(the point may be outside data coverage, or the dates are not yet available)."
        )

    warnings: List[str] = []
    for column in df.columns:
        missing_share = df[column].isna().mean()
        if missing_share > 0.05:
            warnings.append(f"{column}: {missing_share:.0%} of days are missing.")

    # 2) Soil wetness (optional, separate request).
    soil_available = False
    if include_soil_moisture:
        try:
            soil_payload = _request_power(
                lat, lon, start, end, list(SOIL_MOISTURE_PARAMETER), community, timeout
            )
            soil_df, soil_units, soil_absent = _parse_parameters(soil_payload, SOIL_MOISTURE_PARAMETER)
        except NasaPowerError as exc:
            warnings.append(f"Root-zone soil wetness (GWETROOT) could not be retrieved: {exc}")
        else:
            soil_column = SOIL_MOISTURE_PARAMETER["GWETROOT"]
            if soil_absent or soil_df.empty or soil_df[soil_column].notna().sum() == 0:
                warnings.append(
                    "Root-zone soil wetness (GWETROOT) was not available for this request; "
                    "soil moisture anomaly will not be calculated."
                )
            else:
                df = df.join(soil_df, how="left")
                units.update(soil_units)
                soil_available = True

    df.attrs.update(
        {
            "units": units,
            "source": f"NASA POWER Daily API (point, community={community})",
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "soil_moisture_available": soil_available,
            "warnings": warnings,
        }
    )
    return df
