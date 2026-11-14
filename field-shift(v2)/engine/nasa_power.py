"""NASA POWER client: daily rainfall, temperature, root-zone soil wetness (no account or API key needed).
Docs: https://power.larc.nasa.gov/docs/services/api/"""
from __future__ import annotations

import os
import logging

import numpy as np
import pandas as pd
import requests

from .common import DataFetchError, atomic_to_csv, make_session
from .config import CACHE_DIR, POWER_END, POWER_START

log = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
REQUIRED = ("PRECTOTCORR", "T2M_MAX", "T2M_MIN", "GWETROOT")   # rain, max temp, min temp, root-zone wetness
OPTIONAL = ("EVLAND",)                                           # land evaporation (nice to have)
_RAIN = "PRECTOTCORR"


def check_coordinates(lat: float, lon: float) -> None:
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Coordinates out of range: lat={lat}, lon={lon}")


def parse_power_json(payload: dict) -> pd.DataFrame:
    """Turn the API's JSON into a DataFrame indexed by date."""
    try:
        parameters = payload["properties"]["parameter"]
    except (KeyError, TypeError):
        raise DataFetchError(f"Unexpected NASA POWER response: {str(payload)[:200]}") from None
    df = pd.DataFrame(parameters)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.sort_index()


def clean_power(df: pd.DataFrame, max_missing: float = 0.2, max_gap_days: int = 3) -> pd.DataFrame:
    """Replace the -999 'no data' code, fill short gaps in smooth variables, and refuse unusable data.
    Rainfall is never interpolated (that would invent rain); gaps are left empty and treated as dry."""
    absent = [c for c in REQUIRED if c not in df.columns]
    if absent:
        raise DataFetchError(f"NASA POWER response is missing: {', '.join(absent)}")
    df = df.astype(float).replace(-999.0, np.nan)
    smooth = [c for c in df.columns if c != _RAIN]
    df[smooth] = df[smooth].interpolate(limit=max_gap_days, limit_area="inside")
    share = df[list(REQUIRED)].isna().mean()
    if (share > max_missing).any():
        worst = share.idxmax()
        raise DataFetchError(f"Too much missing data in {worst} ({share[worst]:.0%}). Try another location.")
    return df


def fetch_power(lat: float, lon: float, start: str = POWER_START, end: str = POWER_END,
                session: requests.Session | None = None) -> pd.DataFrame:
    """Download and clean daily data for one point."""
    check_coordinates(lat, lon)
    session = session or make_session()
    problem = "no response"
    for params in (REQUIRED + OPTIONAL, REQUIRED):          # second try drops the optional variable
        try:
            r = session.get(POWER_URL, timeout=(10, 120), params=dict(
                parameters=",".join(params), community="AG", latitude=lat, longitude=lon,
                start=start, end=end, format="JSON"))
        except requests.RequestException as e:
            raise DataFetchError(f"Could not reach NASA POWER: {e}") from e
        if r.ok:
            try:
                return clean_power(parse_power_json(r.json()))
            except ValueError as e:                           # body was not valid JSON
                raise DataFetchError(f"NASA POWER returned unreadable data: {e}") from e
        problem = f"HTTP {r.status_code}: {r.text[:200]}"
        log.warning("NASA POWER request failed (%s) for params %s", problem, params)
        if r.status_code not in (400, 422):                   # only a bad-parameter error justifies the retry
            break
    raise DataFetchError(f"NASA POWER request failed ({problem})")


def cache_path(lat: float, lon: float, start: str = POWER_START, end: str = POWER_END,
               cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"power_{lat:.2f}_{lon:.2f}_{start}_{end}.csv")


def fetch_power_cached(lat: float, lon: float, cache_dir: str = CACHE_DIR,
                       start: str = POWER_START, end: str = POWER_END,
                       session: requests.Session | None = None) -> tuple[pd.DataFrame, str]:
    """Use the saved copy if there is one (a fixed 2014-2024 record never changes), else download and save.
    Returns (data, 'saved copy' | 'live NASA POWER download')."""
    path = cache_path(lat, lon, start, end, cache_dir)
    if os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True), "saved copy"
        except Exception as e:                                # corrupt file: fall through and re-download
            log.warning("Ignoring unreadable cache file %s (%s)", path, e)
    df = fetch_power(lat, lon, start, end, session)
    atomic_to_csv(df, path)
    return df, "live NASA POWER download"
