"""MODIS NDVI client (MOD13Q1, 250 m, every 16 days) via the free ORNL DAAC web service, plus bare-field analysis.
NDVI is a satellite 'greenness' score: near 0 = bare soil, ~0.8+ = dense crop."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from .common import DataFetchError, DictAccess, atomic_to_csv, make_session
from .config import CACHE_DIR, NDVI_SPARSE_THRESHOLD
from .nasa_power import check_coordinates

MODIS_URL = "https://modis.ornl.gov/rst/api/v1/MOD13Q1"
BAND = "250m_16_days_NDVI"
SCALE = 0.0001                  # raw values are integers; multiply to get real NDVI
FILL_BELOW = -2000              # -3000 marks "no data"
STEP_DAYS = 16
MAX_DATES_PER_REQUEST = 10      # the service limits how many dates one request may cover
_HEADERS = {"Accept": "application/json"}


@dataclass
class BareFieldSummary(DictAccess):
    sparse_days_per_year: float    # approx. days a year with NDVI below the threshold
    longest_sparse_days: int       # longest unbroken stretch below the threshold
    n_obs: int
    threshold: float


def _get_json(session: requests.Session, url: str, params: dict) -> dict:
    try:
        r = session.get(url, params=params, headers=_HEADERS, timeout=60)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        raise DataFetchError(f"MODIS NDVI service problem: {e}") from e


def fetch_ndvi(lat: float, lon: float, first_year: int = 2022, last_year: int = 2023,
               session: requests.Session | None = None) -> pd.Series:
    """NDVI time series for the 250 m pixel at (lat, lon). Cloud-contaminated/no-data dates are dropped."""
    check_coordinates(lat, lon)
    session = session or make_session()
    listing = _get_json(session, f"{MODIS_URL}/dates", dict(latitude=lat, longitude=lon))
    try:
        wanted = [d["modis_date"] for d in listing["dates"] if first_year <= int(d["calendar_date"][:4]) <= last_year]
    except (KeyError, TypeError, ValueError) as e:
        raise DataFetchError(f"Unexpected MODIS date listing: {e}") from e

    values: dict[pd.Timestamp, float] = {}
    for i in range(0, len(wanted), MAX_DATES_PER_REQUEST):
        chunk = wanted[i:i + MAX_DATES_PER_REQUEST]
        body = _get_json(session, f"{MODIS_URL}/subset", dict(
            latitude=lat, longitude=lon, band=BAND, startDate=chunk[0], endDate=chunk[-1],
            kmAboveBelow=0, kmLeftRight=0))
        for item in body.get("subset", []):
            data = item.get("data") or []
            if data and data[0] > FILL_BELOW:
                values[pd.to_datetime(item["calendar_date"])] = data[0] * SCALE
    return pd.Series(values, dtype=float).sort_index()


def fetch_ndvi_cached(lat: float, lon: float, first_year: int = 2022, last_year: int = 2023,
                      cache_dir: str = CACHE_DIR, session: requests.Session | None = None) -> tuple[pd.Series, str]:
    path = os.path.join(cache_dir, f"ndvi_{lat:.2f}_{lon:.2f}_{first_year}_{last_year}.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)["ndvi"], "saved copy"
        except Exception:
            pass
    series = fetch_ndvi(lat, lon, first_year, last_year, session)
    if not series.empty:
        atomic_to_csv(series.rename("ndvi"), path, header=True)
    return series, "live MODIS download"


def bare_field_summary(ndvi: pd.Series | None, threshold: float = NDVI_SPARSE_THRESHOLD,
                       step_days: int = STEP_DAYS) -> BareFieldSummary:
    """How much of the year the ground looks bare or fallow. Low NDVI also happens right after harvest,
    so read this as 'exposed soil', not as a fault."""
    if ndvi is None or len(ndvi) == 0:
        return BareFieldSummary(0.0, 0, 0, threshold)
    sparse = (np.asarray(ndvi, dtype=float) < threshold)
    longest = run = 0
    for flag in sparse:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return BareFieldSummary(float(sparse.mean() * 365), int(longest * step_days), int(len(ndvi)), threshold)
