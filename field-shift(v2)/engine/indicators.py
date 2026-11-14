"""Turn daily weather into the per-season indicators the scoring uses."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (DRY_DAY_MM, DRY_SOIL_PERCENTILE, FROST_TEMP_C, HEAT_THRESHOLD, KHARIF_MONTHS)


def longest_run(mask) -> int:
    """Length of the longest unbroken run of True values."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    edges = np.diff(np.concatenate(([0], m.astype(int), [0])))
    return int((np.where(edges == -1)[0] - np.where(edges == 1)[0]).max())


def dry_soil_threshold(df: pd.DataFrame, percentile: float = DRY_SOIL_PERCENTILE) -> float:
    """Soil wetness below this value counts as a 'dry-soil day' (driest 20% of days on record)."""
    return float(df["GWETROOT"].quantile(percentile))


def season_table(df: pd.DataFrame, rain_scale: float = 1.0, d_temp: float = 0.0,
                 wet: pd.Series | None = None, wet_thr: float | None = None) -> pd.DataFrame:
    """One row per (year, season): rain, hot days, frost days, longest dry spell, mean soil wetness, dry-soil days.

    rain_scale and d_temp implement the what-if sliders. `wet` can be an ML-adjusted soil-wetness series;
    `wet_thr` is the wetness below which a day counts as dry soil.
    Kharif = May-Oct, Rabi = Nov-Apr (a Rabi season belongs to the year it started in)."""
    d = df.copy()
    d["rain"] = d["PRECTOTCORR"] * rain_scale
    d["tmax"] = d["T2M_MAX"] + d_temp
    d["tmin"] = d["T2M_MIN"] + d_temp
    d["wet"] = df["GWETROOT"] if wet is None else wet
    month = d.index.month
    d["season"] = np.where(np.isin(month, list(KHARIF_MONTHS)), "Kharif", "Rabi")
    d["year"] = np.where(month >= KHARIF_MONTHS.start, d.index.year, d.index.year - 1)

    rows = []
    for (year, season), g in d.groupby(["year", "season"]):
        rows.append(dict(
            year=int(year), season=season,
            rain=float(g["rain"].sum()),
            heat=int((g["tmax"] > HEAT_THRESHOLD[season]).sum()),
            frost=int((g["tmin"] <= FROST_TEMP_C).sum()),
            dry_spell=longest_run(g["rain"].fillna(0) < DRY_DAY_MM),
            wetness=float(g["wet"].mean()),
            dry_soil=int((g["wet"] < wet_thr).sum()) if wet_thr is not None else 0))
    return pd.DataFrame(rows).set_index(["year", "season"])


def monthly_climate(df: pd.DataFrame) -> pd.DataFrame:
    """Average year, month by month (index 1-12)."""
    m = pd.DataFrame({
        "Rain (mm)": df["PRECTOTCORR"].resample("MS").sum(),
        "Max temp (°C)": df["T2M_MAX"].resample("MS").mean(),
        "Min temp (°C)": df["T2M_MIN"].resample("MS").mean(),
        "Soil wetness (0-1)": df["GWETROOT"].resample("MS").mean()})
    return m.groupby(m.index.month).mean()
