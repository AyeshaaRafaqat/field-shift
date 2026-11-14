"""Climate indicators calculated from daily NASA POWER data.

Everything here works on a pandas DataFrame with a daily DatetimeIndex and
these columns (see services/nasa_data.py for units):

    precip_mm_day, tmax_c, tmin_c, and optionally soil_wetness_root

No network access happens in this module.

Seasons
-------
A season is a (start "MM-DD", end "MM-DD") window. If the start is later in
the year than the end (e.g. 11-01 to 04-30) the season wraps over New Year and
is labelled with the year in which it STARTS (Jan-Apr 2021 belongs to the
season that started in 2020).

DEFAULT_SEASONS is only an illustrative split of the calendar year so the
code runs out of the box. It is an editable assumption, not a crop calendar.

Coverage
--------
Seasons at the edges of the downloaded period can be partial, and some days
can be missing. Every table therefore reports `days_with_data`,
`days_in_season` and `coverage` so that incomplete seasons are visible
instead of silently looking like low rainfall or few heat days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import parse_month_day

PRECIP_COL = "precip_mm_day"
TMAX_COL = "tmax_c"
TMIN_COL = "tmin_c"
SOIL_COL = "soil_wetness_root"

SeasonDefs = Dict[str, Tuple[str, str]]

# Illustrative, editable assumption (NOT a crop calendar).
DEFAULT_SEASONS: SeasonDefs = {
    "Rabi": ("11-01", "04-30"),
    "Kharif": ("05-01", "10-31"),
}

# Analytical assumptions (all configurable):
DEFAULT_DRY_DAY_THRESHOLD_MM = 1.0   # a day with rain below this counts as "dry"
DEFAULT_HEAT_THRESHOLD_C = 35.0      # illustrative only; NOT suitable for every crop
DEFAULT_FROST_THRESHOLD_C = 0.0      # frost day: minimum temperature below 0 C
DEFAULT_MIN_COVERAGE = 0.8           # seasons below this are not used for soil baselines/anomalies
DEFAULT_MIN_BASELINE_YEARS = 3       # need at least this many seasons to form a baseline


# --------------------------------------------------------------------------
# Preparation helpers
# --------------------------------------------------------------------------

def _prepare_daily(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    """Validate the input and return a copy with one row per calendar day.

    Missing calendar days are added as NaN rows, so a gap can never be
    mistaken for consecutive days (important for dry spells).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Climate data must be a non-empty DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Climate data must have a DatetimeIndex (one row per day).")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Climate data is missing required column(s): {', '.join(missing)}.")

    out = df.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out.asfreq("D")


def _season_bounds(seasons: SeasonDefs) -> Dict[str, Tuple[int, int]]:
    """Convert season definitions to (start, end) as month*100+day integers."""
    if not seasons:
        raise ValueError("At least one season must be defined.")
    bounds = {}
    for name, (start_text, end_text) in seasons.items():
        sm, sd = parse_month_day(start_text)
        em, ed = parse_month_day(end_text)
        bounds[name] = (sm * 100 + sd, em * 100 + ed)
    return bounds


def _label_index(index: pd.DatetimeIndex, seasons: SeasonDefs) -> pd.DataFrame:
    """Return a frame (same index) with `season` and `season_year` columns.

    Days outside every season get season = None. Overlapping seasons raise.
    """
    bounds = _season_bounds(seasons)
    md = (index.month * 100 + index.day).to_numpy()
    year = index.year.to_numpy()

    season_col = np.full(len(index), None, dtype=object)
    year_col = np.zeros(len(index), dtype=int)

    for name, (start, end) in bounds.items():
        if start <= end:
            mask = (md >= start) & (md <= end)
            label_year = year
        else:  # wraps over New Year
            mask = (md >= start) | (md <= end)
            label_year = np.where(md >= start, year, year - 1)

        if np.any(season_col[mask] != None):  # noqa: E711 - object array comparison
            raise ValueError(f"Season '{name}' overlaps another season.")
        season_col[mask] = name
        year_col[mask] = label_year[mask]

    return pd.DataFrame({"season": season_col, "season_year": year_col}, index=index)


def label_seasons(df: pd.DataFrame, seasons: SeasonDefs) -> pd.DataFrame:
    """Add `season` and `season_year` columns and drop days outside all seasons."""
    labels = _label_index(df.index, seasons)
    out = df.join(labels)
    return out[out["season"].notna()]


def season_label(season: str, season_year: int, seasons: SeasonDefs) -> str:
    """Human-readable label, e.g. 'Kharif 2021' or 'Rabi 2020-21'."""
    start, end = _season_bounds(seasons)[season]
    if start > end:
        return f"{season} {season_year}-{str(season_year + 1)[-2:]}"
    return f"{season} {season_year}"


def _expected_days(seasons: SeasonDefs, years: pd.Series) -> pd.Series:
    """Number of calendar days each (season, season_year) should contain."""
    first, last = int(years.min()) - 1, int(years.max()) + 1
    calendar = pd.date_range(f"{first}-01-01", f"{last + 1}-12-31", freq="D")
    labels = _label_index(calendar, seasons)
    labels = labels[labels["season"].notna()]
    return labels.groupby(["season", "season_year"]).size()


def _per_season(
    df: pd.DataFrame,
    seasons: SeasonDefs,
    column: str,
    reducer: Callable[[pd.Series], float],
    output_name: str,
) -> pd.DataFrame:
    """Apply `reducer` to `column` separately for every (season, season_year)."""
    labelled = label_seasons(df, seasons)
    if labelled.empty:
        return pd.DataFrame(
            columns=["season", "season_year", output_name, "days_with_data", "days_in_season", "coverage"]
        )
    expected = _expected_days(seasons, labelled["season_year"])

    rows = []
    for (season, year), group in labelled.groupby(["season", "season_year"], sort=True):
        values = group[column]
        days_in_season = int(expected[(season, year)])
        days_with_data = int(values.notna().sum())
        rows.append(
            {
                "season": season,
                "season_year": int(year),
                output_name: reducer(values),
                "days_with_data": days_with_data,
                "days_in_season": days_in_season,
                "coverage": days_with_data / days_in_season,
            }
        )
    return pd.DataFrame(rows)


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False])).astype(int)
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return int((ends - starts).max())


def _check_number(value, name: str, minimum: Optional[float] = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return number


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def seasonal_rainfall(df: pd.DataFrame, seasons: SeasonDefs = None) -> pd.DataFrame:
    """Total precipitation (mm) per season and year.

    Missing days are skipped in the sum (so a season with gaps is
    under-counted; check `coverage`). A season with no data at all gives NaN.
    """
    seasons = seasons or DEFAULT_SEASONS
    data = _prepare_daily(df, [PRECIP_COL])
    return _per_season(
        data, seasons, PRECIP_COL, lambda v: v.sum(min_count=1), "total_precip_mm"
    )


def max_dry_spell(
    df: pd.DataFrame,
    seasons: SeasonDefs = None,
    threshold_mm: float = DEFAULT_DRY_DAY_THRESHOLD_MM,
) -> pd.DataFrame:
    """Longest run of consecutive dry days (days) per season and year.

    A "dry day" is a day with precipitation strictly below `threshold_mm`
    (default 1 mm/day - an analytical assumption, not a universal agronomic
    definition). Days with missing data are NOT counted as dry: they end a
    spell, so the result is a lower bound when data are missing. Spells are
    measured inside a season and are cut at the season boundaries.
    """
    seasons = seasons or DEFAULT_SEASONS
    threshold = _check_number(threshold_mm, "threshold_mm", minimum=0.0)
    data = _prepare_daily(df, [PRECIP_COL])

    def reducer(values: pd.Series) -> float:
        if values.notna().sum() == 0:
            return np.nan
        dry = (values < threshold) & values.notna()
        return _longest_run(dry.to_numpy())

    return _per_season(data, seasons, PRECIP_COL, reducer, "max_dry_spell_days")


def heat_days(
    df: pd.DataFrame,
    seasons: SeasonDefs = None,
    threshold_c: float = DEFAULT_HEAT_THRESHOLD_C,
) -> pd.DataFrame:
    """Number of days per season and year with maximum temperature above `threshold_c`.

    The default (35 degrees C) is only an illustrative starting point. Heat
    tolerance differs between crops and growth stages, so the threshold
    should be chosen for the crop being assessed.
    """
    seasons = seasons or DEFAULT_SEASONS
    threshold = _check_number(threshold_c, "threshold_c")
    data = _prepare_daily(df, [TMAX_COL])

    def reducer(values: pd.Series) -> float:
        if values.notna().sum() == 0:
            return np.nan
        return int((values > threshold).sum())

    return _per_season(data, seasons, TMAX_COL, reducer, "heat_days")


def frost_days(
    df: pd.DataFrame,
    seasons: SeasonDefs = None,
    threshold_c: float = DEFAULT_FROST_THRESHOLD_C,
) -> pd.DataFrame:
    """Number of days per season and year with minimum temperature below 0 degrees C.

    Note: this is based on gridded 2 m air temperature, which can differ from
    frost at crop level.
    """
    seasons = seasons or DEFAULT_SEASONS
    threshold = _check_number(threshold_c, "threshold_c")
    data = _prepare_daily(df, [TMIN_COL])

    def reducer(values: pd.Series) -> float:
        if values.notna().sum() == 0:
            return np.nan
        return int((values < threshold).sum())

    return _per_season(data, seasons, TMIN_COL, reducer, "frost_days")


@dataclass
class SoilMoistureAnomalyResult:
    """Result of the soil moisture anomaly calculation.

    available : False if no suitable soil wetness data exist (nothing is calculated).
    reason    : explanation when `available` is False, or a note on limitations.
    table     : one row per season and year (empty if not available).
    """

    available: bool
    reason: Optional[str] = None
    table: pd.DataFrame = field(default_factory=pd.DataFrame)


def soil_moisture_anomaly(
    df: pd.DataFrame,
    seasons: SeasonDefs = None,
    column: str = SOIL_COL,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_baseline_years: int = DEFAULT_MIN_BASELINE_YEARS,
) -> SoilMoistureAnomalyResult:
    """Compare each season's mean root-zone soil wetness with the multi-year baseline.

    For each season (e.g. all Kharif seasons in the data):
      baseline_mean = mean of the seasonal means over the available years
      anomaly       = this year's seasonal mean - baseline_mean   (wetness fraction units)
      anomaly_zscore= anomaly / standard deviation of the seasonal means (if defined)

    Only seasons with `coverage >= min_coverage` take part in the baseline and
    receive an anomaly; partial seasons are shown but get NaN, because comparing
    a partial season with full ones would mix in the seasonal cycle.
    If the soil wetness column is missing or empty, nothing is calculated.
    """
    seasons = seasons or DEFAULT_SEASONS
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return SoilMoistureAnomalyResult(
            available=False,
            reason=f"Soil wetness column '{column}' is not in the data, so no anomaly was calculated.",
        )
    if df[column].notna().sum() == 0:
        return SoilMoistureAnomalyResult(
            available=False,
            reason="Soil wetness values are all missing, so no anomaly was calculated.",
        )

    data = _prepare_daily(df, [column])
    table = _per_season(data, seasons, column, lambda v: v.mean(), "mean_soil_wetness")

    table["used_for_baseline"] = table["coverage"] >= min_coverage
    usable = table[table["used_for_baseline"] & table["mean_soil_wetness"].notna()]
    stats = usable.groupby("season")["mean_soil_wetness"].agg(
        baseline_mean="mean", baseline_std=lambda s: s.std(ddof=1), baseline_years="count"
    )
    table = table.merge(stats, left_on="season", right_index=True, how="left")

    enough = table["baseline_years"].fillna(0) >= min_baseline_years
    can_compare = table["used_for_baseline"] & table["mean_soil_wetness"].notna() & enough
    table["anomaly"] = np.where(can_compare, table["mean_soil_wetness"] - table["baseline_mean"], np.nan)
    std_ok = can_compare & (table["baseline_std"] > 0)
    table["anomaly_zscore"] = np.where(std_ok, table["anomaly"] / table["baseline_std"], np.nan)
    table.loc[~enough, ["baseline_mean", "baseline_std"]] = np.nan

    table["season_label"] = [
        season_label(s, int(y), seasons) for s, y in zip(table["season"], table["season_year"])
    ]
    table = _sort_by_time(table, seasons)

    notes = []
    if not enough.any():
        notes.append(
            f"Fewer than {min_baseline_years} seasons with at least {min_coverage:.0%} coverage "
            "are available, so no baseline or anomaly could be formed."
        )
    return SoilMoistureAnomalyResult(
        available=True, reason=" ".join(notes) or None, table=table.reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Combined analysis
# --------------------------------------------------------------------------

@dataclass
class ClimateAnalysis:
    """All climate indicators, ready to display.

    seasonal_indicators : one row per season and year (rainfall, dry spell, heat days, frost days)
    soil_moisture       : SoilMoistureAnomalyResult (may be "not available")
    assumptions         : thresholds and season definitions that were used
    notes               : caveats to show next to the numbers
    """

    seasonal_indicators: pd.DataFrame
    soil_moisture: SoilMoistureAnomalyResult
    assumptions: Dict[str, object]
    notes: List[str]


def _sort_by_time(table: pd.DataFrame, seasons: SeasonDefs) -> pd.DataFrame:
    """Sort by year, then by the season's start date within the year."""
    bounds = _season_bounds(seasons)
    start_key = table["season"].map(lambda s: bounds[s][0])
    return table.assign(_start=start_key).sort_values(["season_year", "_start"]).drop(columns="_start")


def analyse_climate(
    df: pd.DataFrame,
    seasons: SeasonDefs = None,
    dry_day_threshold_mm: float = DEFAULT_DRY_DAY_THRESHOLD_MM,
    heat_threshold_c: float = DEFAULT_HEAT_THRESHOLD_C,
    frost_threshold_c: float = DEFAULT_FROST_THRESHOLD_C,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> ClimateAnalysis:
    """Calculate all indicators and return them in one labelled structure."""
    seasons = seasons or DEFAULT_SEASONS

    parts = [
        seasonal_rainfall(df, seasons),
        max_dry_spell(df, seasons, dry_day_threshold_mm),
        heat_days(df, seasons, heat_threshold_c),
        frost_days(df, seasons, frost_threshold_c),
    ]
    metric_names = ["total_precip_mm", "max_dry_spell_days", "heat_days", "frost_days"]

    trimmed = []
    for part, metric in zip(parts, metric_names):
        trimmed.append(part[["season", "season_year", metric, "coverage"]].rename(
            columns={"coverage": f"coverage_{metric}"}
        ))
    merged = reduce(lambda a, b: a.merge(b, on=["season", "season_year"], how="outer"), trimmed)

    coverage_cols = [f"coverage_{m}" for m in metric_names]
    merged["coverage"] = merged[coverage_cols].min(axis=1)
    merged = merged.drop(columns=coverage_cols)
    merged["season_label"] = [
        season_label(s, int(y), seasons) for s, y in zip(merged["season"], merged["season_year"])
    ]
    merged["complete"] = merged["coverage"] >= min_coverage
    merged = _sort_by_time(merged, seasons).reset_index(drop=True)
    merged = merged[["season_label", "season", "season_year", *metric_names, "coverage", "complete"]]

    soil = soil_moisture_anomaly(df, seasons, min_coverage=min_coverage)

    assumptions = {
        "seasons": dict(seasons),
        "dry_day_threshold_mm": dry_day_threshold_mm,
        "heat_threshold_c": heat_threshold_c,
        "frost_threshold_c": frost_threshold_c,
        "min_coverage": min_coverage,
    }
    notes = [
        f"A dry day is a day with less than {dry_day_threshold_mm:g} mm of rain (analytical assumption). "
        "Days with missing data end a dry spell instead of extending it.",
        f"Heat days are days with maximum temperature above {heat_threshold_c:g} degrees C. "
        "This threshold is an illustrative setting, not a limit that suits every crop or growth stage.",
        "Season dates are editable assumptions, not a crop calendar.",
        f"Seasons with less than {min_coverage:.0%} data coverage are marked incomplete "
        "(typically the first/last season of the downloaded period).",
        "Values come from gridded NASA POWER estimates, not from a weather station at the field.",
    ]
    return ClimateAnalysis(merged, soil, assumptions, notes)
