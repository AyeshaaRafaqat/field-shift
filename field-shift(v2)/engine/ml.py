"""Machine learning: a random forest that learns how soil moisture responds to weather at one location,
used to estimate soil moisture under what-if climate scenarios."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .common import DictAccess

FEATURE_LABELS = {"rain_1d": "Rain today", "rain_7d": "Rain, last 7 days", "rain_30d": "Rain, last 30 days",
                  "rain_90d": "Rain, last 90 days", "tmax": "Max temp today", "tmax_30d": "Max temp, 30-day avg",
                  "tmin": "Min temp today", "doy_sin": "Time of year (sin)", "doy_cos": "Time of year (cos)"}
DEFAULT_SPLIT = "2022-05-01"    # train before this date, test after


@dataclass
class ModelReport(DictAccess):
    r2: float                    # accuracy on unseen years (1 = perfect)
    mae: float                   # average error in soil-wetness units
    baseline_r2: float           # accuracy of a plain 'average for that month' guess, for comparison
    test_obs: pd.Series
    test_pred: pd.Series
    importance: pd.Series        # which weather features matter most
    feature_min: pd.Series       # weather range seen in training (used to flag extrapolation)
    feature_max: pd.Series
    split: str


def soil_features(df: pd.DataFrame, rain_scale: float = 1.0, d_temp: float = 0.0) -> pd.DataFrame:
    """Weather-only inputs (no soil-wetness inputs), so the model must learn weather -> soil response."""
    rain = df["PRECTOTCORR"].fillna(0) * rain_scale
    tmax, tmin = df["T2M_MAX"] + d_temp, df["T2M_MIN"] + d_temp
    doy = df.index.dayofyear.to_numpy() / 365.25 * 2 * np.pi
    return pd.DataFrame({
        "rain_1d": rain, "rain_7d": rain.rolling(7, min_periods=1).sum(),
        "rain_30d": rain.rolling(30, min_periods=1).sum(), "rain_90d": rain.rolling(90, min_periods=1).sum(),
        "tmax": tmax, "tmax_30d": tmax.rolling(30, min_periods=1).mean(), "tmin": tmin,
        "doy_sin": np.sin(doy), "doy_cos": np.cos(doy)}, index=df.index)


def _forest() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.7, n_jobs=-1, random_state=0)


def train_soil_model(df: pd.DataFrame, split: str = DEFAULT_SPLIT) -> tuple[RandomForestRegressor, ModelReport]:
    """Validate on a time-based hold-out (never shuffle days: neighbours are near-copies and would leak),
    compare against a simple monthly-average baseline, then refit on all data for use."""
    X, y = soil_features(df), df["GWETROOT"]
    ok = y.notna()
    train = ok & (X.index < pd.Timestamp(split))
    test = ok & (X.index >= pd.Timestamp(split))
    if train.sum() < 365 or test.sum() < 90:
        raise ValueError("Not enough data on either side of the train/test split")

    pred = pd.Series(_forest().fit(X[train], y[train]).predict(X[test]), index=X.index[test])
    monthly_mean = y[train].groupby(y[train].index.month).mean()
    baseline = pd.Series(y[test].index.month.map(monthly_mean).to_numpy(dtype=float), index=y[test].index)

    final = _forest().fit(X[ok], y[ok])
    report = ModelReport(
        r2=float(r2_score(y[test], pred)), mae=float(mean_absolute_error(y[test], pred)),
        baseline_r2=float(r2_score(y[test], baseline)), test_obs=y[test], test_pred=pred,
        importance=pd.Series(final.feature_importances_, index=[FEATURE_LABELS[c] for c in X.columns]
                             ).sort_values(ascending=False),
        feature_min=X[ok].min(), feature_max=X[ok].max(), split=split)
    return final, report


def scenario_wetness(df: pd.DataFrame, model: RandomForestRegressor, rain_scale: float, d_temp: float) -> pd.Series:
    """Observed wetness plus the model-predicted CHANGE caused by the what-if weather (the 'delta method')."""
    delta = model.predict(soil_features(df, rain_scale, d_temp)) - model.predict(soil_features(df))
    return (df["GWETROOT"] + pd.Series(delta, index=df.index)).clip(0, 1)


def extrapolation_share(df: pd.DataFrame, report: ModelReport, rain_scale: float, d_temp: float) -> float:
    """Share of days whose scenario weather falls outside the range the model was trained on.
    Random forests cannot extrapolate, so a high share means the scenario result is unreliable."""
    X = soil_features(df, rain_scale, d_temp)
    outside = (X < report.feature_min) | (X > report.feature_max)
    return float(outside.any(axis=1).mean())
