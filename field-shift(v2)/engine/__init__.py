"""Field Shift engine: data download, indicators, ML and rotation scoring.

Module map (read top to bottom for the order data flows through the system):
    config.py       settings and thresholds
    common.py       errors, HTTP retries, safe cache writes
    nasa_power.py   NASA POWER download + cleaning        -> daily weather
    modis.py        MODIS NDVI download + bare-field stats -> vegetation greenness
    crops.py        crop trait table (crops.csv)
    indicators.py   daily weather -> per-season indicators
    ml.py           soil-moisture random forest + what-if scenarios
    scoring.py      score every rotation
    explain.py      plain-language reasons, benefits and risks

Everything the app needs is re-exported here, so `import engine as E` then `E.season_table(...)` works.
"""
from .common import DataFetchError
from .config import (CACHE_DIR, DEFAULT_SCORING, SITES, SOIL_MULT, WATER_LIMIT, ScoringConfig)
from .crops import load_crops, validate_crops
from .explain import benefits, risks, why
from .indicators import dry_soil_threshold, longest_run, monthly_climate, season_table
from .ml import (FEATURE_LABELS, ModelReport, extrapolation_share, scenario_wetness, soil_features,
                 train_soil_model)
from .modis import BareFieldSummary, bare_field_summary, fetch_ndvi, fetch_ndvi_cached
from .nasa_power import clean_power, fetch_power, fetch_power_cached, parse_power_json
from .scoring import RotationScore, evaluate, evaluate_all, pick_strategies

__all__ = [
    "DataFetchError", "CACHE_DIR", "DEFAULT_SCORING", "SITES", "SOIL_MULT", "WATER_LIMIT", "ScoringConfig",
    "load_crops", "validate_crops", "benefits", "risks", "why",
    "dry_soil_threshold", "longest_run", "monthly_climate", "season_table",
    "FEATURE_LABELS", "ModelReport", "extrapolation_share", "scenario_wetness", "soil_features", "train_soil_model",
    "BareFieldSummary", "bare_field_summary", "fetch_ndvi", "fetch_ndvi_cached",
    "clean_power", "fetch_power", "fetch_power_cached", "parse_power_json",
    "RotationScore", "evaluate", "evaluate_all", "pick_strategies",
]
