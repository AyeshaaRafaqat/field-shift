"""All tunable settings and constants in one place (no magic numbers elsewhere)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# ---- locations (name -> (latitude, longitude)) --------------------------------------------------
SITES: dict[str, tuple[float, float]] = {
    "Lahore": (31.52, 74.36), "Faisalabad": (31.42, 73.08), "Multan": (30.20, 71.47),
    "Bahawalpur": (29.40, 71.68), "Sargodha": (32.08, 72.67), "Rahim Yar Khan": (28.42, 70.30),
}

# ---- farmer inputs ------------------------------------------------------------------------------
SOIL_MULT = {"Sandy loam": 1.15, "Loam": 1.0, "Clay loam": 0.9}   # sandy soils drain fast -> more irrigation
WATER_LIMIT = {"Low": 500, "Medium": 900, "High": 1500}            # irrigation mm per year

# ---- data windows and folders -------------------------------------------------------------------
POWER_START = "20140501"        # 10 full agricultural years: May 2014 - April 2024
POWER_END = "20240430"
CACHE_DIR = "data_cache"

# ---- indicator definitions ----------------------------------------------------------------------
KHARIF_MONTHS = range(5, 11)    # May-Oct; every other month is Rabi (Nov-Apr)
HEAT_THRESHOLD = {"Kharif": 40.0, "Rabi": 35.0}   # daily max temp (deg C) counted as a "hot day"
FROST_TEMP_C = 2.0              # daily min temp at or below this counts as a frost day
DRY_DAY_MM = 1.0                # a day with less rain than this counts as dry
DRY_SOIL_PERCENTILE = 0.20      # "dry soil" = driest 20% of days on record
NDVI_SPARSE_THRESHOLD = 0.25    # NDVI below this suggests bare or fallow ground


@dataclass(frozen=True)
class ScoringConfig:
    """Thresholds and weights behind the 0-10 scores. These are reasoned defaults, not calibrated
    constants, so expect to tune them with local experts."""
    effective_rain: float = 0.8            # share of rainfall that counts as useful for crops
    irrigation_scale_mm: float = 1500.0    # irrigation need at which the water score reaches 0
    hot_days_scale: float = 30.0           # hot days at which heat stress saturates
    frost_days_scale: float = 15.0
    dry_spell_scale: float = 30.0          # longest dry spell (days) at which drought stress saturates
    dry_soil_scale: float = 60.0           # dry-soil days at which drought stress saturates
    w_heat: float = 0.4                    # how heat, frost and drought combine into one stress index
    w_frost: float = 0.3
    w_dry: float = 0.3
    dry_spell_weight: float = 0.5          # drought stress = blend of dry spells and dry-soil days
    dry_soil_weight: float = 0.5
    soil_weights: Mapping[str, float] = field(default_factory=lambda: dict(
        fixer=0.30, residue=0.20, root=0.15, diversity=0.10, pest=0.15, cover=0.10))
    confidence_high: float = 1.2           # score spread across years below this = High confidence
    confidence_medium: float = 2.4

    def __post_init__(self):
        for name, parts in {"stress weights": (self.w_heat, self.w_frost, self.w_dry),
                            "drought blend": (self.dry_spell_weight, self.dry_soil_weight),
                            "soil weights": tuple(self.soil_weights.values())}.items():
            if abs(sum(parts) - 1.0) > 1e-9:
                raise ValueError(f"ScoringConfig {name} must sum to 1, got {sum(parts):.3f}")


DEFAULT_SCORING = ScoringConfig()
