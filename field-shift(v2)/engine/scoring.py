"""Score every Kharif x Rabi crop pair on soil health, water fit, resilience and income, for each historical year."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import DictAccess
from .config import DEFAULT_SCORING, ScoringConfig

SEASONS = ("Kharif", "Rabi")


@dataclass(eq=False)
class RotationScore(DictAccess):
    """Result for one rotation. All scores are 0-10 (higher is better) except irr/bare/stress."""
    key: tuple[str, str]
    label: str
    K: pd.Series                 # Kharif crop traits
    R: pd.Series                 # Rabi crop traits
    soil: float
    water: float
    resil: float
    income: float
    total: float                 # weighted by the farmer's priorities, averaged over historical years
    lo: float                    # worst historical year
    hi: float                    # best historical year
    confidence: str              # High / Medium / Low, from the spread between lo and hi
    irr: float                   # estimated irrigation gap, mm per year
    fixer: float                 # 1 if either crop fixes nitrogen
    pest: float                  # 0, 0.5 or 1: how much the crop families differ from the current rotation
    cover: float                 # share of the year with a crop in the ground
    bare: float                  # estimated bare days per year
    stress: float                # heat + frost + drought stress index, 0-1 (lower is better)


def climate_by_season(tbl: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the season table into one year-indexed table per season."""
    out = {s: tbl.xs(s, level="season").sort_index() for s in SEASONS}
    if not out["Kharif"].index.equals(out["Rabi"].index):
        raise ValueError("Kharif and Rabi tables cover different years")
    return out


def _stress(crop: pd.Series, clim: pd.DataFrame, cfg: ScoringConfig) -> np.ndarray:
    """Stress index (0-1) per year for one crop in its season; weaker tolerance ratings (1-5) raise stress."""
    heat = np.minimum(1, clim["heat"].to_numpy() / cfg.hot_days_scale * (6 - crop.heat_tol) / 3)
    frost = np.minimum(1, clim["frost"].to_numpy() / cfg.frost_days_scale * (6 - crop.frost_tol) / 3)
    drought = (cfg.dry_spell_weight * np.minimum(1, clim["dry_spell"].to_numpy() / cfg.dry_spell_scale)
               + cfg.dry_soil_weight * np.minimum(1, clim["dry_soil"].to_numpy() / cfg.dry_soil_scale)
               ) * (6 - crop.drought_tol) / 5
    return cfg.w_heat * heat + cfg.w_frost * frost + cfg.w_dry * drought


def _score(k: str, r: str, cur: tuple[str, str], crops: pd.DataFrame, clim: dict[str, pd.DataFrame],
           soil_mult: float, weights, cfg: ScoringConfig) -> RotationScore:
    K, R = crops.loc[k], crops.loc[r]
    CK, CR = crops.loc[cur[0]], crops.loc[cur[1]]
    w = np.asarray(weights, dtype=float)
    if w.shape != (4,) or (w < 0).any():
        raise ValueError("weights must be four non-negative numbers: soil, water, resilience, income")
    w = w if w.sum() > 0 else np.ones(4)

    # soil health: does not vary by year
    sw = cfg.soil_weights
    fixer = float(max(K.n_fixer, R.n_fixer))
    pest = (int(K.family != CK.family) + int(R.family != CR.family)) / 2
    cover = min(1.0, (K.duration_days + R.duration_days) / 365)
    soil = 10 * (sw["fixer"] * fixer
                 + sw["residue"] * (K.residue + R.residue) / 10
                 + sw["root"] * (K.root + R.root) / 10
                 + sw["diversity"] * float(K.family != R.family)
                 + sw["pest"] * pest
                 + sw["cover"] * cover)
    income = 10 * (K.income + R.income) / 10

    # climate-dependent scores: one value per historical year
    irr = (np.maximum(0, K.water_mm * soil_mult - cfg.effective_rain * clim["Kharif"]["rain"].to_numpy())
           + np.maximum(0, R.water_mm * soil_mult - cfg.effective_rain * clim["Rabi"]["rain"].to_numpy()))
    water = 10 * (1 - np.minimum(1, irr / cfg.irrigation_scale_mm))
    stress = (_stress(K, clim["Kharif"], cfg) + _stress(R, clim["Rabi"], cfg)) / 2
    resil = 10 * (1 - stress)
    total = (w[0] * soil + w[1] * water + w[2] * resil + w[3] * income) / w.sum()

    spread = float(total.max() - total.min())
    confidence = "High" if spread < cfg.confidence_high else "Medium" if spread < cfg.confidence_medium else "Low"
    return RotationScore(
        key=(k, r), label=f"{k} → {r}", K=K, R=R, soil=soil, water=float(water.mean()),
        resil=float(resil.mean()), income=income, total=float(total.mean()),
        lo=float(total.min()), hi=float(total.max()), confidence=confidence, irr=float(irr.mean()),
        fixer=fixer, pest=pest, cover=cover, bare=365 * (1 - cover), stress=float(stress.mean()))


def evaluate(k: str, r: str, cur: tuple[str, str], crops: pd.DataFrame, tbl: pd.DataFrame,
             soil_mult: float, weights, cfg: ScoringConfig = DEFAULT_SCORING) -> RotationScore:
    """Score one rotation (Kharif crop k, Rabi crop r) against the farmer's current rotation `cur`."""
    return _score(k, r, cur, crops, climate_by_season(tbl), soil_mult, weights, cfg)


def evaluate_all(crops: pd.DataFrame, tbl: pd.DataFrame, cur: tuple[str, str], soil_mult: float, weights,
                 cfg: ScoringConfig = DEFAULT_SCORING) -> list[RotationScore]:
    """Score every Kharif x Rabi pair."""
    clim = climate_by_season(tbl)
    kharif = crops.index[crops.season == "Kharif"]
    rabi = crops.index[crops.season == "Rabi"]
    return [_score(k, r, cur, crops, clim, soil_mult, weights, cfg) for k in kharif for r in rabi]


def pick_strategies(options: list[RotationScore], current_key: tuple[str, str],
                    limit: float) -> list[tuple[str, RotationScore]]:
    """The current rotation plus up to three different alternatives that fit the farmer's irrigation limit (mm/yr)."""
    current = next(o for o in options if o.key == current_key)
    pool = [o for o in options if o.key != current_key and o.irr <= limit]
    picks, used = [("Your current rotation", current)], set()
    for title, rank in (("Best match for your priorities", lambda o: o.total),
                        ("Soil-health focus", lambda o: o.soil),
                        ("Water-saving focus", lambda o: -o.irr)):
        for o in sorted(pool, key=rank, reverse=True):
            if o.key not in used:
                used.add(o.key)
                picks.append((title, o))
                break
    return picks
