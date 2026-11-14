"""Monte Carlo rank-stability engine for crop rotation uncertainty analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class UncertaintySummary:
    rotation_id: str
    rotation_name: str
    median_score: float
    p10_score: float
    p90_score: float
    win_rate_pct: float
    rank_stability_score: float
    confidence_level: str
    score_spread: float


def run_monte_carlo_analysis(
    cards: Dict[str, any],
    n_simulations: int = 500,
    random_seed: int = 42,
) -> Dict[str, UncertaintySummary]:
    """Run N stochastic simulations to quantify rotation rank stability."""
    rng = np.random.default_rng(random_seed)
    keys = list(cards.keys())
    simulated_scores = {k: np.zeros(n_simulations) for k in keys}

    for i in range(n_simulations):
        rain_factor = rng.uniform(0.70, 0.95)
        heat_shift = rng.uniform(-1.0, 1.0)
        market_shift = rng.uniform(-0.15, 0.15)
        n_credit_shift = rng.uniform(-0.20, 0.20)

        for k in keys:
            card = cards[k]
            base_water = card.water_saving_score * rain_factor / 0.85
            base_soil = card.soil_health_score * (1.0 + (n_credit_shift if card.has_legume else 0.0))
            base_resil = card.climate_resilience_score - (heat_shift * 2.5 if heat_shift > 0 else 0.0)
            base_income = card.income_score * (1.0 + market_shift)
            score_draw = 0.30 * base_water + 0.30 * base_soil + 0.20 * base_resil + 0.20 * base_income
            simulated_scores[k][i] = np.clip(score_draw, 10.0, 99.0)

    results: Dict[str, UncertaintySummary] = {}
    for k in keys:
        s = simulated_scores[k]
        win_count = sum(
            1 for i in range(n_simulations)
            if simulated_scores[k][i] == max(simulated_scores[kk][i] for kk in keys)
        )
        p10, median, p90 = float(np.percentile(s, 10)), float(np.median(s)), float(np.percentile(s, 90))
        spread = p90 - p10
        win_rate = (win_count / n_simulations) * 100.0
        stability = max(0.0, 100.0 - spread * 2.5)
        confidence = "High" if spread < 12 else ("Medium" if spread < 22 else "Low")
        results[k] = UncertaintySummary(
            rotation_id=k,
            rotation_name=cards[k].rotation_name,
            median_score=median,
            p10_score=p10,
            p90_score=p90,
            win_rate_pct=win_rate,
            rank_stability_score=stability,
            confidence_level=confidence,
            score_spread=spread,
        )
    return results

