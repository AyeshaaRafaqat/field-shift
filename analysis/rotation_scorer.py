"""Crop Rotation Scoring Engine.

Combines rule-based agronomic constraints with a Random Forest model to evaluate
crop rotation viability, water balance, soil health impacts, and climate resilience
under 10-year historical climate conditions and what-if sensitivity scenarios.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DEFAULT_CROP_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "crop_data.csv"
)

# Soil texture permeability & drainage suitability multipliers (0.0 to 1.0)
SOIL_SUITABILITY: Dict[str, Dict[str, float]] = {
    "Sandy Loam": {
        "wheat": 0.90, "rice": 0.40, "cotton": 0.95, "maize": 0.90,
        "sugarcane": 0.70, "chickpea": 0.95, "mungbean": 0.95, "mustard": 0.90, "lentil": 0.85,
    },
    "Loam": {
        "wheat": 1.00, "rice": 0.70, "cotton": 1.00, "maize": 1.00,
        "sugarcane": 0.90, "chickpea": 0.90, "mungbean": 1.00, "mustard": 0.95, "lentil": 0.95,
    },
    "Clay Loam": {
        "wheat": 0.95, "rice": 0.95, "cotton": 0.80, "maize": 0.85,
        "sugarcane": 1.00, "chickpea": 0.65, "mungbean": 0.80, "mustard": 0.85, "lentil": 0.75,
    },
    "Silt Loam": {
        "wheat": 0.95, "rice": 0.75, "cotton": 0.90, "maize": 0.95,
        "sugarcane": 0.90, "chickpea": 0.80, "mungbean": 0.90, "mustard": 0.90, "lentil": 0.90,
    },
    "Sandy": {
        "wheat": 0.60, "rice": 0.15, "cotton": 0.75, "maize": 0.50,
        "sugarcane": 0.40, "chickpea": 0.90, "mungbean": 0.85, "mustard": 0.80, "lentil": 0.80,
    },
    "Clay": {
        "wheat": 0.80, "rice": 1.00, "cotton": 0.50, "maize": 0.60,
        "sugarcane": 0.90, "chickpea": 0.30, "mungbean": 0.55, "mustard": 0.65, "lentil": 0.45,
    },
}

# Water source irrigation supply indices (effective mm equivalent delivery reliability)
WATER_SUPPLY_FACTORS: Dict[str, float] = {
    "Canal Irrigation (Perennial)": 1.0,
    "Canal (Non-perennial) + Tube Well": 0.85,
    "Tube Well (Fresh Groundwater)": 0.80,
    "Tube Well (Marginal / Saline)": 0.55,
    "Rainfed (Barani)": 0.25,
}

# Pre-defined agricultural candidate rotations
ROTATION_CATALOG: Dict[str, Dict] = {
    "rice_wheat": {
        "name": "Basmati Rice → Wheat",
        "crops": ["rice", "wheat"],
        "category": "Traditional Cereal Intensive",
        "description": "Standard intensive cereal rotation across canal-irrigated alluvial plains.",
    },
    "cotton_wheat": {
        "name": "Cotton → Wheat",
        "crops": ["cotton", "wheat"],
        "category": "Cash Crop & Food Security",
        "description": "Core cropping pattern across southern and central Punjab.",
    },
    "maize_wheat": {
        "name": "Grain Maize → Wheat",
        "crops": ["maize", "wheat"],
        "category": "High-Yield Cereal Feed",
        "description": "High nutrient demanding intensive grain cycle popular in dairy belts.",
    },
    "sugarcane_wheat": {
        "name": "Sugarcane (Annual) → Wheat",
        "crops": ["sugarcane", "wheat"],
        "category": "Heavy Irrigation Industrial",
        "description": "High water and fertilizer footprint with long field occupancy.",
    },
    "cotton_mung_wheat": {
        "name": "Cotton → Mungbean (Catch) → Wheat",
        "crops": ["cotton", "mungbean", "wheat"],
        "category": "Legume-Intensified Diversification",
        "description": "Short-duration summer pulse inserted to fix nitrogen and cover bare ground.",
    },
    "rice_chickpea": {
        "name": "Basmati Rice → Desi Chickpea",
        "crops": ["rice", "chickpea"],
        "category": "Soil Restorative Pulse",
        "description": "Replaces winter wheat with a deep-rooting, drought-hardy nitrogen-fixing legume.",
    },
    "maize_mustard": {
        "name": "Grain Maize → Mustard / Canola",
        "crops": ["maize", "mustard"],
        "category": "Low-Water Oilseed Diversification",
        "description": "Replaces winter wheat with drought-tolerant brassica oilseed.",
    },
    "mung_wheat": {
        "name": "Mungbean (Kharif) → Wheat",
        "crops": ["mungbean", "wheat"],
        "category": "Groundwater-Saving Legume",
        "description": "Drastically reduces Kharif pumping by replacing rice/cotton with green gram.",
    },
    "cotton_chickpea": {
        "name": "Cotton → Desi Chickpea",
        "crops": ["cotton", "chickpea"],
        "category": "Climate-Resilient Low-Input",
        "description": "Combines deep-rooted cotton with high-drought tolerant winter pulse.",
    },
}


@dataclass
class RotationScoreCard:
    """Complete evaluation card for one rotation option."""
    rotation_id: str
    rotation_name: str
    category: str
    crop_names: List[str]
    total_water_need_mm: float
    water_saving_score: float      # 0 to 100
    soil_health_score: float       # 0 to 100
    climate_resilience_score: float  # 0 to 100
    income_score: float            # 0 to 100
    composite_score: float         # 0 to 100
    has_legume: bool
    nitrogen_credit_kg_ha: float
    why_suggested: str
    evidence: List[str]
    benefits: List[str]
    risks: List[str]
    information_gaps: List[str]
    agronomist_note: str


class RotationScorer:
    """Evaluates crop rotations using agronomic logic and Random Forest regression."""

    def __init__(self, crop_csv_path: str = DEFAULT_CROP_DATA_PATH):
        self.crop_csv_path = crop_csv_path
        self.crops_df = self._load_crops(crop_csv_path)
        self.rf_model = self._train_surrogate_model()

    def _load_crops(self, path: str) -> pd.DataFrame:
        """Load and validate the 9-crop dataset."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Crop dataset not found at: {path}")

        df = pd.read_csv(path)
        required_cols = [
            "crop_id", "crop_name", "season", "water_requirement_mm",
            "drought_tolerance", "heat_tolerance", "frost_tolerance",
            "soil_health_score", "nitrogen_fixing", "root_depth_m",
            "growth_days", "economic_return_index",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Crop dataset is missing columns: {', '.join(missing)}")

        df["nitrogen_fixing"] = df["nitrogen_fixing"].astype(bool)
        df = df.set_index("crop_id")
        return df

    def _train_surrogate_model(self) -> RandomForestRegressor:
        """Train a lightweight Random Forest regressor on synthetic agronomic parameter variations."""
        np.random.seed(42)
        n_samples = 450

        # Features: [water_demand, drought_tol, heat_tol, root_depth, has_legume,
        #            annual_rain, dry_spell, heat_days, rain_delta, temp_delta]
        X = []
        y = []

        for _ in range(n_samples):
            w_demand = np.random.uniform(400, 2000)
            d_tol = np.random.uniform(1.5, 4.8)
            h_tol = np.random.uniform(2.0, 4.5)
            root_d = np.random.uniform(0.4, 1.6)
            has_leg = np.random.choice([0, 1])
            rain = np.random.uniform(150, 750)
            dry_days = np.random.uniform(10, 55)
            h_days = np.random.uniform(20, 85)
            rain_delta = np.random.uniform(-0.30, 0.30)
            temp_delta = np.random.uniform(0.0, 4.0)

            # Agronomic ground-truth relationship with non-linear penalties
            effective_rain = rain * (1.0 + rain_delta)
            water_stress_penalty = max(0, (w_demand - effective_rain * 2.2) / 35.0)
            drought_buffering = (d_tol * 7.0) + (root_d * 8.0)
            heat_stress_penalty = max(0, (h_days + temp_delta * 12.0 - h_tol * 20.0) * 0.4)
            legume_bonus = 14.0 if has_leg else 0.0

            resilience = 60.0 + drought_buffering + legume_bonus - water_stress_penalty - heat_stress_penalty
            resilience = float(np.clip(resilience + np.random.normal(0, 2.5), 15.0, 98.0))

            X.append([
                w_demand, d_tol, h_tol, root_d, has_leg,
                rain, dry_days, h_days, rain_delta, temp_delta
            ])
            y.append(resilience)

        rf = RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42)
        rf.fit(np.array(X), np.array(y))
        return rf

    def score_single_rotation(
        self,
        rotation_key: str,
        soil_texture: str,
        water_source: str,
        annual_rainfall_mm: float,
        max_dry_spell_days: float,
        annual_heat_days: float,
        priority_weights: Dict[str, float],
        rainfall_scenario_pct: float = 0.0,
        temperature_scenario_c: float = 0.0,
    ) -> RotationScoreCard:
        """Calculate complete metric scores and qualitative card details for one rotation."""
        if rotation_key not in ROTATION_CATALOG:
            raise ValueError(f"Unknown rotation key: {rotation_key}")

        rot_info = ROTATION_CATALOG[rotation_key]
        crop_ids = rot_info["crops"]
        crop_subset = self.crops_df.loc[crop_ids]

        # 1. Basic agronomic aggregations
        total_water_need = float(crop_subset["water_requirement_mm"].sum())
        avg_drought_tol = float(crop_subset["drought_tolerance"].mean())
        avg_heat_tol = float(crop_subset["heat_tolerance"].mean())
        max_root_depth = float(crop_subset["root_depth_m"].max())
        has_legume = bool(crop_subset["nitrogen_fixing"].any())
        crop_names = list(crop_subset["crop_name"])

        # Soil texture compatibility
        soil_lookup = SOIL_SUITABILITY.get(soil_texture, {})
        soil_fit_factors = [soil_lookup.get(cid, 0.8) for cid in crop_ids]
        mean_soil_fit = float(np.mean(soil_fit_factors))

        # Water source reliability factor
        water_factor = WATER_SUPPLY_FACTORS.get(water_source, 0.75)

        # 2. Water Saving Score (0-100)
        # Benchmark reference: rice-wheat uses ~1630 mm, sugarcane-wheat ~1980 mm
        effective_annual_rain = annual_rainfall_mm * (1.0 + rainfall_scenario_pct / 100.0)
        water_saving_pct = max(0.0, min(100.0, (1.0 - (total_water_need / 1900.0)) * 100.0))
        # Penalty if water availability is low and crop needs heavy irrigation
        water_deficit_penalty = 0.0
        if water_factor < 0.6 and total_water_need > 900:
            water_deficit_penalty = (total_water_need - 900) / 18.0
        water_score = max(5.0, min(98.0, (water_saving_pct * 0.85 + water_factor * 15.0) - water_deficit_penalty))

        # 3. Soil Health Score (0-100)
        base_soil_score = float(crop_subset["soil_health_score"].mean()) * 10.0
        legume_n_bonus = 22.0 if has_legume else 0.0
        taproot_bonus = 8.0 if max_root_depth >= 1.2 else 0.0
        soil_score = max(10.0, min(99.0, (base_soil_score * 0.7 + legume_n_bonus + taproot_bonus) * mean_soil_fit))

        # 4. Income / Economic Score (0-100)
        base_econ = float(crop_subset["economic_return_index"].mean()) * 10.0
        # If severe water shortage, high-water crops suffer yield loss
        econ_penalty = 0.0
        if water_factor < 0.6 and total_water_need > 1000:
            econ_penalty = 20.0
        income_score = max(10.0, min(98.0, base_econ * mean_soil_fit - econ_penalty))

        # 5. Machine Learning Climate Resilience Score (Random Forest)
        rf_features = np.array([[
            total_water_need,
            avg_drought_tol,
            avg_heat_tol,
            max_root_depth,
            1 if has_legume else 0,
            annual_rainfall_mm,
            max_dry_spell_days,
            annual_heat_days,
            rainfall_scenario_pct / 100.0,
            temperature_scenario_c,
        ]])
        predicted_resilience = float(self.rf_model.predict(rf_features)[0])
        # Modulate by soil fit
        resilience_score = max(10.0, min(98.0, predicted_resilience * (0.8 + 0.2 * mean_soil_fit)))

        # 6. Composite Score based on Farmer Priorities
        w_water = priority_weights.get("water_saving", 1.0)
        w_soil = priority_weights.get("soil_health", 1.0)
        w_resil = priority_weights.get("climate_resilience", 1.0)
        w_income = priority_weights.get("income", 1.0)
        sum_w = max(0.001, w_water + w_soil + w_resil + w_income)

        composite = (
            w_water * water_score +
            w_soil * soil_score +
            w_resil * resilience_score +
            w_income * income_score
        ) / sum_w

        # Nitrogen credit estimation
        n_credit = 0.0
        if "chickpea" in crop_ids:
            n_credit += 35.0
        if "mungbean" in crop_ids:
            n_credit += 28.0
        if "lentil" in crop_ids:
            n_credit += 25.0

        # Construct textual card explanations
        why, evidence, benefits, risks, gaps = self._build_card_texts(
            rot_info, crop_subset, total_water_need, soil_texture, water_source,
            water_score, soil_score, resilience_score, income_score, n_credit,
            annual_rainfall_mm, max_dry_spell_days, annual_heat_days,
            rainfall_scenario_pct, temperature_scenario_c,
        )

        agronomist_note = (
            "Verify recommended sowing window (Kharif/Rabi turn), certified seed variety "
            "(e.g., heat-tolerant or short-duration cultivars), and localized soil salinity levels "
            "with your district Agriculture Extension Markaz / OFWM agronomist before sowing."
        )

        return RotationScoreCard(
            rotation_id=rotation_key,
            rotation_name=rot_info["name"],
            category=rot_info["category"],
            crop_names=crop_names,
            total_water_need_mm=round(total_water_need, 1),
            water_saving_score=round(water_score, 1),
            soil_health_score=round(soil_score, 1),
            climate_resilience_score=round(resilience_score, 1),
            income_score=round(income_score, 1),
            composite_score=round(composite, 1),
            has_legume=has_legume,
            nitrogen_credit_kg_ha=round(n_credit, 1),
            why_suggested=why,
            evidence=evidence,
            benefits=benefits,
            risks=risks,
            information_gaps=gaps,
            agronomist_note=agronomist_note,
        )

    def _build_card_texts(
        self, rot_info, crop_subset, water_need, soil_texture, water_source,
        water_score, soil_score, resilience_score, income_score, n_credit,
        rain, dry_days, heat_days, rain_delta, temp_delta,
    ) -> Tuple[str, List[str], List[str], List[str], List[str]]:
        """Generate tailored agronomic descriptions."""
        name = rot_info["name"]
        crops = list(crop_subset.index)

        # Why suggested
        if "mungbean" in crops or "chickpea" in crops:
            why = (
                f"Selected as a high-benefit restorative rotation for {soil_texture} soils. "
                f"Integrating a biological nitrogen-fixing pulse addresses declining organic matter "
                f"while cutting tubewell diesel/electricity pumping costs under {water_source}."
            )
        elif water_score > 70:
            why = (
                f"Engineered specifically for water efficiency. At {water_need:.0f} mm total water demand, "
                f"it cushions against dry spells ({dry_days:.0f} consecutive dry days on record) "
                f"and protects the farm against groundwater drawdown."
            )
        elif "rice" in crops:
            why = (
                f"Evaluated as the high-input baseline cereal pattern. "
                f"Requires substantial irrigation access ({water_need:.0f} mm/yr) but provides high calorie "
                f"output and established procurement markets in canal commands."
            )
        else:
            why = (
                f"Balanced rotation designed to sustain cash flow while stabilizing yield against "
                f"extreme heat stress ({heat_days:.0f} hot days/year historically)."
            )

        # Evidence
        evidence = [
            f"Annual crop water requirement: {water_need:.0f} mm (regional historical rainfall: {rain:.0f} mm/yr).",
            f"Machine Learning Resilience Score: {resilience_score:.1f}/100 under current + scenario climate conditions.",
            f"Soil compatibility index on {soil_texture}: {SOIL_SUITABILITY.get(soil_texture, {}).get(crops[0], 0.8):.0%}.",
        ]
        if n_credit > 0:
            evidence.append(f"Biological nitrogen fixation contribution: ~{n_credit:.0f} kg N/ha into root zone.")
        if rain_delta != 0 or temp_delta != 0:
            evidence.append(f"What-if climate scenario tested: Rainfall {rain_delta:+.0f}%, Temperature {temp_delta:+.1f}°C.")

        # Benefits
        benefits = []
        if n_credit > 0:
            benefits.append(f"Provides estimated ~{n_credit:.0f} kg/ha residual nitrogen, lowering chemical urea expenses for the next crop.")
            benefits.append("Breaks continuous cereal weed and root pathogen cycles.")
        if water_need < 800:
            benefits.append(f"Lowers irrigation depth by {(1 - water_need / 1630):.0%} compared to traditional Rice-Wheat, reducing pumping hours.")
        if "cotton" in crops:
            benefits.append("Deep taproot structure (1.5 m) aerates subsoil hardpans created by continuous shallow ploughing.")
        if "mustard" in crops:
            benefits.append("Winter oilseed with low evapotranspiration footprint and excellent local mill demand.")
        if len(benefits) < 2:
            benefits.append("Well-understood farmer management practices with readily available commercial inputs.")
            benefits.append("High gross marketable volume through established grain market mandi networks.")

        # Risks
        risks = []
        if "chickpea" in crops and soil_texture in ["Clay", "Clay Loam"]:
            risks.append(f"Heavy {soil_texture} soils with poor drainage elevate collar rot and Ascochyta blight risk in chickpea.")
        if "rice" in crops and "Tube Well" in water_source:
            risks.append("Extreme vulnerability to tube well power outages and aquifer drawdown during peak summer Kharif heat.")
        if "mungbean" in crops:
            risks.append("Unseasonal summer showers during pod maturity can cause seed sprouting and pod shattering.")
        if temp_delta > 1.5:
            risks.append("Terminal heat stress in March may hasten grain filling in late-sown winter crops, reducing 1000-grain weight.")
        if not risks:
            risks.append("Price volatility at local procurement centers during seasonal peak harvest arrival.")

        # Information gaps
        gaps = [
            "Local availability of certified short-duration seed varieties suited for narrow planting windows.",
            "Field-specific groundwater salinity (EC/SAR) testing to ensure tube well water does not cause sodicity.",
            "Farm-specific canal turn (Warabandi) schedule alignment with peak flowering water stages.",
        ]

        return why, evidence, benefits, risks, gaps

    def compare_rotations(
        self,
        current_rotation_key: str,
        soil_texture: str,
        water_source: str,
        annual_rainfall_mm: float,
        max_dry_spell_days: float,
        annual_heat_days: float,
        priority_weights: Dict[str, float],
        rainfall_scenario_pct: float = 0.0,
        temperature_scenario_c: float = 0.0,
    ) -> Dict[str, RotationScoreCard]:
        """Score all catalog rotations and return:
        - current: Current baseline rotation
        - soil_health: Top alternative emphasizing soil health and legumes
        - water_saving: Top alternative minimizing water consumption
        - climate_resilient: Top alternative with highest climate stress resilience
        """
        all_cards: Dict[str, RotationScoreCard] = {}
        for r_key in ROTATION_CATALOG:
            card = self.score_single_rotation(
                rotation_key=r_key,
                soil_texture=soil_texture,
                water_source=water_source,
                annual_rainfall_mm=annual_rainfall_mm,
                max_dry_spell_days=max_dry_spell_days,
                annual_heat_days=annual_heat_days,
                priority_weights=priority_weights,
                rainfall_scenario_pct=rainfall_scenario_pct,
                temperature_scenario_c=temperature_scenario_c,
            )
            all_cards[r_key] = card

        current_card = all_cards.get(
            current_rotation_key,
            self.score_single_rotation(
                "rice_wheat", soil_texture, water_source,
                annual_rainfall_mm, max_dry_spell_days, annual_heat_days,
                priority_weights, rainfall_scenario_pct, temperature_scenario_c
            )
        )

        # Exclude current rotation from alternatives
        candidates = {k: v for k, v in all_cards.items() if k != current_rotation_key}

        # 1. Soil Health Option: Highest soil health score with a legume
        legume_candidates = {k: v for k, v in candidates.items() if v.has_legume}
        if legume_candidates:
            soil_health_key = max(legume_candidates, key=lambda k: legume_candidates[k].soil_health_score)
        else:
            soil_health_key = max(candidates, key=lambda k: candidates[k].soil_health_score)
        soil_health_card = candidates[soil_health_key]

        # 2. Water Saving Option: Lowest water need / highest water score (distinct if possible)
        water_candidates = {k: v for k, v in candidates.items() if k != soil_health_key}
        if not water_candidates:
            water_candidates = candidates
        water_saving_key = max(water_candidates, key=lambda k: water_candidates[k].water_saving_score)
        water_saving_card = water_candidates[water_saving_key]

        # 3. Climate Resilient Option: Highest resilience under scenario
        resil_candidates = {
            k: v for k, v in candidates.items()
            if k not in (soil_health_key, water_saving_key)
        }
        if not resil_candidates:
            resil_candidates = candidates
        resil_key = max(resil_candidates, key=lambda k: resil_candidates[k].climate_resilience_score)
        resil_card = resil_candidates[resil_key]

        return {
            "current": current_card,
            "soil_health": soil_health_card,
            "water_saving": water_saving_card,
            "climate_resilient": resil_card,
        }
