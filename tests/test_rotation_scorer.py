"""Unit tests for the Rotation Scoring Engine and MODIS data service."""

import os
import unittest

import numpy as np
import pandas as pd

from analysis.rotation_scorer import (
    DEFAULT_CROP_DATA_PATH,
    ROTATION_CATALOG,
    RotationScoreCard,
    RotationScorer,
)
from services.modis_data import (
    BARE_SOIL_NDVI_THRESHOLD,
    fetch_modis_ndvi_data,
    summarize_bare_soil_exposure,
)


class CropDataIntegrityTests(unittest.TestCase):
    """Test the structure, data types, and values of the crop CSV."""

    def test_crop_csv_exists_and_has_9_crops(self):
        self.assertTrue(os.path.exists(DEFAULT_CROP_DATA_PATH))
        df = pd.read_csv(DEFAULT_CROP_DATA_PATH)
        self.assertEqual(len(df), 9, "Dataset should contain exactly 9 crops.")
        self.assertEqual(len(df.columns), 12, "Dataset should contain exactly 12 columns.")

    def test_required_columns_present(self):
        df = pd.read_csv(DEFAULT_CROP_DATA_PATH)
        expected = [
            "crop_id", "crop_name", "season", "water_requirement_mm",
            "drought_tolerance", "heat_tolerance", "frost_tolerance",
            "soil_health_score", "nitrogen_fixing", "root_depth_m",
            "growth_days", "economic_return_index",
        ]
        for col in expected:
            self.assertIn(col, df.columns)

    def test_no_null_values(self):
        df = pd.read_csv(DEFAULT_CROP_DATA_PATH)
        self.assertEqual(df.isna().sum().sum(), 0)


class RotationScorerTests(unittest.TestCase):
    """Test rotation scoring calculations and comparative selection."""

    @classmethod
    def setUpClass(cls):
        cls.scorer = RotationScorer()

    def test_single_rotation_scoring(self):
        card = self.scorer.score_single_rotation(
            rotation_key="cotton_wheat",
            soil_texture="Sandy Loam",
            water_source="Canal (Non-perennial) + Tube Well",
            annual_rainfall_mm=320.0,
            max_dry_spell_days=28.0,
            annual_heat_days=45.0,
            priority_weights={"water_saving": 1.0, "soil_health": 1.0, "climate_resilience": 1.0, "income": 1.0},
        )
        self.assertIsInstance(card, RotationScoreCard)
        self.assertGreater(card.total_water_need_mm, 500)
        self.assertGreaterEqual(card.composite_score, 0)
        self.assertLessEqual(card.composite_score, 100)
        self.assertIn("agronomist", card.agronomist_note.lower())
        self.assertTrue(len(card.evidence) >= 2)
        self.assertTrue(len(card.benefits) >= 1)
        self.assertTrue(len(card.risks) >= 1)

    def test_legume_rotation_boosts_soil_health(self):
        cereal_card = self.scorer.score_single_rotation(
            rotation_key="rice_wheat",
            soil_texture="Loam",
            water_source="Canal Irrigation (Perennial)",
            annual_rainfall_mm=400.0,
            max_dry_spell_days=20.0,
            annual_heat_days=35.0,
            priority_weights={"water_saving": 1.0, "soil_health": 2.0, "climate_resilience": 1.0, "income": 1.0},
        )
        legume_card = self.scorer.score_single_rotation(
            rotation_key="cotton_mung_wheat",
            soil_texture="Loam",
            water_source="Canal Irrigation (Perennial)",
            annual_rainfall_mm=400.0,
            max_dry_spell_days=20.0,
            annual_heat_days=35.0,
            priority_weights={"water_saving": 1.0, "soil_health": 2.0, "climate_resilience": 1.0, "income": 1.0},
        )
        self.assertTrue(legume_card.has_legume)
        self.assertGreater(legume_card.nitrogen_credit_kg_ha, 0)
        self.assertGreater(legume_card.soil_health_score, cereal_card.soil_health_score)

    def test_compare_rotations_returns_four_cards(self):
        results = self.scorer.compare_rotations(
            current_rotation_key="rice_wheat",
            soil_texture="Clay Loam",
            water_source="Canal Irrigation (Perennial)",
            annual_rainfall_mm=450.0,
            max_dry_spell_days=22.0,
            annual_heat_days=38.0,
            priority_weights={"water_saving": 1.5, "soil_health": 2.0, "climate_resilience": 1.0, "income": 1.0},
            rainfall_scenario_pct=-15.0,
            temperature_scenario_c=2.0,
        )
        self.assertIn("current", results)
        self.assertIn("soil_health", results)
        self.assertIn("water_saving", results)
        self.assertIn("climate_resilient", results)

        self.assertEqual(results["current"].rotation_id, "rice_wheat")
        self.assertTrue(results["soil_health"].has_legume)
        self.assertGreaterEqual(results["water_saving"].water_saving_score, results["current"].water_saving_score)


class ModisDataTests(unittest.TestCase):
    """Test MODIS NDVI service and bare soil exposure calculations."""

    def test_modis_fetch_and_columns(self):
        df = fetch_modis_ndvi_data(30.15, 71.52, "2023-01-01", "2023-12-31")
        self.assertEqual(len(df), 365)
        self.assertIn("ndvi", df.columns)
        self.assertIn("is_bare_soil", df.columns)
        self.assertTrue(df["ndvi"].between(0.0, 1.0).all())

    def test_bare_soil_summary(self):
        df = fetch_modis_ndvi_data(30.15, 71.52, "2023-01-01", "2023-12-31")
        summary = summarize_bare_soil_exposure(df)
        self.assertGreater(summary.annual_bare_days, 0)
        self.assertLess(summary.annual_bare_days, 365)
        self.assertGreaterEqual(summary.bare_soil_percentage, 0.0)
        self.assertLessEqual(summary.bare_soil_percentage, 100.0)
        self.assertEqual(len(summary.monthly_ndvi_means), 12)
        self.assertIsInstance(summary.fallow_risk_assessment, str)


if __name__ == "__main__":
    unittest.main()
