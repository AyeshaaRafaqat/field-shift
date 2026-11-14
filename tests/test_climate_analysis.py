"""Tests for analysis/climate_analysis.py using small SYNTHETIC data.

All numbers below are invented for testing the calculations only. They are
not NASA data and must never be shown to users as climate results.

Run from the project root:  python -m unittest discover -v
"""

import unittest

import numpy as np
import pandas as pd

from analysis.climate_analysis import (
    analyse_climate,
    frost_days,
    heat_days,
    max_dry_spell,
    seasonal_rainfall,
    soil_moisture_anomaly,
)

WET = {"Wet": ("06-01", "09-30")}                      # simple season inside one year
WRAPPED = {"Winter": ("11-01", "04-30"), "Summer": ("05-01", "10-31")}


def make_frame(start, end, precip=0.0, tmax=25.0, tmin=10.0, soil=None):
    index = pd.date_range(start, end, freq="D", name="date")
    data = {
        "precip_mm_day": np.full(len(index), float(precip)),
        "tmax_c": np.full(len(index), float(tmax)),
        "tmin_c": np.full(len(index), float(tmin)),
    }
    if soil is not None:
        data["soil_wetness_root"] = np.full(len(index), float(soil))
    return pd.DataFrame(data, index=index)


def row(table, year, season=None):
    sel = table[table["season_year"] == year]
    if season is not None:
        sel = sel[sel["season"] == season]
    assert len(sel) == 1, f"expected one row for {year}/{season}, got {len(sel)}"
    return sel.iloc[0]


class SeasonalRainfallTests(unittest.TestCase):
    def test_total_for_each_season_year(self):
        df = make_frame("2020-01-01", "2021-12-31", precip=0.0)
        in_2020 = (df.index >= "2020-06-01") & (df.index <= "2020-09-30")
        df.loc[in_2020, "precip_mm_day"] = 2.0          # 122 days * 2 mm
        result = seasonal_rainfall(df, WET)
        self.assertAlmostEqual(row(result, 2020)["total_precip_mm"], 244.0)
        self.assertAlmostEqual(row(result, 2021)["total_precip_mm"], 0.0)
        self.assertEqual(row(result, 2020)["days_in_season"], 122)
        self.assertAlmostEqual(row(result, 2020)["coverage"], 1.0)

    def test_wrapped_season_belongs_to_start_year(self):
        df = make_frame("2020-11-01", "2021-04-30", precip=0.0)
        df.loc["2021-01-15", "precip_mm_day"] = 10.0     # January 2021 -> Winter 2020
        result = seasonal_rainfall(df, WRAPPED)
        winter = row(result, 2020, "Winter")
        self.assertAlmostEqual(winter["total_precip_mm"], 10.0)
        self.assertNotIn(2021, set(result.loc[result["season"] == "Winter", "season_year"]))


class DrySpellTests(unittest.TestCase):
    def test_longest_run_between_rain_days(self):
        df = make_frame("2020-06-01", "2020-09-30", precip=0.0)
        df.loc["2020-06-10", "precip_mm_day"] = 5.0
        df.loc["2020-07-20", "precip_mm_day"] = 5.0
        # Runs: Jun1-9 = 9, Jun11-Jul19 = 39, Jul21-Sep30 = 72
        result = max_dry_spell(df, WET, threshold_mm=1.0)
        self.assertEqual(row(result, 2020)["max_dry_spell_days"], 72)

    def test_threshold_is_configurable_and_strict(self):
        df = make_frame("2020-06-01", "2020-06-30", precip=0.5)
        df.loc["2020-06-15", "precip_mm_day"] = 1.0       # exactly the threshold -> not dry
        result = max_dry_spell(df, {"June": ("06-01", "06-30")}, threshold_mm=1.0)
        self.assertEqual(row(result, 2020)["max_dry_spell_days"], 15)   # Jun16-30
        result_low = max_dry_spell(df, {"June": ("06-01", "06-30")}, threshold_mm=0.1)
        self.assertEqual(row(result_low, 2020)["max_dry_spell_days"], 0)

    def test_missing_days_break_a_spell(self):
        df = make_frame("2020-06-01", "2020-06-30", precip=0.0)
        df.loc["2020-06-11", "precip_mm_day"] = np.nan
        result = max_dry_spell(df, {"June": ("06-01", "06-30")})
        self.assertEqual(row(result, 2020)["max_dry_spell_days"], 19)   # Jun12-30

    def test_gap_in_index_is_treated_as_missing_not_consecutive(self):
        df = make_frame("2020-06-01", "2020-06-30", precip=0.0)
        df = df.drop(pd.Timestamp("2020-06-11"))          # row removed entirely
        result = max_dry_spell(df, {"June": ("06-01", "06-30")})
        self.assertEqual(row(result, 2020)["max_dry_spell_days"], 19)
        self.assertEqual(row(result, 2020)["days_with_data"], 29)


class HeatAndFrostTests(unittest.TestCase):
    def test_heat_days_use_strictly_greater_than(self):
        df = make_frame("2020-06-01", "2020-09-30", tmax=25.0)
        df.loc["2020-07-01":"2020-07-05", "tmax_c"] = 36.0   # 5 days above
        df.loc["2020-08-01":"2020-08-03", "tmax_c"] = 35.0   # 3 days exactly at threshold
        self.assertEqual(row(heat_days(df, WET, threshold_c=35.0), 2020)["heat_days"], 5)
        self.assertEqual(row(heat_days(df, WET, threshold_c=30.0), 2020)["heat_days"], 8)

    def test_frost_days_below_zero_only(self):
        df = make_frame("2020-11-01", "2021-04-30", tmin=5.0)
        df.loc["2020-12-10":"2020-12-13", "tmin_c"] = -1.0   # 4 frost days
        df.loc["2021-01-05":"2021-01-06", "tmin_c"] = 0.0    # exactly 0 -> not frost
        result = frost_days(df, WRAPPED)
        self.assertEqual(row(result, 2020, "Winter")["frost_days"], 4)

    def test_all_missing_gives_nan_not_zero(self):
        df = make_frame("2020-06-01", "2020-09-30")
        df["tmax_c"] = np.nan
        self.assertTrue(np.isnan(row(heat_days(df, WET), 2020)["heat_days"]))


class MissingDataAndCoverageTests(unittest.TestCase):
    def test_missing_rain_days_reduce_coverage_and_are_skipped_in_sum(self):
        df = make_frame("2020-06-01", "2020-09-30", precip=1.0)
        df.loc["2020-06-01":"2020-06-12", "precip_mm_day"] = np.nan   # 12 missing days
        r = row(seasonal_rainfall(df, WET), 2020)
        self.assertAlmostEqual(r["total_precip_mm"], 110.0)
        self.assertEqual(r["days_with_data"], 110)
        self.assertAlmostEqual(r["coverage"], 110 / 122)

    def test_all_missing_season_gives_nan_total(self):
        df = make_frame("2020-06-01", "2020-09-30")
        df["precip_mm_day"] = np.nan
        self.assertTrue(np.isnan(row(seasonal_rainfall(df, WET), 2020)["total_precip_mm"]))

    def test_partial_season_at_start_of_data_is_flagged(self):
        df = make_frame("2020-08-01", "2021-09-30")
        result = seasonal_rainfall(df, WET)
        self.assertLess(row(result, 2020)["coverage"], 0.6)     # Aug-Sep only
        self.assertAlmostEqual(row(result, 2021)["coverage"], 1.0)


class SoilMoistureTests(unittest.TestCase):
    def test_not_calculated_when_column_absent(self):
        df = make_frame("2018-01-01", "2021-12-31")
        res = soil_moisture_anomaly(df, WET)
        self.assertFalse(res.available)
        self.assertTrue(res.table.empty)
        self.assertIn("not in the data", res.reason)

    def test_not_calculated_when_all_missing(self):
        df = make_frame("2018-01-01", "2021-12-31", soil=np.nan)
        res = soil_moisture_anomaly(df, WET)
        self.assertFalse(res.available)

    def test_anomaly_against_multi_year_baseline(self):
        df = make_frame("2018-01-01", "2021-12-31", soil=0.5)
        for year, value in {2018: 0.4, 2019: 0.5, 2020: 0.6, 2021: 0.5}.items():
            df.loc[f"{year}-06-01":f"{year}-09-30", "soil_wetness_root"] = value
        res = soil_moisture_anomaly(df, WET)
        self.assertTrue(res.available)
        t = res.table
        self.assertAlmostEqual(row(t, 2018)["baseline_mean"], 0.5)
        self.assertAlmostEqual(row(t, 2018)["anomaly"], -0.1)
        self.assertAlmostEqual(row(t, 2019)["anomaly"], 0.0)
        self.assertAlmostEqual(row(t, 2020)["anomaly"], 0.1)
        self.assertAlmostEqual(row(t, 2020)["anomaly_zscore"], 0.1 / np.std([0.4, 0.5, 0.6, 0.5], ddof=1))

    def test_partial_season_gets_no_anomaly_and_is_excluded_from_baseline(self):
        df = make_frame("2018-08-15", "2021-12-31", soil=0.5)   # 2018 season is partial
        df.loc["2018-08-15":"2018-09-30", "soil_wetness_root"] = 0.9
        res = soil_moisture_anomaly(df, WET)
        t = res.table
        self.assertTrue(np.isnan(row(t, 2018)["anomaly"]))
        self.assertAlmostEqual(row(t, 2019)["baseline_mean"], 0.5)   # 0.9 not included

    def test_too_few_years_gives_no_baseline(self):
        df = make_frame("2020-01-01", "2021-12-31", soil=0.5)       # only 2 seasons
        res = soil_moisture_anomaly(df, WET)
        self.assertTrue(res.available)
        self.assertTrue(res.table["anomaly"].isna().all())
        self.assertIn("Fewer than", res.reason)


class InputValidationTests(unittest.TestCase):
    def test_missing_column_raises(self):
        df = make_frame("2020-06-01", "2020-09-30").drop(columns="tmax_c")
        with self.assertRaises(ValueError):
            heat_days(df, WET)

    def test_non_datetime_index_raises(self):
        df = make_frame("2020-06-01", "2020-09-30").reset_index(drop=True)
        with self.assertRaises(ValueError):
            seasonal_rainfall(df, WET)

    def test_empty_frame_raises(self):
        with self.assertRaises(ValueError):
            seasonal_rainfall(make_frame("2020-06-01", "2020-06-02").iloc[0:0], WET)

    def test_overlapping_seasons_raise(self):
        df = make_frame("2020-06-01", "2020-09-30")
        with self.assertRaises(ValueError):
            seasonal_rainfall(df, {"A": ("05-01", "07-31"), "B": ("07-01", "09-30")})

    def test_bad_season_date_raises(self):
        df = make_frame("2020-06-01", "2020-09-30")
        with self.assertRaises(ValueError):
            seasonal_rainfall(df, {"A": ("13-01", "07-31")})

    def test_bad_thresholds_raise(self):
        df = make_frame("2020-06-01", "2020-09-30")
        with self.assertRaises(ValueError):
            max_dry_spell(df, WET, threshold_mm=-1)
        with self.assertRaises(ValueError):
            heat_days(df, WET, threshold_c="hot")
        with self.assertRaises(ValueError):
            heat_days(df, WET, threshold_c=float("nan"))


class CombinedAnalysisTests(unittest.TestCase):
    def test_combined_structure(self):
        df = make_frame("2016-01-01", "2020-12-31", soil=0.4)
        result = analyse_climate(df, WRAPPED, dry_day_threshold_mm=1.0, heat_threshold_c=35.0)
        table = result.seasonal_indicators
        for column in ("season_label", "total_precip_mm", "max_dry_spell_days",
                       "heat_days", "frost_days", "coverage", "complete"):
            self.assertIn(column, table.columns)
        self.assertTrue(result.soil_moisture.available)
        self.assertEqual(result.assumptions["heat_threshold_c"], 35.0)
        self.assertIn("Winter 2016-17", set(table["season_label"]))
        # first Winter (started Nov 2015) and last Winter (ends Apr 2021) are partial
        self.assertFalse(row(table, 2015, "Winter")["complete"])
        self.assertTrue(row(table, 2017, "Winter")["complete"])
        self.assertTrue(table["season_year"].is_monotonic_increasing)

    def test_combined_without_soil_column(self):
        df = make_frame("2016-01-01", "2020-12-31")
        result = analyse_climate(df, WRAPPED)
        self.assertFalse(result.soil_moisture.available)


if __name__ == "__main__":
    unittest.main()
