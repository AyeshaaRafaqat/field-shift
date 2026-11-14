"""Tests for utils/helpers.py (no network, synthetic inputs)."""

import unittest
from datetime import date

import numpy as np
import pandas as pd

from utils.helpers import (
    data_quality_report,
    parse_month_day,
    to_date,
    validate_coordinates,
    validate_date_range,
)


class HelperTests(unittest.TestCase):
    def test_coordinates_ok_and_cast_to_float(self):
        self.assertEqual(validate_coordinates("31.5", 74), (31.5, 74.0))

    def test_coordinates_bad(self):
        for lat, lon in [(91, 0), (0, 181), (None, 0), ("a", 0), (float("inf"), 0)]:
            with self.assertRaises(ValueError):
                validate_coordinates(lat, lon)

    def test_to_date_formats(self):
        self.assertEqual(to_date("2024-03-05"), date(2024, 3, 5))
        self.assertEqual(to_date("20240305"), date(2024, 3, 5))
        self.assertEqual(to_date(pd.Timestamp("2024-03-05")), date(2024, 3, 5))
        with self.assertRaises(ValueError):
            to_date("05/03/2024")

    def test_date_range(self):
        self.assertEqual(
            validate_date_range("2016-01-01", "2020-12-31"), (date(2016, 1, 1), date(2020, 12, 31))
        )
        with self.assertRaises(ValueError):
            validate_date_range("2020-01-02", "2020-01-01")
        with self.assertRaises(ValueError):
            validate_date_range("1980-12-31", "1990-01-01")

    def test_month_day(self):
        self.assertEqual(parse_month_day("11-01"), (11, 1))
        self.assertEqual(parse_month_day("02-29"), (2, 29))
        for bad in ("13-01", "11-31", "1101", "", "a-b"):
            with self.assertRaises(ValueError):
                parse_month_day(bad)

    def test_quality_report(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
        report = data_quality_report(df)
        self.assertEqual(int(report.loc[0, "missing_days"]), 2)
        self.assertEqual(report.loc[0, "missing_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
