"""Tests for services/nasa_data.py error handling and parsing.

IMPORTANT: the payloads below are SYNTHETIC. They imitate the documented
structure of a NASA POWER daily JSON response so the parsing and error
handling code can be exercised without network access. They are not real
NASA data. The real API is checked with tests/check_live_nasa_api.py.

Run from the project root:  python -m unittest discover -v
"""

import unittest
from unittest import mock

import numpy as np
import requests

from services import nasa_data
from services.nasa_data import NasaPowerError, fetch_nasa_power_data


def synthetic_payload(parameters, days=("20240101", "20240102", "20240103"), values=None, fill=-999.0):
    """Build a payload shaped like a POWER response (synthetic values)."""
    values = values or {}
    parameter_block = {}
    for name in parameters:
        series = values.get(name, [1.0, 2.0, 3.0])
        parameter_block[name] = dict(zip(days, series))
    return {
        "type": "Feature",
        "properties": {"parameter": parameter_block},
        "header": {"fill_value": fill},
        "parameters": {name: {"units": f"unit-of-{name}"} for name in parameters},
    }


def fake_response(status=200, json_body=None, text=""):
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.text = text
    if json_body is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = json_body
    return response


CORE = ["PRECTOTCORR", "T2M_MAX", "T2M_MIN"]


class ValidationTests(unittest.TestCase):
    @mock.patch("services.nasa_data.requests.get")
    def test_invalid_coordinates_never_hit_the_network(self, get):
        for lat, lon in [(95, 70), (-91, 70), (30, 181), (30, -181), ("x", 70), (None, 70), (float("nan"), 70)]:
            with self.assertRaises(ValueError):
                fetch_nasa_power_data(lat, lon, "2024-01-01", "2024-01-03")
        get.assert_not_called()

    @mock.patch("services.nasa_data.requests.get")
    def test_invalid_dates_never_hit_the_network(self, get):
        for start, end in [("2024-02-01", "2024-01-01"), ("1970-01-01", "1980-01-01"),
                           ("2024-01-01", "2999-01-01"), ("not-a-date", "2024-01-01")]:
            with self.assertRaises(ValueError):
                fetch_nasa_power_data(30, 70, start, end)
        get.assert_not_called()


class ParsingTests(unittest.TestCase):
    @mock.patch("services.nasa_data.requests.get")
    def test_fill_values_become_nan_and_columns_are_renamed(self, get):
        core = synthetic_payload(CORE, values={"PRECTOTCORR": [0.0, -999.0, 4.2]})
        get.return_value = fake_response(200, core)
        df = fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03", include_soil_moisture=False)
        self.assertEqual(list(df.columns), ["precip_mm_day", "tmax_c", "tmin_c"])
        self.assertTrue(np.isnan(df["precip_mm_day"].iloc[1]))
        self.assertEqual(df["precip_mm_day"].iloc[2], 4.2)
        self.assertEqual(str(df.index[0].date()), "2024-01-01")
        self.assertEqual(df.index.name, "date")
        self.assertFalse(df.attrs["soil_moisture_available"])
        self.assertEqual(get.call_count, 1)

    @mock.patch("services.nasa_data.requests.get")
    def test_request_uses_expected_query(self, get):
        get.return_value = fake_response(200, synthetic_payload(CORE))
        fetch_nasa_power_data(31.25, 72.5, "2024-01-01", "2024-01-03", include_soil_moisture=False)
        _, kwargs = get.call_args
        query = kwargs["params"]
        self.assertEqual(query["parameters"], "PRECTOTCORR,T2M_MAX,T2M_MIN")
        self.assertEqual(query["community"], "AG")
        self.assertEqual((query["latitude"], query["longitude"]), (31.25, 72.5))
        self.assertEqual((query["start"], query["end"]), ("20240101", "20240103"))
        self.assertIsNotNone(kwargs["timeout"])

    @mock.patch("services.nasa_data.requests.get")
    def test_soil_wetness_added_when_available(self, get):
        get.side_effect = [
            fake_response(200, synthetic_payload(CORE)),
            fake_response(200, synthetic_payload(["GWETROOT"], values={"GWETROOT": [0.3, 0.4, -999.0]})),
        ]
        df = fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")
        self.assertTrue(df.attrs["soil_moisture_available"])
        self.assertIn("soil_wetness_root", df.columns)
        self.assertTrue(np.isnan(df["soil_wetness_root"].iloc[2]))

    @mock.patch("services.nasa_data.requests.get")
    def test_soil_wetness_rejected_keeps_core_data_and_warns(self, get):
        get.side_effect = [
            fake_response(200, synthetic_payload(CORE)),
            fake_response(422, {"messages": ["parameter not valid"]}),
        ]
        df = fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")
        self.assertFalse(df.attrs["soil_moisture_available"])
        self.assertNotIn("soil_wetness_root", df.columns)
        self.assertTrue(any("GWETROOT" in w for w in df.attrs["warnings"]))

    @mock.patch("services.nasa_data.requests.get")
    def test_soil_wetness_all_fill_values_is_unavailable(self, get):
        get.side_effect = [
            fake_response(200, synthetic_payload(CORE)),
            fake_response(200, synthetic_payload(["GWETROOT"], values={"GWETROOT": [-999.0] * 3})),
        ]
        df = fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")
        self.assertFalse(df.attrs["soil_moisture_available"])
        self.assertNotIn("soil_wetness_root", df.columns)

    @mock.patch("services.nasa_data.requests.get")
    def test_many_missing_days_produce_warning(self, get):
        core = synthetic_payload(CORE, values={"PRECTOTCORR": [-999.0, -999.0, 1.0]})
        get.return_value = fake_response(200, core)
        df = fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03", include_soil_moisture=False)
        self.assertTrue(any("precip_mm_day" in w for w in df.attrs["warnings"]))


class ErrorHandlingTests(unittest.TestCase):
    @mock.patch("services.nasa_data.time.sleep")
    @mock.patch("services.nasa_data.requests.get")
    def test_http_422_is_not_retried_and_reports_message(self, get, _sleep):
        get.return_value = fake_response(422, {"messages": ["latitude out of range"]})
        with self.assertRaises(NasaPowerError) as ctx:
            fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")
        self.assertIn("422", str(ctx.exception))
        self.assertIn("latitude out of range", str(ctx.exception))
        self.assertEqual(get.call_count, 1)

    @mock.patch("services.nasa_data.time.sleep")
    @mock.patch("services.nasa_data.requests.get")
    def test_timeouts_are_retried_then_reported(self, get, _sleep):
        get.side_effect = requests.exceptions.Timeout()
        with self.assertRaises(NasaPowerError) as ctx:
            fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")
        self.assertEqual(get.call_count, nasa_data.MAX_RETRIES + 1)
        self.assertIn("timed out", str(ctx.exception))

    @mock.patch("services.nasa_data.time.sleep")
    @mock.patch("services.nasa_data.requests.get")
    def test_connection_error_is_reported(self, get, _sleep):
        get.side_effect = requests.exceptions.ConnectionError("no route")
        with self.assertRaises(NasaPowerError) as ctx:
            fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")
        self.assertIn("could not connect", str(ctx.exception))

    @mock.patch("services.nasa_data.time.sleep")
    @mock.patch("services.nasa_data.requests.get")
    def test_server_error_then_success(self, get, _sleep):
        get.side_effect = [fake_response(503, text="busy"), fake_response(200, synthetic_payload(CORE))]
        df = fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03", include_soil_moisture=False)
        self.assertEqual(len(df), 3)
        self.assertEqual(get.call_count, 2)

    @mock.patch("services.nasa_data.requests.get")
    def test_invalid_json_is_reported(self, get):
        get.return_value = fake_response(200, None, text="<html>oops</html>")
        with self.assertRaises(NasaPowerError):
            fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")

    @mock.patch("services.nasa_data.requests.get")
    def test_unexpected_structure_is_reported(self, get):
        get.return_value = fake_response(200, {"something": "else"})
        with self.assertRaises(NasaPowerError):
            fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")

    @mock.patch("services.nasa_data.requests.get")
    def test_missing_core_parameter_is_reported(self, get):
        get.return_value = fake_response(200, synthetic_payload(["PRECTOTCORR", "T2M_MAX"]))
        with self.assertRaises(NasaPowerError) as ctx:
            fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")
        self.assertIn("T2M_MIN", str(ctx.exception))

    @mock.patch("services.nasa_data.requests.get")
    def test_all_fill_values_means_no_data(self, get):
        all_fill = {name: [-999.0] * 3 for name in CORE}
        get.return_value = fake_response(200, synthetic_payload(CORE, values=all_fill))
        with self.assertRaises(NasaPowerError) as ctx:
            fetch_nasa_power_data(30, 70, "2024-01-01", "2024-01-03")
        self.assertIn("no valid values", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
