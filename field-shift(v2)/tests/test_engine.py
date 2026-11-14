"""Offline tests: synthetic weather and mocked network calls, so no internet is needed.
Run with:  python -m pytest -q     or     python tests/test_engine.py"""
import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd

import engine as E
from engine import nasa_power

CROPS_PATH = os.path.join(os.path.dirname(__file__), "..", "crops.csv")
CROPS = E.load_crops(CROPS_PATH)
CUR = ("Rice", "Wheat")


def synthetic_weather():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2014-05-01", "2024-04-30")
    doy = idx.dayofyear.values
    monsoon = (idx.month >= 7) & (idx.month <= 9)
    rain = pd.Series(np.where(rng.random(len(idx)) < np.where(monsoon, .45, .08), rng.gamma(2, 6, len(idx)), 0), index=idx)
    tmax = 25 + 14 * np.sin((doy - 100) / 365 * 2 * np.pi) + rng.normal(0, 2, len(idx))
    wet = (0.2 + 0.0025 * rain.rolling(30, min_periods=1).sum() - 0.004 * (tmax - 25)).clip(0.05, 0.95)
    return pd.DataFrame({"PRECTOTCORR": rain, "T2M_MAX": tmax, "T2M_MIN": tmax - 12, "GWETROOT": wet}, index=idx)


DF = synthetic_weather()
THR = E.dry_soil_threshold(DF)


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text
        self.ok = status < 400

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def power_payload(with_evland=True, n=10):
    dates = pd.date_range("2020-01-01", periods=n).strftime("%Y%m%d")
    p = {"PRECTOTCORR": dict(zip(dates, [0.0] * n)), "T2M_MAX": dict(zip(dates, [30.0] * n)),
         "T2M_MIN": dict(zip(dates, [18.0] * n)), "GWETROOT": dict(zip(dates, [0.4] * n))}
    if with_evland:
        p["EVLAND"] = dict(zip(dates, [2.0] * n))
    return {"properties": {"parameter": p}}


# ------------------------------------------------------------------ indicators and scoring
def test_longest_run():
    assert E.longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    assert E.longest_run([0, 0]) == 0 and E.longest_run([1, 1, 1]) == 3


def test_season_table_shape():
    t = E.season_table(DF, wet_thr=THR)
    assert len(t) == 20 and set(t.index.get_level_values("season")) == {"Kharif", "Rabi"}
    assert (t.rain >= 0).all() and (t.dry_spell >= 0).all()


def test_scores_in_range_and_current_first():
    t = E.season_table(DF, wet_thr=THR)
    opts = E.evaluate_all(CROPS, t, CUR, 1.0, [7, 6, 5, 4])
    assert len(opts) == 20
    for o in opts:
        for key in ("soil", "water", "resil", "income", "total"):
            assert 0 <= o[key] <= 10, (o.label, key, o[key])
        assert o.lo <= o.total <= o.hi
    picks = E.pick_strategies(opts, CUR, E.WATER_LIMIT["Medium"])
    assert picks[0][1].key == CUR and 2 <= len(picks) <= 4
    assert len({o.key for _, o in picks}) == len(picks)          # no duplicates


def test_dict_style_access_still_works():
    t = E.season_table(DF, wet_thr=THR)
    o = E.evaluate("Maize", "Wheat", CUR, CROPS, t, 1.0, [1, 1, 1, 1])
    assert o["total"] == o.total and o["label"] == "Maize → Wheat"


def test_bad_weights_rejected_and_zero_weights_fallback():
    t = E.season_table(DF, wet_thr=THR)
    assert E.evaluate("Maize", "Wheat", CUR, CROPS, t, 1.0, [0, 0, 0, 0]).total > 0
    for bad in ([1, 2, 3], [-1, 1, 1, 1]):
        try:
            E.evaluate("Maize", "Wheat", CUR, CROPS, t, 1.0, bad)
            raise AssertionError("should have raised")
        except ValueError:
            pass


def test_less_water_option_needs_less_irrigation():
    t = E.season_table(DF, wet_thr=THR)
    rice_wheat = E.evaluate("Rice", "Wheat", CUR, CROPS, t, 1.0, [1, 1, 1, 1])
    mung_chick = E.evaluate("Mungbean", "Chickpea", CUR, CROPS, t, 1.0, [1, 1, 1, 1])
    assert mung_chick.irr < rice_wheat.irr


def test_config_validation():
    try:
        E.ScoringConfig(w_heat=0.9)
        raise AssertionError("weights not summing to 1 should be rejected")
    except ValueError:
        pass


def test_explanations_run_for_every_option():
    t = E.season_table(DF, wet_thr=THR)
    opts = E.evaluate_all(CROPS, t, CUR, 1.0, [1, 1, 1, 1])
    cur = next(o for o in opts if o.key == CUR)
    for o in opts:
        assert E.why(o, cur) and E.benefits(o) and E.risks(o, "Loam", 900)


# ------------------------------------------------------------------ ML
def test_ml_model_learns_and_scenarios_move_the_right_way():
    model, rep = E.train_soil_model(DF)
    assert rep.r2 > 0.7 and rep.r2 > rep.baseline_r2          # beats a plain monthly average
    drier = E.scenario_wetness(DF, model, 0.7, 0.0).mean()
    base = E.scenario_wetness(DF, model, 1.0, 0.0).mean()
    hotter = E.scenario_wetness(DF, model, 1.0, 3.0).mean()
    assert drier < base and hotter < base
    assert np.isclose(base, DF.GWETROOT.mean())               # no change when scenario is neutral


def test_extrapolation_share():
    _, rep = E.train_soil_model(DF)
    assert E.extrapolation_share(DF, rep, 1.0, 0.0) == 0.0
    assert E.extrapolation_share(DF, rep, 1.0, 6.0) > 0.0


# ------------------------------------------------------------------ crops table
def test_crop_validation_catches_problems():
    good = pd.read_csv(CROPS_PATH)
    assert E.validate_crops(good) == []
    bad = good.copy()
    bad.loc[0, "season"] = "Summer"
    bad.loc[1, "heat_tol"] = 9
    bad.loc[2, "n_fixer"] = 2
    problems = " | ".join(E.validate_crops(bad))
    assert "season" in problems and "heat_tol" in problems and "n_fixer" in problems
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bad.csv")
        bad.to_csv(path, index=False)
        try:
            E.load_crops(path)
            raise AssertionError("load_crops should reject a bad table")
        except ValueError:
            pass


# ------------------------------------------------------------------ NASA POWER (mocked network)
def test_parse_and_clean_power():
    payload = power_payload()
    payload["properties"]["parameter"]["T2M_MAX"]["20200105"] = -999        # gap in a smooth variable
    payload["properties"]["parameter"]["PRECTOTCORR"]["20200105"] = -999    # gap in rainfall
    df = nasa_power.clean_power(nasa_power.parse_power_json(payload))
    assert df.loc["2020-01-05", "T2M_MAX"] == 30.0                           # interpolated
    assert np.isnan(df.loc["2020-01-05", "PRECTOTCORR"])                     # rain is never invented
    try:
        nasa_power.clean_power(pd.DataFrame({"PRECTOTCORR": [0.0]}))
        raise AssertionError("missing columns should be rejected")
    except E.DataFetchError:
        pass


def test_fetch_power_falls_back_without_evland():
    session = mock.Mock()
    session.get.side_effect = [FakeResponse(422, text="bad parameter"), FakeResponse(200, power_payload(False))]
    df = E.fetch_power(31.5, 74.3, session=session)
    assert "EVLAND" not in df.columns and session.get.call_count == 2


def test_fetch_power_server_error_is_reported_clearly():
    session = mock.Mock()
    session.get.return_value = FakeResponse(500, text="server on fire")
    try:
        E.fetch_power(31.5, 74.3, session=session)
        raise AssertionError("should raise")
    except E.DataFetchError as e:
        assert "500" in str(e)
    assert session.get.call_count == 1                                       # no pointless second attempt


def test_fetch_power_rejects_bad_coordinates():
    try:
        E.fetch_power(95, 0, session=mock.Mock())
        raise AssertionError("should raise")
    except ValueError:
        pass


def test_cache_round_trip():
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.object(nasa_power, "fetch_power", return_value=DF.head(50)) as fetch:
            _, first = E.fetch_power_cached(31.5, 74.3, cache_dir=d)
            df, second = E.fetch_power_cached(31.5, 74.3, cache_dir=d)
        assert first == "live NASA POWER download" and second == "saved copy"
        assert fetch.call_count == 1 and len(df) == 50 and not [f for f in os.listdir(d) if f.endswith(".tmp")]


# ------------------------------------------------------------------ MODIS
def test_bare_field_summary():
    ndvi = pd.Series([0.1, 0.1, 0.1, 0.6, 0.7, 0.1, 0.6, 0.6], index=pd.date_range("2022-01-01", periods=8, freq="16D"))
    s = E.bare_field_summary(ndvi)
    assert s.longest_sparse_days == 3 * 16 and np.isclose(s.sparse_days_per_year, 4 / 8 * 365)
    empty = E.bare_field_summary(pd.Series(dtype=float))
    assert empty.sparse_days_per_year == 0 and empty.n_obs == 0


def test_fetch_ndvi_parses_and_drops_fill_values():
    dates = {"dates": [{"modis_date": "A2022001", "calendar_date": "2022-01-01"},
                       {"modis_date": "A2022017", "calendar_date": "2022-01-17"},
                       {"modis_date": "A2019001", "calendar_date": "2019-01-01"}]}
    subset = {"subset": [{"calendar_date": "2022-01-01", "data": [4000]},
                         {"calendar_date": "2022-01-17", "data": [-3000]}]}
    session = mock.Mock()
    session.get.side_effect = [FakeResponse(200, dates), FakeResponse(200, subset)]
    series = E.fetch_ndvi(31.5, 74.3, 2022, 2023, session=session)
    assert len(series) == 1 and np.isclose(series.iloc[0], 0.4)


if __name__ == "__main__":
    passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            passed += 1
            print("PASS", name)
    print(f"\n{passed} tests passed")
