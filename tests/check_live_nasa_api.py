"""Live smoke test against the REAL NASA POWER API (needs internet access).

Run from the project root, for example:

    python tests/check_live_nasa_api.py
    python tests/check_live_nasa_api.py 31.45 73.14 2020-01-01 2024-12-31

Default location is a point in southern Punjab, Pakistan (30.1575, 71.5249)
and the default period is one year, to keep the check quick.

It prints what the API actually returned: columns, units, date range, missing
values, value ranges, and whether root-zone soil wetness (GWETROOT) was
available. Nothing here is simulated.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from services.nasa_data import NasaPowerError, fetch_nasa_power_data  # noqa: E402
from utils.helpers import data_quality_report  # noqa: E402


def main(argv):
    lat = float(argv[1]) if len(argv) > 1 else 30.1575
    lon = float(argv[2]) if len(argv) > 2 else 71.5249
    start = argv[3] if len(argv) > 3 else "2023-01-01"
    end = argv[4] if len(argv) > 4 else "2023-12-31"

    print(f"Requesting NASA POWER daily data for ({lat}, {lon}) from {start} to {end} ...")
    try:
        df = fetch_nasa_power_data(lat, lon, start, end)
    except (ValueError, NasaPowerError) as exc:
        print(f"FAILED: {exc}")
        return 1

    pd.set_option("display.width", 120)
    print("\nColumns / dtypes:")
    print(df.dtypes)
    print("\nUnits reported:")
    for column, unit in df.attrs["units"].items():
        print(f"  {column}: {unit}")
    print(f"\nRows: {len(df)} | {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Expected rows for the period: {(pd.Timestamp(end) - pd.Timestamp(start)).days + 1}")
    print(f"Soil wetness available: {df.attrs['soil_moisture_available']}")
    for warning in df.attrs["warnings"]:
        print(f"WARNING: {warning}")
    print("\nFirst rows:")
    print(df.head())
    print("\nData quality:")
    print(data_quality_report(df).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
