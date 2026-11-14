"""Run once with internet:  python prefetch.py
Downloads NASA POWER data for every preset location into data_cache/ so the app starts instantly
and works offline (and on a hosted demo) afterwards."""
import engine as E

for name, (lat, lon) in E.SITES.items():
    try:
        _, src = E.fetch_power_cached(lat, lon)
        print(f"{name:16s} OK ({src})")
    except (E.DataFetchError, ValueError) as e:
        print(f"{name:16s} FAILED: {e}")
