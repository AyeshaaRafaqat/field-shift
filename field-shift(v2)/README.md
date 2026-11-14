# 🌾 Field Shift: Crop Rotation Explorer

A decision-support tool for the NASA **Field Shift** challenge. It uses NASA Earth observation data, local soil
information, crop traits and farmer priorities to compare crop-rotation strategies for soil health and climate
resilience. It supports, and does not replace, farmer and agronomist judgment.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python prefetch.py               # optional, needs internet: saves data for all preset locations
streamlit run app.py
```

## What it does
1. **Field conditions:** 10 years of NASA POWER rainfall, temperature and soil wetness, optional MODIS NDVI (bare-field periods).
2. **Compare rotations:** your current rotation vs three alternatives, scored on soil health, water fit, resilience and income,
   with why / evidence / benefits / risks / information gaps for each.
3. **ML soil-moisture model:** a random forest, validated on held-out years, used to estimate soil moisture under what-if climate sliders.
4. **About & limits:** data, scoring method, limitations.

## Project layout
| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface |
| `engine/` | The logic, split by job (see below) |
| `engine/nasa_power.py` | NASA POWER download, cleaning, saved copies |
| `engine/modis.py` | MODIS NDVI download and bare-field statistics |
| `engine/indicators.py` | Daily weather to per-season indicators |
| `engine/ml.py` | Soil-moisture random forest and what-if scenarios |
| `engine/scoring.py`, `engine/explain.py` | Rotation scores and plain-language reasons |
| `engine/config.py` | All settings and thresholds in one place |
| `engine/crops.py`, `engine/common.py` | Crop table loading/validation; shared helpers |
| `crops.csv` | Crop traits (edit to match your region) |
| `CROP_DATA_SOURCES.md` | Sources for every crop value, confidence, and what is still judgment |
| `prefetch.py` | Downloads and saves data for preset locations |
| `tests/test_engine.py` | Offline tests with synthetic weather |
| `data_cache/` | Saved NASA POWER data (created on first run) |
| `PROJECT_EXPLANATION.txt` | Full write-up: what was built, how, and how to publish on GitHub |

## Tests
```bash
python tests/test_engine.py      # or: python -m pytest -q
```

## Data
NASA POWER (https://power.larc.nasa.gov) and MODIS MOD13Q1 NDVI via ORNL DAAC. Crop traits were cross-checked against FAO, ICAR and Pakistani/Indian Punjab sources (see
`CROP_DATA_SOURCES.md`); ratings and the income index are still judgment, so replace them with local figures.
