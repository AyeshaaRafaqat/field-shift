# Field Shift: Comprehensive Repository Audit (Task 1)

Date: Saturday 14 Nov 2026, 14:30 PKT  
Auditor: Engineering Tech Lead & Story Head (astrooLogic, Space Apps 2026)  
Target: Real codebase state before executing Master Prompt workstreams.

---

## 1. Actual Module Map and Data Flow

```
User (Browser)
      │
      ▼
   app.py (Streamlit single-page script)
      ├──► services/nasa_data.py (fetch_nasa_power_data) ──► HTTP GET power.larc.nasa.gov
      ├──► services/modis_data.py (fetch_modis_ndvi_data) ──► In-memory heuristic simulation
      ├──► analysis/climate_analysis.py (analyse_climate) ──► Pure pandas/numpy seasonal indicators
      └──► analysis/rotation_scorer.py (RotationScorer) ──► In-memory RandomForestRegressor
                 └── reads data/crop_data.csv (9 crops x 12 columns)
```

### Verified Module Roles:
1. `app.py`: Streamlit frontend. Configures inputs (location, soil, water, current rotation, sliders). Calls `fetch_nasa_power_data` at runtime, computes seasonal indicators, calls `fetch_modis_ndvi_data`, and runs `RotationScorer.compare_rotations()`.
2. `services/nasa_data.py`: NASA POWER daily point API client. Issues two HTTP requests: one for core (`PRECTOTCORR`, `T2M_MAX`, `T2M_MIN`) and one for `GWETROOT`. Implements retry logic, validation, fill-value mapping (`-999` $\to$ `NaN`), and caching via Streamlit in `app.py`.
3. `services/modis_data.py`: Contains `fetch_modis_ndvi_data` and `summarize_bare_soil_exposure`. **Finding:** Currently generates a synthetic agro-climatic double-cropping curve using mathematical sinusoidal/exponential formulas rather than querying real satellite rasters/time-series.
4. `analysis/climate_analysis.py`: Pure function library. Calculates seasonal precipitation totals, longest dry spell runs ($< 1\text{ mm}$), heat days ($> 35^\circ\text{C}$), frost days ($< 0^\circ\text{C}$), and root-zone soil wetness multi-year baseline and anomaly $z$-scores.
5. `analysis/rotation_scorer.py`: Hybrid scoring engine. Initializes a 40-tree `RandomForestRegressor` trained on 450 synthetic variations generated on the fly. Applies rule-based multipliers for soil suitability and water source delivery, outputs 4 cards: `current`, `soil_health`, `water_saving`, `climate_resilient`.
6. `data/crop_data.csv`: CSV containing 9 crops and 12 columns (`crop_id`, `crop_name`, `season`, `water_requirement_mm`, `drought_tolerance`, `heat_tolerance`, `frost_tolerance`, `soil_health_score`, `nitrogen_fixing`, `root_depth_m`, `growth_days`, `economic_return_index`).
7. `utils/helpers.py`: Validation functions for coordinates, date conversions, date ranges, and dataframe quality profiling.

---

## 2. Runtime Network Calls Audit

| File | Line | Network Target | Trigger / Condition | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| `services/nasa_data.py` | 118 | `https://power.larc.nasa.gov/api/temporal/daily/point` | Called on app startup or button press when session state or cache is cold. | **CRITICAL (P0)**: Live call failure or NASA timeout kills judging demo. |
| `app.py` | 150 | `load_climate_data(lat, lon)` | Indirectly invokes `services/nasa_data.py`. | **CRITICAL (P0)**: Judged demo violates Rule 4. |

*Note: No other runtime network calls exist (no telemetry, no external font CDNs, no LLM calls).*

---

## 3. Hard-Coded Values and Magic Numbers Audit

| File | Line(s) | Hard-Coded Value / Parameter | Agronomic / Engineering Context | Remediation Plan |
| :--- | :--- | :--- | :--- | :--- |
| `services/nasa_data.py` | 42-43 | `DEFAULT_START_DATE="2016-01-01"`, `DEFAULT_END_DATE="2025-12-31"` | 10-year period assumption | Move to `config/params.yaml`; extend to 2001–2025. |
| `services/nasa_data.py` | 47-49 | `REQUEST_TIMEOUT=(10, 90)`, `MAX_RETRIES=2`, `RETRY_BACKOFF_SECONDS=2` | Network resilience settings | Move to network config. |
| `analysis/climate_analysis.py` | 49-51 | `DEFAULT_SEASONS={"Rabi": ("11-01", "04-30"), "Kharif": ("05-01", "10-31")}` | Fixed seasonal dates | Move to `config/crops.yaml` / `params.yaml`. |
| `analysis/climate_analysis.py` | 54-58 | `DRY_DAY=1.0 mm`, `HEAT=35.0 C`, `FROST=0.0 C`, `COVERAGE=0.8`, `BASELINE_YEARS=3` | Climate thresholds | Move to `config/params.yaml`. |
| `services/modis_data.py` | 22-23 | `BARE_SOIL_NDVI=0.20`, `PEAK_CANOPY_NDVI=0.55` | Remote sensing thresholds | Move to `config/params.yaml`. |
| `services/modis_data.py` | 49-70 | `0.14`, `0.48`, `0.52`, `-0.05`, `0.008` | Synthetic curves for NDVI | Replace with real processed MODIS data. |
| `analysis/rotation_scorer.py` | 23-53 | `SOIL_SUITABILITY` dictionary | Soil texture fit matrix | Move to `config/params.yaml` with citations. |
| `analysis/rotation_scorer.py` | 56-62 | `WATER_SUPPLY_FACTORS` dictionary | Irrigation supply delivery factor | Move to `config/params.yaml`. |
| `analysis/rotation_scorer.py` | 65-115 | `ROTATION_CATALOG` dictionary | Crop sequences | Move to `config/strategies.yaml`. |
| `analysis/rotation_scorer.py` | 150-184 | Synthetic generator bounds (`400-2000`, `1.5-4.8`, etc.) | Training distributions for RF | Replace with honest emulator or ML on real NDVI. |
| `analysis/rotation_scorer.py` | 224-239 | `1900.0 mm`, `900 mm`, `1.2 m`, `1000 mm` | Scoring penalties and bonuses | Move to `config/params.yaml`. |
| `analysis/rotation_scorer.py` | 262-267 | Chickpea `35 kg`, Mungbean `28 kg`, Lentil `25 kg` | Biological Nitrogen credits | Move to `data/crop_data.csv` `n_fix_kg_ha`. |
| `app.py` | 98-132 | Hard-coded district coordinates and soil types | Punjab preset locations | Move to `config/sites.yaml`. |

---

## 4. Test Suite and Coverage Audit

### Pytest Execution (`python -m pytest -q`):
- **Pass Count**: 55 passed in 9.05 seconds.
- **Failures / Errors**: 0.
- **Backend Coverage**: **97%** overall:
  - `analysis/climate_analysis.py`: 98%
  - `analysis/rotation_scorer.py`: 94%
  - `services/modis_data.py`: 88%
  - `services/nasa_data.py`: 97%
  - `utils/helpers.py`: 97%

### What Is Currently Untested:
1. `app.py`: No Streamlit AppTest coverage (`streamlit.testing.v1.AppTest`).
2. Monte Carlo parameter sampling, rank stability, and tornado sensitivity (engine does not exist).
3. Penman-Monteith FAO-56 ET0 and net irrigation demand (functions do not exist).
4. Sowing window overlap and harvest-timing conflict assertions.
5. Multipage navigation and session state persistence.
6. Offline mode startup verification (app currently crashes if network is blocked on first visit).

---

## 5. Section 5 Loophole & Gap Audit Table

| # | Master Prompt Gap | Severity | Status in Repo | Exact File & Line / Finding |
| :--- | :--- | :---: | :---: | :--- |
| **1** | **Demo format risk** (Must be 7-slide deck or $\le 30$ s video) | **P0** | **CONFIRMED GAP** | No 7-slide deck or 30 s script exists in the repo yet. |
| **2** | **Submission not drafted** | **P0** | **CONFIRMED GAP** | `docs/SUBMISSION.md` does not exist yet. |
| **3** | **No public deployed app** | **P0** | **CONFIRMED GAP** | App is only running on local machine (`http://localhost:8501`). Needs Streamlit Community Cloud. |
| **4** | **Runtime dependency on live NASA APIs** | **P0** | **CONFIRMED GAP** | `app.py`:150 calls `fetch_nasa_power_data` at runtime. No offline pre-fetched data exists. |
| **5** | **Unverified crop table** | **P0** | **CONFIRMED GAP** | `data/crop_data.csv` lacks `source` column; missing schema validator on load. |
| **6** | **Circular ML risk** | **P1** | **CONFIRMED GAP** | `analysis/rotation_scorer.py`:137-178 trains Random Forest on synthetic rule equations in-memory. |
| **7** | **No uncertainty shown** | **P1** | **CONFIRMED GAP** | Only point scores shown; no Monte Carlo, no interannual spread, no win-rate. |
| **8** | **Satellite vs ground not explained** | **P1** | **CONFIRMED GAP** | No Data & Method page; GWETROOT not labeled as model-derived. |
| **9** | **Weak harmonization story** | **P1** | **CONFIRMED GAP** | No common daily site-level table; resolution table absent. |
| **10** | **Baseline is single fixed input** | **P1** | **PARTIAL** | Farmer can select current rotation, but alternatives do not show full win rates. |
| **11** | **Bare-field calendar chart** | **P1** | **PARTIAL** | Metric pill exists, but Plotly day-of-year bare probability chart is absent. |
| **12** | **Seasons of vulnerability chart** | **P1** | **CONFIRMED GAP** | No timeline overlaying heat-stress windows onto crop growth stages. |
| **13** | **Scenario toggle & Archetype years** | **P1** | **PARTIAL** | Slider exists in expander, but KMeans historical archetype clustering is absent. |
| **14** | **Farmer priorities incomplete** | **P1** | **PARTIAL** | Sliders exist, but presets ("Save water", "Build soil", etc.) and constraint toggles are absent. |
| **15** | **Glossary** (EN + UR) | **P1** | **CONFIRMED GAP** | No glossary panel in UI. |
| **16** | **Persistent banner** (supports not replaces) | **P1** | **PARTIAL** | Disclaimer is in tiny footer caption, not a persistent banner. |
| **17** | **UI layout mismatch** (Mockup Section 8B) | **P2** | **CONFIRMED GAP** | UI is single-page, lacks left WORKSPACE sidebar (Overview, My Farm, NASA Insights, Crop Rotations, Reports). |
| **18** | **Docs broken / incomplete** | **P2** | **PARTIAL** | `README.md` exists but lacks FAO-56 equations, architecture diagram, and `METHOD.md`. |
| **19** | **Test gaps** (AppTest, ET0, schema) | **P2** | **CONFIRMED GAP** | 55 tests pass, but AppTest, ET0, and schema validation tests are missing. |
| **20** | **Localization** (Urdu RTL toggle) | **P2** | **CONFIRMED GAP** | English only; no Urdu strings or RTL support. |
| **21** | **Generic explanations** | **P2** | **PARTIAL** | Templates generate text, but numbers are not strictly regex-tested against metrics. |

---

## 6. Bigger Architectural Problems Identified

1. **Simulated MODIS Data**: `services/modis_data.py` does not read satellite data; it uses mathematical equations. If a judge checks the code, this will destroy credibility under "Best Use of Data" and "Validity". Real pre-processed MODIS NDVI timeseries must be committed.
2. **Missing FAO-56 Physical Hydrology**: The current water metric is just a static sum of crop table numbers rather than daily Penman-Monteith reference ET0 and crop coefficient ($K_c$) water balance against real precipitation.
3. **App Architecture**: Currently a monolithic `app.py`. Must be refactored into the 5-item WORKSPACE layout (Overview, My Farm, NASA Insights, Crop Rotations, Reports) per Section 8B with `ui/components.py`.
