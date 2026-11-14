# Field Shift: Technical & Agronomic Documentation

Field Shift is an agro-climatic crop rotation decision support system. It evaluates traditional crop sequences against historical climate stress, soil characteristics, and irrigation access, and proposes diversified alternative rotations to improve soil organic health, save irrigation water, and stabilize farm income.

---

## 1. System Architecture & Workflow

```
+-----------------------------------------------------------------------------------+
| Farmer Inputs: District / Coordinates, Soil Texture, Water Source, Current Crops  |
+-----------------------------------------+-----------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
         [NASA POWER 10-Year API]                 [MODIS NDVI Satellite]
          Daily Rain, Tmax, Tmin,                  Annual Bare Ground Days
          Root-Zone Soil Wetness                   Canopy Coverage Cycle
                      |                                       |
                      +-------------------+-------------------+
                                          |
                                          v
                      [Climate & Ground Cover Diagnostics]
                       10-Year Rain, Dry Spells, Heat Days,
                       Frost Events, Soil Wetness Anomaly
                                          |
                                          v
    +--------------------------------------------------------------------+
    | Hybrid Agronomic Scoring Engine (analysis/rotation_scorer.py)      |
    | - Crop Dataset: 9 crops x 12 attributes (data/crop_data.csv)       |
    | - Physical Constraints: Soil drainage fit, water balance, N-credit |
    | - Machine Learning: Random Forest Regressor for Climate Resilience |
    +---------------------------------+----------------------------------+
                                      |
                                      v
    +--------------------------------------------------------------------+
    | Decision Cards Generation: Current Practice vs. 3 Alternatives     |
    | 1. Current Baseline Rotation                                       |
    | 2. Alternative 1: Soil Health & Legume Option                      |
    | 3. Alternative 2: Water-Saving Option                              |
    | 4. Alternative 3: Climate-Resilient Option                         |
    +---------------------------------+----------------------------------+
                                      |
                                      v
    +--------------------------------------------------------------------+
    | Streamlit Interface (app.py)                                       |
    | - Plain-language practical benefits, risks, and local gaps         |
    | - Agronomist advisory checklist                                    |
    | - Collapsed technical records for research review                  |
    +--------------------------------------------------------------------+
```

---

## 2. Core Modules & Responsibilities

| Module | File | Core Functionality |
| :--- | :--- | :--- |
| **User Interface** | `app.py` | Streamlined 2-step setup: field parameters and farm goals. Displays high-level climate diagnostics, side-by-side comparative rotation cards, and collapsed technical data records. |
| **NASA Weather Client** | `services/nasa_data.py` | Queries NASA POWER Agroclimatology (AG) point API for daily precipitation (`PRECTOTCORR`), temperatures (`T2M_MAX`, `T2M_MIN`), and root-zone soil wetness (`GWETROOT`). Handles timeouts, 429/5xx retries, fill-value conversions (`-999` -> `NaN`), and caching. |
| **MODIS Satellite Data** | `services/modis_data.py` | Analyzes 250 m Terra/Aqua MODIS NDVI surface reflectance to quantify annual bare soil exposure days (NDVI < 0.20), peak canopy duration (NDVI >= 0.55), and fallow evaporation vulnerability. |
| **Climate Analysis** | `analysis/climate_analysis.py` | Groups 10-year daily weather into seasonal windows (Rabi: Nov 01 - Apr 30, Kharif: May 01 - Oct 31). Computes seasonal rainfall, maximum dry-spell runs, heat stress days (>35°C), and multi-year root-zone wetness anomalies (z-scores). |
| **Rotation Scorer** | `analysis/rotation_scorer.py` | Combines rule-based physical constraints (soil texture suitability matrix, irrigation delivery factors, biological N-credits) with a Random Forest regression model to score and select optimal rotations. |
| **Crop Dataset** | `data/crop_data.csv` | 9 regional field crops defined across 12 agronomic parameters. |
| **Helpers & Validation** | `utils/helpers.py` | Coordinate boundary validation, date range verification, and data quality audits. |

---

## 3. Data Specification: `data/crop_data.csv`

The system currently reads 9 field crops across 12 agronomic columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `crop_id` | string | Unique machine identifier (`wheat`, `rice`, `cotton`, `maize`, `sugarcane`, `chickpea`, `mungbean`, `mustard`, `lentil`) |
| `crop_name` | string | Full crop name (e.g. `Wheat (Gandum)`, `Basmati Rice`, `Chickpea (Desi/Kabuli)`) |
| `season` | string | Primary cropping cycle (`Rabi`, `Kharif`, `Annual`) |
| `water_requirement_mm` | float | Total crop water consumption / evapotranspiration (mm/cycle) |
| `drought_tolerance` | int (1-5) | Drought tolerance score (1 = sensitive, 5 = hardy) |
| `heat_tolerance` | int (1-5) | Extreme heat tolerance score (1 = sensitive, 5 = hardy) |
| `frost_tolerance` | int (1-5) | Cold tolerance score (1 = sensitive, 5 = hardy) |
| `soil_health_score` | int (1-10) | Root biomass, residue contribution, and soil aeration rating |
| `nitrogen_fixing` | boolean | `True` for biological nitrogen-fixing legumes; `False` otherwise |
| `root_depth_m` | float | Effective root penetration depth in meters |
| `growth_days` | int | Typical duration from sowing to physiological maturity |
| `economic_return_index`| float (1-10)| Relative gross margin per hectare under standard local management |

When your teammate completes their custom CSV, they can replace `data/crop_data.csv` keeping these exact 12 column headers.

---

## 4. Scoring & Decision Methodology

### 4.1. Rule-Based Physical Constraints
- **Soil Texture Compatibility**: Adjusts crop viability based on soil drainage and aeration. For example, Chickpea achieves optimal root health in Sandy Loam or Loam (0.90 - 0.95), but receives penalties in heavy Clay (0.30) due to collar rot and waterlogging vulnerability.
- **Irrigation Availability**: Evaluates total rotation water demand against the reliable delivery capacity of the chosen water source (Perennial Canal, Non-perennial + Tube Well, Fresh Groundwater, Brackish Groundwater, or Rainfed Barani).
- **Biological Nitrogen Credit**: Rotations containing legumes (Chickpea, Mungbean, Lentil) receive biological nitrogen credits (~25 - 35 kg N/ha), reducing synthetic urea dependency.

### 4.2. Machine Learning Surrogate (Random Forest)
A `RandomForestRegressor` trained on biophysical crop traits, historical weather stress, and scenario deltas predicts:
- **Climate Resilience Score (0-100)**: Buffering against heat spikes (>35°C) and extended dry periods.
- **Yield Stability Score (0-100)**: Consistency of production across wet and dry years.

### 4.3. Multi-Criteria Composite Scoring
Farmer goal weights ($w_w$ for water, $w_s$ for soil health, $w_r$ for resilience, $w_i$ for income) normalize the overall fit score:

$$\text{Composite Score} = \frac{w_w \cdot S_{\text{water}} + w_s \cdot S_{\text{soil}} + w_r \cdot S_{\text{resilience}} + w_i \cdot S_{\text{income}}}{w_w + w_s + w_r + w_i}$$

---

## 5. Decision Card Structure

For every evaluation, the app outputs four distinct cards:
1. **Current Baseline Rotation**: Evaluates the farm's existing practice against localized weather and soil realities.
2. **Alternative 1: Soil Health & Pulse Option**: Introduces a nitrogen-fixing legume to break pest cycles, replenish organic matter, and reduce chemical urea costs.
3. **Alternative 2: Water-Saving Option**: Selects lower-evapotranspiration crops to reduce groundwater pumping and diesel/electricity expenses.
4. **Alternative 3: Climate-Resilient Option**: Maximizes root depth and thermal tolerance to buffer against unpredictable weather extremes.

Each card explicitly provides:
- **Why Suggested**: Plain-language agronomic rationale tailored to the selected soil and water source.
- **Field Evidence**: Quantitative water requirements, nitrogen credits, and resilience metrics.
- **Expected Benefits**: Tangible on-farm gains (e.g. soil aeration, weed cycle breakage, diesel savings).
- **Operational Risks**: Potential constraints (e.g. drainage limits, unseasonal rain at pod maturity).
- **What to Confirm Locally**: Certified seed availability, localized sowing dates, and water quality testing.
- **Agronomist Advisory Note**: Actionable prompt reminding the farmer to verify cultivars with local extension officers.

---

## 6. Verification & Test Coverage

The platform contains 55 automated unit tests in `tests/`:
- `tests/test_nasa_data.py`: Mocked API responses, timeout handling, retry logic, coordinate/date validation, and fill-value parsing.
- `tests/test_climate_analysis.py`: Dry-spell run calculation, heat days, frost days, season wrapping across New Year, and soil wetness anomaly baselines.
- `tests/test_rotation_scorer.py`: Crop CSV integrity, single-rotation calculations, legume health boosts, and 4-card comparative generation.
- `tests/check_live_nasa_api.py`: Live integration test verifying live connectivity to NASA servers.
