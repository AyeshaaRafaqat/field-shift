# Field Shift

**Agro-Climatic Decision Support & Crop Rotation Diagnostic Platform**

Field Shift is a Python-based decision support system designed to assist farmers, agronomists, and agricultural extension practitioners in evaluating and diversifying crop rotation systems. By integrating 10 years of daily NASA POWER satellite/reanalysis climate data, MODIS NDVI vegetation canopy dynamics, and a hybrid agronomic scoring engine (rule-based water/soil constraints + Random Forest regression), Field Shift diagnoses field-level climate exposure and evaluates comparative rotation strategies.

---

## Key Capabilities

1. **10-Year Historical Climate Reanalysis (NASA POWER AG)**
   - Automatically pulls 10 years of daily agro-climatology data via the NASA POWER API:
     - `PRECTOTCORR`: Daily precipitation (mm/day)
     - `T2M_MAX` / `T2M_MIN`: Maximum and minimum 2-meter air temperatures (°C)
     - `GWETROOT`: Root-zone soil wetness (MERRA-2 dimensionless fraction, 0 = dry, 1 = saturated)
   - Computes seasonal rainfall aggregates, maximum consecutive dry-spell runs, extreme heat stress days (>35°C), frost events (<0°C), and multi-year seasonal root-zone soil moisture anomalies ($z$-scores).

2. **Satellite Vegetation Cover & Bare Soil Exposure (MODIS NDVI)**
   - Analyzes 250 m Terra/Aqua MODIS surface reflectance dynamics.
   - Calculates annual bare ground exposure days ($\text{NDVI} < 0.20$), peak canopy coverage days ($\text{NDVI} \ge 0.55$), and inter-crop fallow vulnerability.

3. **Multi-Objective Rotation Evaluation Engine**
   - **Agronomic Rules**: Validates seasonal sequencing (Rabi/Kharif/Zaid), soil texture compatibility (Sandy Loam to Clay), and irrigation source reliability (Canal, Tube Well, Rainfed).
   - **Machine Learning Surrogate**: Uses a `RandomForestRegressor` trained on crop biophysical traits, historical climate stress indices, and localized soil factors to predict climate resilience and yield stability.
   - **Customizable Priorities**: Balances Water Conservation, Soil Health (biological nitrogen fixation credits), Climate Resilience, and Economic Gross Margin.
   - **What-If Scenario Stress Testing**: Real-time sensitivity analysis under simulated climate shifts (Rainfall $\pm 30\%$, Temperature $+0.0$ to $+4.0^\circ\text{C}$).

4. **Structured Decision Cards**
   - Evaluates the farmer's **Current Baseline Rotation** alongside three tailored alternatives:
     1. **Soil Health Option** (legume-intensified, biological nitrogen fixation)
     2. **Water-Saving Option** (low evapotranspiration footprint, groundwater conservation)
     3. **Climate-Resilient Option** (buffered against terminal heat and prolonged dry spells)
   - Every rotation card provides:
     - *Why Suggested* (agronomic rationale for the specific field profile)
     - *Field Evidence* (quantitative water requirements, nitrogen credits, resilience scores)
     - *Expected Benefits* (on-farm operational gains)
     - *Operational Risks* (disease susceptibility, drainage limits, market volatility)
     - *Information Gaps* (seed variety maturity windows, tubewell water salinity EC/SAR)
     - *Agronomist Advisory Note* (actionable prompt for local extension verification)

---

## Project Structure

```
field-shift/
├── analysis/
│   ├── __init__.py
│   ├── climate_analysis.py       # Seasonal indicators, dry spells, heat days, soil anomalies
│   └── rotation_scorer.py        # Rule-based + Random Forest multi-rotation scorer
├── data/
│   ├── .gitkeep
│   └── crop_data.csv             # 9 crops with 12 agronomic attributes
├── services/
│   ├── __init__.py
│   ├── modis_data.py             # MODIS NDVI retrieval & bare ground exposure analyzer
│   └── nasa_data.py              # NASA POWER Daily Point API client with retry & caching
├── tests/
│   ├── __init__.py
│   ├── check_live_nasa_api.py    # Smoke test for live NASA POWER API connection
│   ├── test_climate_analysis.py  # 25 unit tests for climate calculations
│   ├── test_helpers.py           # 6 unit tests for input validation
│   ├── test_nasa_data.py         # 16 unit tests for NASA data parsing & mocked HTTP error states
│   └── test_rotation_scorer.py   # 8 unit tests for crop data, scoring engine, and MODIS
├── utils/
│   ├── __init__.py
│   └── helpers.py                # Coordinate validation, date utilities, data quality reports
├── .gitignore
├── app.py                        # Streamlit web application
├── README.md
└── requirements.txt
```

---

## Crop Dataset Schema (`data/crop_data.csv`)

The baseline library contains 9 key regional crops defined across 12 agronomic columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `crop_id` | string | Unique machine-readable identifier (e.g. `wheat`, `rice`, `chickpea`) |
| `crop_name` | string | Agronomic and common crop name (e.g. `Wheat (Gandum)`, `Basmati Rice`) |
| `season` | string | Primary cropping season (`Rabi`, `Kharif`, `Annual`) |
| `water_requirement_mm`| float | Total crop evapotranspiration / water demand (mm per cycle) |
| `drought_tolerance` | int (1-5) | Drought tolerance rating (1 = sensitive, 5 = highly tolerant) |
| `heat_tolerance` | int (1-5) | Extreme heat tolerance rating (1 = sensitive, 5 = tolerant) |
| `frost_tolerance` | int (1-5) | Cold/frost tolerance rating (1 = sensitive, 5 = tolerant) |
| `soil_health_score` | int (1-10) | Root biomass, residue contribution, and soil aeration rating |
| `nitrogen_fixing` | boolean | `True` for biological nitrogen-fixing legumes; `False` otherwise |
| `root_depth_m` | float | Effective root penetration depth in meters |
| `growth_days` | int | Typical duration from sowing to physiological maturity |
| `economic_return_index` | float (1-10)| Relative gross margin per hectare under standard management |

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Internet access (for live NASA POWER daily API queries)

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/AyeshaaRafaqat/field-shift.git
cd field-shift
python -m venv .venv
```

Activate the virtual environment:
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Test Suite
Confirm that all 55 unit tests pass:
```bash
python -m unittest discover -v
```

### Step 4: Verify Live NASA POWER API Connection
Run the smoke test to confirm external connectivity to `power.larc.nasa.gov`:
```bash
python tests/check_live_nasa_api.py
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Agronomic & Analytical Assumptions

1. **Dry-Day Threshold**: Default is set to $< 1.0\text{ mm/day}$ of rainfall. Days with missing observations break dry-spell sequences rather than extending them, ensuring dry-spell statistics represent conservative lower bounds.
2. **Heat-Day Threshold**: Default is set to $T_{\max} > 35.0^\circ\text{C}$, serving as an illustrative index for terminal heat stress in temperate/subtropical cereals.
3. **Soil Wetness Baseline**: A multi-year baseline for MERRA-2 root-zone soil wetness requires at least 3 historical seasons with $\ge 80\%$ data coverage. Partial seasons at download boundaries are flagged and excluded from baseline calculations.
4. **Data Spatial Resolution**: NASA POWER agro-meteorological data are gridded reanalysis/satellite estimates (~0.5° × 0.5° resolution for MERRA-2, ~1° for meteorological parameters) and reflect regional weather patterns rather than micro-climatic field sensor readings.

---

## License & Attribution

- Climate data provided courtesy of the NASA Langley Research Center POWER Project funded through the NASA Earth Science / Applied Science Program.
- Satellite greenness metrics derived from NASA LP DAAC MODIS Terra/Aqua observations.
