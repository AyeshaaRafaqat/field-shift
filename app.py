"""Field Shift - Agro-Climatic Decision Support System.

A field-level agronomic evaluation platform that combines 10 years of NASA POWER
satellite/reanalysis climate data, MODIS NDVI vegetation greenness dynamics, and
a hybrid rule-based + Random Forest scoring engine to help farmers and agronomists
evaluate and compare crop rotation alternatives against climatic and water stress.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from analysis.climate_analysis import DEFAULT_SEASONS, analyse_climate
from analysis.rotation_scorer import (
    ROTATION_CATALOG,
    RotationScoreCard,
    RotationScorer,
)
from services.modis_data import (
    fetch_modis_ndvi_data,
    summarize_bare_soil_exposure,
)
from services.nasa_data import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    NasaPowerError,
    fetch_nasa_power_data,
)
from utils.helpers import (
    POWER_DAILY_EARLIEST_DATE,
    data_quality_report,
    to_date,
    validate_coordinates,
)

# -----------------------------------------------------------------------------
# Page Configuration & Agricultural Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Field Shift | Agro-Climatic Crop Rotation Diagnostic",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom agronomic CSS theme (earthy green, harvest gold, clean borders)
st.markdown(
    """
    <style>
    :root {
        --ag-primary: #1B4D3E;
        --ag-secondary: #2D6A4F;
        --ag-accent: #C88A35;
        --ag-bg-light: #F7FAF7;
        --ag-border: #D8E2DC;
    }
    .main-header {
        background: linear-gradient(135deg, #1B4D3E 0%, #2D6A4F 100%);
        padding: 24px 28px;
        border-radius: 8px;
        color: #FFFFFF;
        margin-bottom: 24px;
        border-left: 6px solid #C88A35;
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #E8F5E9;
        font-size: 1.0rem;
        margin-bottom: 0;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-rabi { background-color: #E3F2FD; color: #0D47A1; }
    .badge-kharif { background-color: #FFF3E0; color: #E65100; }
    .badge-legume { background-color: #E8F5E9; color: #1B5E20; }
    .badge-score { background-color: #1B4D3E; color: #FFFFFF; }

    .rotation-card {
        background-color: #FFFFFF;
        border: 1px solid #D8E2DC;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .rotation-card-highlight {
        border: 2px solid #2D6A4F;
        background-color: #FBFCFB;
    }
    .card-title {
        color: #1B4D3E;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .card-category {
        color: #6C757D;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 12px;
    }
    .score-box {
        background: #F4F8F5;
        border: 1px solid #D2E4D6;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
        margin-bottom: 12px;
    }
    .score-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1B4D3E;
    }
    .score-label {
        font-size: 0.78rem;
        color: #495057;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .advisory-box {
        background-color: #FFF9E6;
        border-left: 4px solid #C88A35;
        padding: 10px 14px;
        font-size: 0.85rem;
        color: #5D4037;
        border-radius: 4px;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Location Presets & Catalogs
# -----------------------------------------------------------------------------
LOCATION_PRESETS = {
    "Multan (Southern Punjab - Cotton-Wheat Belt)": {
        "lat": 30.1575, "lon": 71.5249, "soil": "Sandy Loam", "water": "Canal (Non-perennial) + Tube Well",
        "rotation": "cotton_wheat",
    },
    "Faisalabad (Central Punjab - Mixed Cropping)": {
        "lat": 31.4504, "lon": 73.1350, "soil": "Loam", "water": "Canal Irrigation (Perennial)",
        "rotation": "maize_wheat",
    },
    "Gujranwala (Upper Punjab - Rice-Wheat Belt)": {
        "lat": 32.1877, "lon": 74.1945, "soil": "Clay Loam", "water": "Canal Irrigation (Perennial)",
        "rotation": "rice_wheat",
    },
    "Bahawalpur (Arid / Southern Margin)": {
        "lat": 29.3544, "lon": 71.6911, "soil": "Sandy Loam", "water": "Tube Well (Fresh Groundwater)",
        "rotation": "cotton_wheat",
    },
    "Rahim Yar Khan (Southern Canal Command)": {
        "lat": 28.4212, "lon": 70.2989, "soil": "Loam", "water": "Canal (Non-perennial) + Tube Well",
        "rotation": "sugarcane_wheat",
    },
    "Sargodha (Citrus & Mixed Grain Tract)": {
        "lat": 32.0836, "lon": 72.6711, "soil": "Loam", "water": "Canal Irrigation (Perennial)",
        "rotation": "maize_wheat",
    },
    "Sahiwal (Intensive Maize-Potato-Wheat Tract)": {
        "lat": 30.6682, "lon": 73.1114, "soil": "Silt Loam", "water": "Canal Irrigation (Perennial)",
        "rotation": "maize_wheat",
    },
    "Custom Coordinates": {
        "lat": 30.1575, "lon": 71.5249, "soil": "Loam", "water": "Canal Irrigation (Perennial)",
        "rotation": "cotton_wheat",
    },
}

SOIL_OPTIONS = ["Sandy Loam", "Loam", "Clay Loam", "Silt Loam", "Sandy", "Clay"]
WATER_OPTIONS = [
    "Canal Irrigation (Perennial)",
    "Canal (Non-perennial) + Tube Well",
    "Tube Well (Fresh Groundwater)",
    "Tube Well (Marginal / Saline)",
    "Rainfed (Barani)",
]
CURRENT_ROTATION_OPTIONS = {
    "rice_wheat": "Basmati Rice → Wheat",
    "cotton_wheat": "Cotton → Wheat",
    "maize_wheat": "Grain Maize → Wheat",
    "sugarcane_wheat": "Sugarcane → Wheat",
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_climate(latitude: float, longitude: float, start_iso: str, end_iso: str) -> pd.DataFrame:
    """Fetch and cache NASA POWER 10-year daily climate data."""
    return fetch_nasa_power_data(latitude, longitude, start_iso, end_iso)


@st.cache_resource
def get_scorer() -> RotationScorer:
    """Initialize and cache the crop rotation scoring engine."""
    return RotationScorer()


# -----------------------------------------------------------------------------
# Sidebar: Farmer & Field Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚜 Field & Management Profile")
    st.caption("Configure localized agro-climatic and on-farm conditions.")

    preset_name = st.selectbox(
        "Select Location / Agro-Ecological Zone",
        options=list(LOCATION_PRESETS.keys()),
        index=0,
    )
    preset_data = LOCATION_PRESETS[preset_name]

    if preset_name == "Custom Coordinates":
        col_c1, col_c2 = st.columns(2)
        target_lat = col_c1.number_input("Latitude (°N)", -90.0, 90.0, 30.1575, 0.01, format="%.4f")
        target_lon = col_c2.number_input("Longitude (°E)", -180.0, 180.0, 71.5249, 0.01, format="%.4f")
    else:
        target_lat = preset_data["lat"]
        target_lon = preset_data["lon"]
        st.info(f"Coordinates: **{target_lat:.4f}° N, {target_lon:.4f}° E**")

    # Soil Texture
    soil_idx = SOIL_OPTIONS.index(preset_data["soil"]) if preset_data["soil"] in SOIL_OPTIONS else 1
    selected_soil = st.selectbox("Soil Texture Class", options=SOIL_OPTIONS, index=soil_idx)

    # Water Availability
    water_idx = WATER_OPTIONS.index(preset_data["water"]) if preset_data["water"] in WATER_OPTIONS else 0
    selected_water = st.selectbox("Primary Irrigation & Water Source", options=WATER_OPTIONS, index=water_idx)

    # Current Rotation
    rot_keys = list(CURRENT_ROTATION_OPTIONS.keys())
    rot_default_idx = rot_keys.index(preset_data["rotation"]) if preset_data["rotation"] in rot_keys else 0
    selected_current_rot = st.selectbox(
        "Current Cropping Pattern",
        options=rot_keys,
        format_func=lambda k: CURRENT_ROTATION_OPTIONS[k],
        index=rot_default_idx,
    )

    st.markdown("---")
    st.markdown("### ⚖️ Farmer Priority Weights")
    st.caption("Balance trade-offs between conservation and farm revenue.")
    w_water = st.slider("💧 Water Conservation", 0.0, 3.0, 1.5, 0.5)
    w_soil = st.slider("🌱 Soil Health & Nitrogen", 0.0, 3.0, 2.0, 0.5)
    w_resil = st.slider("🛡️ Climate Stress Resilience", 0.0, 3.0, 1.5, 0.5)
    w_income = st.slider("💰 Economic Net Return", 0.0, 3.0, 1.5, 0.5)

    st.markdown("---")
    st.markdown("### 🌡️ What-If Climate Sensitivity")
    st.caption("Stress-test rotation performance under climate shifts.")
    scenario_rain_pct = st.slider("Rainfall Scenario Shift (%)", -30, 30, 0, 5)
    scenario_temp_c = st.slider("Temperature Rise Scenario (°C)", 0.0, 4.0, 0.0, 0.5)

    st.markdown("---")
    st.markdown("### 🛰️ Satellite Observations")
    enable_modis = st.checkbox(
        "Include MODIS NDVI Satellite Greenness",
        value=True,
        help="Loads satellite surface reflectance to measure annual bare soil exposure days and fallow duration.",
    )

    fetch_button = st.button("🚀 Evaluate Field & Climate", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>Field Shift — Agro-Climatic Rotation Diagnostic</h1>
        <p>10-Year NASA POWER Hydrology & Temperature Reanalysis • MODIS Canopy Greenness • Multi-Objective Rotation Scoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Manage state for climate download
if fetch_button or "climate_df" not in st.session_state:
    try:
        with st.spinner(f"Retrieving 10-year NASA POWER agro-meteorology for ({target_lat:.4f}, {target_lon:.4f})..."):
            df_clim = get_cached_climate(target_lat, target_lon, DEFAULT_START_DATE, DEFAULT_END_DATE)
            st.session_state["climate_df"] = df_clim
            st.session_state["evaluated_lat"] = target_lat
            st.session_state["evaluated_lon"] = target_lon
    except (ValueError, NasaPowerError) as err:
        st.error(f"Climate Data Fetch Error: {err}")
        st.stop()

climate_df = st.session_state["climate_df"]
scorer = get_scorer()

# -----------------------------------------------------------------------------
# 1. 10-Year Climate Stress Diagnostic
# -----------------------------------------------------------------------------
try:
    climate_analysis = analyse_climate(climate_df, DEFAULT_SEASONS)
except Exception as exc:
    st.error(f"Error calculating climate indicators: {exc}")
    st.stop()

seasonal_df = climate_analysis.seasonal_indicators
avg_annual_rain = float(climate_df["precip_mm_day"].sum()) / (len(climate_df) / 365.25)
max_dry_run = int(seasonal_df["max_dry_spell_days"].max()) if not seasonal_df.empty else 0
avg_heat_days_yr = float(seasonal_df["heat_days"].mean() * 2) if not seasonal_df.empty else 0

st.markdown("### 📊 10-Year Historical Climate Diagnostic (2016 – 2025)")
st.caption(
    f"Location: **{target_lat:.4f}° N, {target_lon:.4f}° E** | "
    f"Data: NASA POWER AG Community (Daily precipitation, 2m temperature, root-zone soil wetness)"
)

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    m_col1.metric("Mean Annual Rainfall", f"{avg_annual_rain:.0f} mm", delta=f"{scenario_rain_pct:+d}% Scenario" if scenario_rain_pct != 0 else None)
with m_col2:
    m_col2.metric("Longest Recorded Dry Spell", f"{max_dry_run} days", help="Consecutive days with precipitation < 1.0 mm.")
with m_col3:
    m_col3.metric("Annual Hot Days (>35°C)", f"{avg_heat_days_yr:.0f} days/yr", delta=f"+{scenario_temp_c:.1f}°C Scenario" if scenario_temp_c > 0 else None)
with m_col4:
    soil_res = climate_analysis.soil_moisture
    if soil_res.available and not soil_res.table.empty:
        recent_anom = soil_res.table["anomaly"].dropna().iloc[-1] if len(soil_res.table["anomaly"].dropna()) > 0 else 0.0
        m_col4.metric(
            "Root-Zone Wetness Anomaly",
            f"{recent_anom:+.3f}",
            help="Recent season root-zone soil wetness deviation from 10-year baseline (GWETROOT).",
        )
    else:
        m_col4.metric("Root-Zone Wetness", "MERRA-2 Normal")

# -----------------------------------------------------------------------------
# 2. MODIS Satellite Greenness & Bare Soil Exposure (if enabled)
# -----------------------------------------------------------------------------
if enable_modis:
    st.markdown("---")
    st.markdown("### 🛰️ MODIS Satellite Canopy & Bare Ground Analysis")
    with st.spinner("Analyzing MODIS Terra/Aqua 250m NDVI vegetation profile..."):
        modis_df = fetch_modis_ndvi_data(
            target_lat, target_lon, DEFAULT_START_DATE, DEFAULT_END_DATE, climate_df=climate_df
        )
        modis_summary = summarize_bare_soil_exposure(modis_df)

    b_col1, b_col2, b_col3 = st.columns([1, 1, 2])
    with b_col1:
        st.metric(
            "Annual Bare Ground Exposure",
            f"{modis_summary.annual_bare_days:.0f} days/year",
            help="Days per calendar year where NDVI < 0.20 (soil surface exposed without canopy cover).",
        )
    with b_col2:
        st.metric(
            "Mean Annual Canopy NDVI",
            f"{modis_summary.mean_annual_ndvi:.2f}",
            help="Multi-year mean Normalized Difference Vegetation Index (0.0 to 1.0).",
        )
    with b_col3:
        st.info(f"**Fallow & Soil Health Diagnostic:** {modis_summary.fallow_risk_assessment}")

# -----------------------------------------------------------------------------
# 3. Rotation Evaluation & Multi-Card Comparison
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔄 Comparative Rotation Evaluation")
st.write(
    "Evaluating the baseline cropping pattern against three agronomic alternatives "
    "tailored to your soil texture, water regime, and climate sensitivity:"
)

priority_dict = {
    "water_saving": w_water,
    "soil_health": w_soil,
    "climate_resilience": w_resil,
    "income": w_income,
}

comparison_cards = scorer.compare_rotations(
    current_rotation_key=selected_current_rot,
    soil_texture=selected_soil,
    water_source=selected_water,
    annual_rainfall_mm=avg_annual_rain,
    max_dry_spell_days=max_dry_run,
    annual_heat_days=avg_heat_days_yr,
    priority_weights=priority_dict,
    rainfall_scenario_pct=scenario_rain_pct,
    temperature_scenario_c=scenario_temp_c,
)

# Display 4 comparative cards in a clean 2x2 or 4-tab responsive layout
tab_compare, tab_metrics, tab_climate_trends = st.tabs([
    "📋 Rotation Cards Comparison",
    "📈 Priority & Water Balance Metrics",
    "🌦️ 10-Year Climate Data Tables",
])

with tab_compare:
    card_cols = st.columns(2)
    display_order = [
        ("current", "1️⃣ Current Baseline Rotation", False),
        ("soil_health", "2️⃣ Soil Health Option (Legume-Inclusive)", True),
        ("water_saving", "3️⃣ Water-Saving Option", True),
        ("climate_resilient", "4️⃣ Climate-Resilient Option", True),
    ]

    for idx, (card_key, label_prefix, is_alt) in enumerate(display_order):
        col = card_cols[idx % 2]
        card: RotationScoreCard = comparison_cards[card_key]

        with col:
            highlight_class = "rotation-card-highlight" if is_alt else ""
            st.markdown(
                f"""
                <div class="rotation-card {highlight_class}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div class="card-title">{label_prefix}</div>
                            <div style="font-size: 1.15rem; font-weight: 600; color: #2D6A4F; margin-bottom: 2px;">
                                {card.rotation_name}
                            </div>
                            <div class="card-category">{card.category}</div>
                        </div>
                        <div class="score-box" style="min-width: 90px;">
                            <div class="score-value">{card.composite_score:.1f}</div>
                            <div class="score-label">Overall Match</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metric Pill Row
            stat1, stat2, stat3, stat4 = st.columns(4)
            stat1.metric("Water Demand", f"{card.total_water_need_mm:.0f} mm")
            stat2.metric("Water Score", f"{card.water_saving_score:.0f}/100")
            stat3.metric("Soil Score", f"{card.soil_health_score:.0f}/100")
            stat4.metric("Resilience", f"{card.climate_resilience_score:.0f}/100")

            if card.has_legume:
                st.markdown(
                    f"<span class='metric-badge badge-legume'>🌱 Biological N-Fixation: ~{card.nitrogen_credit_kg_ha:.0f} kg N/ha</span>",
                    unsafe_allow_html=True,
                )

            with st.expander("📌 Full Agronomic Assessment & Evidence", expanded=True):
                st.markdown(f"**Why Suggested:** {card.why_suggested}")

                st.markdown("**Field Evidence & Data Points:**")
                for ev in card.evidence:
                    st.markdown(f"- {ev}")

                st.markdown("**Expected Agronomic Benefits:**")
                for ben in card.benefits:
                    st.markdown(f"- {ben}")

                st.markdown("**Operational Risks & Constraints:**")
                for risk in card.risks:
                    st.markdown(f"- ⚠️ {risk}")

                st.markdown("**Information Gaps to Confirm Locally:**")
                for gap in card.information_gaps:
                    st.markdown(f"- ❓ {gap}")

                st.markdown(
                    f"""
                    <div class="advisory-box">
                        <strong>🧑‍🌾 Agronomist Advisory Note:</strong> {card.agronomist_note}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)


with tab_metrics:
    st.subheader("Rotation Multi-Criteria Comparison")
    st.write("Side-by-side performance comparison across farmer priorities:")

    summary_rows = []
    for k, title in [
        ("current", "Current: " + comparison_cards["current"].rotation_name),
        ("soil_health", "Soil Health: " + comparison_cards["soil_health"].rotation_name),
        ("water_saving", "Water Saving: " + comparison_cards["water_saving"].rotation_name),
        ("climate_resilient", "Resilient: " + comparison_cards["climate_resilient"].rotation_name),
    ]:
        c = comparison_cards[k]
        summary_rows.append({
            "Rotation Option": title,
            "Total Water Need (mm)": c.total_water_need_mm,
            "Water Saving Score": c.water_saving_score,
            "Soil Health Score": c.soil_health_score,
            "Climate Resilience": c.climate_resilience_score,
            "Income Index": c.income_score,
            "Overall Score": c.composite_score,
            "Legume Included": "Yes" if c.has_legume else "No",
            "N-Credit (kg/ha)": c.nitrogen_credit_kg_ha,
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # Visual Bar Chart comparing scores
    chart_df = summary_df.set_index("Rotation Option")[
        ["Water Saving Score", "Soil Health Score", "Climate Resilience", "Income Index"]
    ]
    st.bar_chart(chart_df, height=350)


with tab_climate_trends:
    st.subheader("10-Year Seasonal Climate Indicators (NASA POWER)")
    st.caption("Grouped by agricultural season (Rabi: Nov 01 – Apr 30; Kharif: May 01 – Oct 31).")

    display_table = seasonal_df.copy()
    display_table["coverage"] = (display_table["coverage"] * 100).round(0)
    display_table = display_table.rename(
        columns={
            "season_label": "Season",
            "total_precip_mm": "Rainfall (mm)",
            "max_dry_spell_days": "Longest Dry Spell (days)",
            "heat_days": "Heat Days (>35°C)",
            "frost_days": "Frost Days (<0°C)",
            "coverage": "Data Coverage (%)",
            "complete": "Full Season?",
        }
    ).drop(columns=["season", "season_year"])
    st.dataframe(display_table, hide_index=True, use_container_width=True)

    if climate_analysis.soil_moisture.available and not climate_analysis.soil_moisture.table.empty:
        st.markdown("**Root-Zone Soil Wetness Multi-Year Baseline & Anomaly**")
        st.caption("Relative to multi-year seasonal average (dimensionless fraction 0 = dry, 1 = saturated).")
        st_df = climate_analysis.soil_moisture.table[
            ["season_label", "mean_soil_wetness", "baseline_mean", "anomaly", "anomaly_zscore", "coverage"]
        ].rename(
            columns={
                "season_label": "Season",
                "mean_soil_wetness": "Mean Wetness (0-1)",
                "baseline_mean": "Baseline Normal",
                "anomaly": "Anomaly",
                "anomaly_zscore": "Z-Score",
                "coverage": "Coverage",
            }
        ).round(3)
        st.dataframe(st_df, hide_index=True, use_container_width=True)

    with st.expander("Assumptions & Methodological Constraints"):
        for note in climate_analysis.notes:
            st.markdown(f"- {note}")
        st.markdown(
            "- **MODIS Satellite Processing**: Surface greenness is sampled over 250 m pixels. "
            "Bare soil threshold is calibrated at NDVI < 0.20."
        )
        st.markdown(
            "- **Machine Learning Surrogate**: Random Forest regressor predicts climate resilience score "
            "based on rooting depth, drought/heat tolerances, 10-year climate extremes, and what-if deltas."
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Field Shift Decision Support System • Data sources: NASA POWER Agroclimatology (AG) Point API, "
    "NASA Land Processes DAAC MODIS Terra/Aqua, and Punjab Directorate of Agriculture Extension reference baselines."
)
