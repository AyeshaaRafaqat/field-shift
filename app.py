"""Field Shift - Agro-Climatic Crop Rotation Decision Support Tool.

A professional, practical evaluation platform designed for farmers, agronomists,
and field officers to compare crop rotations against 10-year historical climate
conditions, water availability, and soil characteristics.
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

# -----------------------------------------------------------------------------
# Configuration & Minimal Professional Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Field Shift | Crop Rotation Decision Tool",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Clean, high-contrast, professional typography & layout */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1F2937;
    }
    .top-bar {
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 16px;
        margin-bottom: 24px;
    }
    .top-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .top-subtitle {
        font-size: 0.95rem;
        color: #4B5563;
        margin-top: 4px;
        margin-bottom: 0;
    }
    .step-heading {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #E5E7EB;
    }
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-featured {
        border: 1px solid #15803D;
        background-color: #FBFDFB;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 2px;
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .stat-pill {
        display: inline-block;
        background: #F3F4F6;
        border: 1px solid #E5E7EB;
        border-radius: 4px;
        padding: 6px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .stat-pill-label {
        font-size: 0.72rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
        display: block;
    }
    .stat-pill-val {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
    }
    .stat-pill-highlight {
        background: #ECFDF5;
        border-color: #A7F3D0;
    }
    .stat-pill-highlight .stat-pill-val {
        color: #065F46;
    }
    .advisory-box {
        background-color: #FFFBEB;
        border-left: 3px solid #D97706;
        padding: 10px 14px;
        font-size: 0.88rem;
        color: #78350F;
        border-radius: 2px;
        margin-top: 14px;
    }
    .section-summary {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data Presets
# -----------------------------------------------------------------------------
LOCATION_PRESETS = {
    "Multan (Southern Punjab)": {
        "lat": 30.1575, "lon": 71.5249, "soil": "Sandy Loam",
        "water": "Canal (Non-perennial) + Tube Well", "rotation": "cotton_wheat",
    },
    "Faisalabad (Central Punjab)": {
        "lat": 31.4504, "lon": 73.1350, "soil": "Loam",
        "water": "Canal Irrigation (Perennial)", "rotation": "maize_wheat",
    },
    "Gujranwala (Upper Punjab)": {
        "lat": 32.1877, "lon": 74.1945, "soil": "Clay Loam",
        "water": "Canal Irrigation (Perennial)", "rotation": "rice_wheat",
    },
    "Bahawalpur (Southern Margin)": {
        "lat": 29.3544, "lon": 71.6911, "soil": "Sandy Loam",
        "water": "Tube Well (Fresh Groundwater)", "rotation": "cotton_wheat",
    },
    "Rahim Yar Khan": {
        "lat": 28.4212, "lon": 70.2989, "soil": "Loam",
        "water": "Canal (Non-perennial) + Tube Well", "rotation": "sugarcane_wheat",
    },
    "Sargodha (Mixed Grain / Citrus)": {
        "lat": 32.0836, "lon": 72.6711, "soil": "Loam",
        "water": "Canal Irrigation (Perennial)", "rotation": "maize_wheat",
    },
    "Sahiwal (Maize-Potato Belt)": {
        "lat": 30.6682, "lon": 73.1114, "soil": "Silt Loam",
        "water": "Canal Irrigation (Perennial)", "rotation": "maize_wheat",
    },
    "Custom Location (Enter Coordinates)": {
        "lat": 30.1575, "lon": 71.5249, "soil": "Loam",
        "water": "Canal Irrigation (Perennial)", "rotation": "cotton_wheat",
    },
}

SOIL_OPTIONS = [
    "Sandy Loam (Light, well-drained, lower water holding)",
    "Loam (Balanced texture, good water and nutrient retention)",
    "Clay Loam (Medium-heavy, holds moisture well)",
    "Silt Loam (Fine textured, high water capacity)",
    "Sandy (Light, drains quickly, prone to drying)",
    "Clay (Heavy soil, slow drainage, retains water for ponding)",
]

SOIL_KEY_MAP = {
    "Sandy Loam (Light, well-drained, lower water holding)": "Sandy Loam",
    "Loam (Balanced texture, good water and nutrient retention)": "Loam",
    "Clay Loam (Medium-heavy, holds moisture well)": "Clay Loam",
    "Silt Loam (Fine textured, high water capacity)": "Silt Loam",
    "Sandy (Light, drains quickly, prone to drying)": "Sandy",
    "Clay (Heavy soil, slow drainage, retains water for ponding)": "Clay",
}

WATER_OPTIONS = [
    "Canal Irrigation (Perennial - year-round)",
    "Canal (Non-perennial) + Tube Well backup",
    "Tube Well only (Fresh groundwater)",
    "Tube Well only (Marginal / Saline groundwater)",
    "Rainfed only (Barani - no irrigation)",
]

WATER_KEY_MAP = {
    "Canal Irrigation (Perennial - year-round)": "Canal Irrigation (Perennial)",
    "Canal (Non-perennial) + Tube Well backup": "Canal (Non-perennial) + Tube Well",
    "Tube Well only (Fresh groundwater)": "Tube Well (Fresh Groundwater)",
    "Tube Well only (Marginal / Saline groundwater)": "Tube Well (Marginal / Saline)",
    "Rainfed only (Barani - no irrigation)": "Rainfed (Barani)",
}

CURRENT_ROTATION_CHOICES = {
    "rice_wheat": "Rice then Wheat (Kharif rice followed by Rabi wheat)",
    "cotton_wheat": "Cotton then Wheat (Kharif cotton followed by Rabi wheat)",
    "maize_wheat": "Maize then Wheat (Kharif maize followed by Rabi wheat)",
    "sugarcane_wheat": "Sugarcane then Wheat (Annual sugarcane cycle)",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_climate_data(latitude: float, longitude: float) -> pd.DataFrame:
    """Download 10-year NASA POWER daily climate data."""
    return fetch_nasa_power_data(latitude, longitude, DEFAULT_START_DATE, DEFAULT_END_DATE)


@st.cache_resource
def load_rotation_scorer() -> RotationScorer:
    """Load the scoring engine and dataset."""
    return RotationScorer()


# -----------------------------------------------------------------------------
# Top Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="top-bar">
        <h1 class="top-title">Field Shift</h1>
        <p class="top-subtitle">Crop Rotation Planning Tool | NASA 10-Year Weather Data & Field-Level Soil Diagnostics</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Step-by-Step Farmer Input Section
# -----------------------------------------------------------------------------
st.markdown("<div class='step-heading'>Step 1: Your Location & Field Conditions</div>", unsafe_allow_html=True)
c_loc, c_soil, c_water = st.columns(3)

with c_loc:
    selected_preset = st.selectbox("Select Area", options=list(LOCATION_PRESETS.keys()), index=0)
    preset = LOCATION_PRESETS[selected_preset]

    if selected_preset == "Custom Location (Enter Coordinates)":
        c_lat, c_lon = st.columns(2)
        lat = c_lat.number_input("Latitude", -90.0, 90.0, 30.1575, format="%.4f")
        lon = c_lon.number_input("Longitude", -180.0, 180.0, 71.5249, format="%.4f")
    else:
        lat = preset["lat"]
        lon = preset["lon"]
        st.caption(f"Coordinates: {lat:.4f}° N, {lon:.4f}° E")

with c_soil:
    default_soil_label = [opt for opt in SOIL_OPTIONS if SOIL_KEY_MAP[opt] == preset["soil"]][0]
    soil_input = st.selectbox("Soil Type", options=SOIL_OPTIONS, index=SOIL_OPTIONS.index(default_soil_label))
    selected_soil = SOIL_KEY_MAP[soil_input]

with c_water:
    default_water_label = [opt for opt in WATER_OPTIONS if WATER_KEY_MAP[opt] == preset["water"]][0]
    water_input = st.selectbox("Water Source", options=WATER_OPTIONS, index=WATER_OPTIONS.index(default_water_label))
    selected_water = WATER_KEY_MAP[water_input]


st.markdown("<div class='step-heading' style='margin-top: 14px;'>Step 2: Current Crop Rotation & Farm Priorities</div>", unsafe_allow_html=True)
c_rot, c_prio = st.columns([1, 2])

with c_rot:
    rot_keys = list(CURRENT_ROTATION_CHOICES.keys())
    rot_idx = rot_keys.index(preset["rotation"]) if preset["rotation"] in rot_keys else 0
    selected_rot = st.selectbox(
        "Current Cropping Practice",
        options=rot_keys,
        format_func=lambda k: CURRENT_ROTATION_CHOICES[k],
        index=rot_idx,
    )

with c_prio:
    st.write("**What are your main goals?**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        w_water = st.select_slider("Save Water", options=["Low", "Medium", "High"], value="High")
    with p_col2:
        w_soil = st.select_slider("Improve Soil", options=["Low", "Medium", "High"], value="High")
    with p_col3:
        w_resil = st.select_slider("Climate Safety", options=["Low", "Medium", "High"], value="Medium")
    with p_col4:
        w_income = st.select_slider("Farm Income", options=["Low", "Medium", "High"], value="High")

# Convert sliders to numerical weights
scale_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
priority_weights = {
    "water_saving": scale_map[w_water],
    "soil_health": scale_map[w_soil],
    "climate_resilience": scale_map[w_resil],
    "income": scale_map[w_income],
}

# Optional settings in a simple expander
with st.expander("Optional: Weather Testing & Satellite Check", expanded=False):
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        scenario_rain_pct = st.slider("Test rainfall change (%)", -30, 30, 0, 5, help="Simulate drier or wetter years.")
    with opt_col2:
        scenario_temp_c = st.slider("Test temperature increase (°C)", 0.0, 4.0, 0.0, 0.5, help="Simulate hotter summer or spring temperatures.")
    with opt_col3:
        include_modis = st.checkbox("Check satellite ground cover (MODIS)", value=True, help="Estimates how many days a year the field sits bare without crop canopy.")

submit_eval = st.button("Compare Crop Rotations", type="primary")

# -----------------------------------------------------------------------------
# Data Retrieval & Calculations
# -----------------------------------------------------------------------------
if submit_eval or "active_climate" not in st.session_state:
    try:
        with st.spinner("Loading 10-year NASA climate records for your location..."):
            climate_df = load_climate_data(lat, lon)
            st.session_state["active_climate"] = climate_df
            st.session_state["active_lat"] = lat
            st.session_state["active_lon"] = lon
    except (ValueError, NasaPowerError) as err:
        st.error(f"Weather Data Error: {err}")
        st.stop()

climate_df = st.session_state["active_climate"]
scorer = load_rotation_scorer()

try:
    climate_analysis = analyse_climate(climate_df, DEFAULT_SEASONS)
except Exception as exc:
    st.error(f"Error analyzing climate records: {exc}")
    st.stop()

# Key aggregate metrics
seasonal_df = climate_analysis.seasonal_indicators
avg_annual_rain = float(climate_df["precip_mm_day"].sum()) / (len(climate_df) / 365.25)
max_dry_run = int(seasonal_df["max_dry_spell_days"].max()) if not seasonal_df.empty else 0
avg_heat_days_yr = float(seasonal_df["heat_days"].mean() * 2) if not seasonal_df.empty else 0

# MODIS summary if enabled
modis_summary = None
if include_modis:
    modis_df = fetch_modis_ndvi_data(lat, lon, DEFAULT_START_DATE, DEFAULT_END_DATE, climate_df=climate_df)
    modis_summary = summarize_bare_soil_exposure(modis_df)

# Run comparison
cards = scorer.compare_rotations(
    current_rotation_key=selected_rot,
    soil_texture=selected_soil,
    water_source=selected_water,
    annual_rainfall_mm=avg_annual_rain,
    max_dry_spell_days=max_dry_run,
    annual_heat_days=avg_heat_days_yr,
    priority_weights=priority_weights,
    rainfall_scenario_pct=scenario_rain_pct,
    temperature_scenario_c=scenario_temp_c,
)

# -----------------------------------------------------------------------------
# Results Section
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Field Summary & 10-Year Climate Exposure")

# 3 or 4 clear summary tiles
s_col1, s_col2, s_col3, s_col4 = st.columns(4)
with s_col1:
    s_col1.markdown(
        f"""
        <div class="stat-pill" style="width: 100%;">
            <span class="stat-pill-label">Average Annual Rain</span>
            <span class="stat-pill-val">{avg_annual_rain:.0f} mm</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s_col2:
    s_col2.markdown(
        f"""
        <div class="stat-pill" style="width: 100%;">
            <span class="stat-pill-label">Longest Dry Spell on Record</span>
            <span class="stat-pill-val">{max_dry_run} days</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s_col3:
    s_col3.markdown(
        f"""
        <div class="stat-pill" style="width: 100%;">
            <span class="stat-pill-label">Hot Days per Year (>35°C)</span>
            <span class="stat-pill-val">{avg_heat_days_yr:.0f} days</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with s_col4:
    if modis_summary:
        s_col4.markdown(
            f"""
            <div class="stat-pill" style="width: 100%;">
                <span class="stat-pill-label">Satellite Bare Soil Exposure</span>
                <span class="stat-pill-val">{modis_summary.annual_bare_days:.0f} days / yr</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        s_col4.markdown(
            f"""
            <div class="stat-pill" style="width: 100%;">
                <span class="stat-pill-label">Soil & Water Regime</span>
                <span class="stat-pill-val">{selected_soil}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

if modis_summary and modis_summary.annual_bare_days > 80:
    st.info(f"Field Notice: Satellite records show your soil sits unplanted for about {modis_summary.annual_bare_days:.0f} days each year. Introducing a short pulse or cover crop helps keep the soil moist and prevents organic matter loss from extreme sun exposure.")


# -----------------------------------------------------------------------------
# Rotation Cards: Current + 3 Alternatives
# -----------------------------------------------------------------------------
st.markdown("### Crop Rotation Comparison")
st.write("Comparing your current practice against three practical alternatives suited for your soil and water access:")

card_order = [
    ("current", "Current Baseline Practice", False),
    ("soil_health", "Alternative 1: Soil Health & Pulse Option", True),
    ("water_saving", "Alternative 2: Water-Saving Option", True),
    ("climate_resilient", "Alternative 3: Climate-Resilient Option", True),
]

col_left, col_right = st.columns(2)

for i, (key, heading, is_featured) in enumerate(card_order):
    card_data: RotationScoreCard = cards[key]
    target_col = col_left if (i % 2 == 0) else col_right

    with target_col:
        box_style = "card card-featured" if is_featured else "card"
        st.markdown(
            f"""
            <div class="{box_style}">
                <div class="card-subtitle">{heading}</div>
                <div class="card-title">{card_data.rotation_name}</div>
                <div style="font-size: 0.88rem; color: #4B5563; margin-bottom: 12px;">{card_data.category}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clean Metric Pills
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
                <div class="stat-pill" style="width: 100%;">
                    <span class="stat-pill-label">Water Needed</span>
                    <span class="stat-pill-val">{card_data.total_water_need_mm:.0f} mm</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="stat-pill {'stat-pill-highlight' if card_data.water_saving_score > 60 else ''}" style="width: 100%;">
                    <span class="stat-pill-label">Water Efficiency</span>
                    <span class="stat-pill-val">{card_data.water_saving_score:.0f} / 100</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="stat-pill {'stat-pill-highlight' if card_data.nitrogen_credit_kg_ha > 0 else ''}" style="width: 100%;">
                    <span class="stat-pill-label">Nitrogen Added</span>
                    <span class="stat-pill-val">{card_data.nitrogen_credit_kg_ha:.0f} kg / ha</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Details, Benefits & What to Check", expanded=True):
            st.markdown(f"**Why this rotation:** {card_data.why_suggested}")

            st.markdown("**Direct Practical Benefits:**")
            for b in card_data.benefits:
                st.markdown(f"- {b}")

            st.markdown("**Things to Watch Out For (Risks):**")
            for r in card_data.risks:
                st.markdown(f"- {r}")

            st.markdown("**What to Confirm Locally:**")
            for g in card_data.information_gaps:
                st.markdown(f"- {g}")

            st.markdown(
                f"""
                <div class="advisory-box">
                    <strong>Agronomist Note:</strong> {card_data.agronomist_note}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# High-Level Side-by-Side Summary Table
# -----------------------------------------------------------------------------
st.markdown("### Summary Table")

tbl_data = []
for k, label in [
    ("current", "Current: " + cards["current"].rotation_name),
    ("soil_health", "Soil Health: " + cards["soil_health"].rotation_name),
    ("water_saving", "Water Saving: " + cards["water_saving"].rotation_name),
    ("climate_resilient", "Climate Resilient: " + cards["climate_resilient"].rotation_name),
]:
    c = cards[k]
    tbl_data.append({
        "Option": label,
        "Total Water Need (mm)": f"{c.total_water_need_mm:.0f} mm",
        "Water Score": f"{c.water_saving_score:.0f}/100",
        "Soil Health Score": f"{c.soil_health_score:.0f}/100",
        "Resilience Score": f"{c.climate_resilience_score:.0f}/100",
        "Nitrogen Credit": f"{c.nitrogen_credit_kg_ha:.0f} kg/ha" if c.has_legume else "None",
        "Overall Fit": f"{c.composite_score:.1f}/100",
    })

st.dataframe(pd.DataFrame(tbl_data), hide_index=True, use_container_width=True)

# -----------------------------------------------------------------------------
# Technical Records (Collapsed for farmers; available for agronomists)
# -----------------------------------------------------------------------------
with st.expander("Detailed 10-Year Weather Records (NASA POWER Daily Data)", expanded=False):
    st.write("Seasonal breakdown (Rabi: Nov-Apr, Kharif: May-Oct):")
    clean_seasonal = seasonal_df[[
        "season_label", "total_precip_mm", "max_dry_spell_days", "heat_days", "frost_days"
    ]].rename(columns={
        "season_label": "Season",
        "total_precip_mm": "Total Rainfall (mm)",
        "max_dry_spell_days": "Longest Dry Run (days)",
        "heat_days": "Days Above 35°C",
        "frost_days": "Days Below 0°C",
    })
    st.dataframe(clean_seasonal, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption(
    "Field Shift Decision Support Tool | Data: NASA POWER Agro-Climatology Daily API & MODIS Terra/Aqua Surface Reflectance."
)
