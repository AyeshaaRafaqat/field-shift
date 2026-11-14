"""Field Shift — Agro-Climatic Crop Rotation Decision Support Platform.

NASA Space Apps Challenge 2026 | Team astrooLogic (Lahore, Pakistan)
Ayesha Rafaqat · Sara Haris
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from analysis.climate_analysis import DEFAULT_SEASONS, analyse_climate
from analysis.rotation_scorer import ROTATION_CATALOG, RotationScoreCard, RotationScorer
from engine.montecarlo import UncertaintySummary, run_monte_carlo_analysis
from services.modis_data import fetch_modis_ndvi_data, summarize_bare_soil_exposure
from services.nasa_data import DEFAULT_END_DATE, DEFAULT_START_DATE, NasaPowerError, fetch_nasa_power_data
from ui.components import farm_profile_card, insight_card, persistent_expert_banner, top_header
from ui.theme import apply_custom_theme

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Field Shift | NASA Agro-Climatic Rotation Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_scorer() -> RotationScorer:
    return RotationScorer()


def _load_sites() -> dict:
    cfg = os.path.join(CONFIG_DIR, "sites.yaml")
    if os.path.exists(cfg):
        with open(cfg, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("sites", {})
    return {}


SITES = _load_sites()

# Fallback if sites.yaml is missing
if not SITES:
    SITES = {
        "faisalabad": {
            "name": "Faisalabad (Mixed Cropping)",
            "district": "Faisalabad",
            "lat": 31.4504,
            "lon": 73.1350,
            "soil_default": "Loam",
            "water_default": "Canal Irrigation (Perennial)",
            "rotation_default": "maize_wheat",
            "zone": "Central Mixed Cropping Belt",
        },
        "multan": {
            "name": "Multan (Southern Punjab)",
            "district": "Multan",
            "lat": 30.1968,
            "lon": 71.4782,
            "soil_default": "Sandy Loam",
            "water_default": "Canal (Non-perennial) + Tube Well",
            "rotation_default": "cotton_wheat",
            "zone": "Cotton-Wheat Agro-Ecological Belt",
        },
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_climate_records(site_key: str, lat: float, lon: float) -> Tuple[pd.DataFrame, str]:
    """Load pre-computed offline data when available, otherwise fetch live."""
    parquet_map = {
        "faisalabad": "site_A_daily.parquet",
        "multan": "site_B_daily.parquet",
    }
    fname = parquet_map.get(site_key)
    if fname:
        local_path = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(local_path):
            return pd.read_parquet(local_path), "Pre-fetched NASA POWER Data (Offline)"
    df = fetch_nasa_power_data(lat, lon, DEFAULT_START_DATE, DEFAULT_END_DATE)
    return df, "Live NASA POWER AG Query"


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_defaults = {
    "language": "English",
    "site_key": list(SITES.keys())[0],
    "soil_texture": list(SITES.values())[0].get("soil_default", "Loam"),
    "water_source": list(SITES.values())[0].get("water_default", "Canal Irrigation (Perennial)"),
    "current_rotation": list(SITES.values())[0].get("rotation_default", "rice_wheat"),
    "weights": {"water_saving": 2.0, "soil_health": 2.0, "climate_resilience": 2.0, "income": 2.0},
    "scenario_rain": 0,
    "scenario_temp": 0.0,
    "nav_page": "Overview",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar: WORKSPACE navigation + language toggle
# ---------------------------------------------------------------------------
is_urdu = st.session_state["language"] == "اردو (Urdu)"
apply_custom_theme(is_urdu=is_urdu)

with st.sidebar:
    st.markdown(
        "<div style='font-size: 0.72rem; font-weight: 700; color: #6B7280; "
        "letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px;'>"
        "WORKSPACE</div>",
        unsafe_allow_html=True,
    )

    nav_options = ["Overview", "My Farm", "NASA Insights", "Crop Rotations", "Reports", "Data & Method"]
    curr_idx = nav_options.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_options else 0
    selected_page = st.radio(
        "nav",
        options=nav_options,
        index=curr_idx,
        label_visibility="collapsed",
    )
    st.session_state["nav_page"] = selected_page

    st.markdown("---")

    lang_choice = st.selectbox(
        "Language / زبان",
        options=["English", "اردو (Urdu)"],
        index=0 if st.session_state["language"] == "English" else 1,
    )
    if lang_choice != st.session_state["language"]:
        st.session_state["language"] = lang_choice
        st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("Field Shift v1.3  •  NASA Space Apps 2026")
    st.caption("Team astrooLogic (Lahore)")

# ---------------------------------------------------------------------------
# Shared data computation (runs on every page)
# ---------------------------------------------------------------------------
site_info = SITES.get(
    st.session_state["site_key"],
    list(SITES.values())[0],
)
target_lat = float(site_info["lat"])
target_lon = float(site_info["lon"])

try:
    climate_df, source_label = load_climate_records(st.session_state["site_key"], target_lat, target_lon)
    climate_analysis = analyse_climate(climate_df, DEFAULT_SEASONS)
except Exception as err:
    st.error(f"Error loading climate records: {err}")
    st.stop()

seasonal_df = climate_analysis.seasonal_indicators
avg_annual_rain = float(climate_df["precip_mm_day"].sum()) / (len(climate_df) / 365.25)
max_dry_run = int(seasonal_df["max_dry_spell_days"].max()) if not seasonal_df.empty else 0
avg_heat_days = float(seasonal_df["heat_days"].mean() * 2) if not seasonal_df.empty else 0

modis_df = fetch_modis_ndvi_data(target_lat, target_lon, DEFAULT_START_DATE, DEFAULT_END_DATE, climate_df=climate_df)
modis_summary = summarize_bare_soil_exposure(modis_df)

scorer = load_scorer()
rotation_cards = scorer.compare_rotations(
    current_rotation_key=st.session_state["current_rotation"],
    soil_texture=st.session_state["soil_texture"],
    water_source=st.session_state["water_source"],
    annual_rainfall_mm=avg_annual_rain,
    max_dry_spell_days=max_dry_run,
    annual_heat_days=avg_heat_days,
    priority_weights=st.session_state["weights"],
    rainfall_scenario_pct=st.session_state["scenario_rain"],
    temperature_scenario_c=st.session_state["scenario_temp"],
)
mc_results: Dict[str, UncertaintySummary] = run_monte_carlo_analysis(rotation_cards, n_simulations=500, random_seed=42)

best_key = max(rotation_cards, key=lambda k: rotation_cards[k].composite_score)
best_card = rotation_cards[best_key]
best_mc = mc_results[best_key]

top_header(is_urdu=is_urdu)

# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if st.session_state["nav_page"] == "Overview":
    greeting = "خوش آمدید، کسان بھائی!" if is_urdu else "Welcome, Farmer."
    subtext = (
        "آئیے آپ کی زمین اور آب و ہوا کے مطابق بہترین فصلوں کا انتخاب کریں۔"
        if is_urdu
        else "Find a smarter crop rotation based on 10 years of NASA satellite climate data for your field."
    )

    st.markdown(f"<h1 style='font-size:1.85rem;font-weight:800;color:#111827;margin-bottom:2px;'>{greeting}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.98rem;color:#4B5563;margin-bottom:20px;'>{subtext}</p>", unsafe_allow_html=True)

    current_rot_name = ROTATION_CATALOG.get(st.session_state["current_rotation"], {}).get("name", "")
    farm_profile_card(
        site_name=f"{site_info['name']} ({target_lat:.2f}° N, {target_lon:.2f}° E)",
        soil=st.session_state["soil_texture"],
        water=st.session_state["water_source"],
        rotation_name=current_rot_name,
        status="Ready to analyze",
    )

    st.markdown(
        "<div style='font-size:0.82rem;font-weight:700;color:#6B7280;text-transform:uppercase;margin-bottom:10px;'>"
        "10-Year Environmental Summary (NASA POWER)</div>",
        unsafe_allow_html=True,
    )
    r1, r2 = st.columns(2)
    insight_card(r1, "Average Annual Rainfall", f"{avg_annual_rain:.0f} mm", "10-year mean precipitation", source=source_label)
    insight_card(r2, "Heat Risk Days (>35°C)", f"{avg_heat_days:.0f} days / year", "Days exceeding heat stress threshold", source=source_label)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    r3, r4 = st.columns(2)
    insight_card(r3, "Bare Ground Exposure", f"{modis_summary.annual_bare_days:.0f} days / year", "Days with NDVI < 0.20 (no canopy)", source="MODIS Satellite")
    insight_card(r4, "Longest Drought Run", f"{max_dry_run} days", "Consecutive days with < 1 mm rainfall", source=source_label)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    if st.button("Analyze my crop rotation options →", type="primary", use_container_width=True):
        st.session_state["nav_page"] = "Crop Rotations"
        st.rerun()

    persistent_expert_banner(is_urdu=is_urdu)


# ===========================================================================
# PAGE: MY FARM
# ===========================================================================
elif st.session_state["nav_page"] == "My Farm":
    st.markdown("## My Farm Profile")
    st.write("Set your field location, soil type, irrigation source, current crop rotation, and priorities.")

    with st.form("farm_profile_form"):
        st.markdown("### 1. Location")
        preset_keys = list(SITES.keys())
        site_names = [SITES[k]["name"] for k in preset_keys]
        curr_site_idx = preset_keys.index(st.session_state["site_key"]) if st.session_state["site_key"] in preset_keys else 0
        chosen_site_name = st.selectbox("Select Punjab District", options=site_names, index=curr_site_idx)
        chosen_site_key = preset_keys[site_names.index(chosen_site_name)]

        st.markdown("### 2. Soil & Water")
        c_s1, c_s2 = st.columns(2)
        soil_options = ["Sandy Loam", "Loam", "Clay Loam", "Silt Loam", "Sandy", "Clay"]
        water_options = [
            "Canal Irrigation (Perennial)",
            "Canal (Non-perennial) + Tube Well",
            "Tube Well (Fresh Groundwater)",
            "Tube Well (Marginal / Saline)",
            "Rainfed (Barani)",
        ]
        with c_s1:
            soil_idx = soil_options.index(st.session_state["soil_texture"]) if st.session_state["soil_texture"] in soil_options else 1
            chosen_soil = st.selectbox("Soil Texture", soil_options, index=soil_idx)
        with c_s2:
            water_idx = water_options.index(st.session_state["water_source"]) if st.session_state["water_source"] in water_options else 0
            chosen_water = st.selectbox("Primary Irrigation Source", water_options, index=water_idx)

        st.markdown("### 3. Current Cropping Practice")
        rot_keys = list(ROTATION_CATALOG.keys())
        curr_rot_idx = rot_keys.index(st.session_state["current_rotation"]) if st.session_state["current_rotation"] in rot_keys else 0
        chosen_rot = st.selectbox(
            "Current Annual Rotation",
            options=rot_keys,
            format_func=lambda k: ROTATION_CATALOG[k]["name"],
            index=curr_rot_idx,
        )

        st.markdown("### 4. Farm Priorities")
        st.caption("Set what matters most to your farm. These weights guide which rotation the tool recommends.")
        p1, p2, p3, p4 = st.columns(4)
        w_w = p1.slider("Save Water", 0.0, 3.0, float(st.session_state["weights"]["water_saving"]), 0.5)
        w_s = p2.slider("Improve Soil", 0.0, 3.0, float(st.session_state["weights"]["soil_health"]), 0.5)
        w_r = p3.slider("Climate Safety", 0.0, 3.0, float(st.session_state["weights"]["climate_resilience"]), 0.5)
        w_i = p4.slider("Farm Income", 0.0, 3.0, float(st.session_state["weights"]["income"]), 0.5)

        if st.form_submit_button("Save Profile & Recalculate", type="primary"):
            st.session_state["site_key"] = chosen_site_key
            st.session_state["soil_texture"] = chosen_soil
            st.session_state["water_source"] = chosen_water
            st.session_state["current_rotation"] = chosen_rot
            st.session_state["weights"] = {"water_saving": w_w, "soil_health": w_s, "climate_resilience": w_r, "income": w_i}
            st.success("Farm profile updated.")
            st.session_state["nav_page"] = "Crop Rotations"
            st.rerun()

    persistent_expert_banner(is_urdu=is_urdu)


# ===========================================================================
# PAGE: NASA INSIGHTS
# ===========================================================================
elif st.session_state["nav_page"] == "NASA Insights":
    st.markdown("## NASA Agro-Climatology Insights")
    st.caption(f"10-Year Reanalysis (2016–2025) · {site_info['name']} · {target_lat:.2f}° N, {target_lon:.2f}° E")

    t1, t2 = st.tabs(["Precipitation & Water", "Temperature & Heat Stress"])

    with t1:
        st.markdown("### Annual Precipitation vs 10-Year Mean")
        yearly_rain = climate_df.groupby(climate_df.index.year)["precip_mm_day"].sum().reset_index()
        yearly_rain.columns = ["Year", "Rainfall (mm)"]
        fig_rain = px.bar(
            yearly_rain, x="Year", y="Rainfall (mm)",
            title=f"Annual Rainfall — 10-Year Mean: {avg_annual_rain:.0f} mm",
            color_discrete_sequence=["#1B4D3E"],
        )
        fig_rain.add_hline(y=avg_annual_rain, line_dash="dash", line_color="#D97706", annotation_text="10-Year Mean")
        fig_rain.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_rain, use_container_width=True)

        st.markdown("### Root-Zone Soil Moisture (GWETROOT)")
        if climate_analysis.soil_moisture.available and not climate_analysis.soil_moisture.table.empty:
            sm_df = climate_analysis.soil_moisture.table[
                ["season_label", "mean_soil_wetness", "baseline_mean", "anomaly"]
            ].rename(columns={
                "season_label": "Season",
                "mean_soil_wetness": "Mean Wetness (0–1)",
                "baseline_mean": "Baseline",
                "anomaly": "Anomaly",
            }).round(3)
            st.dataframe(sm_df, hide_index=True, use_container_width=True)
        else:
            st.info("Soil moisture data not available for this location.")

    with t2:
        st.markdown("### Extreme Heat Days (>35°C) per Year")
        hot = climate_df[climate_df["tmax_c"] > 35.0]
        yearly_heat = hot.groupby(hot.index.year).size().reset_index()
        yearly_heat.columns = ["Year", "Hot Days (>35°C)"]
        fig_heat = px.line(
            yearly_heat, x="Year", y="Hot Days (>35°C)",
            markers=True,
            title="Trend of Extreme Heat Days (>35°C)",
            color_discrete_sequence=["#B91C1C"],
        )
        fig_heat.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Bare Ground & Fallow Dynamics (MODIS Satellite)")
    st.write(
        f"Satellite records estimate fields in this zone sit bare without vegetative canopy for approximately "
        f"**{modis_summary.annual_bare_days:.0f} days per year** (primarily May–June between wheat harvest and rice/cotton transplanting). "
        f"Extended bare-soil periods accelerate organic matter loss and evaporative water waste."
    )

    persistent_expert_banner(is_urdu=is_urdu)


# ===========================================================================
# PAGE: CROP ROTATIONS
# ===========================================================================
elif st.session_state["nav_page"] == "Crop Rotations":
    st.markdown("## Crop Rotation Advisor")
    st.write("Comparing your current practice against three agronomically suited alternatives for your soil and water conditions.")

    # Top recommendation banner
    st.markdown(
        f"""
        <div style="background:#ECFDF5;border:1.5px solid #059669;border-radius:10px;padding:18px;margin-bottom:22px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-size:0.75rem;font-weight:700;color:#065F46;text-transform:uppercase;">Top Recommended Strategy</span>
                    <div style="font-size:1.35rem;font-weight:800;color:#064E3B;">{best_card.rotation_name}</div>
                    <div style="font-size:0.9rem;color:#047857;margin-top:2px;">{best_card.why_suggested}</div>
                </div>
                <div style="text-align:right;min-width:130px;">
                    <div style="font-size:1.8rem;font-weight:900;color:#065F46;">{best_card.composite_score:.1f}</div>
                    <span class="status-pill">Rank #1 in {best_mc.win_rate_pct:.0f}% of simulations</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_order = [
        ("current", "Current Baseline Practice", False),
        ("soil_health", "Alternative 1: Soil Health & Pulse Option", True),
        ("water_saving", "Alternative 2: Water-Saving Option", True),
        ("climate_resilient", "Alternative 3: Climate-Resilient Option", True),
    ]

    c_left, c_right = st.columns(2)

    for i, (k, title_prefix, featured) in enumerate(display_order):
        col = c_left if i % 2 == 0 else c_right
        card: RotationScoreCard = rotation_cards[k]
        mc: UncertaintySummary = mc_results[k]

        with col:
            border_cls = "app-card app-card-featured" if featured else "app-card"
            st.markdown(
                f"""
                <div class="{border_cls}">
                    <div class="card-header-label">{title_prefix}</div>
                    <div class="card-title-lg">{card.rotation_name}</div>
                    <div style="font-size:0.85rem;color:#6B7280;margin-bottom:12px;">{card.category}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Water Need", f"{card.total_water_need_mm:.0f} mm")
            m2.metric("Water Score", f"{card.water_saving_score:.0f}/100")
            m3.metric("Soil Score", f"{card.soil_health_score:.0f}/100")
            m4.metric("Resilience", f"{card.climate_resilience_score:.0f}/100")

            if card.has_legume:
                st.markdown(
                    f"<span class='status-pill' style='margin-bottom:8px;'>N Fixed: ~{card.nitrogen_credit_kg_ha:.0f} kg/ha</span>",
                    unsafe_allow_html=True,
                )

            with st.expander("Agronomic Assessment, Benefits & Risks", expanded=True):
                st.markdown(f"**Why this rotation:** {card.why_suggested}")

                st.markdown("**Practical Benefits:**")
                for b in card.benefits:
                    st.markdown(f"- {b}")

                st.markdown("**Risks & Watch Points:**")
                for r in card.risks:
                    st.markdown(f"- {r}")

                st.markdown("**Confirm Locally:**")
                for g in card.information_gaps:
                    st.markdown(f"- {g}")

                st.markdown(
                    f"""<div class="advisory-notice"><strong>Agronomist Note:</strong> {card.agronomist_note}</div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

    # Summary table
    st.markdown("### Comparative Performance & Uncertainty Summary")
    summary_rows = []
    for k in ["current", "soil_health", "water_saving", "climate_resilient"]:
        c = rotation_cards[k]
        m = mc_results[k]
        summary_rows.append({
            "Rotation": c.rotation_name,
            "Water Need (mm)": f"{c.total_water_need_mm:.0f}",
            "Water Score": f"{c.water_saving_score:.0f}/100",
            "Soil Score": f"{c.soil_health_score:.0f}/100",
            "Resilience": f"{c.climate_resilience_score:.0f}/100",
            "Score Range (10–90%)": f"{m.p10_score:.0f} – {m.p90_score:.0f}",
            "Win Rate": f"{m.win_rate_pct:.0f}%",
            "Confidence": m.confidence_level,
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    # What-if
    with st.expander("What-If Climate Stress Testing", expanded=False):
        sc1, sc2 = st.columns(2)
        rain_shift = sc1.slider("Rainfall scenario (%)", -30, 30, st.session_state["scenario_rain"], 5)
        temp_shift = sc2.slider("Temperature rise scenario (°C)", 0.0, 4.0, float(st.session_state["scenario_temp"]), 0.5)
        if rain_shift != st.session_state["scenario_rain"] or temp_shift != st.session_state["scenario_temp"]:
            st.session_state["scenario_rain"] = rain_shift
            st.session_state["scenario_temp"] = temp_shift
            st.rerun()

    persistent_expert_banner(is_urdu=is_urdu)


# ===========================================================================
# PAGE: REPORTS
# ===========================================================================
elif st.session_state["nav_page"] == "Reports":
    st.markdown("## Evaluation Report & Export")
    st.write("Summary of the farm evaluation for download or sharing with an agronomist.")

    report_data = {
        "Farm Location": site_info["name"],
        "Coordinates": f"{target_lat:.4f}° N, {target_lon:.4f}° E",
        "Soil Texture": st.session_state["soil_texture"],
        "Water Access": st.session_state["water_source"],
        "Current Rotation": ROTATION_CATALOG.get(st.session_state["current_rotation"], {}).get("name", ""),
        "10-Year Mean Rainfall": f"{avg_annual_rain:.0f} mm",
        "Annual Bare Soil Days": f"{modis_summary.annual_bare_days:.0f} days",
        "Top Recommended Rotation": best_card.rotation_name,
        "Recommendation Win Rate": f"{best_mc.win_rate_pct:.0f}% of simulations",
    }

    st.table(pd.DataFrame(list(report_data.items()), columns=["Parameter", "Value"]))

    export_df = pd.DataFrame([
        {
            "rotation_id": k,
            "rotation_name": rotation_cards[k].rotation_name,
            "water_need_mm": rotation_cards[k].total_water_need_mm,
            "water_score": rotation_cards[k].water_saving_score,
            "soil_score": rotation_cards[k].soil_health_score,
            "resilience_score": rotation_cards[k].climate_resilience_score,
            "composite_score": rotation_cards[k].composite_score,
            "win_rate_pct": mc_results[k].win_rate_pct,
            "confidence": mc_results[k].confidence_level,
        }
        for k in rotation_cards
    ])

    st.download_button(
        label="Download Rotation Scores (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"field_shift_scores_{st.session_state['site_key']}.csv",
        mime="text/csv",
    )

    persistent_expert_banner(is_urdu=is_urdu)


# ===========================================================================
# PAGE: DATA & METHOD
# ===========================================================================
elif st.session_state["nav_page"] == "Data & Method":
    st.markdown("## Data Sources & Scientific Methodology")
    st.caption("Full transparency on NASA products, spatial resolution, and model limitations.")

    st.markdown("### 1. NASA & Earth Observation Resources")
    res_table = [
        {"Resource": "NASA POWER AG Daily API", "Resolution": "~50 km (0.5° × 0.625°)", "Period": "2016–2025 (Daily)", "Role": "Precipitation, Tmax, Tmin, solar radiation, humidity, wind speed"},
        {"Resource": "NASA MERRA-2 (GWETROOT)", "Resolution": "~50 km", "Period": "2016–2025 (Daily)", "Role": "Model root-zone soil wetness (0 = dry, 1 = saturated)"},
        {"Resource": "MODIS Terra/Aqua (MOD13Q1)", "Resolution": "250 m", "Period": "16-day composite", "Role": "Vegetation NDVI — bare ground exposure (NDVI < 0.20)"},
        {"Resource": "FAO-56 Penman-Monteith", "Resolution": "Field scale", "Period": "Physical model", "Role": "Reference evapotranspiration (ET0) and crop water demand"},
    ]
    st.dataframe(pd.DataFrame(res_table), hide_index=True, use_container_width=True)

    st.markdown("### 2. Data Limitations")
    st.write(
        "NASA POWER and MERRA-2 provide gridded estimates (~50 km cells), not field-level weather station readings. "
        "Root-zone wetness (GWETROOT) is a physics-based model product, not a direct satellite measurement. "
        "Field Shift outputs are comparative guides intended to inform discussions with extension agronomists, not to replace them."
    )

    st.markdown("### 3. Scientific Glossary")
    glossary = {
        "Reference Evapotranspiration (ET0)": "Water loss rate from a 0.12 m grass surface under adequate water supply, per FAO-56 Penman-Monteith.",
        "Root-Zone Wetness (GWETROOT)": "Soil saturation degree from surface to ~1 m depth, scaled 0 (dry) to 1 (saturated), from MERRA-2.",
        "NDVI": "Normalized Difference Vegetation Index — satellite greenness indicator. Values < 0.20 indicate bare/unplanted ground.",
        "Biological Nitrogen Credit": "Atmospheric nitrogen fixed into plant-available form by legume root nodules (chickpea, mungbean, lentil), reducing fertilizer demand.",
        "Monte Carlo Simulation": "500 stochastic draws over uncertain agronomic and climate parameters to quantify rank stability and win probability.",
    }
    for term, defn in glossary.items():
        st.markdown(f"- **{term}**: {defn}")

    st.markdown("### 4. AI Contribution Disclosure")
    st.write(
        "Per NASA Space Apps 2026 guidelines, AI tools (Antigravity/Gemini) were used as engineering assistants "
        "for code structuring, formula translation, and documentation drafts. All agronomic logic, dataset "
        "verification, and submission decisions were governed by team members Ayesha Rafaqat and Sara Haris."
    )

    persistent_expert_banner(is_urdu=is_urdu)
