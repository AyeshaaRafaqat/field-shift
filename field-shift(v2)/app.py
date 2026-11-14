import pandas as pd
import streamlit as st
import engine as E

st.set_page_config(page_title="Field Shift – Rotation Explorer", page_icon="🌾", layout="wide")
st.title("🌾 Field Shift: Crop Rotation Explorer")
st.info("**Decision support, not a prescription.** This tool highlights patterns in NASA data to help you "
        "explore rotation options. Local agronomists and your own field knowledge should make the final call.")

SITES = E.SITES


@st.cache_data(show_spinner="Fetching 10 years of NASA POWER data…", ttl=86400)
def get_power(lat, lon):
    return E.fetch_power_cached(lat, lon)


@st.cache_data(show_spinner="Fetching MODIS NDVI…", ttl=86400)
def get_ndvi(lat, lon):
    return E.fetch_ndvi_cached(lat, lon)


@st.cache_resource(show_spinner="Training the ML soil-moisture model…")
def get_model(lat, lon):
    return E.train_soil_model(get_power(lat, lon)[0])


crops = E.load_crops("crops.csv")
kharif = list(crops.index[crops.season == "Kharif"])
rabi = list(crops.index[crops.season == "Rabi"])

# ------------------------------------------------------------ sidebar inputs
sb = st.sidebar
sb.header("1 · Your field")
site = sb.selectbox("Location", list(SITES) + ["Custom"])
if site == "Custom":
    lat = sb.number_input("Latitude", value=31.5, format="%.3f")
    lon = sb.number_input("Longitude", value=74.3, format="%.3f")
else:
    lat, lon = SITES[site]
soil = sb.selectbox("Soil texture", list(E.SOIL_MULT), index=1)
water = sb.selectbox("Irrigation water available", list(E.WATER_LIMIT), index=1)
limit = E.WATER_LIMIT[water]

sb.header("2 · Current rotation")
cur_k = sb.selectbox("Summer crop (Kharif, May–Oct)", kharif, index=kharif.index("Rice"))
cur_r = sb.selectbox("Winter crop (Rabi, Nov–Apr)", rabi, index=rabi.index("Wheat"))
cur = (cur_k, cur_r)

sb.header("3 · What matters to you?")
w_soil = sb.slider("Soil health", 0, 10, 7)
w_water = sb.slider("Saving water", 0, 10, 6)
w_res = sb.slider("Resilience to heat / frost / dry spells", 0, 10, 5)
w_inc = sb.slider("Income", 0, 10, 4)

sb.header("4 · What-if climate")
rain_pct = sb.slider("Rainfall change (%)", -30, 30, 0, 5)
d_temp = sb.slider("Warming (°C)", 0.0, 4.0, 0.0, 0.5)
use_ml = sb.checkbox("Use ML model to estimate how soil moisture would respond", value=True)

# ------------------------------------------------------------ data
try:
    raw, source = get_power(lat, lon)
except Exception as e:
    st.error(f"Could not fetch NASA POWER data (check your internet connection). Details: {e}")
    st.stop()

st.caption(f"Climate data: NASA POWER 2014–2024 · source: {source}")
model, info = get_model(lat, lon)
wet_thr = E.dry_soil_threshold(raw)                   # "dry soil" = driest 20% of days on record
wet = E.scenario_wetness(raw, model, 1 + rain_pct / 100, d_temp) if use_ml else None
tbl = E.season_table(raw, 1 + rain_pct / 100, d_temp, wet=wet, wet_thr=wet_thr)
if rain_pct or d_temp:
    st.warning(f"What-if scenario active: rainfall {rain_pct:+d}%, temperature +{d_temp}°C"
               + (" · soil moisture re-estimated by the ML model" if use_ml else ""))

tab1, tab2, tabml, tab3 = st.tabs(["1 · What is happening in my field?", "2 · Compare rotations",
                                    "3 · ML soil-moisture model", "4 · About & limits"])

# ------------------------------------------------------------ tab 1
with tab1:
    left, right = st.columns([1, 2])
    left.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
    summ = tbl.groupby("season").mean().rename(columns={
        "rain": "Rain (mm)", "heat": "Hot days", "frost": "Frost days",
        "dry_spell": "Longest dry spell (days)", "wetness": "Root-zone soil wetness (0–1)",
        "dry_soil": "Dry-soil days"}).round(2)
    right.subheader("10-year average per season")
    right.dataframe(summ, use_container_width=True)
    right.caption("Hot days = above 40 °C in Kharif, above 35 °C in Rabi. Frost days = minimum temperature ≤ 2 °C. "
                  "Dry spell = longest run of days with under 1 mm rain. Dry-soil day = root-zone wetness in the driest 20% of the record.")

    clim = E.monthly_climate(raw)
    c1, c2, c3 = st.columns(3)
    c1.subheader("Rainfall by month"); c1.bar_chart(clim["Rain (mm)"])
    c2.subheader("Temperature by month"); c2.line_chart(clim[["Max temp (°C)", "Min temp (°C)"]])
    c3.subheader("Soil wetness by month"); c3.line_chart(clim["Soil wetness (0-1)"])

    st.subheader("Rain varies a lot from year to year")
    st.bar_chart(tbl["rain"].unstack("season"))

    st.subheader("Bare-field periods (optional: satellite vegetation)")
    if st.checkbox("Load MODIS NDVI for this location (slower)"):
        try:
            ndvi, _ = get_ndvi(lat, lon)
            if ndvi.empty:
                st.info("No NDVI returned for this location.")
            else:
                st.line_chart(pd.DataFrame({"NDVI": ndvi, "Bare-soil line (0.25)": 0.25}))
                bare = E.bare_field_summary(ndvi)
                b1, b2 = st.columns(2)
                b1.metric("Approx. days per year with sparse vegetation (NDVI < 0.25)", f"{bare.sparse_days_per_year:.0f}")
                b2.metric("Longest sparse stretch (days)", f"{bare.longest_sparse_days}")
                st.caption("NDVI measures how green and dense vegetation is. Low values mean bare or fallow soil, "
                           "which is exposed to erosion and moisture loss.")
        except Exception as e:
            st.info(f"NDVI service unavailable right now, so skipping this layer. ({e})")

    with st.expander("Plain-language glossary"):
        st.markdown(
            "- **NDVI**: a satellite 'greenness' score, from about 0 (bare soil) to 0.9 (dense crop).\n"
            "- **Root-zone soil wetness**: how full of water the soil is where roots grow (0 = dry, 1 = saturated).\n"
            "- **Dry spell**: consecutive days with almost no rain.\n"
            "- **Satellite vs ground data**: satellites cover everywhere consistently but are coarse and indirect. "
            "A rain gauge or soil test on your farm is more precise but covers one spot. Using both is best.")

# ------------------------------------------------------------ tab 2
with tab2:
    options = E.evaluate_all(crops, tbl, cur, E.SOIL_MULT[soil], [w_soil, w_water, w_res, w_inc])
    picks = E.pick_strategies(options, cur, limit)
    base = picks[0][1]
    if base["irr"] > limit:
        st.warning(f"Your current rotation needs about {base['irr']:.0f} mm/yr of irrigation, "
                   f"more than your '{water}' water setting ({limit} mm/yr).")
    if len(picks) < 4:
        st.info("Few alternatives fit your water limit. Try raising it to see more options.")

    cols = st.columns(len(picks))
    for col, (title, o) in zip(cols, picks):
        with col:
            st.markdown(f"##### {title}")
            st.markdown(f"**{o['label']}**")
            delta = None if o is base else f"{o['total'] - base['total']:+.1f} vs current"
            st.metric("Overall score (0–10)", f"{o['total']:.1f}", delta)
            st.caption(f"Range over 10 years: {o['lo']:.1f}–{o['hi']:.1f} · confidence: {o['confidence']}")
            for name, key in [("Soil health", "soil"), ("Water fit", "water"),
                              ("Resilience", "resil"), ("Income (rough)", "income")]:
                st.progress(min(1.0, o[key] / 10), text=f"{name}: {o[key]:.1f}")
            with st.expander("Why suggested"):
                for b in E.why(o, base): st.markdown(f"- {b}")
            with st.expander("Evidence & benefits"):
                st.markdown(f"- Irrigation gap about **{o['irr']:.0f} mm/yr** (rain counted at 80%)\n"
                            f"- Crop cover about **{o['cover'] * 365:.0f} days/yr** (≈{o['bare']:.0f} bare)\n"
                            f"- Stress index {o['stress']:.2f} (heat + frost + dry spells; lower is better)")
                for b in E.benefits(o): st.markdown(f"- {b}")
            with st.expander("Risks & gaps: needs local review"):
                for b in E.risks(o, soil, limit): st.markdown(f"- {b}")

    st.subheader("Side-by-side")
    chart = pd.DataFrame({f"{t}: {o['label']}": {"Soil health": o["soil"], "Water fit": o["water"],
                                                 "Resilience": o["resil"], "Income (rough)": o["income"]}
                          for t, o in picks})
    st.bar_chart(chart)

    st.warning("**Where human judgment matters most:** pest pressure, seed and market availability, "
               "machinery and labour, canal/tubewell reliability, and your own experience of this field.")

# ------------------------------------------------------------ ML tab
with tabml:
    st.markdown(
        "**What the model does.** A random-forest model learns, from this location's own 10-year record, how "
        "root-zone soil moisture responds to weather (recent rainfall, temperature, time of year). "
        "For the what-if sliders we change the weather, ask the model what soil moisture would then be, "
        "and add that *change* to the observed record. That feeds the dry-soil days used in the resilience score.")
    quality = "good" if info["r2"] >= 0.8 else "moderate" if info["r2"] >= 0.5 else "weak"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² on unseen 2022–24 data", f"{info['r2']:.2f}")
    m2.metric("Average error (soil wetness units)", f"{info['mae']:.3f}")
    m3.metric("Model quality here", quality.capitalize())
    m4.metric("Plain monthly-average guess (R²)", f"{info['baseline_r2']:.2f}")
    if info["r2"] <= info["baseline_r2"]:
        st.warning("The ML model does not beat a plain monthly average here, so it adds little for this location.")
    if quality == "weak":
        st.warning("The model explains soil moisture poorly at this location, so treat what-if soil results as very uncertain.")

    st.subheader("Does it work? Predicted vs. actual on held-out data")
    st.line_chart(pd.DataFrame({"Actual": info["test_obs"], "Predicted": info["test_pred"]}))
    st.caption("The model was trained only on data before May 2022, then asked to predict May 2022–April 2024 without seeing it.")

    st.subheader("What drives soil moisture here?")
    st.bar_chart(info["importance"])

    st.subheader("Baseline vs. scenario soil wetness (monthly average)")
    if use_ml and (rain_pct or d_temp):
        share = E.extrapolation_share(raw, info, 1 + rain_pct / 100, d_temp)
        if share > 0.05:
            st.warning(f"{share:.0%} of days in this scenario are hotter or wetter than anything the model saw in "
                       "training, so treat this scenario's soil estimates with extra caution.")
        mm = raw.index.month
        st.line_chart(pd.DataFrame({"Observed baseline": raw["GWETROOT"].groupby(mm).mean(),
                                    "Under your scenario": wet.groupby(mm).mean()}))
    else:
        st.info("Move the rainfall or warming sliders in the sidebar (with the ML option ticked) to see the model's estimate.")
    st.caption("Limits: a random forest cannot extrapolate beyond weather it has seen, so large changes (e.g. +4 °C) are less reliable. "
               "NASA POWER soil wetness is itself a model product, not a field measurement: the ML learns that relationship, "
               "and a soil-moisture probe on the farm would be the best way to check it.")

# ------------------------------------------------------------ tab 3
with tab3:
    st.markdown("""
### Data used
- **NASA POWER** (daily, 2014–2024, ~50 km grid): rainfall, max/min temperature, root-zone soil wetness.
- **MODIS NDVI** (250 m, optional): vegetation greenness for bare-field periods.
- **Local inputs:** soil texture class, water availability, current crops, priorities.
- **Crop traits table** (`crops.csv`): water needs and season lengths cross-checked against FAO, ICAR and Punjab (Pakistan/India) sources, listed in `CROP_DATA_SOURCES.md`. The 1–5 ratings and the income index are judgment, so replace them with local figures.

### How scores work (all transparent, no black box)
Each rotation (one Kharif + one Rabi crop) is scored 0–10 on four criteria, then combined using your sliders.
- **Soil health:** nitrogen fixation, residue cover, root depth, crop-family diversity, pest-cycle break vs your current rotation, days of ground cover.
- **Water fit:** estimated irrigation need = crop water need (adjusted for soil) minus 80% of seasonal rain.
- **Resilience:** exposure of each crop to hot days, frost days, dry spells and dry-soil days in the 10-year record. Under a what-if scenario, dry-soil days come from the ML model.
- **Income:** a rough 1–5 crop index, **not real prices**.

Scores are recomputed for each historical year, so the range shows how much weather variability affects each option.

### Machine learning
A random-forest soil-moisture model (see tab 3) is validated on a time-based hold-out and used for what-if scenarios.
The rotation scores themselves stay rule-based so every number is explainable.

### Limitations
- Climate data comes from a ~50 km model grid, not from your field. Satellite estimates differ from ground measurements.
- Soil is a texture class only: no organic carbon, salinity or pH.
- Income, pest pressure, seed and market access are not modelled in detail.
- Future work: SMAP soil moisture, ECOSTRESS heat stress, GRACE groundwater trends, SRTM terrain/erosion risk, GEDI biomass.
""")
