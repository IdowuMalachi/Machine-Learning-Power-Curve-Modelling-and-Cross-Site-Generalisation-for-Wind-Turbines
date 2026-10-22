"""Wind turbine power prediction app (Penmanshiel, Senvion MM82).

Run with:  streamlit run app.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core import (RATED_KW, KNOWN_TURBINES, read_upload, guess_columns, standardise,
                  normal_operation_mask, choose_model, predict, scores)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "power_models.joblib")
TRAIN_YEARS = range(2017, 2023)
ACCENT, GREY = "#2a78d6", "#8a8a8a"

st.set_page_config(page_title="Wind Turbine Power Prediction", layout="wide")


@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)


bundle = load_models()

st.title("Wind turbine power prediction")
st.write(
    "Upload 10-minute turbine data to predict power output with the gradient boosting model from the report "
    "*Predicting Wind Turbine Power Output in Future Years*. If the file includes measured power, the app also "
    "scores the predictions."
)

with st.sidebar:
    st.header("About the model")
    st.write(f"**Trained on:** {bundle['info']['trained_on']} ({bundle['info']['rows']:,} rows).")
    st.write(f"**Model:** {bundle['info']['model']}.")
    st.write(
        "**Two versions:** a turbine-aware model, used when every row is Penmanshiel turbine 11 to 15, "
        "and a general model (wind speed, direction, temperature, hour of day) for any other turbine."
    )
    st.write(
        "**Valid for:** Senvion MM82-type turbines (2,050 kW, 82 m rotor) in conditions like Penmanshiel's. "
        "Other turbine types will be predicted with a Penmanshiel power curve."
    )

st.subheader("1. Upload data")
st.caption(
    "Accepted: the cleaned yearly CSVs (penmanshiel_<year>_clean.csv), a raw Greenbyte file from Zenodo "
    "(Turbine_Data_Penmanshiel_NN_...csv), or any CSV with time, wind speed, wind direction and temperature columns."
)
uploaded = st.file_uploader("CSV file", type=["csv", "txt"])

if uploaded is None:
    st.info("Upload a file to start.")
    st.stop()

try:
    raw, fmt = read_upload(uploaded.getvalue(), uploaded.name)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the file: {exc}")
    st.stop()

st.success(f"Read {len(raw):,} rows ({fmt}).")
with st.expander("Preview the first rows"):
    st.dataframe(raw.head(20), use_container_width=True)

st.subheader("2. Check the columns")
guess = guess_columns(raw)
options = ["(none)"] + list(raw.columns)
labels = {
    "time": "Timestamp (required)",
    "wind_speed": "Wind speed, m/s (required)",
    "wind_dir": "Wind direction, degrees (required)",
    "temp": "Ambient temperature, °C (required)",
    "power": "Measured power, kW (optional: for scoring)",
    "turbine": "Turbine number (optional)",
    "pitch": "Blade pitch angle, degrees (optional: for filtering)",
    "stopped": "Stopped flag, True/False (optional: for filtering)",
}
cols = st.columns(4)
mapping = {}
for i, (key, label) in enumerate(labels.items()):
    default = options.index(guess[key]) if guess.get(key) in options else 0
    choice = cols[i % 4].selectbox(label, options, index=default, key=f"map_{key}")
    mapping[key] = None if choice == "(none)" else choice

missing = [labels[k] for k in ["time", "wind_speed", "wind_dir", "temp"] if mapping[k] is None]
if missing:
    st.warning("Choose a column for: " + "; ".join(missing))
    st.stop()

df = standardise(raw, mapping)
if df["time"].isna().all():
    st.error("The timestamp column could not be read as dates.")
    st.stop()

has_power = "power" in df and df["power"].notna().any()

st.subheader("3. Options")
c1, c2 = st.columns(2)
auto_kind = choose_model(df)
kind = c1.radio(
    "Model",
    ["turbine", "general"],
    index=0 if auto_kind == "turbine" else 1,
    format_func=lambda k: "Turbine-aware (Penmanshiel turbines 11–15)" if k == "turbine" else "General (any turbine)",
    disabled=auto_kind != "turbine",
    help="The turbine-aware model is only available when every row has a turbine number from 11 to 15.",
)
filter_rows = c2.checkbox(
    "Score only normal operation",
    value=True,
    disabled=not has_power,
    help="Applies the report's cleaning rules before scoring: missing and impossible values, near-zero power, "
         "pitched-out blades, capped power and outliers, plus logged stops when a stopped column is mapped. "
         "Predictions are still made for every row.",
)

df["predicted_power"] = predict(bundle, df, kind)
n_pred = int(df["predicted_power"].notna().sum())
if n_pred == 0:
    st.error("No row has all of wind speed, direction, temperature and time, so nothing could be predicted.")
    st.stop()

years = sorted(df["time"].dt.year.dropna().unique().astype(int))
in_training = [y for y in years if y in TRAIN_YEARS]
if kind == "turbine" and in_training:
    st.warning(
        f"Years {', '.join(map(str, in_training))} were part of the training data for turbines 11–15, so scores on "
        "them are optimistic. For a fair test, use a year or turbine the model has not seen."
    )
elif in_training:
    st.info(
        "If this file is Penmanshiel turbine 11–15 data from 2017–2022, it was part of the training data and the "
        "scores will be optimistic. Data from any other turbine or year is a fair test."
    )

st.subheader("4. Results")
st.write(f"Predicted power for **{n_pred:,}** of {len(df):,} rows "
         f"({'turbine-aware' if kind == 'turbine' else 'general'} model).")

scored = df
if has_power:
    if filter_rows:
        keep = normal_operation_mask(df)
        scored = df[keep]
        st.caption(f"Scoring {len(scored):,} rows of normal operation; {len(df) - len(scored):,} rows filtered out.")
    s = scores(scored["power"], scored["predicted_power"])
    if s:
        m = st.columns(5)
        m[0].metric("MAE", f"{s['MAE (kW)']:.1f} kW")
        m[1].metric("MAE, % of rated", f"{s['MAE (% of rated)']:.2f}%")
        m[2].metric("RMSE", f"{s['RMSE (kW)']:.1f} kW")
        m[3].metric("R²", f"{s['R²']:.4f}")
        m[4].metric("Bias", f"{s['bias (kW)']:+.1f} kW", help="Predicted minus measured; positive = over-predicting.")
        st.caption("For reference, the report's test error on unseen years was 30–36 kW MAE (1.5–1.8% of rated).")
else:
    st.info("No measured power column, so the app shows predictions only.")

tab1, tab2, tab3 = st.tabs(["Power curve", "Predicted vs measured", "Over time"])
sample = df.dropna(subset=["predicted_power"]).sample(min(20000, n_pred), random_state=0)

with tab1:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if has_power:
        ax.scatter(sample["wind_speed"], sample["power"], s=2, alpha=0.25, color=GREY, label="Measured")
    ax.scatter(sample["wind_speed"], sample["predicted_power"], s=2, alpha=0.4, color=ACCENT, label="Predicted")
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power (kW)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, markerscale=5)
    st.pyplot(fig)

with tab2:
    if has_power:
        sc = scored.dropna(subset=["predicted_power", "power"])
        sc = sc.sample(min(20000, len(sc)), random_state=0)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(sc["power"], sc["predicted_power"], s=2, alpha=0.3, color=ACCENT)
        ax.plot([0, RATED_KW], [0, RATED_KW], color="0.4", linewidth=1)
        ax.set_xlabel("Measured power (kW)")
        ax.set_ylabel("Predicted power (kW)")
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=False)
    else:
        st.write("Needs a measured power column.")

with tab3:
    daily = df.set_index("time").resample("D")[["predicted_power"] + (["power"] if has_power else [])].mean()
    daily = daily.rename(columns={"predicted_power": "Predicted (kW)", "power": "Measured (kW)"})
    st.line_chart(daily, color=[ACCENT, GREY][: daily.shape[1]])
    st.caption("Daily mean power.")
    if has_power:
        monthly = scored.assign(err=scored["predicted_power"] - scored["power"]).set_index("time")["err"]
        monthly = monthly.resample("MS").agg(["mean", lambda e: e.abs().mean()])
        monthly.columns = ["Bias (kW)", "MAE (kW)"]
        monthly.index = monthly.index.strftime("%Y-%m")
        st.write("Error by month (rows scored):")
        st.dataframe(monthly.round(1), use_container_width=True)

st.subheader("5. Download")
out = df.copy()
out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
if has_power:
    out["error_kw"] = out["predicted_power"] - out["power"]
st.download_button(
    "Download predictions (CSV)",
    out.to_csv(index=False).encode("utf-8"),
    file_name=os.path.splitext(uploaded.name)[0] + "_predictions.csv",
    mime="text/csv",
)
