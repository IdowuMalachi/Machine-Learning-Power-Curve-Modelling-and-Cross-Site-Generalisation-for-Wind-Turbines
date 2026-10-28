"""Wind turbine power prediction app for the CSCI 6505 project
"Predicting Wind Turbine Power Output in Future Years" (Idowu Malachi and Sharon Sarker).

Run with:  streamlit run app.py
"""
import io
import os
import json
import glob

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import core
from core import RATED_KW

HERE = os.path.dirname(os.path.abspath(__file__))
# Your own notebook output comes first (all 14 turbines); the copy in models/ is the fallback.
MODEL_CANDIDATES = [
    os.path.join(HERE, "results_split", "power_models.joblib"),
    os.path.join(HERE, "..", "results_split", "power_models.joblib"),
    os.path.join(HERE, "models", "power_models.joblib"),
    os.path.join(HERE, "power_models.joblib"),
]
RESULT_CANDIDATES = {
    "Your notebook run (results_split, next to the app)": os.path.join(HERE, "results_split"),
    "Your notebook run (results_split, one folder up)": os.path.join(HERE, "..", "results_split"),
    "Built-in copy of the report run": os.path.join(HERE, "results_builtin"),
}
SAMPLE_FILE = os.path.join(HERE, "sample_penmanshiel13_jan2019.csv")

BLUE, GREY, ORANGE, RED, GREEN = "#2a78d6", "#8a8a8a", "#e8772e", "#c0392b", "#2e9d5b"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "axes.titlesize": 11})

st.set_page_config(page_title="Wind Turbine Power Prediction", page_icon="🌬️", layout="wide")


# =================================================================== loading

@st.cache_resource(show_spinner="Loading the model…")
def load_bundle_from_path(path):
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading the model…")
def load_bundle_from_bytes(raw):
    return joblib.load(io.BytesIO(raw))


@st.cache_data(show_spinner="Reading the file…")
def read_files(files):
    """files: tuple of (name, bytes). Returns one table, the formats found, and per-file row counts."""
    frames, fmts, counts = [], set(), []
    for name, raw in files:
        df, fmt = core.read_upload(raw, name)
        df["source_file"] = name
        frames.append(df)
        fmts.add(fmt)
        counts.append((name, len(df)))
    return pd.concat(frames, ignore_index=True), sorted(fmts), counts


@st.cache_data(show_spinner="Predicting power…")
def cached_predict(_bundle, bundle_key, df, kind, turbines, ws_scale):
    return core.predict(_bundle, df, kind, list(turbines), ws_scale)


@st.cache_data(show_spinner="Applying the cleaning rules…")
def cached_flags(df):
    return core.cleaning_flags(df)


def fig_out(fig):
    st.pyplot(fig)
    plt.close(fig)


# =================================================================== sidebar: model

with st.sidebar:
    st.title("🌬️ Wind power")
    page = st.radio("Page", ["Overview", "Predict from data", "What-if and energy", "Model results",
                             "Data and methods"], label_visibility="collapsed")
    st.divider()

    bundle, bundle_key = None, None
    upload_model = st.file_uploader("Use a different model file (optional)", type=["joblib"],
                                    help="The power_models.joblib saved by Train_Chosen_Years.ipynb, in results_split.")
    if upload_model is not None:
        bundle, bundle_key = load_bundle_from_bytes(upload_model.getvalue()), "upload:" + upload_model.name
    else:
        for p in MODEL_CANDIDATES:
            if os.path.exists(p):
                bundle, bundle_key = load_bundle_from_path(p), os.path.abspath(p)
                break

    if bundle is None:
        st.error("No model file found. Put power_models.joblib in the models folder, or upload it above.")
        st.stop()

    TURBINES = core.bundle_turbines(bundle)
    TRAIN_YEARS = core.bundle_train_years(bundle)
    info = bundle.get("info", {})
    st.caption("**Model in use**")
    st.caption(f"Gradient boosting, trained on Penmanshiel turbines {', '.join(map(str, TURBINES))}, "
               f"years {', '.join(map(str, TRAIN_YEARS))} ({info.get('rows', 0):,} rows).")
    st.caption(f"scikit-learn {info.get('sklearn_version', '?')}")


def turbine_label(k):
    return "Turbine-aware (Penmanshiel turbines)" if k == "turbine" else "General (any turbine)"


# =================================================================== page: overview

def page_overview():
    st.title("Predicting wind turbine power output in future years")
    st.caption("CSCI 6505 Machine Learning, Fall 2026 · Idowu Malachi and Sharon Sarker · Dalhousie University")

    st.markdown(
        "**Research question.** How can we best predict a wind turbine's power output in years the model has "
        "never seen, and which model, inputs and training-data choices work best?\n\n"
        "A turbine's power curve is the standard way to estimate output from wind speed. This project tests whether "
        "machine learning, using wind speed plus direction, temperature, time of day and which turbine it is, "
        "predicts future years better than the industry-standard binned power curve."
    )

    c = st.columns(4)
    c[0].metric("Best model", "Gradient boosting")
    c[1].metric("Test error (2023-2024)", "32.0 kW MAE", help="Mean of 31.0 (2023) and 32.9 (2024).")
    c[2].metric("Share of rated power", "1.56%")
    c[3].metric("Better than power curve", "20%", help="Binned curve per turbine: 39.9 kW mean MAE on the same years.")

    st.subheader("How the study was set up")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "1. **Data.** Penmanshiel wind farm, Scotland: 14 Senvion MM82 turbines (2,050 kW, 82 m rotor), "
            "10-minute SCADA records 2016-2024, open on Zenodo (Cubico, CC-BY-4.0).\n"
            "2. **Cleaning.** Seven rules keep only normal operation: logged stops, missing and impossible values, "
            "non-producing, pitched-out, capped and outlier rows are removed.\n"
            "3. **Split by year.** Training years 2016, 2017, 2018, 2020, 2021, 2022. Test years 2023 and 2024 are "
            "never used to choose anything.\n"
            "4. **Model selection.** Leave-one-year-out cross-validation inside the training years: each training "
            "year is held out in turn, so whole seasons are tested and neighbouring rows never straddle a split.\n"
            "5. **Final test.** The chosen model is refitted on all training years and scored on each test year, per "
            "turbine and by wind speed band."
        )
    with right:
        fig, ax = plt.subplots(figsize=(4.6, 2.6))
        years = list(range(2016, 2025))
        colors = [BLUE if y in TRAIN_YEARS else (ORANGE if y in (2023, 2024) else "#d0d0d0") for y in years]
        ax.bar([str(y) for y in years], [1] * len(years), color=colors)
        ax.set_yticks([])
        ax.grid(False)
        for s in ["left"]:
            ax.spines[s].set_visible(False)
        ax.set_title("Blue: training · orange: test · grey: not used")
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig_out(fig)

    st.subheader("What you can do here")
    st.markdown(
        "- **Predict from data:** upload 10-minute SCADA data (cleaned, raw Greenbyte, or any CSV) and get "
        "predicted power, accuracy scores, expected energy and a day-by-day performance check.\n"
        "- **What-if and energy:** set wind speed, direction, temperature and time of day and see the predicted "
        "power curve, then estimate a year's energy from an average wind speed.\n"
        "- **Model results:** the cross-validation and test results from the training notebook.\n"
        "- **Data and methods:** the cleaning rules, inputs, models and limits."
    )


# =================================================================== page: predict

def page_predict():
    st.title("Predict power from turbine data")

    st.subheader("1. Upload data")
    st.caption("Accepted: the cleaned yearly CSVs (penmanshiel_<year>_clean.csv), raw Greenbyte files from Zenodo "
               "(Turbine_Data_Penmanshiel_NN_….csv, unzipped), or any CSV with time, wind speed, wind direction and "
               "temperature. You can upload several files at once, for example one per turbine.")
    uploads = st.file_uploader("CSV files", type=["csv", "txt"], accept_multiple_files=True)
    use_sample = False
    if not uploads and os.path.exists(SAMPLE_FILE):
        use_sample = st.checkbox("No file? Use the sample: Penmanshiel turbine 13, January 2019")
    if not uploads and not use_sample:
        st.info("Upload a file to start.")
        return

    files = tuple((u.name, u.getvalue()) for u in uploads) if uploads else \
        ((os.path.basename(SAMPLE_FILE), open(SAMPLE_FILE, "rb").read()),)
    try:
        raw, fmts, counts = read_files(files)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the file: {exc}")
        return
    st.success(f"Read {len(raw):,} rows from {len(counts)} file(s) ({', '.join(fmts)}).")
    with st.expander("Preview the first rows"):
        st.dataframe(raw.head(20))

    # ---------- columns
    st.subheader("2. Check the columns")
    guess = core.guess_columns(raw)
    options = ["(none)"] + [c for c in raw.columns if c != "source_file"]
    labels = {
        "time": "Timestamp (required)",
        "wind_speed": "Wind speed, m/s (required)",
        "wind_dir": "Wind direction, degrees (required)",
        "temp": "Ambient temperature, °C (required)",
        "power": "Measured power, kW (for scoring)",
        "turbine": "Turbine number (optional)",
        "pitch": "Blade pitch, degrees (for cleaning)",
        "stopped": "Stopped flag, True/False (for cleaning)",
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
        return

    df = core.standardise(raw, mapping)
    df["source_file"] = raw["source_file"].values
    if df["time"].isna().all():
        st.error("The timestamp column could not be read as dates.")
        return
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if "turbine" not in df and len(counts) > 1:
        st.info("Several files but no turbine column: each file is treated as a different, unknown turbine.")

    # ---------- options
    st.subheader("3. Options")
    o1, o2, o3 = st.columns(3)
    tmin, tmax = df["time"].min().date(), df["time"].max().date()
    date_range = o1.date_input("Date range", (tmin, tmax), min_value=tmin, max_value=tmax)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        df = df[(df["time"].dt.date >= date_range[0]) & (df["time"].dt.date <= date_range[1])].reset_index(drop=True)
    if "turbine" in df and df["turbine"].notna().any():
        present = sorted(int(t) for t in df["turbine"].dropna().unique())
        chosen = o1.multiselect("Turbines", present, default=present)
        df = df[df["turbine"].isin(chosen)].reset_index(drop=True)
    if df.empty:
        st.warning("No rows left after the date and turbine filters.")
        return

    auto_kind = core.choose_model(df, TURBINES)
    kind = o2.radio("Model", ["turbine", "general"], index=0 if auto_kind == "turbine" else 1,
                    format_func=turbine_label, disabled=auto_kind != "turbine",
                    help=f"The turbine-aware model needs every row to be one of Penmanshiel turbines "
                         f"{', '.join(map(str, TURBINES))}. Anything else uses the general model.")
    ws_scale = 1.0
    if kind == "general":
        rotor = o2.number_input("Rotor diameter of the uploaded turbine (m)", 40.0, 200.0, float(core.ROTOR_M), 1.0,
                                help="For a different turbine type, wind speed is rescaled by "
                                     "(diameter / 82 m)^(2/3) so the same wind gives the same power per swept area. "
                                     "Kelmarsh's MM92 is 92 m.")
        ws_scale = (rotor / core.ROTOR_M) ** (2 / 3)
        if abs(ws_scale - 1) > 1e-9:
            o2.caption(f"Wind speed multiplied by {ws_scale:.3f}.")
    has_power = "power" in df and df["power"].notna().any()
    filter_rows = o3.checkbox("Score only normal operation", value=True, disabled=not has_power,
                              help="Applies the seven cleaning rules before scoring. Predictions are still made "
                                   "for every row. Rules 1 and 5 need a stopped and a pitch column.")

    # ---------- predict
    df["predicted_power"] = cached_predict(bundle, bundle_key, df, kind, tuple(TURBINES), ws_scale).values
    n_pred = int(df["predicted_power"].notna().sum())
    if n_pred == 0:
        st.error("No row has wind speed, direction, temperature and time, so nothing could be predicted.")
        return

    years = sorted(df["time"].dt.year.unique().astype(int))
    seen_years = [y for y in years if y in TRAIN_YEARS]
    seen_turb = "turbine" in df and df["turbine"].isin(TURBINES).any()
    if seen_years and seen_turb:
        st.warning(f"Years {', '.join(map(str, seen_years))} of Penmanshiel turbines were used to train the model, "
                   "so scores on them are optimistic. For a fair test, use another year (2019, 2023, 2024) "
                   "or another turbine.")

    scored = df
    if has_power:
        df["clean_rule"] = cached_flags(df).values
        if filter_rows:
            scored = df[df["clean_rule"] == ""]

    # ---------- results
    st.subheader("4. Results")
    st.write(f"Predicted power for **{n_pred:,}** of {len(df):,} rows with the **{turbine_label(kind).lower()}** "
             f"model, {df['time'].min():%d %b %Y} to {df['time'].max():%d %b %Y}.")
    if has_power:
        s = core.scores(scored["power"], scored["predicted_power"])
        if s:
            m = st.columns(5)
            m[0].metric("MAE", f"{s['MAE']:.1f} kW")
            m[1].metric("MAE, % of rated", f"{s['MAE %']:.2f}%")
            m[2].metric("RMSE", f"{s['RMSE']:.1f} kW")
            m[3].metric("R²", f"{s['R2']:.4f}")
            m[4].metric("Bias", f"{s['bias']:+.1f} kW", help="Predicted minus measured; + means over-predicting.")
            st.caption(f"Scored on {s['rows']:,} rows"
                       f"{' of normal operation' if filter_rows else ''}. For reference, the report's test error on "
                       "2023 and 2024 was 31.0 and 32.9 kW MAE (about 1.5-1.6% of rated).")
        e = st.columns(3)
        e_pred, e_meas = core.energy_mwh(scored["predicted_power"]), core.energy_mwh(scored["power"])
        e[0].metric("Predicted energy", f"{e_pred:,.1f} MWh")
        e[1].metric("Measured energy", f"{e_meas:,.1f} MWh")
        e[2].metric("Difference", f"{(e_pred - e_meas) / e_meas * 100:+.2f}%" if e_meas else "n/a")
        st.caption("Energy assumes 10-minute rows: each row's mean power × 1/6 hour, over the rows scored.")
    else:
        st.info("No measured power column, so the app shows predictions only.")
        st.metric("Predicted energy over the file", f"{core.energy_mwh(df['predicted_power']):,.1f} MWh")

    names = ["Power curve", "Predicted vs measured", "Over time", "Error analysis", "Performance check",
             "Cleaning"]
    tabs = st.tabs(names)
    sample = df.dropna(subset=["predicted_power"]).sample(min(20000, n_pred), random_state=0)

    with tabs[0]:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        if has_power:
            ax.scatter(sample["wind_speed"], sample["power"], s=2, alpha=0.2, color=GREY, label="Measured")
            bc = core.binned_curve(scored["wind_speed"], scored["power"])
            ax.plot(bc["ws"], bc["p"], color="black", lw=1.5, label="Measured, binned")
        ax.scatter(sample["wind_speed"], sample["predicted_power"], s=2, alpha=0.3, color=BLUE, label="Predicted")
        bp = core.binned_curve(df["wind_speed"], df["predicted_power"])
        ax.plot(bp["ws"], bp["p"], color=BLUE, lw=2, label="Predicted, binned")
        ax.axhline(RATED_KW, color=GREY, ls=":", lw=1)
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Power (kW)")
        ax.legend(frameon=False, markerscale=5, loc="lower right")
        plt.tight_layout()
        fig_out(fig)
        st.caption("Points: up to 20,000 random rows. Lines: mean power in 0.5 m/s wind speed bins. The spread of "
                   "predicted points at the same wind speed comes from direction, temperature, time of day and turbine.")

    with tabs[1]:
        if has_power:
            sc = scored.dropna(subset=["predicted_power", "power"])
            c1, c2 = st.columns(2)
            fig, ax = plt.subplots(figsize=(5, 5))
            hb = ax.hexbin(sc["power"], sc["predicted_power"], gridsize=60, bins="log", cmap="Blues", mincnt=1)
            ax.plot([0, RATED_KW], [0, RATED_KW], color=RED, lw=1)
            ax.set_xlabel("Measured power (kW)")
            ax.set_ylabel("Predicted power (kW)")
            ax.set_title("Each hexagon: number of rows (log scale)")
            fig.colorbar(hb, ax=ax, shrink=0.8)
            plt.tight_layout()
            with c1:
                fig_out(fig)
            err = sc["predicted_power"] - sc["power"]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hist(err.clip(-400, 400), bins=80, color=BLUE)
            ax.axvline(0, color="black", lw=1)
            ax.set_xlabel("Error, predicted − measured (kW, clipped at ±400)")
            ax.set_ylabel("Rows")
            ax.set_title(f"Error distribution · median {err.median():+.1f} kW")
            plt.tight_layout()
            with c2:
                fig_out(fig)
        else:
            st.write("Needs a measured power column.")

    with tabs[2]:
        rule = {"Hour": "h", "Day": "D", "Week": "W", "Month": "MS"}
        step = st.radio("Average over", list(rule), index=1, horizontal=True)
        base = scored if has_power and filter_rows else df
        series = base.set_index("time")[["predicted_power"] + (["power"] if has_power else [])]
        agg = series.resample(rule[step]).mean().rename(columns={"predicted_power": "Predicted (kW)",
                                                                   "power": "Measured (kW)"})
        st.line_chart(agg, color=[BLUE, GREY][: agg.shape[1]])
        st.caption(f"Mean power per {step.lower()}" + (", rows of normal operation." if has_power and filter_rows else "."))

    with tabs[3]:
        if has_power:
            c1, c2 = st.columns(2)
            band = core.error_by_band(scored)
            with c1:
                st.markdown("**By wind speed band**")
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                ax.bar(band.index, band["MAE (kW)"], color=BLUE)
                ax.set_ylabel("MAE (kW)")
                plt.xticks(rotation=30)
                plt.tight_layout()
                fig_out(fig)
                st.dataframe(band.round(1))
            with c2:
                st.markdown("**By month**")
                e = scored.assign(err=scored["predicted_power"] - scored["power"]).set_index("time")["err"]
                monthly = pd.DataFrame({"rows": e.resample("MS").size(), "MAE (kW)": e.abs().resample("MS").mean(),
                                        "bias (kW)": e.resample("MS").mean()}).dropna()
                monthly.index = monthly.index.strftime("%Y-%m")
                st.dataframe(monthly.round(1))
                if "turbine" in scored and scored["turbine"].notna().any():
                    st.markdown("**By turbine**")
                    rows = []
                    for t, g in scored.groupby("turbine"):
                        r = core.scores(g["power"], g["predicted_power"])
                        if r:
                            rows.append({"turbine": int(t), "rows": r["rows"], "MAE (kW)": r["MAE"],
                                         "bias (kW)": r["bias"], "R²": r["R2"],
                                         "seen in training": "yes" if int(t) in TURBINES else "no"})
                    st.dataframe(pd.DataFrame(rows).set_index("turbine").round(3))
            st.caption("The middle of the power curve (about 8-12 m/s), where power changes fastest with wind speed, "
                       "has the largest errors; near cut-in and at rated power errors are small.")
        else:
            st.write("Needs a measured power column.")

    with tabs[4]:
        if has_power:
            st.markdown("Compares measured energy with the model's expected energy for each day, using all rows "
                        "(including stops and curtailment). Days well below 100% point to lost production: "
                        "downtime, curtailment, icing or a fault.")
            thr = st.slider("Flag days below this share of expected energy (%)", 50, 100, 90)
            group = ["turbine"] if "turbine" in df and df["turbine"].notna().any() else []
            d = df.dropna(subset=["power", "predicted_power"]).copy()
            d["day"] = d["time"].dt.floor("D")
            daily = d.groupby(group + ["day"]).agg(measured=("power", "sum"), expected=("predicted_power", "sum"),
                                                   rows=("power", "size"))
            daily[["measured", "expected"]] = daily[["measured", "expected"]] / 6000  # kW·10 min -> MWh
            daily = daily[daily["expected"] > 0.5]
            daily["ratio %"] = daily["measured"] / daily["expected"] * 100
            daily["lost MWh"] = (daily["expected"] - daily["measured"]).clip(lower=0)
            flagged = daily[daily["ratio %"] < thr].sort_values("lost MWh", ascending=False)
            c = st.columns(3)
            c[0].metric("Days checked", f"{len(daily):,}")
            c[1].metric(f"Days below {thr}%", f"{len(flagged):,}")
            c[2].metric("Energy lost on those days", f"{flagged['lost MWh'].sum():,.1f} MWh")
            fig, ax = plt.subplots(figsize=(9, 3.5))
            dd = daily.reset_index()
            ax.scatter(dd["day"], dd["ratio %"].clip(0, 150), s=8,
                       c=np.where(dd["ratio %"] < thr, RED, BLUE))
            ax.axhline(100, color="black", lw=1)
            ax.axhline(thr, color=RED, lw=1, ls="--")
            ax.set_ylabel("Measured / expected energy (%)")
            plt.tight_layout()
            fig_out(fig)
            show = flagged.reset_index()
            show["day"] = show["day"].dt.strftime("%Y-%m-%d")
            st.dataframe(show.round(2).head(200))
        else:
            st.write("Needs a measured power column.")

    with tabs[5]:
        if has_power:
            counts_rule = df["clean_rule"].replace("", "kept (normal operation)").value_counts()
            tbl = pd.DataFrame({"rows": counts_rule, "share %": counts_rule / len(df) * 100})
            st.dataframe(tbl.round(2))
            skipped = [r for r, c in [("1 (needs a stopped column)", "stopped"), ("5 (needs a pitch column)", "pitch")]
                       if c not in df]
            if skipped:
                st.caption("Rules skipped: " + ", ".join(skipped) + ".")
            s2 = df.sample(min(20000, len(df)), random_state=0)
            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            k = s2["clean_rule"] == ""
            ax.scatter(s2.loc[k, "wind_speed"], s2.loc[k, "power"], s=2, alpha=0.3, color=BLUE, label="Kept")
            ax.scatter(s2.loc[~k, "wind_speed"], s2.loc[~k, "power"], s=3, alpha=0.5, color=RED, label="Removed")
            ax.set_xlabel("Wind speed (m/s)")
            ax.set_ylabel("Power (kW)")
            ax.legend(frameon=False, markerscale=5)
            plt.tight_layout()
            fig_out(fig)
        else:
            st.write("Needs a measured power column.")

    st.subheader("5. Download")
    out = df.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if has_power:
        out["error_kw"] = out["predicted_power"] - out["power"]
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="power_predictions.csv", mime="text/csv")


# =================================================================== page: what-if

def page_whatif():
    st.title("What-if: power curve and annual energy")
    st.markdown("Set the conditions and see the model's predicted power. Wind speed matters most; direction, "
                "temperature (air density) and time of day (atmospheric stability) shift the curve slightly.")

    c1, c2, c3 = st.columns(3)
    kind = c1.radio("Model", ["turbine", "general"], format_func=turbine_label)
    turbine = c1.selectbox("Turbine", TURBINES, index=len(TURBINES) - 3 if len(TURBINES) > 2 else 0,
                           disabled=kind != "turbine")
    wind_dir = c2.slider("Wind direction (degrees, 0 = north)", 0, 359, 240)
    temp = c2.slider("Temperature (°C)", -10, 30, 8)
    hour = c3.slider("Hour of day", 0, 23, 12)
    ws = c3.slider("Wind speed for a single prediction (m/s)", 0.0, 25.0, 9.0, 0.1)

    t = turbine if kind == "turbine" else None
    p = core.predict_grid(bundle, kind, TURBINES, np.array([ws]), wind_dir, temp, hour, t)[0]
    m = st.columns(3)
    m[0].metric(f"Predicted power at {ws:.1f} m/s", f"{max(p, 0):,.0f} kW")
    m[1].metric("Share of rated", f"{max(p, 0) / RATED_KW * 100:.1f}%")
    m[2].metric("Energy if held for 1 hour", f"{max(p, 0) / 1000:.2f} MWh")

    grid = np.round(np.arange(0, 25.01, 0.25), 2)
    curve = core.predict_grid(bundle, kind, TURBINES, grid, wind_dir, temp, hour, t)

    st.subheader("Predicted power curve")
    compare = st.multiselect("Compare with", ["Opposite wind direction", "Temperature −15 °C colder",
                                              "Night (03:00)", "General model"], default=[])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(grid, curve, color=BLUE, lw=2.2, label="Your settings")
    extra = {
        "Opposite wind direction": dict(wind_dir=(wind_dir + 180) % 360),
        "Temperature −15 °C colder": dict(temp=temp - 15),
        "Night (03:00)": dict(hour=3),
    }
    colors = iter([ORANGE, GREEN, RED, GREY])
    for name in compare:
        if name == "General model":
            y = core.predict_grid(bundle, "general", TURBINES, grid, wind_dir, temp, hour, None)
        else:
            args = dict(wind_dir=wind_dir, temp=temp, hour=hour) | extra[name]
            y = core.predict_grid(bundle, kind, TURBINES, grid, args["wind_dir"], args["temp"], args["hour"], t)
        ax.plot(grid, y, lw=1.5, color=next(colors), label=name)
    ax.scatter([ws], [max(p, 0)], color="black", zorder=5)
    ax.axhline(RATED_KW, color=GREY, ls=":", lw=1)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Predicted power (kW)")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig_out(fig)
    st.caption("Beyond about 20 m/s the training data are thin, so the curve there is less reliable. "
               "The Senvion MM82 cuts in near 3.5 m/s, reaches rated power near 14 m/s and cuts out at 25 m/s.")

    st.subheader("Annual energy from an average wind speed")
    st.markdown("Wind speeds over a year roughly follow a Weibull distribution. Combining it with the curve above "
                "gives the expected annual energy (AEP) and capacity factor.")
    e1, e2, e3 = st.columns(3)
    mean_ws = e1.slider("Annual mean wind speed at hub height (m/s)", 4.0, 12.0, 7.5, 0.1)
    k = e2.slider("Weibull shape k (2 = typical inland site)", 1.2, 3.5, 2.0, 0.1)
    avail = e3.slider("Availability (%)", 80, 100, 97) / 100
    aep, cf = core.annual_energy(grid, curve, mean_ws, k, avail)
    m = st.columns(3)
    m[0].metric("Annual energy", f"{aep:,.0f} MWh")
    m[1].metric("Capacity factor", f"{cf * 100:.1f}%")
    m[2].metric("Homes supplied (at 3.5 MWh/yr each)", f"{aep / 3.5:,.0f}")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    v = np.linspace(0.01, 25, 300)
    ax.fill_between(v, core.weibull_pdf(v, mean_ws, k), color=BLUE, alpha=0.3)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Share of time")
    ax.set_title(f"Weibull distribution, mean {mean_ws} m/s, k = {k}")
    plt.tight_layout()
    fig_out(fig)
    st.caption("An estimate for one turbine. Home figure assumes about 3,500 kWh a year per UK household.")


# =================================================================== page: results

def page_results():
    st.title("Model results")
    available = {k: v for k, v in RESULT_CANDIDATES.items()
                 if os.path.exists(os.path.join(v, "model_selection_leave_one_year_out.csv"))}
    if not available:
        st.error("No results found. Put the results_split folder from the notebook next to the app.")
        return
    src = st.selectbox("Results from", list(available))
    folder = available[src]

    def read(name):
        p = os.path.join(folder, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    cv, test, pt = read("model_selection_leave_one_year_out.csv"), read("test_results_by_year.csv"), \
        read("best_model_per_turbine.csv")
    run_info = None
    if os.path.exists(os.path.join(folder, "run_info.json")):
        run_info = json.load(open(os.path.join(folder, "run_info.json")))

    tabs = st.tabs(["Model selection", "Test years", "Per turbine", "Inputs", "Data used"])

    with tabs[0]:
        st.markdown("**Leave-one-year-out cross-validation inside the training years.** Each model is trained on "
                    "five training years and scored on the sixth, once per year. Lower MAE is better; *spread* is "
                    "the standard deviation across held-out years.")
        cv = cv.sort_values("CV MAE")
        cv["label"] = cv["model"] + " [" + cv["inputs"] + "]"
        fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(cv) + 1))
        colors = [BLUE if i == 0 else "#b8b8b8" for i in range(len(cv))]
        ax.barh(cv["label"][::-1], cv["CV MAE"][::-1], xerr=cv["spread"][::-1], color=colors[::-1],
                error_kw=dict(ecolor="#555", lw=1))
        for i, v in enumerate(cv["CV MAE"][::-1]):
            ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)
        ax.set_xlabel("Cross-validation MAE (kW)")
        ax.grid(axis="y", alpha=0)
        plt.tight_layout()
        fig_out(fig)
        st.dataframe(cv.drop(columns="label").set_index(["model", "inputs"]).round(1))
        yc = [c for c in cv.columns if c.startswith("MAE ")]
        if yc:
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
            for i, (_, r) in enumerate(cv.head(5).iterrows()):
                ax.plot([c[4:] for c in yc], r[yc].values, marker="o", lw=2 if i == 0 else 1,
                        color=BLUE if i == 0 else plt.cm.Greys(0.4 + 0.12 * i), label=r["label"])
            ax.set_ylabel("MAE on held-out year (kW)")
            ax.set_xlabel("Held-out year")
            ax.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            fig_out(fig)
            st.caption("The ranking of models is the same in every held-out year. 2022 is the hardest year for "
                       "every model, so year-to-year differences in conditions matter more than the model choice.")

    with tabs[1]:
        if test is not None:
            st.markdown("**Final models**, refitted on all training years and scored on test years they never saw.")
            mae = test.pivot(index="model", columns="year", values="MAE")
            mae["mean"] = (mae.mean(axis=1) + 1e-9).round(1)
            mae = mae.sort_values("mean")
            bias = test.pivot(index="model", columns="year", values="bias").loc[mae.index]
            yrs = [c for c in mae.columns if c != "mean"]
            fig, ax = plt.subplots(figsize=(9, 4.3))
            w = 0.8 / len(mae)
            for i, lab in enumerate(mae.index):
                ax.bar(np.arange(len(yrs)) + i * w, mae.loc[lab, yrs], w, label=lab,
                       color=BLUE if i == 0 else plt.cm.Greys(0.3 + 0.5 * i / len(mae)))
            ax.set_xticks(np.arange(len(yrs)) + w * (len(mae) - 1) / 2)
            ax.set_xticklabels(yrs)
            ax.set_ylabel("MAE (kW)")
            ax.set_xlabel("Test year")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            plt.tight_layout()
            fig_out(fig)
            c1, c2 = st.columns(2)
            c1.markdown("MAE (kW)")
            c1.dataframe(mae.round(1))
            c2.markdown("Bias (kW, + = over-predicting)")
            c2.dataframe(bias.round(1))
            best, base = mae.index[0], [i for i in mae.index if "Binned" in i]
            if base:
                gain = (1 - mae.loc[best, "mean"] / mae.loc[base[0], "mean"]) * 100
                st.success(f"**{best}** has the lowest test error, {mae.loc[best, 'mean']:.1f} kW mean MAE: "
                           f"{gain:.0f}% lower than the binned power curve ({mae.loc[base[0], 'mean']:.1f} kW).")
            st.caption("All models over-predict slightly in the test years (positive bias), more in 2024, which "
                       "suggests turbines produce a little less over time (ageing or changed conditions).")

    with tabs[2]:
        if pt is not None:
            piv = pt.pivot(index="turbine", columns="year", values="MAE")
            fig, ax = plt.subplots(figsize=(9, 3.8))
            w = 0.8 / len(piv.columns)
            for i, y in enumerate(piv.columns):
                ax.bar(np.arange(len(piv)) + i * w, piv[y], w, label=str(y), color=[BLUE, ORANGE, GREEN][i % 3])
            ax.set_xticks(np.arange(len(piv)) + w * (len(piv.columns) - 1) / 2)
            ax.set_xticklabels(piv.index)
            ax.set_xlabel("Turbine")
            ax.set_ylabel("MAE (kW)")
            ax.legend(frameon=False)
            plt.tight_layout()
            fig_out(fig)
            st.dataframe(piv.round(1))
            st.caption("Best model per turbine. 2024 data exist only for turbines 11-15.")

    with tabs[3]:
        imp_path = os.path.join(folder, "input_importance.csv")
        if not os.path.exists(imp_path):
            imp_path = os.path.join(HERE, "results_builtin", "input_importance.csv")
        imp = pd.read_csv(imp_path).set_index("input").iloc[:, 0].sort_values()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(imp.index, imp.values, color=BLUE)
        ax.set_xscale("log")
        ax.set_xlabel("Increase in MAE when the input is shuffled (kW, log scale)")
        for i, v in enumerate(imp.values):
            ax.text(v * 1.1, i, f"{v:.1f}", va="center")
        plt.tight_layout()
        fig_out(fig)
        st.markdown("Permutation importance on the test years. Wind speed dominates. Direction, temperature and "
                    "turbine each add a few kW of accuracy; together they explain the gap between input sets A "
                    "(39.4 kW) and D (31.7 kW) in cross-validation.")

    with tabs[4]:
        if run_info:
            pres = pd.DataFrame(run_info["presence"], index=run_info["turbines"])
            pres.index.name = "turbine"
            st.markdown(f"Rows of cleaned 10-minute data per turbine and year. Training: "
                        f"{', '.join(map(str, run_info['train_years']))} ({run_info['training_rows']:,} rows). "
                        f"Test: {', '.join(map(str, run_info['test_years']))}.")
            st.dataframe(pres)
        else:
            st.write("Row counts are in the notebook, section 4.")


# =================================================================== page: methods

def page_methods():
    st.title("Data and methods")
    st.subheader("Data")
    st.markdown(
        "- **Penmanshiel wind farm** (Scottish Borders): 14 Senvion MM82 turbines numbered 1, 2 and 4-15, "
        "2,050 kW rated, 82 m rotor, 59 m hub height. 10-minute SCADA data and status (stop) logs, 2016-2024.\n"
        "- **Kelmarsh wind farm** (England): 6 Senvion MM92 turbines (92 m rotor), used only to test transfer to "
        "a different turbine type.\n"
        "- Source: Cubico Sustainable Investments, published on Zenodo under CC-BY-4.0."
    )
    st.subheader("The seven cleaning rules")
    st.markdown("A row is removed by the first rule that applies. The remaining rows are normal operation, which "
                "is what a power curve describes.")
    st.table(pd.DataFrame(core.CLEANING_RULES, columns=["Rule", "Name", "Removes rows where…"]).set_index("Rule"))

    st.subheader("Inputs")
    st.table(pd.DataFrame([
        ["A", "Wind speed", "The classic power curve"],
        ["B", "A + wind direction (sine, cosine) + temperature", "Wake effects and air density"],
        ["C", "B + one 0/1 column per turbine", "Each turbine's own behaviour"],
        ["D", "C + hour of day (sine, cosine)", "Day/night atmospheric stability"],
        ["General", "D without the turbine columns", "For turbines the model never saw"],
    ], columns=["Set", "Inputs", "Why"]).set_index("Set"))
    st.caption("Rotor speed and blade pitch are not used as inputs: they are set by the turbine's controller in "
               "response to the wind, so they would leak the answer and are not known in advance for a future year.")

    st.subheader("Models compared")
    st.table(pd.DataFrame([
        ["Binned power curve", "IEC 61400-12-1 method of bins, 0.25 m/s, one curve per turbine (the baseline)"],
        ["Logistic curve", "5-parameter logistic function of wind speed"],
        ["Gradient boosting", "HistGradientBoosting: learning rate 0.02, 31 leaves, 1,500 trees, min leaf 200, L2 1.0"],
        ["Random forest", "100 trees, min leaf 5, up to 260,000 rows per tree"],
        ["Extra trees", "100 randomised trees, min leaf 5"],
        ["Neural network", "2 hidden layers of 64, L2 0.01, early stopping, scaled inputs"],
    ], columns=["Model", "Settings"]).set_index("Model"))

    st.subheader("Evaluation")
    st.markdown(
        "- **Model selection:** leave-one-year-out cross-validation inside the training years.\n"
        "- **Final test:** years never used for any choice (2023, 2024).\n"
        "- **Scores:** MAE (main), RMSE, R² and bias (predicted − measured), in kW and % of rated power."
    )
    st.subheader("Limits")
    st.markdown(
        "- The model describes **normal operation**. Downtime, curtailment and icing are not predicted; the "
        "*Performance check* tab uses that gap to find lost production.\n"
        "- It needs **measured** wind speed. For true forecasting, wind speed must come from a weather forecast, "
        "whose error would add to the model's.\n"
        "- Trained on one turbine type at one site. Other types need the rotor-size correction and still carry "
        "larger errors.\n"
        "- A small positive bias in later years suggests slow performance change over time; retraining each year "
        "on recent data keeps it small."
    )


{"Overview": page_overview, "Predict from data": page_predict, "What-if and energy": page_whatif,
 "Model results": page_results, "Data and methods": page_methods}[page]()
