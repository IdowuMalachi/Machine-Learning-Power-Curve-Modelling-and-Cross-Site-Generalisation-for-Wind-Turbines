"""Core logic for the Penmanshiel power prediction app: reading files, inputs, cleaning rules,
prediction, scoring and energy. Kept free of Streamlit so it can be tested on its own."""
import io
import re
import numpy as np
import pandas as pd

RATED_KW = 2050
ROTOR_M = 82
DEFAULT_TURBINES = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
DEFAULT_TRAIN_YEARS = [2016, 2017, 2018, 2020, 2021, 2022]
BASE_FEATURES = ["wind_speed", "dir_sin", "dir_cos", "temp", "hour_sin", "hour_cos"]

# Raw Greenbyte column names -> short names
RAW_COLUMNS = {
    "# Date and time": "time",
    "Wind speed (m/s)": "wind_speed",
    "Wind direction (°)": "wind_dir",
    "Nacelle ambient temperature (°C)": "temp",
    "Power (kW)": "power",
    "Blade angle (pitch position) A (°)": "pitch",
    "Rotor speed (RPM)": "rotor_rpm",
}

CLEANING_RULES = [
    ("1", "Logged stop", "The 10-minute interval overlaps a stop in the turbine's status log."),
    ("2", "Missing value", "Wind speed, direction, temperature or power is missing."),
    ("3", "Impossible value", "Wind speed below 0 or above 40 m/s, power above 105% of rated or below -50 kW, "
                              "direction outside 0-360 degrees."),
    ("4", "Not producing", "Power below 50 kW while wind speed is above 4.5 m/s."),
    ("5", "Pitched out", "Blade pitch above 5 degrees at 5-9 m/s, where blades should be at fine pitch."),
    ("6", "Capped (curtailed)", "Power below 95% of rated at 14 m/s or more, where the turbine should be at full power."),
    ("7", "Statistical outlier", "Power more than 3 IQR outside the middle 50% of its 0.5 m/s wind speed bin."),
]


# ---------------------------------------------------------------- model bundle

def bundle_turbines(bundle):
    info = bundle.get("info", {})
    if info.get("turbines"):
        return [int(t) for t in info["turbines"]]
    m = re.search(r"turbines \[([\d,\s]+)\]", info.get("trained_on", ""))
    if m:
        return [int(t) for t in m.group(1).split(",")]
    n = getattr(bundle["turbine"], "n_features_in_", None)
    if n == len(BASE_FEATURES) + 5:
        return [11, 12, 13, 14, 15]
    return DEFAULT_TURBINES


def bundle_train_years(bundle):
    info = bundle.get("info", {})
    if info.get("train_years"):
        return [int(y) for y in info["train_years"]]
    m = re.search(r"years \[?([\d,\s]+)\]?", info.get("trained_on", ""))
    if m:
        return [int(y) for y in re.findall(r"\d{4}", m.group(1))]
    return DEFAULT_TRAIN_YEARS


def feature_names(kind, turbines):
    return BASE_FEATURES + ([f"T{t}" for t in turbines] if kind == "turbine" else [])


# ---------------------------------------------------------------- reading files

def read_upload(raw_bytes, filename=""):
    """Read a cleaned CSV, a raw Greenbyte Turbine_Data file, or a tab-separated table.
    Returns (dataframe, description of the format found)."""
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    first_lines = text.splitlines()[:15]
    header_idx = next((i for i, l in enumerate(first_lines) if l.startswith("# Date and time")), None)
    if header_idx is not None:  # raw Greenbyte export
        cols = pd.read_csv(io.StringIO(text), skiprows=header_idx, nrows=0).columns
        use = [c for c in RAW_COLUMNS if c in cols]
        df = pd.read_csv(io.StringIO(text), skiprows=header_idx, usecols=use).rename(columns=RAW_COLUMNS)
        fmt = "raw Greenbyte turbine file"
        m = re.search(r"Penmanshiel_(\d+)_", filename) or re.search(r"Penmanshiel (\d+)", text[:3000])
        if m and "turbine" not in df.columns:
            df["turbine"] = int(m.group(1))
    else:
        sep = "\t" if first_lines and first_lines[0].count("\t") > first_lines[0].count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
        fmt = "table with a header row"
    return df, fmt


def guess_columns(df):
    """Guess which column holds each signal, by name."""
    lower = {c.lower(): c for c in df.columns}

    def find(*keys):
        for k in keys:
            if k in lower:
                return lower[k]
        for k in keys:
            for lc, c in lower.items():
                if k in lc:
                    return c
        return None

    return {
        "time": find("time", "date and time", "timestamp", "date"),
        "wind_speed": find("wind_speed", "wind speed", "windspeed", "ws"),
        "wind_dir": find("wind_dir", "wind direction", "direction", "wd"),
        "temp": find("temp", "temperature"),
        "power": find("power", "power (kw)"),
        "turbine": find("turbine"),
        "pitch": find("pitch", "blade angle"),
        "stopped": find("stopped"),
    }


def standardise(df, mapping):
    """Rename mapped columns to standard names and parse types. `mapping` maps standard name -> column or None."""
    out = pd.DataFrame(index=df.index)
    for std, col in mapping.items():
        if col:
            out[std] = df[col]
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    for c in ["wind_speed", "wind_dir", "temp", "power", "pitch", "turbine"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "stopped" in out:
        out["stopped"] = out["stopped"].astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- inputs and prediction

def add_features(df, turbines):
    d = df.copy()
    rad = np.deg2rad(d["wind_dir"])
    d["dir_sin"], d["dir_cos"] = np.sin(rad), np.cos(rad)
    hour = 2 * np.pi * (d["time"].dt.hour + d["time"].dt.minute / 60) / 24
    d["hour_sin"], d["hour_cos"] = np.sin(hour), np.cos(hour)
    turb = d["turbine"] if "turbine" in d else pd.Series(np.nan, index=d.index)
    for t in turbines:
        d[f"T{t}"] = (turb == t).astype(int)
    return d


def choose_model(df, turbines):
    """Turbine-aware model when every row is one of the model's turbines, else the general model."""
    if "turbine" in df and df["turbine"].notna().all() and df["turbine"].isin(turbines).all():
        return "turbine"
    return "general"


def predict(bundle, df, kind, turbines, ws_scale=1.0):
    feats = feature_names(kind, turbines)
    d = add_features(df, turbines)
    d["wind_speed"] = d["wind_speed"] * ws_scale
    ok = d[feats].notna().all(axis=1)
    pred = pd.Series(np.nan, index=d.index)
    if ok.any():
        pred[ok] = np.clip(bundle[kind].predict(d.loc[ok, feats].values), -50, RATED_KW * 1.02)
    return pred


def predict_grid(bundle, kind, turbines, wind_speeds, wind_dir, temp, hour, turbine=None):
    """Predicted power over a range of wind speeds with the other inputs held fixed (what-if curve)."""
    n = len(wind_speeds)
    t = pd.Timestamp("2025-01-01") + pd.to_timedelta(hour, unit="h")
    df = pd.DataFrame({"time": [t] * n, "wind_speed": wind_speeds, "wind_dir": wind_dir, "temp": temp})
    if turbine is not None:
        df["turbine"] = turbine
    return predict(bundle, df, kind, turbines).values


def binned_curve(ws, power, width=0.5, min_n=10):
    """Measured power curve (method of bins) from the uploaded data."""
    d = pd.DataFrame({"ws": ws, "p": power}).dropna()
    g = d.groupby(np.floor(d["ws"] / width)).agg(ws=("ws", "mean"), p=("p", "mean"), n=("p", "size"))
    return g[g["n"] >= min_n].reset_index(drop=True)


# ---------------------------------------------------------------- cleaning and scoring

def cleaning_flags(df):
    """Which rule removes each row (first rule that applies), or '' to keep. Rules 1 and 5 need a
    stopped and a pitch column; without them they are skipped."""
    reason = pd.Series("", index=df.index, dtype=object)

    def mark(mask, label):
        m = mask.fillna(False) & (reason == "")
        reason[m] = label

    if "stopped" in df:
        mark(df["stopped"].astype(bool), "1 Logged stop")
    mark(~df[["wind_speed", "wind_dir", "temp", "power"]].notna().all(axis=1), "2 Missing value")
    bad = ~(df["wind_speed"].between(0, 40) & df["power"].between(-50, RATED_KW * 1.05) & df["wind_dir"].between(0, 360))
    mark(bad, "3 Impossible value")
    mark((df["wind_speed"] > 4.5) & (df["power"] < 50), "4 Not producing")
    if "pitch" in df:
        mark(df["wind_speed"].between(5, 9) & (df["pitch"] > 5), "5 Pitched out")
    mark((df["wind_speed"] >= 14) & (df["power"] < 0.95 * RATED_KW), "6 Capped (curtailed)")
    keep = reason == ""
    d = df[keep]
    if len(d):
        b = d["wind_speed"] // 0.5
        g = d.groupby(b)["power"]
        q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        out = (d["power"] < q1 - 3 * iqr) | (d["power"] > q3 + 3 * iqr)
        reason[out[out].index] = "7 Statistical outlier"
    return reason


def scores(y, p):
    y, p = pd.Series(y).reset_index(drop=True), pd.Series(p).reset_index(drop=True)
    m = y.notna() & p.notna()
    y, p = y[m].values, p[m].values
    if len(y) < 2:
        return None
    e = p - y
    return {"rows": len(y), "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
            "R2": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), "bias": np.mean(e),
            "MAE %": np.mean(np.abs(e)) / RATED_KW * 100}


def error_by_band(df, edges=(0, 4, 6, 8, 10, 12, 14, 30)):
    d = df.dropna(subset=["power", "predicted_power", "wind_speed"])
    err = d["predicted_power"] - d["power"]
    bands = pd.cut(d["wind_speed"], list(edges))
    out = pd.DataFrame({"rows": err.groupby(bands, observed=True).size(),
                        "MAE (kW)": err.abs().groupby(bands, observed=True).mean(),
                        "bias (kW)": err.groupby(bands, observed=True).mean()})
    out.index = [f"{int(b.left)}-{int(b.right)} m/s" for b in out.index]
    return out


# ---------------------------------------------------------------- energy

def energy_mwh(power_kw, minutes=10):
    """Energy from a series of mean powers over fixed intervals."""
    return np.nansum(power_kw) * minutes / 60 / 1000


def weibull_pdf(v, mean_ws, k):
    from math import gamma
    c = mean_ws / gamma(1 + 1 / k)
    return (k / c) * (v / c) ** (k - 1) * np.exp(-(v / c) ** k)


def annual_energy(curve_ws, curve_kw, mean_ws, k=2.0, availability=1.0):
    """Annual energy (MWh) and capacity factor from a power curve and a Weibull wind distribution."""
    v = np.linspace(0, 30, 601)
    f = weibull_pdf(np.clip(v, 1e-6, None), mean_ws, k)
    f = f / np.trapezoid(f, v)
    p = np.interp(v, curve_ws, curve_kw, left=0, right=0)
    mean_kw = np.trapezoid(f * np.clip(p, 0, None), v) * availability
    return mean_kw * 8760 / 1000, mean_kw / RATED_KW
