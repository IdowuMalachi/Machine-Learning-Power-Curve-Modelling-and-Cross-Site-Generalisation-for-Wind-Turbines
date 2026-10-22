"""Core logic for the Penmanshiel power prediction app: reading files, features, cleaning, prediction, scoring."""
import io
import numpy as np
import pandas as pd

RATED_KW = 2050
KNOWN_TURBINES = [11, 12, 13, 14, 15]

# Raw Greenbyte column names -> short names
RAW_COLUMNS = {
    "# Date and time": "time",
    "Wind speed (m/s)": "wind_speed",
    "Wind direction (°)": "wind_dir",
    "Nacelle ambient temperature (°C)": "temp",
    "Power (kW)": "power",
    "Blade angle (pitch position) A (°)": "pitch",
}

FEATURES_GENERAL = ["wind_speed", "dir_sin", "dir_cos", "temp", "hour_sin", "hour_cos"]
FEATURES_TURBINE = FEATURES_GENERAL + [f"T{t}" for t in KNOWN_TURBINES]


def read_upload(raw_bytes, filename=""):
    """Read a cleaned CSV, a raw Greenbyte Turbine_Data file, or a tab-separated paste.
    Returns (dataframe, description of the format found)."""
    text = raw_bytes.decode("utf-8", errors="replace")
    first_lines = text.splitlines()[:15]
    header_idx = next((i for i, l in enumerate(first_lines) if l.startswith("# Date and time")), None)
    if header_idx is not None:  # raw Greenbyte export
        cols = pd.read_csv(io.StringIO(text), skiprows=header_idx, nrows=0).columns
        use = [c for c in RAW_COLUMNS if c in cols]
        df = pd.read_csv(io.StringIO(text), skiprows=header_idx, usecols=use).rename(columns=RAW_COLUMNS)
        fmt = "raw Greenbyte turbine file"
        import re
        m = re.search(r"Penmanshiel_(\d+)_", filename) or re.search(r"Turbine: Penmanshiel (\d+)", text[:2000])
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
            for lc, c in lower.items():
                if k in lc:
                    return c
        return None
    return {
        "time": find("time", "date", "timestamp"),
        "wind_speed": find("wind_speed", "wind speed", "windspeed", "ws"),
        "wind_dir": find("wind_dir", "wind direction", "direction", "wd"),
        "temp": find("temp", "temperature"),
        "power": find("power"),
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
    return out


def add_features(df):
    d = df.copy()
    rad = np.deg2rad(d["wind_dir"])
    d["dir_sin"], d["dir_cos"] = np.sin(rad), np.cos(rad)
    hour = 2 * np.pi * (d["time"].dt.hour + d["time"].dt.minute / 60) / 24
    d["hour_sin"], d["hour_cos"] = np.sin(hour), np.cos(hour)
    turb = d["turbine"] if "turbine" in d else pd.Series(np.nan, index=d.index)
    for t in KNOWN_TURBINES:
        d[f"T{t}"] = (turb == t).astype(int)
    return d


def normal_operation_mask(df):
    """Rules 1-7 of the report's cleaning. Rule 1 (logged stops) applies only when a 'stopped' column is present.
    True = keep."""
    keep = df[["wind_speed", "wind_dir", "temp", "power"]].notna().all(axis=1)
    if "stopped" in df:
        keep &= ~df["stopped"]
    keep &= df["wind_speed"].between(0, 40) & df["power"].between(-50, RATED_KW * 1.05) & df["wind_dir"].between(0, 360)
    keep &= ~((df["wind_speed"] > 4.5) & (df["power"] < 50))
    if "pitch" in df:
        keep &= ~(df["wind_speed"].between(5, 9) & (df["pitch"] > 5))
    keep &= ~((df["wind_speed"] >= 14) & (df["power"] < 0.95 * RATED_KW))
    d = df[keep]
    b = d["wind_speed"] // 0.5
    g = d.groupby(b)["power"]
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    outlier = (d["power"] < q1 - 3 * iqr) | (d["power"] > q3 + 3 * iqr)
    keep.loc[outlier[outlier].index] = False
    return keep


def choose_model(df):
    """Use the turbine-aware model when every row is a Penmanshiel turbine 11-15, else the general model."""
    if "turbine" in df and df["turbine"].notna().all() and df["turbine"].isin(KNOWN_TURBINES).all():
        return "turbine"
    return "general"


def predict(bundle, df, kind):
    feats = FEATURES_TURBINE if kind == "turbine" else FEATURES_GENERAL
    d = add_features(df)
    ok = d[feats].notna().all(axis=1)
    pred = pd.Series(np.nan, index=d.index)
    if ok.any():
        pred[ok] = np.clip(bundle[kind].predict(d.loc[ok, feats].values), -50, RATED_KW * 1.02)
    return pred


def scores(y, p):
    m = y.notna() & p.notna()
    y, p = y[m].values, p[m].values
    if len(y) == 0:
        return None
    e = p - y
    return {"rows scored": len(y), "MAE (kW)": np.mean(np.abs(e)), "RMSE (kW)": np.sqrt(np.mean(e ** 2)),
            "R²": 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2), "bias (kW)": np.mean(e),
            "MAE (% of rated)": np.mean(np.abs(e)) / RATED_KW * 100}
