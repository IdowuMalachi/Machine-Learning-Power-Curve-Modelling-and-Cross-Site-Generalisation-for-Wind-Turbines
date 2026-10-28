"""Rebuild models/power_models.joblib from the cleaned yearly CSVs, with the same settings as
Train_Chosen_Years.ipynb (gradient boosting, input set D for the turbine-aware model, no turbine
columns for the general model).

    python train_models.py <folder with penmanshiel_<year>_clean.csv>
"""
import os, sys, json, time
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core import add_features, feature_names

DATA = sys.argv[1] if len(sys.argv) > 1 else "cleaned"
TRAIN_YEARS = [2016, 2017, 2018, 2020, 2021, 2022]
MAX_TRAIN_ROWS = 1_000_000
SEED = 0

data = {y: pd.read_csv(os.path.join(DATA, f"penmanshiel_{y}_clean.csv"), parse_dates=["time"]) for y in TRAIN_YEARS}
turbines = sorted(int(t) for t in set.intersection(*[set(d["turbine"].unique()) for d in data.values()]))
df = pd.concat([d[d["turbine"].isin(turbines)] for d in data.values()], ignore_index=True)
if len(df) > MAX_TRAIN_ROWS:
    df = df.iloc[np.random.default_rng(SEED).choice(len(df), MAX_TRAIN_ROWS, replace=False)]
df = add_features(df, turbines)

def gb():
    return HistGradientBoostingRegressor(learning_rate=0.02, max_leaf_nodes=31, min_samples_leaf=200, max_iter=1500,
                                         l2_regularization=1.0, early_stopping=False, random_state=SEED)

bundle = {}
for kind in ["turbine", "general"]:
    t0 = time.time()
    feats = feature_names(kind, turbines)
    bundle[kind] = gb().fit(df[feats].values, df["power"].values)
    print(kind, round(time.time() - t0), "s")
bundle["info"] = {"trained_on": f"Penmanshiel turbines {turbines}, cleaned, years {TRAIN_YEARS}", "rows": int(len(df)),
                  "sklearn_version": sklearn.__version__, "turbines": turbines, "train_years": TRAIN_YEARS,
                  "model": "HistGradientBoostingRegressor (lr 0.02, 31 leaves, 1500 trees, min leaf 200, L2 1.0)"}
os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
joblib.dump(bundle, os.path.join(HERE, "models", "power_models.joblib"), compress=3)
print(json.dumps(bundle["info"]))
