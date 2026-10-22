"""Train the two app models on cleaned Penmanshiel data, 2017-2022, turbines 11-15 (the report's best solution)."""
import sys, json, time
import pandas as pd, joblib, sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import add_features, FEATURES_GENERAL, FEATURES_TURBINE, KNOWN_TURBINES
DATA = sys.argv[1]; YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
df = pd.concat([pd.read_csv(f"{DATA}/penmanshiel_{y}_clean.csv", parse_dates=["time"]) for y in YEARS])
df = add_features(df[df.turbine.isin(KNOWN_TURBINES)].reset_index(drop=True))
def gb(): return HistGradientBoostingRegressor(learning_rate=0.02, max_leaf_nodes=31, min_samples_leaf=200, max_iter=1500, l2_regularization=1.0, early_stopping=False, random_state=0)
bundle = {}
for kind, feats in [("turbine", FEATURES_TURBINE), ("general", FEATURES_GENERAL)]:
    t0 = time.time(); bundle[kind] = gb().fit(df[feats].values, df.power.values); print(kind, round(time.time() - t0), "s")
bundle["info"] = {"trained_on": "Penmanshiel turbines 11-15, cleaned, years " + ", ".join(map(str, YEARS)), "rows": int(len(df)),
                  "sklearn_version": sklearn.__version__, "model": "HistGradientBoostingRegressor (lr 0.02, 31 leaves, 1500 trees, min leaf 200, L2 1.0)"}
joblib.dump(bundle, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "models", "power_models.joblib"), compress=3)
print(json.dumps(bundle["info"]))
