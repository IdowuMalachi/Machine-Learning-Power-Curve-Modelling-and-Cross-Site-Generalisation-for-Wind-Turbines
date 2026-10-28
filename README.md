# Wind turbine power prediction app

Streamlit app for the CSCI 6505 project *Predicting Wind Turbine Power Output in Future Years*
(Idowu Malachi and Sharon Sarker, Fall 2026). It uses the gradient boosting model from
`Train_Chosen_Years.ipynb` (trained on Penmanshiel 2016, 2017, 2018, 2020, 2021, 2022).

## Pages

| Page | What it does |
|---|---|
| Overview | Research question, study design, headline results |
| Predict from data | Upload one or more CSVs, get predicted power, scores (MAE, RMSE, R², bias), predicted vs measured energy, power curve, error by wind speed / month / turbine, a day-by-day performance check (lost energy), the cleaning rules applied, and a CSV download |
| What-if and energy | Set wind speed, direction, temperature, hour and turbine; see the predicted power curve, compare conditions, and estimate annual energy and capacity factor from a mean wind speed |
| Model results | Cross-validation, test-year, per-turbine and input-importance results from the notebook |
| Data and methods | Data, the seven cleaning rules, inputs, models, evaluation and limits |

## Run it on your computer (Windows, Anaconda Prompt)

1. Put this `wind_app` folder **inside** `C:\Users\idowu\POWER CURVE`, so it sits next to `results_split`.
   The app then finds your own 14-turbine model (`results_split\power_models.joblib`) and your results tables
   automatically.
2. Install Streamlit in the `windml` environment:
   ```
   conda activate windml
   pip install streamlit
   ```
3. Start the app:
   ```
   cd "C:\Users\idowu\POWER CURVE\wind_app"
   streamlit run app.py
   ```
   It opens in your browser at http://localhost:8501. Stop it with Ctrl+C in the black window.
4. Try it: tick *Use the sample* on the Predict page, or upload `..\cleaned\penmanshiel_2023_clean.csv`.

The sidebar shows which model is in use. If it lists turbines 1-15, it is your own model; if it lists 11-15, it is
the fallback in `models\`. Either way the training years are 2016-2018 and 2020-2022.

## Put it online (Streamlit Community Cloud, free)

1. Upload your `results_split\power_models.joblib` to the repository (next to `app.py` or in `models`; any file
   name ending in `.joblib` works). The app looks at every `.joblib` file, **ignores any model trained with 2019,
   2023 or 2024** (such as the first app's old model), and uses the valid model that covers the most turbines,
   so your 14-turbine model wins over the built-in fallback. You may also copy your `results_split\*.csv` files
   into a folder `results_split` inside this folder.
2. Upload every file in this folder to a GitHub repository (including `models`, `results_builtin` and
   `.streamlit`).
3. At https://share.streamlit.io choose **Create app**, pick the repository, main file `app.py`, and deploy.

## What you can upload

- Cleaned yearly files, e.g. `penmanshiel_2023_clean.csv` (several at once is fine).
- Raw Greenbyte files from Zenodo, unzipped, e.g. `Turbine_Data_Penmanshiel_12_2019-01-01_-_2020-01-01_1052.csv`;
  the turbine number is read from the file name.
- Any CSV with time, wind speed (m/s), wind direction (degrees) and temperature (°C). Power, turbine, pitch and a
  stopped flag are optional. The app guesses the columns; you can change its choices.

## Models

- **Turbine-aware:** used when every row is one of the model's Penmanshiel turbines.
- **General:** no turbine columns, for any other turbine. For another turbine type, enter its rotor diameter and
  wind speed is rescaled by (diameter / 82 m)^(2/3); for Kelmarsh (92 m) this halves the error.

Scores on the training years (2016-2018, 2020-2022) of Penmanshiel turbines are optimistic; the app warns you.
Fair tests: 2019, 2023, 2024, or other turbines.

`scikit-learn` must be 1.9.1, the version the model was saved with.

## Files

| File | Purpose |
|---|---|
| `app.py` | The app (five pages) |
| `core.py` | Reading files, inputs, cleaning rules, prediction, scoring, energy |
| `train_models.py` | Rebuilds the fallback model from the cleaned CSVs |
| `models/penmanshiel_gb_2016-2022_no2019.joblib` | Fallback model (turbines 11-15, trained 2016-2018 and 2020-2022, same settings) |
| `results_builtin/` | Copy of the notebook's results tables, used when `results_split` is not found |
| `sample_penmanshiel13_jan2019.csv` | A month of data to try |
| `.streamlit/config.toml` | Allows uploads up to 1 GB, sets the colour |
