# Wind turbine power prediction app

Upload 10-minute turbine data and get predicted power output from the gradient boosting model in the report
*Predicting Wind Turbine Power Output in Future Years* (Idowu Malachi and Sharon Sarker, CSCI 6505, Fall 2026).

## Run it on your computer

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and install the packages:
   ```
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```
   It opens in your browser at http://localhost:8501.
4. Try it with `sample_penmanshiel13_jan2019.csv`.

`scikit-learn` is pinned to 1.9.1 because the saved model was built with that version; a different version may
refuse to load it or give slightly different predictions. If you need another version, retrain (below).

## Put it online (Streamlit Community Cloud, free)

1. Create a GitHub repository and upload every file in this folder, including `models/power_models.joblib`.
2. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**, pick the repository and set the
   main file to `app.py`.
3. Deploy. You get a public link you can share.

## What you can upload

- The cleaned yearly files, e.g. `penmanshiel_2019_clean.csv`.
- A raw Greenbyte file straight from Zenodo, e.g. `Turbine_Data_Penmanshiel_01_2017-01-01_-_2018-01-01_1042.csv`
  (unzipped). The turbine number is read from the file name.
- Any CSV (comma or tab separated) with columns for time, wind speed (m/s), wind direction (degrees) and ambient
  temperature (°C). Measured power, turbine number, pitch angle and a stopped flag are optional. The app guesses
  which column is which; you can change its choices.

## What it does

- **Two models.** A turbine-aware model, used when every row is Penmanshiel turbine 11 to 15, and a general model
  (wind speed, direction, temperature, hour of day) for any other turbine.
- **Scoring.** When measured power is present it reports MAE, RMSE, R² and bias, optionally on normal operation
  only, using the report's cleaning rules.
- **Charts.** Power curve (measured and predicted), predicted against measured, daily power over time, and error
  by month.
- **Download.** A CSV with the predictions and, if power was given, the error for each row.

## Limits

- Trained on Penmanshiel turbines 11 to 15, 2017 to 2022 (Senvion MM82: 2,050 kW, 82 m rotor). Scores on those
  turbines and years are optimistic, because the model has seen them.
- Other turbine types are predicted with a Penmanshiel MM82 power curve, so expect large errors for different
  turbines.
- Check: on Penmanshiel turbine 1 in 2017, which the model never saw, the general model scored an MAE of 33.9 kW
  (1.66% of rated) with a bias of +13.5 kW.

## Retrain the models

`train_models.py` rebuilds `models/power_models.joblib` from the cleaned yearly CSVs:
```
python train_models.py <folder with penmanshiel_<year>_clean.csv files>
```
Edit `YEARS` in the script to change the training years. Training takes a few minutes.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `core.py` | Reading files, inputs, cleaning rules, prediction and scoring |
| `train_models.py` | Retrains the models |
| `models/power_models.joblib` | The trained models |
| `sample_penmanshiel13_jan2019.csv` | A month of data to try |
| `requirements.txt` | Python packages |
