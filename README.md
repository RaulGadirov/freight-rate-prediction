# freight-rate-prediction

Predicting posted freight rates ($) for truckload shipments with gradient boosting (XGBoost + Optuna).
Solution for the **Freight Rate Prediction Challenge**.

## Task

1. Train and validate a model on `data/train_test.csv`.
2. Predict every load in `data/validation.csv` (12,000 loads, unique `load_id`).
3. Save the predictions as `validation_predictions.csv` (`load_id,predicted_rate`).
4. Predict the fixed December 2025 scenario in `data/december_chart_inputs.csv`
   (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb, only the date changes).
5. Validate the output files with `score.py`, which also creates `scorer_results/candidate_december.png`.

## Approach

**Target.** The rate is right-skewed, so the model is trained on `log1p(posted_rate)` and predictions are converted back with `expm1`.

**Features**

| Group | Features |
|---|---|
| Load | `distance`, `weight` (absolute value, negative flag, missing flag), `equipment` (Dry Van / Reefer / Flatbed → 1 / 2 / 3) |
| Market | `market_index` (missing flag, filled by same-day median, then train median), `market_daily`, `market_dev`, `quote_signal`, `quote_x_distance` |
| Lane | `pickup`, `delivery` (categorical), city coordinates (`*_lat`, `*_lon`), `pickup_out`, `delivery_in`, `lane_balance` |
| Calendar | `month`, `dayofweek`, `day_of_year` |

**Model.** `XGBRegressor`, hyperparameters tuned with Optuna (30 trials, 10 min timeout), objective = mean RMSE (log scale) from 5-fold `KFold` cross-validation on the training part only.

**No leakage.**
- The target column is removed from the features (`posted_rate` is not in `X`).
- All statistics used to fill or build features (`weight_median`, `market_median`, `out_cnt`, `in_cnt`, city categories) are computed on train and reused for validation and December.
- Test rows are never used for tuning or clipping.

## Validation

| Split | Description | R²
|---|---|---|---|---|
| Random 80/20 | `train_test_split(test_size=0.2, random_state=42)` 86%

Metrics are computed in the original dollar scale (after `expm1`).
The time-based split is closer to the real setting, because validation and December lie after the training period.

## December scenario

`december_chart_inputs.csv` has no `quote_signal` and `market_index`, so they are set to the median of the last 30 days of training data.
City coordinates are taken from the training data, calendar features are computed from the date.
Because the market inputs are fixed, the variation across the month comes only from the calendar features.

![December chart](scorer_results/candidate_december.png)

## Repository structure

```
.
├── notebook.ipynb                  # EDA, feature engineering, tuning, validation, predictions
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── validation_predictions.csv      # submission file
├── december_chart_inputs.csv       # completed with predicted_rate
├── scorer_results/
│   └── candidate_december.png
├── score.py                        # provided scorer
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/freight-rate-prediction.git
cd freight-rate-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

1. Open `notebook.ipynb` (Jupyter or Google Colab) and run all cells top to bottom.
   It trains the model and writes `validation_predictions.csv` and the completed `december_chart_inputs.csv`.
2. Validate the outputs and create the chart:

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december_chart_inputs.csv
```

Expected output:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

## Requirements

`requirements.txt` must include everything the notebook imports, not only the scorer packages:

```
matplotlib>=3.8,<4
numpy>=1.26,<3
pandas>=2.0,<3
scikit-learn
xgboost
optuna
```

## Limitations

- December 2025 is outside the training period, so tree models cannot extrapolate trends and the December curve is mostly driven by calendar effects and the fixed market assumptions.
- `quote_signal` and `market_index` for December are assumptions, not observed data.
- The random split can overestimate quality compared to the time-based split.

## Deliverables

- Code and run instructions (this repository)
- `validation_predictions.csv`
- PDF/DOCX report with the validation approach, data split and `candidate_december.png`
- 2–3 minute Loom walkthrough: _link_
