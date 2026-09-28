# LSTM Bank Deposit Forecasting

Next-day forecasting of daily bank deposit inflows. An LSTM gave the lowest error of four models tested (persistence baseline, HistGradientBoosting, LSTM, BiLSTM, RNN), cutting mean absolute error by 72% versus the naive "tomorrow equals today" forecast.

> **Data note:** the series is **simulated** (4,383 days, 2014-01-01 to 2025-12-31, in GHS millions per day). It has growth, weekday effects, payday and month-end spikes, December seasonality, a 2020 shock, and autocorrelated noise. Results show the method working, not facts about a real bank. To use real data, place a `deposits_daily.csv` (columns `date`, `deposits`, optional `policy_rate`) next to the notebook.

## Business question
How much will customers deposit tomorrow? Accurate next-day inflow forecasts support liquidity planning and cash management.

## Approach
- **Target:** next-day log-return of deposits, converted back to GHS millions for scoring. Deposits trend upward, and tree models cannot extrapolate a rising level.
- **Features (12):** last two daily returns, deviation from the same weekday last week and from the 28-day average, 7-day volatility, day-of-week and month cyclical encodings, payday and month-end flags for the forecast day, and a policy-rate series.
- **Split:** chronological 80/20 (3,504 fit days, 877 validation days from 2023-08-07 to 2025-12-30), with a gap so fit targets never look into validation.
- **Models:** persistence baseline; HistGradientBoosting in a median-imputer pipeline, tuned with a 32-combination grid and 5-fold `TimeSeriesSplit`; LSTM, BiLSTM, and SimpleRNN on 30-day windows (Huber loss, Adam, early stopping, small grid over units, dropout, and learning rate).
- **Preprocessing for the networks:** median imputation and standard scaling fit on the fit period only; validation windows include the last 29 fit days as context so every validation day is scored.

## Results (validation set, GHS millions per day)

| Model | MAE | RMSE | R² | MAE reduction vs persistence |
|---|---|---|---|---|
| **LSTM** | **6.687** | **8.608** | **0.923** | **72.4%** |
| HistGradientBoosting | 6.809 | 8.876 | 0.918 | 71.9% |
| RNN | 6.953 | 8.966 | 0.917 | 71.3% |
| BiLSTM | 7.361 | 9.464 | 0.907 | 69.6% |
| Persistence (baseline) | 24.203 | 33.259 | -0.148 | 0% |

Best LSTM settings: 32 units, dropout 0.2, learning rate 0.001, 30-day window.
Best HistGradientBoosting settings: learning rate 0.03, max depth 3, 300 iterations, min samples per leaf 20, L2 regularization 1.0.

<img width="1018" height="449" alt="validation_plot" src="https://github.com/user-attachments/assets/d5a377a3-74e9-4660-a39d-63e4bee51d26" />

## Reading the results
Persistence does poorly because deposits swing with the weekday, payday, and month-end, so yesterday is a weak guide to tomorrow. Every model captures those patterns and beats it by roughly 70%. The LSTM leads, but its margin over HistGradientBoosting is small (about 2% on MAE).

## Limitations
- Simulated data with a clean structure; real deposits will be noisier and errors higher.
- One chronological split and one random seed. The gap between the top models is small enough that a different seed could change the ranking.
- The networks' hyperparameters and early stopping used the same validation window that is reported, which slightly favors them. HistGradientBoosting was tuned inside the fit period only. A final untouched test period would give a fairer comparison.
- One-day horizon only; no multi-step forecasts or prediction intervals.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook lstm_bank_deposit_forecasting.ipynb
```
Set `QUICK=1` in the environment (or `QUICK = True` in the config cell) for a fast test run. Trained models and a config file are saved to `saved_models/`.

## Repository contents
- `lstm_bank_deposit_forecasting.ipynb`: full pipeline (data, features, split, tuning, evaluation, saving, next-day forecast)
- `reports/validation_plot.png`: actual vs persistence vs LSTM on the last 120 validation days
- `reports/model_results.csv`: metrics table

**Tools:** Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, matplotlib
