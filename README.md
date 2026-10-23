# Bitcoin Price Prediction with LSTM

An educational time-series forecasting pipeline for daily Bitcoin prices. It downloads historical BTC-USD data, trains an LSTM on chronological windows, evaluates on a held-out tail, and plots predictions. Forecasts are experimental and are not financial advice.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/train.py --start 2017-01-01 --epochs 30
```
The script downloads data from Yahoo Finance and writes `artifacts/forecast.png` and `artifacts/metrics.json`. Network access is required for the initial download.

## Method
- Daily close prices are scaled using training data only.
- Sequences use the previous 60 trading days to predict the next close.
- The latest 20% is held out chronologically; no random shuffling is used.
- Reports MAE and RMSE in USD.

This simple univariate model does not account for market fundamentals, fees, or uncertainty and should not be used to make investment decisions.
