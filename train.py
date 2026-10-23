"""Train a leakage-conscious LSTM baseline on BTC-USD daily closes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()
    frame = yf.download("BTC-USD", start=args.start, auto_adjust=True, progress=False, multi_level_index=False)
    if frame.empty:
        raise RuntimeError("No BTC-USD data returned. Check the date or data connection.")
    close_series = frame["Close"]
    if hasattr(close_series, "columns"):  # tolerate yfinance MultiIndex output
        close_series = close_series.iloc[:, 0]
    close = close_series.dropna().to_numpy(dtype=np.float32).reshape(-1, 1)
    split = int(len(close) * 0.8)
    if split <= args.window or len(close) - split < 2:
        raise ValueError("Not enough observations for this window and holdout split.")
    scaler = MinMaxScaler().fit(close[:split])
    scaled = scaler.transform(close)

    def windows(start: int, stop: int):
        xs, ys = [], []
        for target in range(max(start, args.window), stop):
            xs.append(scaled[target - args.window:target, 0])
            ys.append(scaled[target, 0])
        return np.asarray(xs)[..., None], np.asarray(ys)

    x_train, y_train = windows(args.window, split)
    x_test, y_test = windows(split, len(scaled))
    model = keras.Sequential([
        keras.layers.Input(shape=(args.window, 1)),
        keras.layers.LSTM(64, return_sequences=True), keras.layers.Dropout(0.2),
        keras.layers.LSTM(32), keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=32, validation_split=0.1,
              shuffle=False, callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)
    pred = scaler.inverse_transform(model.predict(x_test, verbose=0))[:, 0]
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
    dates = frame.index[-len(actual):]
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    metrics = {"mae_usd": float(mean_absolute_error(actual, pred)),
               "rmse_usd": float(np.sqrt(mean_squared_error(actual, pred))),
               "test_observations": int(len(actual))}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "
", encoding="utf-8")
    plt.figure(figsize=(12, 6)); plt.plot(dates, actual, label="Actual")
    plt.plot(dates, pred, label="LSTM prediction"); plt.title("BTC-USD daily close: held-out forecast")
    plt.xlabel("Date"); plt.ylabel("Price (USD)"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "forecast.png", dpi=160)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()


