## =========================================================
## FINAL MODEL
## Stable Train/Test + 4 Forecast Strategies + Ensemble
## =========================================================

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas import read_csv

## =========================================================
## CONFIG
## =========================================================

RESULT_DIR = '/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/261017_kr'
DATA_PATH = '/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/nr_kr1_half2.csv'

LOOK_BACK = 6
N_FUTURE = 24
MC_SAMPLES = 100

TRAIN_RATIO = 0.67
EPOCHS = 300
BATCH_SIZE = 32
PATIENCE = 30

os.makedirs(RESULT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

## =========================================================
## HELPER FUNCTIONS
## =========================================================

def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

## ---------- A. Recursive Forecast ----------
def forecast_recursive(model, seed, n_steps, scaler):
    window = seed.reshape(1, LOOK_BACK, 1)
    preds = []

    for _ in range(n_steps):
        p = model.predict(window, verbose=0)
        preds.append(p[0, 0])
        window = np.concatenate([window[:, 1:, :], p.reshape(1, 1, 1)], axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

## ---------- B. Direct Forecast ----------
def forecast_direct(model, seed, n_steps, scaler):
    X = seed.reshape(1, LOOK_BACK, 1)
    preds = []

    for _ in range(n_steps):
        p = model.predict(X, verbose=0)
        preds.append(p[0, 0])

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

## ---------- C. MC Dropout Forecast ----------
def forecast_mc_dropout(model, seed, n_steps, scaler, n_samples):
    runs = []

    for _ in range(n_samples):
        window = seed.reshape(1, LOOK_BACK, 1)
        run = []

        for _ in range(n_steps):
            p = model(window, training=True).numpy()
            run.append(p[0, 0])
            window = np.concatenate([window[:, 1:, :], p.reshape(1, 1, 1)], axis=1)

        runs.append(run)

    runs = np.array(runs)

    mean = scaler.inverse_transform(runs.mean(axis=0).reshape(-1, 1)).flatten()
    low  = scaler.inverse_transform(np.percentile(runs, 5, axis=0).reshape(-1, 1)).flatten()
    high = scaler.inverse_transform(np.percentile(runs, 95, axis=0).reshape(-1, 1)).flatten()

    return mean, low, high, runs

## ---------- D. Residual Bootstrap Forecast ----------
def forecast_residual_bootstrap(model, trainX, trainY, seed, n_steps, scaler, n_samples=200):
    preds_train = model.predict(trainX).flatten()
    residuals = trainY.flatten() - preds_train

    paths = []

    for _ in range(n_samples):
        window = seed.reshape(1, LOOK_BACK, 1)
        path = []

        for _ in range(n_steps):
            base = model.predict(window, verbose=0)[0, 0]
            noise = np.random.choice(residuals)
            val = base + noise

            path.append(val)
            window = np.concatenate(
                [window[:, 1:, :], np.array(val).reshape(1, 1, 1)],
                axis=1
            )

        paths.append(path)

    paths = np.array(paths)
    paths = scaler.inverse_transform(paths.reshape(-1, 1)).reshape(n_samples, n_steps)

    return (
        paths.mean(axis=0),
        np.percentile(paths, 5, axis=0),
        np.percentile(paths, 95, axis=0),
        paths
    )

## =========================================================
## DATA
## =========================================================

df = read_csv(DATA_PATH, usecols=[1])
values = df.values.astype('float32')

train_size = int(len(values) * TRAIN_RATIO)
train_raw = values[:train_size]
test_raw  = values[train_size:]

scaler = MinMaxScaler((0, 1))
train = scaler.fit_transform(train_raw)
test  = scaler.transform(test_raw)

trainX, trainY = create_dataset(train, LOOK_BACK)
testX,  testY  = create_dataset(test, LOOK_BACK)

trainX = trainX.reshape(-1, LOOK_BACK, 1)
testX  = testX.reshape(-1, LOOK_BACK, 1)

## =========================================================
## MODEL
## =========================================================

model = Sequential([
    Input(shape=(LOOK_BACK, 1)),
    Conv1D(32, 3, padding='causal', activation='relu'),
    Dropout(0.1),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.Huber(delta=0.5)
)

model.fit(
    trainX, trainY,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
    verbose=2
)

## =========================================================
## FORECAST EXECUTION
## =========================================================

seed = testX[-1]

rec = forecast_recursive(model, seed, N_FUTURE, scaler)
dirf = forecast_direct(model, seed, N_FUTURE, scaler)

mc_mean, mc_low, mc_high, mc_runs = forecast_mc_dropout(
    model, seed, N_FUTURE, scaler, MC_SAMPLES
)

res_mean, res_low, res_high, res_paths = forecast_residual_bootstrap(
    model, trainX, trainY, seed, N_FUTURE, scaler
)

## =========================================================
## ENSEMBLE FORECAST ⭐
## =========================================================

ensemble_mean = np.mean(
    np.vstack([rec, dirf, mc_mean, res_mean]),
    axis=0
)

ensemble_low = np.min(
    np.vstack([mc_low, res_low]),
    axis=0
)

ensemble_high = np.max(
    np.vstack([mc_high, res_high]),
    axis=0
)

## =========================================================
## SAVE CSV FILES
## =========================================================

def save_forecast_csv(name, mean, low=None, high=None):
    df = pd.DataFrame({
        'Step': np.arange(1, N_FUTURE + 1),
        'Mean': mean
    })
    if low is not None:
        df['Low_5pct'] = low
        df['High_95pct'] = high

    df.to_csv(f'{RESULT_DIR}/{timestamp}_{name}.csv', index=False)

# ---------------------------
# CSV 저장
# ---------------------------
save_forecast_csv('forecast_recursive', rec)
save_forecast_csv('forecast_direct', dirf)
save_forecast_csv('forecast_mc_dropout', mc_mean, mc_low, mc_high)
save_forecast_csv('forecast_residual_bootstrap', res_mean, res_low, res_high)
save_forecast_csv('forecast_ensemble', ensemble_mean, ensemble_low, ensemble_high)

print("✅ 모든 forecast + Ensemble CSV 저장 완료")

# =========================================================
# ✅ ENSEMBLE PLOT (여기부터 추가)
# =========================================================

steps = np.arange(1, N_FUTURE + 1)

plt.figure(figsize=(16, 6))

plt.plot(steps, rec, alpha=0.4, label='Recursive')
plt.plot(steps, dirf, alpha=0.4, label='Direct')
plt.plot(steps, mc_mean, alpha=0.4, label='MC Dropout Mean')
plt.plot(steps, res_mean, alpha=0.4, label='Residual Bootstrap Mean')

plt.plot(
    steps,
    ensemble_mean,
    color='black',
    linewidth=3,
    label='⭐ Ensemble Mean'
)

plt.fill_between(
    steps,
    ensemble_low,
    ensemble_high,
    color='gray',
    alpha=0.25,
    label='Ensemble Confidence Band'
)

plt.title('Ensemble Forecast with Multiple Strategies')
plt.xlabel('Forecast Horizon (Step)')
plt.ylabel('Forecast Value')
plt.grid(alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()

plt.savefig(
    f'{RESULT_DIR}/{timestamp}_forecast_ensemble_plot.png',
    dpi=150,
    bbox_inches='tight'
)
plt.close()

print("✅ Ensemble forecast plot 저장 완료")