## =========================================================
## MULTI-OUTPUT MODEL
## Multi-Output Forecast (6 steps) + Ensemble
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

RESULT_DIR = '/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/261017_kr2'
DATA_PATH = '/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/nr_kr1_half2.csv'

LOOK_BACK = 6
N_FUTURE = 6  # 24에서 6으로 변경 (6개 입력 -> 6개 동시 예측)
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

# 다중 출력(Multi-Output)에 맞게 Y를 N_FUTURE 길이만큼 생성하도록 수정
def create_dataset(dataset, look_back, n_future):
    X, Y = [], []
    for i in range(len(dataset) - look_back - n_future + 1):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back:i + look_back + n_future, 0])
    return np.array(X), np.array(Y)

## ---------- A. Multi-Output Forecast ----------
def forecast_multi_output(model, seed, scaler):
    # 한 번의 예측으로 N_FUTURE 개를 모두 출력
    window = seed.reshape(1, LOOK_BACK, 1)
    pred = model.predict(window, verbose=0)
    return scaler.inverse_transform(pred).flatten()

## ---------- B. MC Dropout Forecast ----------
def forecast_mc_dropout(model, seed, scaler, n_samples):
    runs = []
    window = seed.reshape(1, LOOK_BACK, 1)

    for _ in range(n_samples):
        # Dropout을 활성화한 상태로 여러 번 예측
        p = model(window, training=True).numpy()
        runs.append(p[0])

    runs = np.array(runs)
    runs_inv = scaler.inverse_transform(runs) # 형태: (n_samples, n_future)

    mean = runs_inv.mean(axis=0)
    low  = np.percentile(runs_inv, 5, axis=0)
    high = np.percentile(runs_inv, 95, axis=0)

    return mean, low, high, runs_inv

## ---------- C. Residual Bootstrap Forecast ----------
def forecast_residual_bootstrap(model, trainX, trainY, seed, scaler, n_samples=200):
    # 훈련 데이터에 대한 예측 및 잔차(오차) 계산
    preds_train = model.predict(trainX, verbose=0)
    residuals = trainY - preds_train # 형태: (데이터 수, n_future)

    window = seed.reshape(1, LOOK_BACK, 1)
    base_pred = model.predict(window, verbose=0)[0]

    paths = []
    for _ in range(n_samples):
        # 훈련 데이터의 잔차 중 하나를 무작위로 추출하여 더함
        idx = np.random.randint(0, len(residuals))
        noise = residuals[idx]
        val = base_pred + noise
        paths.append(val)

    paths = np.array(paths)
    paths_inv = scaler.inverse_transform(paths)

    mean = paths_inv.mean(axis=0)
    low = np.percentile(paths_inv, 5, axis=0)
    high = np.percentile(paths_inv, 95, axis=0)

    return mean, low, high, paths_inv

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

# N_FUTURE 파라미터 추가
trainX, trainY = create_dataset(train, LOOK_BACK, N_FUTURE)
testX,  testY  = create_dataset(test, LOOK_BACK, N_FUTURE)

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
    Dense(N_FUTURE) # 1에서 N_FUTURE(6)로 변경하여 한 번에 다중 출력
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

mo_pred = forecast_multi_output(model, seed, scaler)

mc_mean, mc_low, mc_high, mc_runs = forecast_mc_dropout(
    model, seed, scaler, MC_SAMPLES
)

res_mean, res_low, res_high, res_paths = forecast_residual_bootstrap(
    model, trainX, trainY, seed, scaler
)

## =========================================================
## ENSEMBLE FORECAST ⭐
## =========================================================

# 다중 출력 구조에 맞게 앙상블 조합 변경
ensemble_mean = np.mean(
    np.vstack([mo_pred, mc_mean, res_mean]),
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
save_forecast_csv('forecast_multi_output', mo_pred)
save_forecast_csv('forecast_mc_dropout', mc_mean, mc_low, mc_high)
save_forecast_csv('forecast_residual_bootstrap', res_mean, res_low, res_high)
save_forecast_csv('forecast_ensemble', ensemble_mean, ensemble_low, ensemble_high)

print("✅ 모든 forecast + Ensemble CSV 저장 완료")

# =========================================================
# ✅ ENSEMBLE PLOT
# =========================================================

steps = np.arange(1, N_FUTURE + 1)

plt.figure(figsize=(12, 6))

plt.plot(steps, mo_pred, alpha=0.4, label='Multi-Output Direct')
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

plt.title(f'Multi-Output Ensemble Forecast ({LOOK_BACK} inputs -> {N_FUTURE} outputs)')
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