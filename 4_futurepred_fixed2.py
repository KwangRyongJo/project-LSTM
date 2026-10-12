import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. 설정 및 로드 ---
RESULT_DIR = '/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/260523_kr/three/models_w'
DATA_PATH = '/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/nr_kr1_half2.csv'

LOOK_BACK = 6
N_FUTURE = 24

scaler = joblib.load(f'{RESULT_DIR}/scaler.gz')
m_lstm = load_model(f'{RESULT_DIR}/model_lstm.keras')
m_att = load_model(f'{RESULT_DIR}/model_att.keras')
m_nbeats = load_model(f'{RESULT_DIR}/model_nbeats.keras')

# --- 2. 입력 데이터(Seed) 준비 ---
df = pd.read_csv(DATA_PATH)
# 데이터가 2번째 열(index 1)에 있다고 가정
recent_raw = df.iloc[-LOOK_BACK:, 1].values.reshape(-1, 1)
scaled_seed = scaler.transform(recent_raw)
input_seed = scaled_seed.reshape(1, LOOK_BACK, 1)

# --- 3. 예측 함수 ---
def get_forecast(model, seed, n_steps):
    curr_window = seed.copy()
    preds = []
    for _ in range(n_steps):
        p = model.predict(curr_window, verbose=0)
        preds.append(p[0, 0])
        curr_window = np.concatenate([curr_window[:, 1:, :], p.reshape(1, 1, 1)], axis=1)
    return np.array(preds)

print("🔮 모델별 미래 예측 수행 중...")
res_lstm = get_forecast(m_lstm, input_seed, N_FUTURE)
res_att = get_forecast(m_att, input_seed, N_FUTURE)
res_nbeats = get_forecast(m_nbeats, input_seed, N_FUTURE)

# 앙상블 가중 평균 (가중치 조정 가능)
w_lstm, w_att, w_nbeats = 0.4, 0.3, 0.3
ensemble_scaled = (res_lstm * w_lstm) + (res_att * w_att) + (res_nbeats * w_nbeats)

# --- 4. 역스케일링 및 결과 정리 ---
final_df = pd.DataFrame({
    'Step': np.arange(1, N_FUTURE + 1),
    'LSTM': scaler.inverse_transform(res_lstm.reshape(-1, 1)).flatten(),
    'Attention': scaler.inverse_transform(res_att.reshape(-1, 1)).flatten(),
    'NBeats': scaler.inverse_transform(res_nbeats.reshape(-1, 1)).flatten(),
    '⭐Ensemble': scaler.inverse_transform(ensemble_scaled.reshape(-1, 1)).flatten()
})

print("\n--- [미래 예측 상세 결과] ---")
print(final_df.round(4).to_string(index=False))

final_df.to_csv('/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/261017_kr/forecast_261017.csv', index=False)
print(f"\n✅ 'forecast_261017_kr.csv' 저장 완료.")

