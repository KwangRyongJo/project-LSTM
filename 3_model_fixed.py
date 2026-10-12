import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from sklearn.preprocessing import MinMaxScaler

# --- 기초 설정 ---
LOOK_BACK = 6
N_FUTURE = 24
DATA_PATH = '/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/nr_kr1_whole.csv'
RESULT_DIR = '/dongoh_RnD/linux/gvfs/KN_data/raw_data/trimmed/LSTM/261017_kr/three/models_w'
os.makedirs(RESULT_DIR, exist_ok=True)

# 1. 데이터 준비
df = pd.read_csv(DATA_PATH, usecols=[1])
values = df.values.astype('float32')

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)
joblib.dump(scaler, f'{RESULT_DIR}/scaler.gz')

def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(Y)

X, y = create_dataset(scaled_data, LOOK_BACK)

# --- 모델 정의 함수들 ---

def build_lstm():
    model = Sequential([
        Input(shape=(LOOK_BACK, 1)),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_attention_model():
    inputs = Input(shape=(LOOK_BACK, 1))
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    query = Dense(64)(lstm_out)
    att_out = tf.keras.layers.Attention()([query, lstm_out])
    flat = tf.keras.layers.Flatten()(att_out)
    outputs = Dense(1)(flat)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_nbeats_style():
    inputs = Input(shape=(LOOK_BACK, 1))
    # 입력을 1차원으로 펼칩니다 (6, 1) -> (6,)
    x_flatten = tf.keras.layers.Flatten()(inputs)
    
    # 첫 번째 연산: 입력 크기를 128로 확장
    curr_x = Dense(128, activation='relu')(x_flatten)
    
    for _ in range(3):
        # 잔차 블록 (Residual Block)
        d = Dense(128, activation='relu')(curr_x)
        d = Dense(128, activation='relu')(d)
        # 이제 두 입력 모두 (128,) 이므로 뺄셈이 가능합니다.
        curr_x = tf.keras.layers.Subtract()([curr_x, d]) 
    
    outputs = Dense(1)(curr_x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
    
# --- 2. 실행부: 각각 학습 및 저장 ---

models = {
    "model_lstm": build_lstm(),
    "model_att": build_attention_model(),
    "model_nbeats": build_nbeats_style()
}

for name, model in models.items():
    print(f"🚀 {name} 학습 시작...")
    model.fit(X, y, epochs=50, batch_size=32, verbose=0) # 학습 시간 단축을 위해 epochs 조절 가능
    save_path = f'{RESULT_DIR}/{name}.keras'
    model.save(save_path)
    print(f"✅ {name} 저장 완료: {save_path}")

print("\n✨ 모든 모델 학습 및 저장 완료!")