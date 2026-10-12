(tf) [krjo@krjo LSTM]$ python finetunning-1.py
2025-01-22 17:11:42.292669: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-01-22 17:11:42.308873: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1737533502.326057  326362 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1737533502.331065  326362 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-01-22 17:11:42.349645: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX512_FP16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Traceback (most recent call last):
  File "/dongoh_RnD/linux/gvfs/LSTM/finetunning-1.py", line 32, in <module>
    trainX, trainY = create_dataset(train, look_back)
NameError: name 'create_dataset' is not defined
(tf) [krjo@krjo LSTM]$ 



import csv
from datetime import datetime, timedelta

# 시작 날짜와 종료 날짜 설정
start_date = datetime(1990, 1, 1)
end_date = datetime(2000, 12, 31)

# CSV 파일 작성
with open('/poclinux/gvfs/LSTM/dates.csv', 'w', newline='') as csvfile:
    date_writer = csv.writer(csvfile)
    date_writer.writerow(['Date'])  # 헤더 작성

    current_date = start_date
    while current_date <= end_date:
        date_writer.writerow([current_date.strftime('%Y-%m-%d')])
        current_date += timedelta(days=1)

print("CSV 파일이 성공적으로 생성: dates.csv")

