# 입력 파일 읽기
#source tensorflow_env/bin/activate

import csv
from datetime import datetime, timedelta

# 입력 파일 읽기
with open('nr.txt', 'r') as file:
    lines = file.readlines()

# 데이터 처리
processed_data = []
start_date = datetime(1748, 1, 1)
date = start_date

for line in reversed(lines):
    numbers = line.strip().split()
    for number in numbers:
        if date.month <= 12:
            processed_data.append((date.strftime('%Y-%m-%d'), int(number)))
            date += timedelta(days=31)  # 다음 달로 이동
        else:
            # 12월을 넘어가면 9월부터 시작하여 일자를 증가
            date = datetime(date.year, 9, (date.day % 5) + 1)
            processed_data.append((date.strftime('%Y-%m-%d'), int(number)))
            date += timedelta(days=31)  # 다음 달로 이동

# CSV 파일로 저장
with open('nr2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['index', 'value'])  # 헤더 추가
    for row in processed_data:
        writer.writerow(row)

print("nr2.csv 파일이 성공적으로 생성.")

# 생성된 CSV 파일의 내용 출력
print("\nnr2.csv 파일의 내용:")
with open('nr2.csv', 'r') as file:
    print(file.read())

