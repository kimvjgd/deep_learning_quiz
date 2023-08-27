import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')


# 모델 학습을 위한 데이터 전처리
from sklearn.preprocessing import StandardScaler
# print(df.columns)
# # Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
#        'ejection_fraction', 'high_blood_pressure', 'platelets',
#        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
#        'DEATH_EVENT'],
#       dtype='object')

# 수치형 입력 데이터, 범주형 입력 데이터, 출력 데이터 구분
X_num = df[['age', 'creatinine_phosphokinase','ejection_fraction', 'platelets','serum_creatinine', 'serum_sodium']]
X_cat = df[['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']]
y = df['DEATH_EVENT']

# 수치형 입력 데이터를 전처리하고 입력 데이터 통합
scaler = StandardScaler()
print(X_num)
scaler.fit(X_num)

X_scaled = scaler.transform(X_num)
# 이렇게하면 numpy로 바뀌게 된다. 그러면 index & column 정보가 빠지게 된다.
X_scaled = pd.DataFrame(data=X_scaled, index=X_num.index, columns=X_num.columns)

X = pd.concat([X_scaled, X_cat], axis=1)

print(X.head())