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


# train, test data Separation
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(X, y, test_size=0.3, random_state=1)


# Classification 모델 학습하기
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(train_input, train_target)

# Classification 모델 학습 결과
from sklearn.metrics import classification_report
pred = model_lr.predict(test_input)
# print(classification_report(test_target, pred))

# XGBoost 모델 생성 & 학습
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(train_input, train_target)

pred = model_xgb.predict(test_input)
# print(classification_report(test_target, pred))

# 특징의 중요도 확인하기
# XGBClassifier 모델의 feature_importances_를 이용하여 중요도 plot
plt.bar(X.columns, model_xgb.feature_importances_)
plt.xticks(rotation=90)
plt.show()