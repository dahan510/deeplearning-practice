import os
import numpy as np
import pandas as pd

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터"
)  # data/수업용데이터/
df = pd.read_csv(os.path.join(DATA, "11_설비센서_ai4i.csv"), encoding="utf-8-sig")

특징이름 = ["공기온도", "회전수", "토크", "공구마모"]
X = df[특징이름].values.astype(float)
y = df["고장여부"].values.astype(float)
print(X.shape)
print("*" * 10)

rng = np.random.RandomState(3)
고장idx = rng.permutation(np.where(y == 1)[0])
정상idx = rng.permutation(np.where(y == 0)[0])  # 정상 192대 번호를 섞음

# concatenate = 두 목록을 이어 붙이기. 학습: 고장 6 + 정상 134 = 140
tr = np.concatenate([고장idx[:6], 정상idx[:134]])
te = np.concatenate([고장idx[6:], 정상idx[134:]])  # 시험: 고장 2 + 정상 58 = 60
X_train, X_test, y_train, y_test = (
    X[tr],
    X[te],
    y[tr],
    y[te],
)
print(X_train.shape)
