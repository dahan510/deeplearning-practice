import os
import pandas as pd
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터")


def 경로(파일명):
    return os.path.join(DATA, 파일명)


x = 3.14159
a = np.array([10, 20, 30, 40])
for n in [1, 2, 3]:
    제곱 = n * n
쌍 = [("온도", 0.5), ("압력", -2.0), ("진동", 1.2)]
print(sorted(쌍, key=lambda t: -abs(t[1])))

점수 = 0.83
b = np.array([1, 2, 3, 4, 5])
M = np.array([[1, 2, 3], [4, 5, 6]])

df = pd.read_csv(경로("11_설비센서_ai4i.csv"), encoding="utf-8-sig")

고장난것 = df[df["고장여부"] == 1]
df["고장여부"].value_counts()
고장난것["공구마모"].mean()
df["공구마모"].mean()

숫자열 = df.drop(columns=["설비ID", "타입"])
가짜 = df.copy()
가짜["설비번호"] = range(1, len(df) + 1)

dirty.isna().sum()
dirty[dirty["진동"] > 10]
dirty[dirty.duplicated(keep=False)]
clean.loc[clean["회전수"] < 0, "회전수"] = np.nan


def 예측(x, w, b):
    return w * x + b


def 손실(x, y, w, b):
    틀린정도 = y - 예측(x, w, b)
    return np.mean(틀린정도**2)


def 기울기_밟아보기(x, y, w, b):
    gw = 손실(x, t, w + h, b)
