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
    gw = (손실(x, y, w + h, b) - 손실(x, y, w - h, b)) / (2 * h)
    gb = (손실(x, y, w, b + h) - 손실(x, y, w, b - h)) / (2 * h)
    return gw, gb


gw, gb = 기울기_밟아보기(x, y, 1.0, 9.0)

lr = 0.00001
w, b = 1.0, 9.0
w = w - lr * gw
b = b - lr * gb

w, b = 1.0, 9.0
with np.errstate(over="ignore", invalid="ignore"):
    for _ in range(20):
        gw, gb = 기울기_밟아보기(x, y, w, b)
        w, b = (
            (w - 0.001 * gw),
            b - 0.001 * gb,
        )  # 기울기의 반대 방향으로 w와 b를 조금 이동시켜 손실을 줄임

m = x.mean()
s = x.std()

z = (x - m) / s


def 학습(z, y, lr=0.1, epochs=300, 보여주기=True):
    w, b = 0.0, 0.0
    for epoch in range(epochs):
        gw, gb = 기울기_밟아보기(z, y, w, b)
        w = w - lr * gw
        b = b - lr * gb
        if 보여주기 and epoch in (0, 1, 2, 5, 10, 30, 100, 299):
            print(
                f"    epoch {epoch:3d}  w={w:7.3f}  b={b:8.3f}  손실 {손실(z, y, w, b):10.4f}"
            )  # 자릿수 지정은 00 미리보기 ①
    return w, b


w_x, b_z = 학습(z, y)

w_gd = w_x / s
b_gd = b_z - w_z * m / s

for lr in [0.001, 0.01, 0.1, 1.05]:
    w_t, b_t = 학습(z, y, lr=lr, 보여주기 = False)
    L = 손실(z, y, w_t, b_t)
    if np.isnan(L) or L > 1e6:
        판정 = "발산(튕겨 나감)"
    elif L > 1.0:
        판정 = "굼벵이 (300걸음으론 아직 멀었음)"
    else:
        판정 = "좋음 (바닥)"
    print(f"    lr={lr:<6} → 손실 {L:>14.4f}  {판정}")

w_공식 = np.sum((x - x.mean())* (y-y.mean())) / np.sum((x- x.mean())**2)
b_공식 = y.mean() - w_공식 * x.mean()

for 새온도 in [298.0, 300.0, 303.0]:
    print({새온도} {예측(새온도, w_gd, b_gd)})

def R2(x, y, w, b):
    못밪힌것 = np.sum((y - 예측(x,w,b))**2)
    평균으로찍기 = np.sum((y-y.mean())**2)
    return 1 - 못맞힌것/평균으로찍기