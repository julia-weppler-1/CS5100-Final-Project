import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# THSI FILE IS VERY FAR FROM DONE, ALL OF THE NUMBERS/VARIABLES WILL LIKELY BE CHANGED

CELL_MONTH_PARQ = "cleaned_data.parquet"

MIN_MONTHS = 2

# train on 48 months
N_TRAIN = 48

# baseline sensor set size, WILL MODIFY
K_SENSORS = 1

ALPHA = 1.0 # regularization strength

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

# load and pivot to month x cell matrix
cell = pd.read_parquet(CELL_MONTH_PARQ)
cell["cell_id"] = cell["cx"].astype(str) + "_" + cell["cy"].astype(str)

X = (cell.pivot(index="month", columns="cell_id", values="do_mean").sort_index())

keep = X.notna().sum(axis=0) >= MIN_MONTHS

X = X.loc[:, keep]

print("Matrix shape after filter:", X.shape)
print("Overall missing %:", X.isna().mean().mean())

# time split
train = X.iloc[:N_TRAIN]
test = X.iloc[N_TRAIN:]
print("Train months:", train.index.min()," to ", train.index.max(), f"({len(train)})")
print("Test months:", test.index.min()," to ", test.index.max(), f"({len(test)})")

variances = train.var(axis=0, skipna=True).sort_values(ascending=False)
sensors = list(variances.head(K_SENSORS).index)
print(f"Selected {K_SENSORS} sensors (highest train variance): {sensors}")

X_train_s = train[sensors]
X_test_s = test[sensors]

train_ok = X_train_s.notna().all(axis=1)
test_ok = X_test_s.notna().all(axis=1)
print("Train rows with complete sensor inputs:", int(train_ok.sum()), "out of", len(train))
print("Test rows with complete sensor inputs:", int(test_ok.sum()), "out of", len(test))

y_true_all = []
y_pred_all = []

model = Ridge(alpha=ALPHA, fit_intercept=True)
for target in X.columns:
    y_train = train[target]
    ok_rows = train_ok & y_train.notna()
    Xtr = X_train_s.loc[ok_rows].to_numpy()
    ytr = y_train.loc[ok_rows].to_numpy()

    model.fit(Xtr, ytr)
    y_test = test[target]

    ok_test_rows = test_ok & y_test.notna()
    if ok_test_rows.sum() == 0:
        continue

    Xte = X_test_s.loc[ok_test_rows].to_numpy()
    yte = y_test.loc[ok_test_rows].to_numpy
    yhat = model.predict(Xte)

    y_true_all.append(yte)
    y_pred_all.append(yhat)

if not y_true_all:
    raise RuntimeError("No targets had enough data to evaluate")

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

print("\nBaseline Ridge regression results:")
print("Evaluated points:", len(y_true_all))
print("RMSE:", rmse(y_true_all, y_pred_all))