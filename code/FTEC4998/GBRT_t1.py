import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import linregress

RANDOM_SEED = 42
pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 80)

# ------------------------------------------------------------
# Load merged data (CRSP + factors + macro) from merged.parquet / merged.csv
# ------------------------------------------------------------

CANDIDATES = [
    Path(r"C:\Users\johns\OneDrive\Documents\FYP\data\merged.parquet"),
    Path(r"C:\Users\johns\OneDrive\Documents\FYP\data\merged.csv"),
]

def load_first_available(paths):
    for p in paths:
        if p.exists():
            print(f"Reading: {p}")
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            if p.suffix == ".csv":
                return pd.read_csv(p)
    raise FileNotFoundError("No expected merged data file found. Please set CANDIDATES to the correct path.")

df = load_first_available(CANDIDATES)
print("Merged data shape:", df.shape)
print(df.head())

# Replace explicit string missing markers if present (e.g. '\N')
df = df.replace(r"\N", np.nan)

# ------------------------------------------------------------
# Build excess return and feature set
# ------------------------------------------------------------

required = [
    "permno",               # stock identifier, needed for proper lead
    "ret_adj", "RF", "date",
    "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom",
    "market_equity", "turnover", "prc", "vol",
    "tbl", "tms", "dfy", "ntis", "svar", "rsvix"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns in merged data: {missing}. "
        f"Available: {list(df.columns)[:40]} ..."
    )

d = df[required].copy()

# Convert 'date' to datetime; let pandas infer format, coerce bad values to NaT
d["date"] = pd.to_datetime(d["date"], errors="coerce")
d = d.dropna(subset=["date"])

# Scale factor returns and RF from percent to decimals
factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
for col in factor_cols:
    d[col] = pd.to_numeric(d[col], errors="coerce") / 100.0

# Excess stock return at time t
d["ret_adj"] = pd.to_numeric(d["ret_adj"], errors="coerce")
d["RF"]      = pd.to_numeric(d["RF"],      errors="coerce")
d["excess_ret"] = d["ret_adj"] - d["RF"]

# Log size (more stable than raw market_equity)
d["market_equity"] = pd.to_numeric(d["market_equity"], errors="coerce")
d["log_me"] = np.log(d["market_equity"].replace({0: np.nan}))

# Ensure other numeric features are numeric
for col in ["turnover", "prc", "vol", "tbl", "tms", "dfy", "ntis", "svar", "rsvix"]:
    d[col] = pd.to_numeric(d[col], errors="coerce")

# Sort by stock and date so the lead is well-defined
d = d.sort_values(["permno", "date"])

# NEXT-MONTH excess return (prediction target)
d["excess_ret_lead"] = d.groupby("permno")["excess_ret"].shift(-1)

# Feature set for GBRT full (consistent with other full models)
feature_cols = [
    "Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom",
    "log_me", "turnover", "prc", "vol",
    "tbl", "tms", "dfy", "ntis", "svar", "rsvix"
]

# Working sample: features at t, target at t+1
d_small = d[["excess_ret_lead"] + feature_cols + ["date"]].dropna().copy()
print("Working sample shape:", d_small.shape)
print(d_small.describe())

# ------------------------------------------------------------
# Clip extreme excess returns (1% and 99%)
# ------------------------------------------------------------

q_low  = d_small["excess_ret_lead"].quantile(0.01)
q_high = d_small["excess_ret_lead"].quantile(0.99)

d_small["excess_ret_lead"] = d_small["excess_ret_lead"].clip(lower=q_low, upper=q_high)

print("After clipping:")
print(d_small["excess_ret_lead"].describe())

# ------------------------------------------------------------
# Time-based split: train / validation / test (60% / 20% / 20%)
# ------------------------------------------------------------

d_small_sorted = d_small.sort_values("date").reset_index(drop=True)

n = len(d_small_sorted)
n_train = int(0.60 * n)
n_val   = int(0.20 * n)
n_test_start = n_train + n_val

train = d_small_sorted.iloc[:n_train]
val   = d_small_sorted.iloc[n_train:n_test_start]
test  = d_small_sorted.iloc[n_test_start:]

X_train = train[feature_cols]
y_train = train["excess_ret_lead"]
dates_train = train["date"]

X_val = val[feature_cols]
y_val = val["excess_ret_lead"]
dates_val = val["date"]

X_test = test[feature_cols]
y_test = test["excess_ret_lead"]
dates_test = test["date"]

print("Train shape:", X_train.shape,
      "Val shape:",   X_val.shape,
      "Test shape:",  X_test.shape)

print("Train period:", dates_train.min(), "→", dates_train.max())
print("Val period:  ", dates_val.min(),   "→", dates_val.max())
print("Test period: ", dates_test.min(),  "→", dates_test.max())

# ------------------------------------------------------------
# GBRT FULL: tune a few hyperparameters on validation set
# ------------------------------------------------------------

# 1) Use a smaller subsample of the training data just for tuning.
tune_frac = 0.25  # 25% of train
train_tune = train.sample(frac=tune_frac, random_state=RANDOM_SEED)
X_tune = train_tune[feature_cols]
y_tune = train_tune["excess_ret_lead"]

# 2) Use fewer trees while tuning, then more trees for the final model.
n_estimators_tune = 80
n_estimators_final = 250

# Small, sensible grid over key GBRT hyperparameters:
# - learning_rate: step size of boosting
# - max_depth: depth of each tree (complexity)
# - min_samples_leaf: smoothing / regularization
# - subsample: stochastic GBRT (row subsampling)
param_grid = [
    {"learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 20, "subsample": 0.7},
    {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20, "subsample": 0.7},
    {"learning_rate": 0.10, "max_depth": 2, "min_samples_leaf": 20, "subsample": 0.7},
    {"learning_rate": 0.10, "max_depth": 3, "min_samples_leaf": 20, "subsample": 0.7},
]

best_gbrt = None
best_val_mse = np.inf
best_params = None

for params in param_grid:
    gbrt = GradientBoostingRegressor(
        n_estimators=n_estimators_tune,
        random_state=RANDOM_SEED,
        **params
    )
    gbrt.fit(X_tune, y_tune)
    y_val_pred = gbrt.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred)

    print(f"params={params}, val MSE={mse_val:.6f}")

    if mse_val < best_val_mse:
        best_val_mse = mse_val
        best_gbrt = gbrt
        best_params = params

print("Best GBRT params on tuning grid:", best_params)
print(f"Best validation MSE with {n_estimators_tune} trees: {best_val_mse:.6f}")

# 3) Refit final GBRT on the full training set with more trees using best params
gbrt_model = GradientBoostingRegressor(
    n_estimators=n_estimators_final,
    random_state=RANDOM_SEED,
    **best_params
)
gbrt_model.fit(X_train, y_train)

print("Refit final GBRT on full Train with params:", best_params)

# ------------------------------------------------------------
# Evaluate on train and test
# ------------------------------------------------------------

y_pred_tr = gbrt_model.predict(X_train)
y_pred_te = gbrt_model.predict(X_test)

r2_tr = r2_score(y_train, y_pred_tr)
r2_te = r2_score(y_test,  y_pred_te)
rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
rmse_te = np.sqrt(mean_squared_error(y_test,  y_pred_te))
mae_tr  = mean_absolute_error(y_train, y_pred_tr)
mae_te  = mean_absolute_error(y_test,  y_pred_te)

res_test = y_test - y_pred_te

# Raw CAPM beta for reference (OLS on Mkt-RF only)
slope, _, _, _, _ = linregress(test["Mkt-RF"].values.ravel(), y_test.values)
print(f"Raw CAPM β (OLS, test, vs Mkt-RF only): {slope:.4f}")
print(f"Mean residual (test): {res_test.mean():.8f}")
print(f"Train  → R²={r2_tr:.4f}, RMSE={rmse_tr:.6f}, MAE={mae_tr:.6f}")
print(f"Test   → R²={r2_te:.4f}, RMSE={rmse_te:.6f}, MAE={mae_te:.6f}")

print("Predicted (test) summary:")
print(pd.Series(y_pred_te).describe())

print("\nActual (test) summary:")
print(pd.Series(y_test).describe())

band = 0.01
close_to_zero = ((y_pred_te > -band) & (y_pred_te < band)).mean()
print(f"Share of predictions in [-{band}, {band}]: {close_to_zero:.2%}")

print("\nPredictor summaries (working sample):")
print(d_small[feature_cols].describe())

print("\nNext-month stock excess return summary (target):")
print(d_small["excess_ret_lead"].describe())

# ------------------------------------------------------------
# Feature importances
# ------------------------------------------------------------

importances = pd.Series(gbrt_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)
print("\nGBRT feature importances:")
print(importances)

# ------------------------------------------------------------
# Cumulative squared prediction error – Test set
# ------------------------------------------------------------

sq_err = (y_pred_te - y_test) ** 2

test_df = X_test.copy()
test_df["actual"]    = y_test
test_df["predicted"] = y_pred_te
test_df["sq_err"]    = sq_err
test_df["date"]      = dates_test

test_df = test_df.sort_values("date")
test_df["cum_sq_err"] = test_df["sq_err"].cumsum()

plot_df = (test_df
           .assign(year_mon=lambda x: x["date"].dt.to_period("M"))
           .groupby("year_mon")
           .last())

plt.figure(figsize=(7, 3))
plt.plot(plot_df.index.to_timestamp(), plot_df["cum_sq_err"])
plt.title("Cumulative Squared Prediction Error – Test Set (GBRT FULL, merged)")
plt.xlabel("Date")
plt.ylabel("Σ(ŷ – y)²")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Scatter: excess return vs Mkt-RF with GBRT fitted curve
# (holding other predictors at their training means)
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))

plt.scatter(
    test["Mkt-RF"], y_test,
    alpha=0.05, s=3, label="Actual (Test)"
)

xs = np.linspace(test["Mkt-RF"].min(), test["Mkt-RF"].max(), 200)

# Build X_line: vary Mkt-RF, keep other predictors at train means
X_line = pd.DataFrame({col: X_train[col].mean() for col in feature_cols},
                      index=range(len(xs)))
X_line["Mkt-RF"] = xs

y_line = gbrt_model.predict(X_line)

plt.plot(xs, y_line, linewidth=2, label="GBRT FULL fit (others at mean)")

plt.title("GBRT FULL (merged): Excess Return vs Mkt-RF (Test)")
plt.xlabel("Market excess return (Mkt-RF)")
plt.ylabel("Stock excess return")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Actual vs Predicted (Test set)
# ------------------------------------------------------------

plt.figure(figsize=(5, 5))
plt.scatter(y_pred_te, y_test, alpha=0.4, s=12)

min_val = min(y_pred_te.min(), y_test.min())
max_val = max(y_pred_te.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Predicted excess return")
plt.ylabel("Actual excess return")
plt.title("Actual vs Predicted Stock Excess Returns (Test) – GBRT FULL (merged)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Importance table (for template consistency)
# ------------------------------------------------------------

coef_table = pd.DataFrame({
    "feature": importances.index,
    "importance": importances.values
})
print(coef_table)

importances = gbrt_model.feature_importances_
fi = pd.Series(importances, index=feature_cols)
fi_sorted = fi.sort_values(ascending=True)  # ascending for a nice horizontal plot

print("\nGBRT feature importances:")
print(fi.sort_values(ascending=False))

plt.figure(figsize=(8, 5))
y_pos = np.arange(len(fi_sorted))
plt.barh(y_pos, fi_sorted.values)
plt.yticks(y_pos, fi_sorted.index)
plt.xlabel("Importance")
plt.title("GBRT Feature Importances (sorted)")
plt.tight_layout()
plt.show()
