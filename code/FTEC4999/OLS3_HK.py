import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import linregress

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 80)

# -----------------------------
# 1) File paths
# -----------------------------
CANDIDATES = [
    Path("C:\\Users\\johns\\OneDrive\\Documents\\FYP\\t2\\data\\hk_stock_data_final.parquet"),
    Path("C:\\Users\\johns\\OneDrive\\Documents\\FYP\\t2\\data\\hk_stock_data_final.csv"),
]

ID_COL = 'Instrument'
DATE_COL = 'Date'
TARGET_COL = 'Target_Forward_Log_Return'
FACTOR_COLS = ['Mkt - RF_lagged', 'SMB_lagged', 'HML']


def load_first_available(paths):
    for p in paths:
        if p.exists():
            print(f"Reading: {p}")
            if p.suffix.lower() == '.parquet':
                return pd.read_parquet(p)
            if p.suffix.lower() == '.csv':
                return pd.read_csv(p)
    raise FileNotFoundError(
        "No expected data file found. Please update CANDIDATES with the correct HK file path."
    )


def maybe_scale_percent_to_decimal(series: pd.Series, name: str) -> pd.Series:
    """
    Some datasets store returns/factors in percent units (e.g. 5.2) instead of decimals (0.052).
    Divide by 100 only when the magnitude strongly suggests percent units.
    """
    s = pd.to_numeric(series, errors='coerce').copy()
    if s.dropna().empty:
        return s

    q99 = s.abs().quantile(0.99)
    if q99 > 2:
        print(f"{name}: looks like percent units. Dividing by 100.")
        s = s / 100.0
    else:
        print(f"{name}: looks like decimal units already. Keeping as-is.")
    return s


# -----------------------------
# 2) Load data
# -----------------------------
df = load_first_available(CANDIDATES)
print("shape:", df.shape)
print(df.head())

required = [ID_COL, DATE_COL, TARGET_COL] + FACTOR_COLS
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns:\n{list(df.columns)}"
    )

d = df[required].copy()

# -----------------------------
# 3) Clean / standardize types
# -----------------------------
d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors='coerce')

for col in FACTOR_COLS + [TARGET_COL]:
    d[col] = maybe_scale_percent_to_decimal(d[col], col)

d = d.dropna(subset=[ID_COL, DATE_COL, TARGET_COL] + FACTOR_COLS).copy()
d = d.sort_values([ID_COL, DATE_COL])

print("\nWorking sample shape:", d.shape)
print(d[FACTOR_COLS + [TARGET_COL]].describe())

# -----------------------------
# 4) Time-based split by UNIQUE MONTHS
#    (prevents the same month being split across sets)
# -----------------------------
unique_dates = np.array(sorted(d[DATE_COL].dropna().unique()))
n_dates = len(unique_dates)

if n_dates < 10:
    raise ValueError(
        f"Only {n_dates} unique dates found. Need more monthly observations for train/val/test splitting."
    )

n_train_dates = int(0.60 * n_dates)
n_val_dates = int(0.20 * n_dates)

train_dates = unique_dates[:n_train_dates]
val_dates = unique_dates[n_train_dates:n_train_dates + n_val_dates]
test_dates = unique_dates[n_train_dates + n_val_dates:]

train = d[d[DATE_COL].isin(train_dates)].copy()
val = d[d[DATE_COL].isin(val_dates)].copy()
test = d[d[DATE_COL].isin(test_dates)].copy()

print("\nBefore clipping:")
print("Train target summary:")
print(train[TARGET_COL].describe())

# -----------------------------
# 5) Clip extreme target values USING TRAIN ONLY
#    (avoids look-ahead leakage)
# -----------------------------
q_low = train[TARGET_COL].quantile(0.01)
q_high = train[TARGET_COL].quantile(0.99)

train[TARGET_COL] = train[TARGET_COL].clip(lower=q_low, upper=q_high)
val[TARGET_COL] = val[TARGET_COL].clip(lower=q_low, upper=q_high)
test[TARGET_COL] = test[TARGET_COL].clip(lower=q_low, upper=q_high)

print("\nAfter clipping with TRAIN thresholds:")
print("Train target summary:")
print(train[TARGET_COL].describe())
print("Validation target summary:")
print(val[TARGET_COL].describe())
print("Test target summary:")
print(test[TARGET_COL].describe())

# -----------------------------
# 6) Build X / y
# -----------------------------
X_train = train[FACTOR_COLS]
y_train = train[TARGET_COL]
dates_train = train[DATE_COL]

X_val = val[FACTOR_COLS]
y_val = val[TARGET_COL]
dates_val = val[DATE_COL]

X_test = test[FACTOR_COLS]
y_test = test[TARGET_COL]
dates_test = test[DATE_COL]

print("\nTrain shape:", X_train.shape, "Val shape:", X_val.shape, "Test shape:", X_test.shape)
print("Train period:", dates_train.min(), "→", dates_train.max())
print("Val period:  ", dates_val.min(), "→", dates_val.max())
print("Test period: ", dates_test.min(), "→", dates_test.max())

# -----------------------------
# 7) Fit OLS-3
# -----------------------------
linreg = LinearRegression()
linreg.fit(X_train, y_train)

alpha = linreg.intercept_
betas = linreg.coef_

print(f"\nAlpha (intercept): {alpha:.6f}")
print(f"Beta ({FACTOR_COLS[0]}): {betas[0]:.6f}")
print(f"Beta ({FACTOR_COLS[1]}): {betas[1]:.6f}")
print(f"Beta ({FACTOR_COLS[2]}): {betas[2]:.6f}")
print("Fitted HK OLS-3.")

# Predictions
y_pred_tr = linreg.predict(X_train)
y_pred_te = linreg.predict(X_test)

# -----------------------------
# 8) Evaluation
# -----------------------------
r2_tr = r2_score(y_train, y_pred_tr)
r2_te = r2_score(y_test, y_pred_te)
rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_te))
mae_tr = mean_absolute_error(y_train, y_pred_tr)
mae_te = mean_absolute_error(y_test, y_pred_te)

res_test = y_test - y_pred_te

# "Raw" slope of target vs market factor only, just as a simple reference
slope, _, _, _, _ = linregress(X_test[FACTOR_COLS[0]].values.ravel(), y_test.values)

print(f"\nRaw stock β w.r.t {FACTOR_COLS[0]} (test): {slope:.6f}")
print(f"Mean residual (test): {res_test.mean():.8f}")
print(f"Train  → R²={r2_tr:.4f}, RMSE={rmse_tr:.6f}, MAE={mae_tr:.6f}")
print(f"Test   → R²={r2_te:.4f}, RMSE={rmse_te:.6f}, MAE={mae_te:.6f}")

print("\nPredicted (test) summary:")
print(pd.Series(y_pred_te).describe())

print("\nActual (test) summary:")
print(pd.Series(y_test).describe())

band = 0.01
close_to_zero = ((y_pred_te > -band) & (y_pred_te < band)).mean()
print(f"\nShare of predictions in [-{band}, {band}]: {close_to_zero:.2%}")

print("\nPredictor summaries (full working sample):")
print(d[FACTOR_COLS].describe())

print(f"\n{TARGET_COL} summary (full working sample):")
print(d[TARGET_COL].describe())

# -----------------------------
# 9) Paper-style out-of-sample R^2
#    benchmark = cross-sectional mean each month
# -----------------------------
test_os = pd.DataFrame({
    'date': dates_test.values,
    'y': y_test.values,
    'y_hat': y_pred_te
})

test_os['y_bar'] = test_os.groupby('date')['y'].transform('mean')

num = ((test_os['y'] - test_os['y_hat']) ** 2).sum()
den = ((test_os['y'] - test_os['y_bar']) ** 2).sum()
r2_os = 1 - num / den if den != 0 else np.nan
print(f"\nOS R^2 (paper-style): {r2_os:.6f}")

# -----------------------------
# 10) Cumulative squared prediction error
# -----------------------------
sq_err = (y_pred_te - y_test) ** 2

test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['predicted'] = y_pred_te
test_df['sq_err'] = sq_err
test_df['date'] = dates_test.values

test_df = test_df.sort_values('date')
test_df['cum_sq_err'] = test_df['sq_err'].cumsum()

plot_df = (
    test_df
    .assign(year_mon=lambda x: x['date'].dt.to_period('M'))
    .groupby('year_mon')
    .last()
)

plt.figure(figsize=(7, 3))
plt.plot(plot_df.index.to_timestamp(), plot_df['cum_sq_err'])
plt.title('Cumulative Squared Prediction Error – HK OLS-3 Test Set')
plt.xlabel('Date')
plt.ylabel('Σ(ŷ – y)²')
plt.tight_layout()
plt.show()

# -----------------------------
# 11) Scatter: target vs Mkt-RF with fitted line
#     holding SMB and HML at training means
# -----------------------------
plt.figure(figsize=(6, 4))
plt.scatter(X_test['Mkt - RF_lagged'], y_test, alpha=0.05, s=3, label='Actual (Test)')

xs = np.linspace(X_test['Mkt - RF_lagged'].min(), X_test['Mkt - RF_lagged'].max(), 200)
mean_SMB = X_train['SMB_lagged'].mean()
mean_HML = X_train['HML'].mean()

X_line = pd.DataFrame({
    'Mkt - RF_lagged': xs,
    'SMB_lagged': mean_SMB,
    'HML': mean_HML
})

y_line = linreg.predict(X_line)

plt.plot(xs, y_line, linewidth=2, label='OLS-3 fit (SMB,HML at train mean)')
plt.title('HK OLS-3: Target Forward Return vs Mkt - RF (Test)')
plt.xlabel('Market excess return (Mkt - RF)')
plt.ylabel('Target forward log return')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 12) Actual vs predicted
# -----------------------------
plt.figure(figsize=(5, 5))
plt.scatter(y_pred_te, y_test, alpha=0.4, s=12)

min_val = min(y_pred_te.min(), y_test.min())
max_val = max(y_pred_te.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel('Predicted forward log return')
plt.ylabel('Actual forward log return')
plt.title('Actual vs Predicted HK Stock Returns (Test) – OLS-3')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 13) Coefficients table
# -----------------------------
coef_table = pd.DataFrame({
    'feature': FACTOR_COLS,
    'coefficient': betas
})
print("\nCoefficient table:")
print(coef_table)

coef_table_sorted = coef_table.copy()
coef_table_sorted['abs_coef'] = coef_table_sorted['coefficient'].abs()
coef_table_sorted = coef_table_sorted.sort_values('abs_coef', ascending=True)

print("\nOLS-3 'feature importance' (|coefficient|):")
print(coef_table_sorted.sort_values('abs_coef', ascending=False))

plt.figure(figsize=(8, 5))
plt.barh(coef_table_sorted['feature'], coef_table_sorted['abs_coef'])
plt.xlabel('|Coefficient|')
plt.title('HK OLS-3 Feature Importances (absolute coefficients)')
plt.tight_layout()
plt.show()

print("\nInterpretation:")
print(" - Coefficients are the OLS-3 betas on Mkt - RF, SMB, and HML.")
print(" - Alpha is the average pricing error (intercept).")
print(f" - Target is '{TARGET_COL}'.")
print(" - This is an HK OLS-3 adaptation using the prepared forward-return target.")