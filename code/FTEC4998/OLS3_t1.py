import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import linregress

RANDOM_SEED = 42
pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 50)

# Load data
CANDIDATES = [
    Path("C:\\Users\\johns\\OneDrive\\Documents\\FYP\\data\\crsp_msf_with_factors.parquet"),
    Path("C:\\Users\\johns\\OneDrive\\Documents\\FYP\\data\\crsp_msf_with_factors.csv"),
]

def load_first_available(paths):
    for p in paths:
        if p.exists():
            print(f'Reading: {p}')
            if p.suffix == '.parquet':
                return pd.read_parquet(p)
            if p.suffix == '.csv':
                return pd.read_csv(p)
    raise FileNotFoundError('No expected data file found. Please set CANDIDATES to the correct path.')

df = load_first_available(CANDIDATES)
print('shape:', df.shape)
print(df.head())

# ------------------------------------------------------------------
# Build excess return and select predictors for OLS-3 (Mkt-RF, SMB, HML)
# ------------------------------------------------------------------
required = ['permno', 'ret', 'Mkt-RF', 'SMB', 'HML', 'RF', 'date']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing columns in data: {missing}. Available: {list(df.columns)[:30]} ...')

d = df[required].dropna().copy()

# Scale factors and RF to decimals (from percent)
factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
d[factor_cols] = d[factor_cols] / 100.0

# Excess stock return at time t
d['excess_ret'] = d['ret'] - d['RF']

# Proper datetime
d['date'] = pd.to_datetime(d['date'], format='%Y-%m')

# Sort by stock and date so that leading works correctly
d = d.sort_values(['permno', 'date'])

# Excess return at t+1 (prediction target)
d['excess_ret_lead'] = d.groupby('permno')['excess_ret'].shift(-1)

# Working sample: features at t, target at t+1
d_small = d[['excess_ret_lead', 'Mkt-RF', 'SMB', 'HML', 'date']].dropna().copy()
print('Working sample shape:', d_small.shape)
print(d_small.describe())

# ------------------------------------------------------------------
# Clip extreme excess returns (1% and 99%)
# ------------------------------------------------------------------
q_low  = d_small['excess_ret_lead'].quantile(0.01)
q_high = d_small['excess_ret_lead'].quantile(0.99)

d_small_clipped = d_small.copy()
d_small_clipped['excess_ret_lead'] = d_small_clipped['excess_ret_lead'].clip(
    lower=q_low,
    upper=q_high
)

print("After clipping:")
print(d_small_clipped['excess_ret_lead'].describe())

# Use the clipped data as our working sample
d_small = d_small_clipped

# ------------------------------------------------------------------
# Time-based split: train / validation / test (30% / 20% / 50%)
# ------------------------------------------------------------------
d_small_sorted = d_small.sort_values('date').reset_index(drop=True)

n = len(d_small_sorted)
n_train = int(0.60 * n)
n_val   = int(0.20 * n)
n_test_start = n_train + n_val

train = d_small_sorted.iloc[:n_train]
val   = d_small_sorted.iloc[n_train:n_test_start]
test  = d_small_sorted.iloc[n_test_start:]

X_train = train[['Mkt-RF', 'SMB', 'HML']]
y_train = train['excess_ret_lead']
dates_train = train['date']

X_val = val[['Mkt-RF', 'SMB', 'HML']]
y_val = val['excess_ret_lead']
dates_val = val['date']

X_test = test[['Mkt-RF', 'SMB', 'HML']]
y_test = test['excess_ret_lead']
dates_test = test['date']

print('Train shape:', X_train.shape,
      'Val shape:',   X_val.shape,
      'Test shape:',  X_test.shape)

print('Train period:', dates_train.min(), '→', dates_train.max())
print('Val period:  ', dates_val.min(),   '→', dates_val.max())
print('Test period: ', dates_test.min(),  '→', dates_test.max())

# ------------------------------------------------------------------
# OLS-3: Linear regression with three predictors (Mkt-RF, SMB, HML)
# ------------------------------------------------------------------
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred_tr = linreg.predict(X_train)
y_pred_te = linreg.predict(X_test)

alpha = linreg.intercept_
betas = linreg.coef_
print(f"Alpha (intercept): {alpha:.6f}")
print(f"Beta (Mkt-RF):     {betas[0]:.4f}")
print(f"Beta (SMB):        {betas[1]:.4f}")
print(f"Beta (HML):        {betas[2]:.4f}")
print('Fitted OLS-3 (Mkt-RF, SMB, HML).')

# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------
r2_tr = r2_score(y_train, y_pred_tr)
r2_te = r2_score(y_test,  y_pred_te)
rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
rmse_te = np.sqrt(mean_squared_error(y_test,  y_pred_te))
mae_tr  = mean_absolute_error(y_train, y_pred_tr)
mae_te  = mean_absolute_error(y_test,  y_pred_te)

res_test = y_test - y_pred_te

# "Raw" beta of excess returns vs Mkt-RF only (for reference)
slope, _, _, _, _ = linregress(test['Mkt-RF'].values.ravel(), y_test.values)
print(f"Raw stock β w.r.t Mkt-RF (test): {slope:.4f}")
print(f'Mean residual (test): {res_test.mean():.8f}')
print(f'Train  → R²={r2_tr:.4f}, RMSE={rmse_tr:.6f}, MAE={mae_tr:.6f}')
print(f'Test   → R²={r2_te:.4f}, RMSE={rmse_te:.6f}, MAE={mae_te:.6f}')

print("Predicted (test) summary:")
print(pd.Series(y_pred_te).describe())

print("\nActual (test) summary:")
print(pd.Series(y_test).describe())

band = 0.01
close_to_zero = ((y_pred_te > -band) & (y_pred_te < band)).mean()
print(f"Share of predictions in [-{band}, {band}]: {close_to_zero:.2%}")

print("\nPredictor summaries (working sample):")
print(d_small[['Mkt-RF', 'SMB', 'HML']].describe())

print("\nStock excess return summary:")
print(d_small['excess_ret_lead'].describe())

# ------------------------------------------------------------------
# Cumulative squared prediction error – Test set
# ------------------------------------------------------------------
sq_err = (y_pred_te - y_test) ** 2

test_df = X_test.copy()
test_df['actual']    = y_test
test_df['predicted'] = y_pred_te
test_df['sq_err']    = sq_err
test_df['date']      = dates_test

test_df = test_df.sort_values('date')
test_df['cum_sq_err'] = test_df['sq_err'].cumsum()

plot_df = (test_df
           .assign(year_mon=lambda x: x['date'].dt.to_period('M'))
           .groupby('year_mon')
           .last())

plt.figure(figsize=(7, 3))
plt.plot(plot_df.index.to_timestamp(), plot_df['cum_sq_err'])
plt.title('Cumulative Squared Prediction Error – Test Set (OLS-3)')
plt.xlabel('Date')
plt.ylabel('Σ(ŷ – y)²')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Scatter: excess return vs Mkt-RF with fitted line
# (holding SMB, HML at their sample means)
# ------------------------------------------------------------------
plt.figure(figsize=(6,4))

plt.scatter(test['Mkt-RF'], y_test, alpha=0.05, s=3, label='Actual (Test)')

xs = np.linspace(test['Mkt-RF'].min(), test['Mkt-RF'].max(), 200)
mean_SMB = X_train['SMB'].mean()
mean_HML = X_train['HML'].mean()
X_line = pd.DataFrame({
    'Mkt-RF': xs,
    'SMB': mean_SMB,
    'HML': mean_HML
})
y_line = linreg.predict(X_line)

plt.plot(xs, y_line, linewidth=2, label='OLS-3 fit (SMB,HML at mean)')

plt.title('OLS-3: Excess Return vs Mkt-RF (Test)')
plt.xlabel('Market excess return (Mkt-RF)')
plt.ylabel('Stock excess return')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Actual vs Predicted (Test set)
# ------------------------------------------------------------------
plt.figure(figsize=(5,5))
plt.scatter(y_pred_te, y_test, alpha=0.4, s=12)

min_val = min(y_pred_te.min(), y_test.min())
max_val = max(y_pred_te.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel('Predicted excess return')
plt.ylabel('Actual excess return')
plt.title('Actual vs Predicted Stock Excess Returns (Test) – OLS-3')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Coefficients table
# ------------------------------------------------------------------
coef_table = pd.DataFrame({
    'feature': ['Mkt-RF', 'SMB', 'HML'],
    'coefficient': betas
})
print(coef_table)

coef_table_sorted = coef_table.copy()
coef_table_sorted["abs_coef"] = coef_table_sorted["coefficient"].abs()
coef_table_sorted = coef_table_sorted.sort_values("abs_coef", ascending=True)

print("\nOLS3 'feature importance' (|coefficient|):")
print(coef_table_sorted.sort_values("abs_coef", ascending=False))

# Horizontal bar plot
plt.figure(figsize=(8, 5))
plt.barh(coef_table_sorted["feature"], coef_table_sorted["abs_coef"])
plt.xlabel("|Coefficient|")
plt.title("OLS3 Feature Importances (absolute coefficients)")
plt.tight_layout()
plt.show()

print('\nInterpretation:')
print(" - Coefficients are the OLS-3 betas on Mkt-RF, SMB, and HML.")
print(" - Alpha is the average pricing error (intercept).")
