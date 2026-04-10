import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
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

# Build excess return
# Build modeling DataFrame with date inside
required = ['permno', 'ret', 'Mkt-RF', 'RF', 'date']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing columns in data: {missing}. Available: {list(df.columns)[:30]} ...')

d = df[required].dropna().copy()

# Scale factors to decimals (ret is already in decimals)
d['Mkt-RF'] /= 100
d['RF']     /= 100

# Excess return at time t
d['excess_ret'] = d['ret'] - d['RF']
d['date'] = pd.to_datetime(d['date'], format='%Y-%m')

# Sort by stock and date so leading works properly
d = d.sort_values(['permno', 'date'])

# Excess return at t+1 (prediction target)
d['excess_ret_lead'] = d.groupby('permno')['excess_ret'].shift(-1)

# Working sample: features at t, target at t+1
d_small = d[['excess_ret_lead', 'Mkt-RF', 'date']].dropna().copy()
print('Working sample shape:', d_small.shape)
print(d_small.describe())

# Before split: clip extreme excess returns (1% and 99%)
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

# Time-based split: train / validation / test
# ------------------------------------------------

# Sort by date so we preserve temporal ordering
d_small_sorted = d_small.sort_values('date').reset_index(drop=True)

n = len(d_small_sorted)
n_train = int(0.60 * n)
n_val   = int(0.20 * n)
n_test_start = n_train + n_val

train = d_small_sorted.iloc[:n_train]
val   = d_small_sorted.iloc[n_train:n_test_start]
test  = d_small_sorted.iloc[n_test_start:]

X_train = train[['Mkt-RF']]
y_train = train['excess_ret_lead']
dates_train = train['date']

X_val   = val[['Mkt-RF']]
y_val   = val['excess_ret_lead']
dates_val = val['date']

X_test  = test[['Mkt-RF']]
y_test  = test['excess_ret_lead']
dates_test = test['date']

print('Train shape:', X_train.shape,
      'Val shape:',   X_val.shape,
      'Test shape:',  X_test.shape)

print('Train period:', dates_train.min(), '→', dates_train.max())
print('Val period:  ', dates_val.min(),   '→', dates_val.max())
print('Test period: ', dates_test.min(),  '→', dates_test.max())

# Build LINEAR CAPM LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred_tr = linreg.predict(X_train)
y_pred_te = linreg.predict(X_test)

beta = linreg.coef_[0]
alpha = linreg.intercept_
print(f"Alpha (intercept): {alpha:.6f}")
print(f"Beta (Mkt-RF):     {beta:.4f}")
print('Fitted linear CAPM.')

# Evaluate
y_pred_tr = linreg.predict(X_train)
y_pred_te = linreg.predict(X_test)

r2_tr = r2_score(y_train, y_pred_tr)
r2_te = r2_score(y_test, y_pred_te)
rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
rmse_te = np.sqrt(mean_squared_error(y_test,  y_pred_te))
mae_tr  = mean_absolute_error(y_train, y_pred_tr)
mae_te  = mean_absolute_error(y_test,  y_pred_te)
# Quick residual check
res_test = y_test - y_pred_te

beta_fitted_reg = LinearRegression().fit(X_test, y_pred_te)
beta_fitted = beta_fitted_reg.coef_[0]
slope, _, _, _, _ = linregress(X_test.values.ravel(), y_test.values)
print(f"Raw stock β (test)      : {slope:.4f}")
print(f"β_fitted (slope of ŷ on Mkt-RF, test set): {beta_fitted:.4f}")
print(f'Mean residual (test): {res_test.mean():.8f}')
print(f'Train  → R²={r2_tr:.4f}, RMSE={rmse_tr:.6f}, MAE={mae_tr:.6f}')
print(f'Test   → R²={r2_te:.4f}, RMSE={rmse_te:.6f}, MAE={mae_te:.6f}')

print("Predicted (test) summary:")
print(pd.Series(y_pred_te).describe())

print("\nActual (test) summary:")
print(pd.Series(y_test).describe())

band = 0.01   # or 1 if returns are in percent units

close_to_zero = ((y_pred_te > -band) & (y_pred_te < band)).mean()
print(f"Share of predictions in [-{band}, {band}]: {close_to_zero:.2%}")

print("Mkt-RF summary:")
print(d_small['Mkt-RF'].describe())

print("\nStock excess return summary:")
print(d_small['excess_ret_lead'].describe())

# Paper-style out-of-sample R^2: benchmark is cross-sectional mean each month
test_os = pd.DataFrame({
    'date': dates_test,
    'y': y_test,
    'y_hat': y_pred_te
})

# cross-sectional mean return at each date
test_os['y_bar'] = test_os.groupby('date')['y'].transform('mean')

num = ((test_os['y'] - test_os['y_hat'])**2).sum()
den = ((test_os['y'] - test_os['y_bar'])**2).sum()
r2_os = 1 - num / den
print(f"OS R^2 (paper-style): {r2_os:.6f}")

#---------------------------------------------------------------------------------------------------------
# Squared error on the test set
sq_err = (y_pred_te - y_test) ** 2

# Build a test DataFrame for plotting
test_df = X_test.copy()
test_df['actual']    = y_test
test_df['predicted'] = y_pred_te
test_df['sq_err']    = sq_err
test_df['date']      = dates_test   # from the time-based split

# Sort by date BEFORE doing the cumulative sum
test_df = test_df.sort_values('date')
test_df['cum_sq_err'] = test_df['sq_err'].cumsum()

# Aggregate to last value per month (optional, like you had)
plot_df = (test_df
           .assign(year_mon=lambda x: x['date'].dt.to_period('M'))
           .groupby('year_mon')
           .last())

# Plot cumulative squared prediction error over time
plt.figure(figsize=(7, 3))
plt.plot(plot_df.index.to_timestamp(), plot_df['cum_sq_err'])
plt.title('Cumulative Squared Prediction Error – Test Set')
plt.xlabel('Date')
plt.ylabel('Σ(ŷ – y)²')
plt.tight_layout()
plt.show()
# ----------------------------------------------------------------------------------------------------

plt.figure(figsize=(6,4))

# Actual test returns vs market
plt.scatter(X_test['Mkt-RF'], y_test, alpha=0.05, s=3, label='Actual (Test)')

# Smooth x-grid for the fitted line
xs = np.linspace(X_test['Mkt-RF'].min(), X_test['Mkt-RF'].max(), 200)
X_line = pd.DataFrame({'Mkt-RF': xs})
y_line = linreg.predict(X_line)

# CAPM fitted line
plt.plot(xs, y_line, linewidth=2, label='CAPM fit')

plt.title('Linear CAPM: Test Set')
plt.xlabel('Market excess return (Mkt-RF)')
plt.ylabel('Stock excess return')
plt.legend()
plt.tight_layout()
plt.show()

#------
# Actual vs Predicted (Test set)
plt.figure(figsize=(5,5))
plt.scatter(y_pred_te, y_test, alpha=0.4, s=12)

# 45° line
min_val = min(y_pred_te.min(), y_test.min())
max_val = max(y_pred_te.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel('Predicted excess return')
plt.ylabel('Actual excess return')
plt.title('Actual vs Predicted Stock Excess Returns (Test)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
#---------

# Coefficient interpretation (only linear beta)
coef_table = pd.DataFrame({
    'feature': ['Mkt-RF'],
    'coefficient': [beta]
})
print(coef_table)

print('\nInterpretation:')
print(" - 'Mkt-RF'  → linear beta (first-order sensitivity)")
print(" - No higher-order terms ⇒ pure linear CAPM.")

