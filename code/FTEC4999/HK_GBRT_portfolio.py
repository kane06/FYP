import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import linregress

RANDOM_SEED = 42
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', 120)

# ============================================================
# 1) File paths
# ============================================================
CANDIDATES = [
    Path(r"C:\Users\johns\OneDrive\Documents\FYP\t2\data\hk_stock_data_final.csv"),
    Path(r"C:\Users\johns\OneDrive\Documents\FYP\t2\data\hk_stock_data_final.parquet"),
]
BENCHMARK_CANDIDATES = [
    Path(r"C:\Users\johns\OneDrive\Documents\FYP\t2\data\HSI&1M1HIBOR_returns_2021_to_2025.csv"),
    Path(r"HSI&1M1HIBOR_returns_2021_to_2025.csv"),
]

ID_COL = "Instrument"
DATE_COL = "Date"
TARGET_COL = "Target_Forward_Log_Return"
TOP_N_IMPORTANCE = 10
TOP_PCT = 0.25
WEIGHTING_MODE = "dollar_neutral_50_50"  # options: "dollar_neutral_50_50" or "gross_200"

# Backtest realism controls
MIN_PRICE = 1.0                  # remove very low-priced names from the universe
MIN_MC_MILLIONS = 500.0          # remove very small-cap names from the universe
TRANSACTION_COST_BPS = 7         # round-trip cost per month across long and short legs
RETURN_CAP_UPPER = 1.0           # exclude realized monthly returns >= +100%
RETURN_CAP_LOWER = -0.8          # exclude realized monthly returns <= -80%

# ============================================================
# 2) HK GBRT feature set from GBRT_HK_fixed.py
# ============================================================
DESIRED_FEATURE_COLS = [
    "Mkt - RF_lagged", "SMB_lagged", "HML", "MOM_lagged",
    "log_mc", "Price Close", "Daily_Std", "Monthly_Volatility",
    "Price To Book Value Per Share (Daily Time Series Ratio)",
    "Price To Sales Per Share (Daily Time Series Ratio)",
    "Dividend yield", "Revenue_Growth_YoY", "Gross_Profit_Growth_YoY",
    "DP_HSI_lagged", "EP_HSI", "BM_HSI_lagged", "INFL", "log_SVAR",
    "diff_TBL_3MHIBOR_lagged", "diff_LTY_10Y", "diff_TMS_HIBOR",
    "diff_DFY", "diff_RF_Monthly_lagged",
]
FACTOR_COLS = ["Mkt - RF_lagged", "SMB_lagged", "HML", "MOM_lagged"]


# ============================================================
# 3) Helpers
# ============================================================
def load_first_available(paths):
    last_error = None
    for p in paths:
        if not p.exists():
            continue
        print(f"Reading: {p}")
        try:
            if p.suffix.lower() == ".parquet":
                try:
                    print("Trying parquet with pyarrow...")
                    return pd.read_parquet(p, engine="pyarrow")
                except Exception as e:
                    print(f"pyarrow failed: {e}")
                    last_error = e
                    try:
                        print("Trying parquet with fastparquet...")
                        return pd.read_parquet(p, engine="fastparquet")
                    except Exception as e2:
                        print(f"fastparquet failed: {e2}")
                        last_error = e2
                        continue
            elif p.suffix.lower() == ".csv":
                return pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"Failed reading {p.name}: {e}")
            last_error = e
            continue
    raise RuntimeError(f"Could not read any candidate file. Last error: {last_error}")


def maybe_scale_percent_to_decimal(series: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()
    if s.dropna().empty:
        return s
    q99 = s.abs().quantile(0.99)
    if q99 > 2:
        print(f"{name}: looks like percent units. Dividing by 100.")
        s = s / 100.0
    else:
        print(f"{name}: looks like decimal units already. Keeping as-is.")
    return s


def monthwise_permutation_importance(model, X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                                     n_repeats: int = 20, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    X = X.copy()
    y = pd.Series(y).copy()
    dates = pd.Series(dates, index=X.index)
    baseline_pred = model.predict(X)
    baseline_mse = mean_squared_error(y, baseline_pred)
    date_groups = list(dates.groupby(dates).groups.values())
    rows = []
    for col in X.columns:
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            for idx in date_groups:
                idx = list(idx)
                values = X_perm.loc[idx, col].to_numpy(copy=True)
                if len(values) > 1:
                    rng.shuffle(values)
                    X_perm.loc[idx, col] = values
            perm_pred = model.predict(X_perm)
            perm_mse = mean_squared_error(y, perm_pred)
            deltas.append(perm_mse - baseline_mse)
        rows.append({
            "feature": col,
            "perm_importance_mean": float(np.mean(deltas)),
            "perm_importance_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("perm_importance_mean", ascending=False)


def build_long_short_one_month(
    df_month: pd.DataFrame,
    top_pct: float = 0.25,
    weighting_mode: str = "dollar_neutral_50_50",
    transaction_cost_bps: float = 7,
    return_cap_upper: float = 1.0,
    return_cap_lower: float = -0.8,
):
    df_month = df_month.sort_values("predicted_log_return", ascending=False).copy()

    # Remove extreme realized returns that are often data errors or
    # economically uninvestable jumps in small / illiquid names.
    df_month = df_month[
        (df_month["actual_simple_return"] < return_cap_upper) &
        (df_month["actual_simple_return"] > return_cap_lower)
    ].copy()

    n = len(df_month)
    if n < 4:
        return pd.Series({
            "n_stocks": n,
            "n_each_side": 0,
            "long_return": 0.0,
            "short_leg_return": 0.0,
            "portfolio_return": 0.0,
        })

    q = max(int(np.floor(n * top_pct)), 1)
    longs = df_month.head(q)
    shorts = df_month.tail(q)

    long_ret = longs["actual_simple_return"].mean() if not longs.empty else 0.0
    short_stock_ret = shorts["actual_simple_return"].mean() if not shorts.empty else 0.0

    if weighting_mode == "dollar_neutral_50_50":
        portfolio_ret = 0.5 * long_ret - 0.5 * short_stock_ret
    elif weighting_mode == "gross_200":
        portfolio_ret = long_ret - short_stock_ret
    else:
        raise ValueError("weighting_mode must be 'dollar_neutral_50_50' or 'gross_200'")

    tc_rate = transaction_cost_bps / 10000.0
    portfolio_ret = portfolio_ret - (2.0 * tc_rate)

    return pd.Series({
        "n_stocks": n,
        "n_each_side": q,
        "long_return": long_ret,
        "short_leg_return": short_stock_ret,
        "portfolio_return": portfolio_ret,
    })


def performance_metrics(monthly_returns: pd.Series, rf_monthly: pd.Series | None = None, risk_free_self_row: bool = False) -> pd.Series:
    monthly_returns = pd.Series(monthly_returns).dropna()
    n = len(monthly_returns)
    if n == 0:
        return pd.Series({
            "Total Return (%)": np.nan,
            "Annualized Return (%)": np.nan,
            "Annualized Sharpe": np.nan,
            "Monthly Avg Return (%)": np.nan,
            "Monthly Volatility (%)": np.nan,
        })

    total_return = (1 + monthly_returns).prod() - 1
    annualized_return = (1 + total_return) ** (12 / n) - 1
    monthly_avg_return = monthly_returns.mean()
    monthly_volatility = monthly_returns.std(ddof=1)

    if risk_free_self_row:
        annualized_sharpe = 0.0
    elif rf_monthly is None:
        annualized_sharpe = np.sqrt(12) * monthly_avg_return / monthly_volatility if monthly_volatility > 0 else np.nan
    else:
        rf_monthly = pd.Series(rf_monthly).reindex(monthly_returns.index)
        excess = monthly_returns - rf_monthly
        excess_mean = excess.mean()
        excess_vol = excess.std(ddof=1)
        annualized_sharpe = np.sqrt(12) * excess_mean / excess_vol if excess_vol > 0 else np.nan

    return pd.Series({
        "Total Return (%)": total_return * 100,
        "Annualized Return (%)": annualized_return * 100,
        "Annualized Sharpe": annualized_sharpe,
        "Monthly Avg Return (%)": monthly_avg_return * 100,
        "Monthly Volatility (%)": monthly_volatility * 100,
    })


# ============================================================
# 4) Load stock data and benchmark data
# ============================================================
df = load_first_available(CANDIDATES)
print("shape:", df.shape)
print(df.head())
df = df.replace(r"\N", np.nan)

bench = load_first_available(BENCHMARK_CANDIDATES)
bench.columns = [c.strip() for c in bench.columns]
bench["Date"] = pd.to_datetime(bench["Date"], errors="coerce")
required_bench = ["Date", "hsi_simple_return", "rf_monthly_simple_return"]
missing_bench = [c for c in required_bench if c not in bench.columns]
if missing_bench:
    raise ValueError(f"Benchmark file missing columns: {missing_bench}. Found: {list(bench.columns)}")
bench = bench[required_bench].dropna().copy()
bench["year_mon"] = bench["Date"].dt.to_period("M")
bench = bench.drop_duplicates("year_mon").sort_values("Date")
print("\nBenchmark date range:", bench["Date"].min(), "→", bench["Date"].max())

required_base = [ID_COL, DATE_COL, TARGET_COL, "Company Market Cap (Millions)"] + FACTOR_COLS
missing_base = [c for c in required_base if c not in df.columns]
if missing_base:
    raise ValueError(f"Missing required columns: {missing_base}\nAvailable columns:\n{list(df.columns)}")

# ============================================================
# 5) Basic cleaning and feature prep
# ============================================================
d = df.copy()
d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
d = d.dropna(subset=[ID_COL, DATE_COL]).copy()

d["Company Market Cap (Millions)"] = pd.to_numeric(d["Company Market Cap (Millions)"], errors="coerce")
d.loc[d["Company Market Cap (Millions)"] <= 0, "Company Market Cap (Millions)"] = np.nan
d["log_mc"] = np.log(d["Company Market Cap (Millions)"])

for col in FACTOR_COLS + [TARGET_COL]:
    d[col] = maybe_scale_percent_to_decimal(d[col], col)
for col in ["Dividend yield", "Gross Margin, Percent", "INFL"]:
    if col in d.columns:
        d[col] = maybe_scale_percent_to_decimal(d[col], col)

numeric_candidates = [
    "Price Close", "Daily_Std", "Monthly_Volatility",
    "Price To Book Value Per Share (Daily Time Series Ratio)",
    "Price To Sales Per Share (Daily Time Series Ratio)",
    "Dividend yield", "Revenue_Growth_YoY", "Gross_Profit_Growth_YoY",
    "DP_HSI_lagged", "EP_HSI", "BM_HSI_lagged", "INFL", "log_SVAR",
    "diff_TBL_3MHIBOR_lagged", "diff_LTY_10Y", "diff_TMS_HIBOR",
    "diff_DFY", "diff_RF_Monthly_lagged",
]
for col in numeric_candidates:
    if col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")

pre_screen_rows = len(d)
d = d[
    (d["Price Close"] >= MIN_PRICE) &
    (d["Company Market Cap (Millions)"] >= MIN_MC_MILLIONS)
].copy()
print(
    f"\nAfter investability screen (Price Close >= {MIN_PRICE}, "
    f"Market Cap >= {MIN_MC_MILLIONS}M): {len(d):,} rows kept "
    f"out of {pre_screen_rows:,}"
)

feature_cols = [c for c in DESIRED_FEATURE_COLS if c in d.columns]
missing_optional = [c for c in DESIRED_FEATURE_COLS if c not in d.columns]
print("\nFeature columns used:")
print(feature_cols)
if missing_optional:
    print("\nDesired columns not found in file:")
    print(missing_optional)
if len(feature_cols) < 8:
    raise ValueError("Too few usable feature columns found for HK GBRT. Please check the exact column names.")

for col in feature_cols + [TARGET_COL]:
    d[col] = pd.to_numeric(d[col], errors="coerce")
d[feature_cols + [TARGET_COL]] = d[feature_cols + [TARGET_COL]].replace([np.inf, -np.inf], np.nan)

arr = d[feature_cols + [TARGET_COL]].to_numpy(dtype=np.float64)
bad_mask = ~np.isfinite(arr)
bad_counts = pd.Series(bad_mask.sum(axis=0), index=feature_cols + [TARGET_COL]).sort_values(ascending=False)
print("\nNon-finite counts by column:")
print(bad_counts[bad_counts > 0])

d_small = d[[ID_COL, DATE_COL, TARGET_COL] + feature_cols].dropna().copy()
d_small = d_small.sort_values([ID_COL, DATE_COL])
print("\nWorking sample shape after finite-value cleanup:", d_small.shape)

# ============================================================
# 6) Time-based split by unique months
# ============================================================
unique_dates = np.array(sorted(d_small[DATE_COL].dropna().unique()))
n_dates = len(unique_dates)
if n_dates < 10:
    raise ValueError(f"Only {n_dates} unique dates found. Need more monthly observations.")

n_train_dates = int(0.60 * n_dates)
n_val_dates = int(0.20 * n_dates)
train_dates = unique_dates[:n_train_dates]
val_dates = unique_dates[n_train_dates:n_train_dates + n_val_dates]
test_dates = unique_dates[n_train_dates + n_val_dates:]

train = d_small[d_small[DATE_COL].isin(train_dates)].copy()
val = d_small[d_small[DATE_COL].isin(val_dates)].copy()
test = d_small[d_small[DATE_COL].isin(test_dates)].copy()

print("\nBefore clipping:")
print(train[TARGET_COL].describe())

q_low = train[TARGET_COL].quantile(0.01)
q_high = train[TARGET_COL].quantile(0.99)
train[TARGET_COL] = train[TARGET_COL].clip(lower=q_low, upper=q_high)
val[TARGET_COL] = val[TARGET_COL].clip(lower=q_low, upper=q_high)
test[TARGET_COL] = test[TARGET_COL].clip(lower=q_low, upper=q_high)

train_month_counts = train.groupby(DATE_COL)[ID_COL].transform("count")
train_weights = 1.0 / train_month_counts
train_weights = train_weights * (len(train_weights) / train_weights.sum())
val_month_counts = val.groupby(DATE_COL)[ID_COL].transform("count")
val_weights = 1.0 / val_month_counts
val_weights = val_weights * (len(val_weights) / val_weights.sum())

X_train = train[feature_cols]
y_train = train[TARGET_COL]
dates_train = train[DATE_COL]
X_val = val[feature_cols]
y_val = val[TARGET_COL]
dates_val = val[DATE_COL]
X_test = test[feature_cols]
y_test = test[TARGET_COL]
dates_test = test[DATE_COL]

print("\nTrain shape:", X_train.shape, "Val shape:", X_val.shape, "Test shape:", X_test.shape)
print("Train period:", dates_train.min(), "→", dates_train.max())
print("Val period:  ", dates_val.min(), "→", dates_val.max())
print("Test period: ", dates_test.min(), "→", dates_test.max())
print("\nMax absolute value by feature in X_train:")
print(X_train.abs().max().sort_values(ascending=False).head(15))

if not np.isfinite(X_train.to_numpy(dtype=np.float64)).all():
    bad_cols = X_train.columns[~np.isfinite(X_train.to_numpy(dtype=np.float64)).all(axis=0)]
    raise ValueError(f"X_train still contains non-finite values in columns: {list(bad_cols)}")
if not np.isfinite(y_train.to_numpy(dtype=np.float64)).all():
    raise ValueError("y_train still contains non-finite values")

# ============================================================
# 7) GBRT tuning and fitting
# ============================================================
tune_frac = 0.25
train_tune = train.sample(frac=tune_frac, random_state=RANDOM_SEED)
X_tune = train_tune[feature_cols]
y_tune = train_tune[TARGET_COL]
w_tune = train_weights.loc[train_tune.index].values

n_estimators_tune = 80
n_estimators_final = 250
param_grid = [
    {"learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 20, "subsample": 0.7},
    {"learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20, "subsample": 0.7},
    {"learning_rate": 0.10, "max_depth": 2, "min_samples_leaf": 20, "subsample": 0.7},
    {"learning_rate": 0.10, "max_depth": 3, "min_samples_leaf": 20, "subsample": 0.7},
]

best_val_mse = np.inf
best_params = None
for params in param_grid:
    gbrt = GradientBoostingRegressor(n_estimators=n_estimators_tune, random_state=RANDOM_SEED, **params)
    gbrt.fit(X_tune, y_tune, sample_weight=w_tune)
    y_val_pred = gbrt.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred, sample_weight=val_weights.values)
    print(f"params={params}, val MSE={mse_val:.6f}")
    if mse_val < best_val_mse:
        best_val_mse = mse_val
        best_params = params

print("\nBest GBRT params on tuning grid:", best_params)
print(f"Best validation MSE with {n_estimators_tune} trees: {best_val_mse:.6f}")

gbrt_model = GradientBoostingRegressor(n_estimators=n_estimators_final, random_state=RANDOM_SEED, **best_params)
gbrt_model.fit(X_train, y_train, sample_weight=train_weights.values)
print("Refit final HK GBRT on full Train with params:", best_params)

# ============================================================
# 8) Predict and evaluate model
# ============================================================
y_pred_tr = gbrt_model.predict(X_train)
y_pred_te = gbrt_model.predict(X_test)

r2_tr = r2_score(y_train, y_pred_tr)
r2_te = r2_score(y_test, y_pred_te)
rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_te))
mae_tr = mean_absolute_error(y_train, y_pred_tr)
mae_te = mean_absolute_error(y_test, y_pred_te)
res_test = y_test - y_pred_te

if "Mkt - RF_lagged" in X_test.columns:
    slope, _, _, _, _ = linregress(X_test["Mkt - RF_lagged"].values.ravel(), y_test.values)
    print(f"Raw CAPM-like β (test, vs Mkt - RF_lagged only): {slope:.6f}")
print(f"Mean residual (test): {res_test.mean():.8f}")
print(f"Train  → R²={r2_tr:.4f}, RMSE={rmse_tr:.6f}, MAE={mae_tr:.6f}")
print(f"Test   → R²={r2_te:.4f}, RMSE={rmse_te:.6f}, MAE={mae_te:.6f}")

test_os = pd.DataFrame({"date": dates_test.values, "y": y_test.values, "y_hat": y_pred_te})
test_os["y_bar"] = test_os.groupby("date")["y"].transform("mean")
num = ((test_os["y"] - test_os["y_hat"]) ** 2).sum()
den = ((test_os["y"] - test_os["y_bar"]) ** 2).sum()
r2_os = 1 - num / den if den != 0 else np.nan
print(f"OS R^2 (paper-style): {r2_os:.6f}")

importance_table = monthwise_permutation_importance(gbrt_model, X_test, y_test, dates_test, n_repeats=20, random_state=RANDOM_SEED)
importance_table_top = importance_table.head(TOP_N_IMPORTANCE).copy()
print("\nHK GBRT monthwise permutation importances (top 10, test set):")
print(importance_table_top)

# ============================================================
# 9) TASK 5: Backtest long-short portfolio vs HSI and 1M HIBOR
# ============================================================
backtest = test[[ID_COL, DATE_COL, TARGET_COL]].copy()
backtest["predicted_log_return"] = y_pred_te
backtest["actual_simple_return"] = np.exp(backtest[TARGET_COL]) - 1
backtest["year_mon"] = backtest[DATE_COL].dt.to_period("M")

print("\nBacktest realized-return diagnostics before filtering:")
print(backtest["actual_simple_return"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
print(
    f"Rows with actual_simple_return <= {RETURN_CAP_LOWER:.2f}: "
    f"{int((backtest['actual_simple_return'] <= RETURN_CAP_LOWER).sum())}"
)
print(
    f"Rows with actual_simple_return >= {RETURN_CAP_UPPER:.2f}: "
    f"{int((backtest['actual_simple_return'] >= RETURN_CAP_UPPER).sum())}"
)

portfolio_monthly = (
    backtest
    .groupby("year_mon", group_keys=False)
    .apply(
        build_long_short_one_month,
        top_pct=TOP_PCT,
        weighting_mode=WEIGHTING_MODE,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        return_cap_upper=RETURN_CAP_UPPER,
        return_cap_lower=RETURN_CAP_LOWER,
    )
    .reset_index()
)

portfolio_monthly = portfolio_monthly.merge(
    bench[["year_mon", "Date", "hsi_simple_return", "rf_monthly_simple_return"]],
    on="year_mon",
    how="inner"
).rename(columns={"Date": "benchmark_date"})

if portfolio_monthly.empty:
    raise ValueError(
        "No overlap between test-period months and benchmark file months. "
        "Check the stock-data test period and the benchmark date range."
    )

portfolio_monthly["Date"] = portfolio_monthly["year_mon"].dt.to_timestamp("M")
portfolio_monthly = portfolio_monthly.sort_values("Date").reset_index(drop=True)
portfolio_monthly["cum_portfolio"] = (1 + portfolio_monthly["portfolio_return"]).cumprod() - 1
portfolio_monthly["cum_hsi"] = (1 + portfolio_monthly["hsi_simple_return"]).cumprod() - 1
portfolio_monthly["cum_rf"] = (1 + portfolio_monthly["rf_monthly_simple_return"]).cumprod() - 1

print("\nBenchmark overlap summary:")
print("Backtest months used:", len(portfolio_monthly))
print("Overlap period:", portfolio_monthly["Date"].min(), "→", portfolio_monthly["Date"].max())
print("Average stocks per month in test overlap:", round(portfolio_monthly["n_stocks"].mean(), 2))
print("Average stocks per long/short side:", round(portfolio_monthly["n_each_side"].mean(), 2))

plt.figure(figsize=(10, 5))
plt.plot(portfolio_monthly["Date"], portfolio_monthly["cum_portfolio"], label="GBRT Long-Short Portfolio")
plt.plot(portfolio_monthly["Date"], portfolio_monthly["cum_hsi"], linestyle="--", label="Hang Seng Index")
plt.plot(portfolio_monthly["Date"], portfolio_monthly["cum_rf"], linestyle=":", label="Risk-Free (1M HIBOR)")
plt.title("Cumulative Returns: HK GBRT Portfolio vs HSI vs Risk-Free")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 10) TASK 6: Performance metrics
# ============================================================
portfolio_series = portfolio_monthly.set_index("Date")["portfolio_return"]
hsi_series = portfolio_monthly.set_index("Date")["hsi_simple_return"]
rf_series = portfolio_monthly.set_index("Date")["rf_monthly_simple_return"]

summary_table = pd.DataFrame({
    "GBRT Long-Short Portfolio": performance_metrics(portfolio_series, rf_series),
    "Hang Seng Index": performance_metrics(hsi_series, rf_series),
    "Risk-Free": performance_metrics(rf_series, rf_series, risk_free_self_row=True),
})
print("\nPerformance summary (Task 6):")
print(summary_table)

# ============================================================
# 11) Save outputs
# ============================================================
portfolio_monthly.to_csv("gbrt_fixed_portfolio_monthly_results.csv", index=False)
summary_table.to_csv("gbrt_fixed_performance_summary.csv")
importance_table.to_csv("gbrt_fixed_feature_importances.csv", index=False)

print("\nSaved files:")
print(" - gbrt_fixed_portfolio_monthly_results.csv")
print(" - gbrt_fixed_performance_summary.csv")
print(" - gbrt_fixed_feature_importances.csv")

print("\nNotes:")
print(f" - Weighting mode: {WEIGHTING_MODE}")
print(f" - Portfolio rule: long top {int(TOP_PCT*100)}% and short bottom {int(TOP_PCT*100)}% by predicted return each month.")
print(" - Portfolio returns use realized simple returns converted from Target_Forward_Log_Return.")
print(f" - Investability screen: Price Close >= {MIN_PRICE} and Company Market Cap >= {MIN_MC_MILLIONS}M.")
print(f" - Realized-return filter inside each rebalance month: ({RETURN_CAP_LOWER}, {RETURN_CAP_UPPER}) exclusive.")
print(f" - Transaction cost applied each month: {TRANSACTION_COST_BPS} bps per leg (2 legs total).")
print(" - HSI and risk-free are taken directly from the benchmark CSV as simple monthly returns.")
print(" - If you want a 100/100 long-short portfolio instead of 50/50 dollar-neutral, set WEIGHTING_MODE = 'gross_200'.")
