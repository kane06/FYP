import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import linregress

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
TOP_PCT = 0.25
WEIGHTING_MODE = "dollar_neutral_50_50"  # options: "dollar_neutral_50_50" or "gross_200"

# Backtest realism controls
MIN_PRICE = 1.0
MIN_MC_MILLIONS = 500.0
TRANSACTION_COST_BPS = 7
RETURN_CAP_UPPER = 1.0
RETURN_CAP_LOWER = -0.8

# ============================================================
# 2) Safer HK OLS-full feature set
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


def build_long_short_one_month(
    df_month: pd.DataFrame,
    top_pct: float = 0.25,
    weighting_mode: str = "dollar_neutral_50_50",
    transaction_cost_bps: float = 7,
    return_cap_upper: float = 1.0,
    return_cap_lower: float = -0.8,
):
    df_month = df_month.sort_values("predicted_log_return", ascending=False).copy()

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
# 4) Load data
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
    raise ValueError(
        f"Missing required columns: {missing_base}\n"
        f"Available columns:\n{list(df.columns)}"
    )

# ============================================================
# 5) Basic cleaning
# ============================================================
d = df.copy()
d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
d = d.dropna(subset=[ID_COL, DATE_COL]).copy()

d["Company Market Cap (Millions)"] = pd.to_numeric(
    d["Company Market Cap (Millions)"], errors="coerce"
)
d.loc[d["Company Market Cap (Millions)"] <= 0, "Company Market Cap (Millions)"] = np.nan
d["log_mc"] = np.log(d["Company Market Cap (Millions)"])

for col in FACTOR_COLS + [TARGET_COL]:
    d[col] = maybe_scale_percent_to_decimal(d[col], col)
for col in ["Dividend yield", "Gross Margin, Percent", "INFL"]:
    if col in d.columns:
        d[col] = maybe_scale_percent_to_decimal(d[col], col)

numeric_candidates = [
    "Price Close",
    "Daily_Std",
    "Monthly_Volatility",
    "Price To Book Value Per Share (Daily Time Series Ratio)",
    "Price To Sales Per Share (Daily Time Series Ratio)",
    "Dividend yield",
    "Revenue_Growth_YoY",
    "Gross_Profit_Growth_YoY",
    "DP_HSI_lagged",
    "EP_HSI",
    "BM_HSI_lagged",
    "INFL",
    "log_SVAR",
    "diff_TBL_3MHIBOR_lagged",
    "diff_LTY_10Y",
    "diff_TMS_HIBOR",
    "diff_DFY",
    "diff_RF_Monthly_lagged",
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
    raise ValueError(
        "Too few usable feature columns found for HK OLS-full. "
        "Please check the exact column names in your full dataset."
    )

# ============================================================
# 6) Sanitize values before building modeling sample
# ============================================================
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
print(d_small[feature_cols + [TARGET_COL]].describe())

# ============================================================
# 7) Time-based split by unique dates
# ============================================================
unique_dates = np.array(sorted(d_small[DATE_COL].dropna().unique()))
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

train = d_small[d_small[DATE_COL].isin(train_dates)].copy()
val = d_small[d_small[DATE_COL].isin(val_dates)].copy()
test = d_small[d_small[DATE_COL].isin(test_dates)].copy()

print("\nBefore clipping:")
print("Train target summary:")
print(train[TARGET_COL].describe())

# ============================================================
# 8) Clip target using TRAIN ONLY
# ============================================================
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

# ============================================================
# 9) Build X / y
# ============================================================
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
# 10) Fit OLS-full
# ============================================================
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred_tr = linreg.predict(X_train)
y_pred_te = linreg.predict(X_test)

alpha = linreg.intercept_
coefs = pd.Series(linreg.coef_, index=feature_cols)

print(f"\nAlpha (intercept): {alpha:.6f}")
print("Betas (HK OLS-full):")
print(coefs)
print("Fitted HK OLS-full model.")

# ============================================================
# 11) Evaluation
# ============================================================
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

print("\nPredicted (test) summary:")
print(pd.Series(y_pred_te).describe())

print("\nActual (test) summary:")
print(pd.Series(y_test).describe())

band = 0.01
close_to_zero = ((y_pred_te > -band) & (y_pred_te < band)).mean()
print(f"\nShare of predictions in [-{band}, {band}]: {close_to_zero:.2%}")

print("\nPredictor summaries (working sample):")
print(d_small[feature_cols].describe())

print(f"\n{TARGET_COL} summary (working sample):")
print(d_small[TARGET_COL].describe())

# ============================================================
# 12) Paper-style out-of-sample R^2
# ============================================================
test_os = pd.DataFrame({
    "date": dates_test.values,
    "y": y_test.values,
    "y_hat": y_pred_te
})

test_os["y_bar"] = test_os.groupby("date")["y"].transform("mean")

num = ((test_os["y"] - test_os["y_hat"]) ** 2).sum()
den = ((test_os["y"] - test_os["y_bar"]) ** 2).sum()
r2_os = 1 - num / den if den != 0 else np.nan
print(f"\nOS R^2 (paper-style): {r2_os:.6f}")

# ============================================================
# 13) Cumulative squared prediction error
# ============================================================
sq_err = (y_pred_te - y_test) ** 2

test_df = X_test.copy()
test_df["actual"] = y_test.values
test_df["predicted"] = y_pred_te
test_df["sq_err"] = sq_err
test_df["date"] = dates_test.values

test_df = test_df.sort_values("date")
test_df["cum_sq_err"] = test_df["sq_err"].cumsum()

plot_df = (
    test_df
    .assign(year_mon=lambda x: x["date"].dt.to_period("M"))
    .groupby("year_mon")
    .last()
)

plt.figure(figsize=(7, 3))
plt.plot(plot_df.index.to_timestamp(), plot_df["cum_sq_err"])
plt.title("Cumulative Squared Prediction Error – HK OLS-full Test Set")
plt.xlabel("Date")
plt.ylabel("Σ(ŷ – y)²")
plt.tight_layout()
plt.show()

# ============================================================
# 14) Scatter: target vs Mkt - RF with fitted line
# ============================================================
if "Mkt - RF_lagged" in feature_cols:
    plt.figure(figsize=(6, 4))
    plt.scatter(
        X_test["Mkt - RF_lagged"], y_test,
        alpha=0.05, s=3, label="Actual (Test)"
    )

    xs = np.linspace(X_test["Mkt - RF_lagged"].min(), X_test["Mkt - RF_lagged"].max(), 200)

    X_line = pd.DataFrame(
        {col: X_train[col].mean() for col in feature_cols},
        index=range(len(xs))
    )
    X_line["Mkt - RF_lagged"] = xs

    y_line = linreg.predict(X_line)

    plt.plot(xs, y_line, linewidth=2, label="OLS-full fit (others at train mean)")
    plt.title("HK OLS-full: Target Forward Return vs Mkt - RF (Test)")
    plt.xlabel("Market excess return (Mkt - RF)")
    plt.ylabel("Target forward log return")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 15) Actual vs predicted
# ============================================================
mask_plot = (np.abs(y_pred_te) < 0.5) & (np.abs(y_test) < 0.5)

plt.figure(figsize=(5, 5))
plt.scatter(y_pred_te[mask_plot], y_test[mask_plot], alpha=0.4, s=12)

min_val = min(y_pred_te[mask_plot].min(), y_test[mask_plot].min())
max_val = max(y_pred_te[mask_plot].max(), y_test[mask_plot].max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Predicted forward log return")
plt.ylabel("Actual forward log return")
plt.title("Actual vs Predicted (Test) – HK OLS-full (zoomed)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("Share of points outside zoom region:", (~mask_plot).mean())

# ============================================================
# 16) Coefficients table
# ============================================================
coef_table = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": coefs.values
})
print("\nCoefficient table:")
print(coef_table)

coef_table_sorted = coef_table.copy()
coef_table_sorted["abs_coef"] = coef_table_sorted["coefficient"].abs()
coef_table_sorted = coef_table_sorted.sort_values("abs_coef", ascending=True)

print("\nHK OLS-full 'feature importance' (|coefficient|):")
print(coef_table_sorted.sort_values("abs_coef", ascending=False))

plt.figure(figsize=(10, 7))
plt.barh(coef_table_sorted["feature"], coef_table_sorted["abs_coef"])
plt.xlabel("|Coefficient|")
plt.title("HK OLS-full Feature Importances (absolute coefficients)")
plt.tight_layout()
plt.show()

# ============================================================
# 17) TASK 5: Backtest long-short portfolio vs HSI and 1M HIBOR
# ============================================================
backtest = test[[ID_COL, DATE_COL, TARGET_COL]].copy()
backtest["predicted_log_return"] = y_pred_te
backtest["actual_simple_return"] = np.exp(backtest[TARGET_COL]) - 1
backtest["year_mon"] = backtest[DATE_COL].dt.to_period("M")

pre_filter_counts = backtest.groupby("year_mon")["actual_simple_return"].agg(
    total_names="size",
    extreme_names=lambda s: ((s >= RETURN_CAP_UPPER) | (s <= RETURN_CAP_LOWER)).sum(),
)
print("\nExtreme realized-return diagnostics before monthly filtering:")
print(pre_filter_counts.head())
print("Average extreme names per month:", round(pre_filter_counts["extreme_names"].mean(), 2))

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
plt.plot(portfolio_monthly["Date"], portfolio_monthly["cum_portfolio"], label="OLS Long-Short Portfolio")
plt.plot(portfolio_monthly["Date"], portfolio_monthly["cum_hsi"], linestyle="--", label="Hang Seng Index")
plt.plot(portfolio_monthly["Date"], portfolio_monthly["cum_rf"], linestyle=":", label="Risk-Free (1M HIBOR)")
plt.title("Cumulative Returns: HK OLS-full Portfolio vs HSI vs Risk-Free")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 18) TASK 6: Performance metrics
# ============================================================
portfolio_series = portfolio_monthly.set_index("Date")["portfolio_return"]
hsi_series = portfolio_monthly.set_index("Date")["hsi_simple_return"]
rf_series = portfolio_monthly.set_index("Date")["rf_monthly_simple_return"]

summary_table = pd.DataFrame({
    "OLS Long-Short Portfolio": performance_metrics(portfolio_series, rf_series),
    "Hang Seng Index": performance_metrics(hsi_series, rf_series),
    "Risk-Free": performance_metrics(rf_series, rf_series, risk_free_self_row=True),
})
print("\nPerformance summary (Task 6):")
print(summary_table)

# ============================================================
# 19) Save outputs
# ============================================================
portfolio_monthly.to_csv("olsfull_fixed_portfolio_monthly_results.csv", index=False)
summary_table.to_csv("olsfull_fixed_performance_summary.csv")
coef_table.to_csv("olsfull_coefficients.csv", index=False)

print("\nSaved files:")
print(" - olsfull_fixed_portfolio_monthly_results.csv")
print(" - olsfull_fixed_performance_summary.csv")
print(" - olsfull_coefficients.csv")

print("\nInterpretation:")
print(" - Coefficients are the HK OLS-full betas on the available feature set.")
print(" - Alpha is the intercept.")
print(f" - Target is '{TARGET_COL}'.")
print(" - This HK version uses forward log return, not forward excess return.")
print(f" - Investability screen: Price Close >= {MIN_PRICE} and Company Market Cap >= {MIN_MC_MILLIONS}M.")
print(f" - Return filter inside each rebalance month: {RETURN_CAP_LOWER:.0%} < realized simple return < {RETURN_CAP_UPPER:.0%}.")
print(f" - Transaction costs: {TRANSACTION_COST_BPS} bps per leg per month.")
print(f" - Portfolio rule: long top {int(TOP_PCT*100)}% and short bottom {int(TOP_PCT*100)}% by predicted return each month.")
print(" - HSI and risk-free are taken directly from the benchmark CSV as simple monthly returns.")
