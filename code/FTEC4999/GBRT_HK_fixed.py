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

ID_COL = "Instrument"
DATE_COL = "Date"
TARGET_COL = "Target_Forward_Log_Return"
TOP_N_IMPORTANCE = 10

# ============================================================
# 2) Safer HK GBRT feature set
#    (same safer set as OLS-full first pass)
# ============================================================
DESIRED_FEATURE_COLS = [
    # HK asset pricing factors
    "Mkt - RF_lagged", "SMB_lagged", "HML", "MOM_lagged",

    # size / price / volatility
    "log_mc",
    "Price Close",
    "Daily_Std",
    "Monthly_Volatility",

    # valuation / growth
    "Price To Book Value Per Share (Daily Time Series Ratio)",
    "Price To Sales Per Share (Daily Time Series Ratio)",
    "Dividend yield",
    "Revenue_Growth_YoY",
    "Gross_Profit_Growth_YoY",

    # HK macro variables
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
    """
    Divide by 100 only when the magnitude strongly suggests percent units.
    Example: 5.2 -> 0.052, but 0.052 stays unchanged.
    """
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
    """
    Permute each feature *within month* so importance reflects cross-sectional stock selection
    rather than month-level regime timing. Features that are constant within month (like common
    macro/factor series) will receive near-zero importance unless they vary cross-sectionally.
    """
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


# ============================================================
# 4) Load data
# ============================================================
df = load_first_available(CANDIDATES)
print("shape:", df.shape)
print(df.head())

# Replace explicit string missing markers if present
df = df.replace(r"\N", np.nan)

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

# Safe log market cap
d["Company Market Cap (Millions)"] = pd.to_numeric(
    d["Company Market Cap (Millions)"], errors="coerce"
)
d.loc[d["Company Market Cap (Millions)"] <= 0, "Company Market Cap (Millions)"] = np.nan
d["log_mc"] = np.log(d["Company Market Cap (Millions)"])

# Scale factors and target if needed
for col in FACTOR_COLS + [TARGET_COL]:
    d[col] = maybe_scale_percent_to_decimal(d[col], col)

# Some columns may also come in percent units
for col in ["Dividend yield", "Gross Margin, Percent", "INFL"]:
    if col in d.columns:
        d[col] = maybe_scale_percent_to_decimal(d[col], col)

# Convert all desired numeric features to numeric if present
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

# Keep only available desired features
feature_cols = [c for c in DESIRED_FEATURE_COLS if c in d.columns]
missing_optional = [c for c in DESIRED_FEATURE_COLS if c not in d.columns]

print("\nFeature columns used:")
print(feature_cols)

if missing_optional:
    print("\nDesired columns not found in file:")
    print(missing_optional)

if len(feature_cols) < 8:
    raise ValueError(
        "Too few usable feature columns found for HK GBRT. "
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
bad_counts = pd.Series(
    bad_mask.sum(axis=0),
    index=feature_cols + [TARGET_COL]
).sort_values(ascending=False)

print("\nNon-finite counts by column:")
print(bad_counts[bad_counts > 0])

d_small = d[[ID_COL, DATE_COL, TARGET_COL] + feature_cols].dropna().copy()
d_small = d_small.sort_values([ID_COL, DATE_COL])

print("\nWorking sample shape after finite-value cleanup:", d_small.shape)
print(d_small[feature_cols + [TARGET_COL]].describe())

# ============================================================
# 7) Time-based split by UNIQUE MONTHS
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

# Equal-weight months so repeated macro rows do not dominate the fit
train_month_counts = train.groupby(DATE_COL)[ID_COL].transform("count")
train_weights = 1.0 / train_month_counts
train_weights = train_weights * (len(train_weights) / train_weights.sum())

val_month_counts = val.groupby(DATE_COL)[ID_COL].transform("count")
val_weights = 1.0 / val_month_counts
val_weights = val_weights * (len(val_weights) / val_weights.sum())

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
# 10) GBRT tuning on a train subsample
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
    gbrt = GradientBoostingRegressor(
        n_estimators=n_estimators_tune,
        random_state=RANDOM_SEED,
        **params
    )

    gbrt.fit(X_tune, y_tune, sample_weight=w_tune)
    y_val_pred = gbrt.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred, sample_weight=val_weights.values)

    print(f"params={params}, val MSE={mse_val:.6f}")

    if mse_val < best_val_mse:
        best_val_mse = mse_val
        best_params = params

print("\nBest GBRT params on tuning grid:", best_params)
print(f"Best validation MSE with {n_estimators_tune} trees: {best_val_mse:.6f}")

# ============================================================
# 11) Refit final GBRT on full training set
# ============================================================
gbrt_model = GradientBoostingRegressor(
    n_estimators=n_estimators_final,
    random_state=RANDOM_SEED,
    **best_params
)
gbrt_model.fit(X_train, y_train, sample_weight=train_weights.values)

print("Refit final HK GBRT on full Train with params:", best_params)

# ============================================================
# 12) Evaluate on train and test
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
# 13) Paper-style out-of-sample R^2
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
# 14) Monthwise permutation importances
#     (cross-sectional importance within each month)
# ============================================================
importance_table = monthwise_permutation_importance(
    gbrt_model,
    X_test,
    y_test,
    dates_test,
    n_repeats=20,
    random_state=RANDOM_SEED
)

importance_table_top = importance_table.head(TOP_N_IMPORTANCE).copy()

print("\nHK GBRT monthwise permutation importances (top 10, test set):")
print(importance_table_top)

print("\nNote: importance is computed by shuffling each feature within month only.")
print("This measures stock-selection importance and prevents common month-level factors like HML from dominating just because they move over time.")

# ============================================================
# 15) Cumulative squared prediction error
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
plt.title("Cumulative Squared Prediction Error – HK GBRT Test Set")
plt.xlabel("Date")
plt.ylabel("Σ(ŷ – y)²")
plt.tight_layout()
plt.show()

# ============================================================
# 16) Scatter: target vs Mkt - RF with fitted GBRT curve
#     holding other predictors at train means
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

    y_line = gbrt_model.predict(X_line)

    plt.plot(xs, y_line, linewidth=2, label="HK GBRT fit (others at train mean)")
    plt.title("HK GBRT: Target Forward Return vs Mkt - RF_lagged (Test)")
    plt.xlabel("Market excess return (Mkt - RF_lagged)")
    plt.ylabel("Target forward log return")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 17) Actual vs predicted
# ============================================================
plt.figure(figsize=(5, 5))
plt.scatter(y_pred_te, y_test, alpha=0.4, s=12)

min_val = min(y_pred_te.min(), y_test.min())
max_val = max(y_pred_te.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Predicted forward log return")
plt.ylabel("Actual forward log return")
plt.title("Actual vs Predicted HK Stock Returns (Test) – GBRT")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 18) Importance table
# ============================================================
print("\nImportance table (top 10):")
print(importance_table_top)

fi_display = importance_table_top.copy()

# Match your teammate's idea: convert top-10 importance into relative impact ratios
fi_display["importance_for_ratio"] = fi_display["perm_importance_mean"].clip(lower=0)

total_top10_importance = fi_display["importance_for_ratio"].sum()
if total_top10_importance > 0:
    fi_display["Ratio (%)"] = (
        fi_display["importance_for_ratio"] / total_top10_importance
    ) * 100
else:
    fi_display["Ratio (%)"] = 0.0

fi_sorted = fi_display.sort_values("Ratio (%)", ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(
    fi_sorted["feature"],
    fi_sorted["Ratio (%)"],
    color="skyblue",
    edgecolor="black",
    linewidth=0.5
)

for bar in bars:
    plt.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f'{bar.get_width():.1f}%',
        va='center',
        fontsize=10
    )

plt.xlabel('Relative Impact Ratio (%)', fontsize=11, fontweight='bold')
plt.title('Feature Importance: Relative Impact of Top 10 Features', fontsize=13, fontweight='bold')
plt.xlim(0, fi_sorted["Ratio (%)"].max() + 5 if len(fi_sorted) else 5)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nInterpretation:")
print(" - This is the HK adaptation of your GBRT full model.")
print(" - Target is Target_Forward_Log_Return.")
print(" - Importance is based on monthwise permutation importance on the test set, not tree impurity importance.")
print(" - Shuffling is done within each month, so the ranking reflects cross-sectional stock-selection usefulness.")
print(" - This HK version uses forward log return, not forward excess return.")
