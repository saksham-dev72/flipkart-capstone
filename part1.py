"""
Part 1 — Estimation, inference and a regression baseline
Domain: E-commerce (Flipkart)

Covers Tasks 1.1 to 1.7:
  1.1 Mean + 95% CI of order_value_inr_hundreds
  1.2 Z-test against H0: mu = 8.0
  1.3 Sample-size calculation
  1.4 Simple OLS regression (value ~ X4)
  1.5 Gradient descent to fit the same line
  1.6 Funnel check and log-log regression
  1.7 Multiple regression by normal equation
"""

import csv
import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

rows = []
with open(os.path.join("data", "records.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

N = len(rows)  # 800

# Parse columns
X1 = np.array([float(r["customer_tenure_months"]) for r in rows])
X2 = np.array([float(r["complaint_count_90d"]) for r in rows])
X3 = np.array([float(r["discount_order_share"]) for r in rows])
X4 = np.array([float(r["avg_order_value_inr_hundreds"]) for r in rows])
X5 = np.array([float(r["loyalty_score"]) for r in rows])
X6 = np.array([float(r["cod_flag"]) for r in rows])
X7 = np.array([float(r["plus_member_flag"]) for r in rows])
outcome = np.array([int(r["churned"]) for r in rows])
value = np.array([float(r["order_value_inr_hundreds"]) for r in rows])

results = {}

# ---------------------------------------------------------------------------
# Task 1.1 — Estimate with an interval
# ---------------------------------------------------------------------------
n = N
mean_value = float(np.sum(value) / n)
# Sample SD: divide by n-1
sd_value = float(math.sqrt(np.sum((value - mean_value) ** 2) / (n - 1)))
se_value = sd_value / math.sqrt(n)
ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value

results["n"] = n
results["mean_value"] = mean_value
results["sd_value"] = sd_value
results["se_value"] = se_value
results["ci95_low"] = ci95_low
results["ci95_high"] = ci95_high

print(f"[1.1] n={n}, mean_value={mean_value:.4f}  (Checkpoint: 800, 8.9382)")

# ---------------------------------------------------------------------------
# Task 1.2 — Z-test H0: mu = 8.0
# ---------------------------------------------------------------------------
z_stat = (mean_value - 8.0) / se_value
reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

results["z_stat"] = z_stat
results["reject_h0"] = reject_h0

print(f"[1.2] z_stat={z_stat:.4f}, reject_h0={reject_h0}")

# ---------------------------------------------------------------------------
# Task 1.3 — Sample size needed for E=0.5
# ---------------------------------------------------------------------------
E = 0.5
n_required = math.ceil((1.96 * sd_value / E) ** 2)

results["n_required"] = n_required
print(f"[1.3] n_required={n_required}")

# ---------------------------------------------------------------------------
# Task 1.4 — Simple OLS regression: value ~ X4
# ---------------------------------------------------------------------------
# Manual formulas — no numpy.polyfit / cov / corrcoef
mean_X4 = float(np.sum(X4) / n)
mean_val = mean_value

# cov(X4, value) with divisor n-1
cov_X4_val = float(np.sum((X4 - mean_X4) * (value - mean_val)) / (n - 1))
var_X4 = float(np.sum((X4 - mean_X4) ** 2) / (n - 1))

reg_slope = cov_X4_val / var_X4
reg_intercept = mean_val - reg_slope * mean_X4

fitted = reg_slope * X4 + reg_intercept
residuals = value - fitted

tss = float(np.sum((value - mean_val) ** 2))
ess = float(np.sum((fitted - mean_val) ** 2))
rss = float(np.sum(residuals ** 2))

tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
r2 = 1.0 - rss / tss
rmse = math.sqrt(rss / n)
mae = float(np.sum(np.abs(residuals)) / n)

results["reg_slope"] = reg_slope
results["reg_intercept"] = reg_intercept
results["tss"] = tss
results["ess"] = ess
results["rss"] = rss
results["tss_equals_ess_plus_rss"] = tss_equals_ess_plus_rss
results["r2"] = r2
results["rmse"] = rmse
results["mae"] = mae

print(f"[1.4] reg_slope={reg_slope:.4f}, r2={r2:.4f}")

# ---------------------------------------------------------------------------
# Task 1.5 — Gradient descent on standardised X4
# ---------------------------------------------------------------------------
mean_X4_gd = mean_X4
sd_X4_gd = math.sqrt(np.sum((X4 - mean_X4_gd) ** 2) / (n - 1))
z_gd = (X4 - mean_X4_gd) / sd_X4_gd  # standardised predictor

m = 0.0
c = 0.0
eta = 0.1

def gd_cost(m_, c_, z_, y_):
    e = y_ - (m_ * z_ + c_)
    return float(np.sum(e ** 2) / len(y_))

gd_J0 = gd_cost(m, c, z_gd, value)
results["gd_J0"] = gd_J0

costs_at = {1: None, 10: None, 100: None, 500: None}
for step in range(1, 501):
    e = value - (m * z_gd + c)
    grad_m = -2.0 / n * float(np.sum(z_gd * e))
    grad_c = -2.0 / n * float(np.sum(e))
    m -= eta * grad_m
    c -= eta * grad_c
    if step in costs_at:
        costs_at[step] = gd_cost(m, c, z_gd, value)

results["gd_J1"] = costs_at[1]
results["gd_J10"] = costs_at[10]
results["gd_J100"] = costs_at[100]
results["gd_J500"] = costs_at[500]

# Convert back to original scale
gd_slope = m / sd_X4_gd
gd_intercept = c - m * mean_X4_gd / sd_X4_gd
gd_slope_gap = abs(gd_slope - reg_slope)

results["gd_slope"] = gd_slope
results["gd_intercept"] = gd_intercept
results["gd_slope_gap"] = gd_slope_gap

print(f"[1.5] gd_slope={gd_slope:.4f}, gd_slope_gap={gd_slope_gap:.6f}")

# ---------------------------------------------------------------------------
# Task 1.6 — Funnel check and log-log regression
# ---------------------------------------------------------------------------
# Sort by fitted value of Task 1.4 model (stable sort by fitted then row index)
sort_idx = np.argsort(fitted, kind="stable")
lower_idx = sort_idx[:400]
upper_idx = sort_idx[400:]

resid_sd_lower_raw = float(math.sqrt(np.sum((residuals[lower_idx] - np.mean(residuals[lower_idx])) ** 2) / (400 - 1)))
resid_sd_upper_raw = float(math.sqrt(np.sum((residuals[upper_idx] - np.mean(residuals[upper_idx])) ** 2) / (400 - 1)))
resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

results["resid_sd_lower_raw"] = resid_sd_lower_raw
results["resid_sd_upper_raw"] = resid_sd_upper_raw
results["resid_sd_ratio_raw"] = resid_sd_ratio_raw

# Log-log regression: ln(value) ~ ln(X4)
ln_value = np.log(value)
ln_X4 = np.log(X4)
mean_ln_val = float(np.sum(ln_value) / n)
mean_ln_X4 = float(np.sum(ln_X4) / n)

cov_ll = float(np.sum((ln_X4 - mean_ln_X4) * (ln_value - mean_ln_val)) / (n - 1))
var_ll = float(np.sum((ln_X4 - mean_ln_X4) ** 2) / (n - 1))
loglog_slope = cov_ll / var_ll
loglog_intercept = mean_ln_val - loglog_slope * mean_ln_X4

fitted_ll = loglog_slope * ln_X4 + loglog_intercept
resid_ll = ln_value - fitted_ll
tss_ll = float(np.sum((ln_value - mean_ln_val) ** 2))
rss_ll = float(np.sum(resid_ll ** 2))
loglog_r2 = 1.0 - rss_ll / tss_ll
loglog_rmse = math.sqrt(rss_ll / n)

results["loglog_intercept"] = loglog_intercept
results["loglog_slope"] = loglog_slope
results["loglog_r2"] = loglog_r2
results["loglog_rmse"] = loglog_rmse

# Funnel check for log-log
sort_idx_ll = np.argsort(fitted_ll, kind="stable")
lower_idx_ll = sort_idx_ll[:400]
upper_idx_ll = sort_idx_ll[400:]

resid_sd_lower_log = float(math.sqrt(np.sum((resid_ll[lower_idx_ll] - np.mean(resid_ll[lower_idx_ll])) ** 2) / (400 - 1)))
resid_sd_upper_log = float(math.sqrt(np.sum((resid_ll[upper_idx_ll] - np.mean(resid_ll[upper_idx_ll])) ** 2) / (400 - 1)))
resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

results["resid_sd_lower_log"] = resid_sd_lower_log
results["resid_sd_upper_log"] = resid_sd_upper_log
results["resid_sd_ratio_log"] = resid_sd_ratio_log

print(f"[1.6] resid_sd_ratio_raw={resid_sd_ratio_raw:.4f}, resid_sd_ratio_log={resid_sd_ratio_log:.4f}")

# ---------------------------------------------------------------------------
# Task 1.7 — Multiple regression by normal equation
# ---------------------------------------------------------------------------
# ln(value) ~ [1, ln(X4), X1, X2, X3, X5, X6, X7]
X_mr = np.column_stack([
    np.ones(n),
    ln_X4,
    X1, X2, X3, X5, X6, X7
])
Y_mr = ln_value

# Normal equation: beta = (X'X)^-1 X'Y  — using np.linalg.solve
XtX = X_mr.T @ X_mr
XtY = X_mr.T @ Y_mr
beta = np.linalg.solve(XtX, XtY)

for i, b in enumerate(beta):
    results[f"mr_beta_{i}"] = float(b)

fitted_mr = X_mr @ beta
resid_mr = Y_mr - fitted_mr
tss_mr = float(np.sum((Y_mr - np.mean(Y_mr)) ** 2))
rss_mr = float(np.sum(resid_mr ** 2))
mr_r2 = 1.0 - rss_mr / tss_mr
p_mr = 7  # predictors excluding constant
mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p_mr - 1)

results["mr_r2"] = mr_r2
results["mr_adj_r2"] = mr_adj_r2

# Adjusted R2 for log-log (p=1)
loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)
results["loglog_adj_r2"] = loglog_adj_r2

print(f"[1.7] mr_r2={mr_r2:.4f}, mr_adj_r2={mr_adj_r2:.4f}, loglog_adj_r2={loglog_adj_r2:.4f}")

# ---------------------------------------------------------------------------
# Write outputs/part1_results.txt
# ---------------------------------------------------------------------------
int_keys = {"n", "n_required"}
yes_no_keys = {"tss_equals_ess_plus_rss", "reject_h0"}

lines = []
for key, val in results.items():
    if key in int_keys:
        lines.append(f"{key} = {int(val)}")
    elif key in yes_no_keys:
        lines.append(f"{key} = {val}")
    else:
        lines.append(f"{key} = {float(val):.4f}")

with open(os.path.join("outputs", "part1_results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("[Part 1] Done. Wrote outputs/part1_results.txt")
