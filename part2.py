"""
Part 2 — Classification and evaluation
Domain: E-commerce (Flipkart)

Covers Tasks 2.1 to 2.8:
  2.1 Stratified 70/30 split + standardisation
  2.2 Logistic regression from scratch (1000 steps, eta=0.1)
  2.3 Learning-rate effect (eta in {0.01, 0.1, 1.0}, 200 steps)
  2.4 Odds ratios
  2.5 Evaluate at tau=0.5 and tau=0.3
  2.6 Threshold sweep (0.1 … 0.9) → part2_thresholds.csv
  2.7 KNN with 5-fold CV → part2_knn_cv.csv
  2.8 Cost-based threshold selection
"""

import csv
import math
import os

import numpy as np

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------
rows = []
with open(os.path.join("data", "records.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

N = len(rows)

X_all = np.array([
    [float(r["customer_tenure_months"]),
     float(r["complaint_count_90d"]),
     float(r["discount_order_share"]),
     float(r["avg_order_value_inr_hundreds"]),
     float(r["loyalty_score"]),
     float(r["cod_flag"]),
     float(r["plus_member_flag"])]
    for r in rows
])
outcome_all = np.array([int(r["churned"]) for r in rows])

# ---------------------------------------------------------------------------
# Task 2.1 — Stratified 70/30 split
# ---------------------------------------------------------------------------
idx_neg = [i for i in range(N) if outcome_all[i] == 0]
idx_pos = [i for i in range(N) if outcome_all[i] == 1]

n_neg = len(idx_neg)
n_pos = len(idx_pos)
n_neg_train = int(7 * n_neg // 10)
n_pos_train = int(7 * n_pos // 10)

train_idx = sorted(idx_neg[:n_neg_train] + idx_pos[:n_pos_train])
test_idx  = sorted(idx_neg[n_neg_train:] + idx_pos[n_pos_train:])

X_train = X_all[train_idx]
y_train  = outcome_all[train_idx]
X_test   = X_all[test_idx]
y_test   = outcome_all[test_idx]

train_rows = len(train_idx)
train_positives = int(np.sum(y_train))
test_rows  = len(test_idx)
test_positives  = int(np.sum(y_test))

print(f"[2.1] train_rows={train_rows}, train_positives={train_positives}")
print(f"      test_rows={test_rows},  test_positives={test_positives}")
print(f"      Checkpoint: 560, 119, 240, 51")

# Standardise using training mean and sample SD
train_mean = np.sum(X_train, axis=0) / train_rows
train_sd   = np.sqrt(np.sum((X_train - train_mean) ** 2, axis=0) / (train_rows - 1))

X_train_s = (X_train - train_mean) / train_sd
X_test_s  = (X_test  - train_mean) / train_sd

# ---------------------------------------------------------------------------
# Task 2.2 — Logistic regression from scratch
# ---------------------------------------------------------------------------
def sigmoid(z):
    # Numerically stable sigmoid
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def bce_loss(probs, y):
    eps = 1e-15
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

w = np.zeros(7)
b = 0.0
eta_lr = 0.1
n_tr = train_rows

# epoch 0 loss
probs0 = sigmoid(X_train_s @ w + b)
loss_epoch_0 = bce_loss(probs0, y_train)
print(f"[2.2] loss_epoch_0={loss_epoch_0:.4f}  (Checkpoint: 0.6931)")

losses_at = {}
for step in range(1, 1001):
    probs = sigmoid(X_train_s @ w + b)
    err   = probs - y_train
    grad_w = (X_train_s.T @ err) / n_tr
    grad_b = np.sum(err) / n_tr
    w -= eta_lr * grad_w
    b -= eta_lr * float(grad_b)
    if step in (1, 10, 100, 1000):
        losses_at[step] = bce_loss(sigmoid(X_train_s @ w + b), y_train)

final_probs_train = sigmoid(X_train_s @ w + b)
final_probs_test  = sigmoid(X_test_s  @ w + b)
test_bce = bce_loss(final_probs_test, y_test)

lr_results = {
    "train_rows": train_rows,
    "train_positives": train_positives,
    "test_rows": test_rows,
    "test_positives": test_positives,
    "loss_epoch_0": loss_epoch_0,
    "loss_epoch_1": losses_at[1],
    "loss_epoch_10": losses_at[10],
    "loss_epoch_100": losses_at[100],
    "loss_epoch_1000": losses_at[1000],
}
for j in range(7):
    lr_results[f"weight_X{j+1}"] = float(w[j])
lr_results["bias"] = float(b)
lr_results["test_bce"] = test_bce

# Store final weights / bias for use in later tasks
final_w = w.copy()
final_b = float(b)
final_probs_train_lr = final_probs_train.copy()
final_probs_test_lr  = final_probs_test.copy()

# ---------------------------------------------------------------------------
# Task 2.3 — Learning-rate effect
# ---------------------------------------------------------------------------
for eta_test in [0.01, 0.1, 1.0]:
    w_tmp = np.zeros(7)
    b_tmp = 0.0
    for step in range(200):
        probs_tmp = sigmoid(X_train_s @ w_tmp + b_tmp)
        err_tmp   = probs_tmp - y_train
        w_tmp -= eta_test * (X_train_s.T @ err_tmp) / n_tr
        b_tmp -= eta_test * float(np.sum(err_tmp)) / n_tr
    key = f"loss_epoch_200_eta_{eta_test}"
    lr_results[key] = bce_loss(sigmoid(X_train_s @ w_tmp + b_tmp), y_train)
    print(f"[2.3] {key} = {lr_results[key]:.4f}")

# ---------------------------------------------------------------------------
# Task 2.4 — Odds ratios
# ---------------------------------------------------------------------------
for j in range(7):
    lr_results[f"odds_ratio_X{j+1}"] = math.exp(float(final_w[j]))

# ---------------------------------------------------------------------------
# Helper: confusion matrix metrics
# ---------------------------------------------------------------------------
def confusion_metrics(probs, y, tau):
    preds = (probs >= tau).astype(int)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return tp, fp, tn, fn, accuracy, precision, recall, f1, specificity

# ---------------------------------------------------------------------------
# Task 2.5 — Evaluate at tau=0.5 and tau=0.3
# ---------------------------------------------------------------------------
for tau in [0.5, 0.3]:
    tp, fp, tn, fn, acc, prec, rec, f1, spec = confusion_metrics(final_probs_test_lr, y_test, tau)
    tstr = str(tau)
    lr_results[f"tau_{tstr}_tp"] = tp
    lr_results[f"tau_{tstr}_fp"] = fp
    lr_results[f"tau_{tstr}_tn"] = tn
    lr_results[f"tau_{tstr}_fn"] = fn
    lr_results[f"tau_{tstr}_accuracy"]    = acc
    lr_results[f"tau_{tstr}_precision"]   = prec
    lr_results[f"tau_{tstr}_recall"]      = rec
    lr_results[f"tau_{tstr}_f1"]          = f1
    lr_results[f"tau_{tstr}_specificity"] = spec
    print(f"[2.5] tau={tau}: tp={tp}, fp={fp}, tn={tn}, fn={fn}, f1={f1:.4f}")

# Blind accuracy: predict 0 for every record
blind_preds = np.zeros(len(y_test), dtype=int)
blind_accuracy = float(np.sum(blind_preds == y_test) / len(y_test))
lr_results["blind_accuracy"] = blind_accuracy

# ---------------------------------------------------------------------------
# Task 2.6 — Threshold sweep → part2_thresholds.csv
# ---------------------------------------------------------------------------
thresholds_rows = []
for k in range(1, 10):
    tau = round(k * 0.1, 1)
    tp, fp, tn, fn, acc, prec, rec, f1, spec = confusion_metrics(final_probs_test_lr, y_test, tau)
    tpr = rec
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    thresholds_rows.append([f"{tau:.1f}", tp, fp, tn, fn, f"{tpr:.4f}", f"{fpr:.4f}"])

with open(os.path.join("outputs", "part2_thresholds.csv"), "w", newline="") as f:
    w_csv = csv.writer(f)
    w_csv.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
    w_csv.writerows(thresholds_rows)

# ---------------------------------------------------------------------------
# Task 2.7 — K-nearest neighbours with 5-fold CV
# ---------------------------------------------------------------------------
def knn_predict_proba(X_ref, y_ref, X_query, K):
    """Return positive-class probability for each query row."""
    n_query = X_query.shape[0]
    probs = np.zeros(n_query)
    for i in range(n_query):
        diffs = X_ref - X_query[i]
        dists = np.sum(diffs ** 2, axis=1)
        sorted_idx = np.argsort(dists, kind="stable")
        neighbours = y_ref[sorted_idx[:K]]
        probs[i] = float(np.sum(neighbours)) / K
    return probs

def knn_preds(proba, K):
    threshold = (K + 1) / 2
    return (np.round(proba * K) >= threshold).astype(int)

# 5-fold CV on training set
n_tr = train_rows
fold_ids = np.arange(n_tr) % 5  # fold 0,1,2,3,4 by position

K_values = list(range(1, 22, 2))  # 1,3,5,...,21
cv_f1 = {K: [] for K in K_values}

for fold in range(5):
    ref_mask = fold_ids != fold
    val_mask  = fold_ids == fold
    X_ref_fold = X_train_s[ref_mask]
    y_ref_fold = y_train[ref_mask]
    X_val_fold = X_train_s[val_mask]
    y_val_fold = y_train[val_mask]

    for K in K_values:
        proba_val = knn_predict_proba(X_ref_fold, y_ref_fold, X_val_fold, K)
        preds_val = (proba_val >= 0.5).astype(int)  # majority = >= (K+1)/2 out of K
        # More precisely: positive when at least (K+1)/2 neighbours are positive
        preds_val = (proba_val * K >= (K + 1) / 2).astype(int)
        tp = int(np.sum((preds_val == 1) & (y_val_fold == 1)))
        fp = int(np.sum((preds_val == 1) & (y_val_fold == 0)))
        fn = int(np.sum((preds_val == 0) & (y_val_fold == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_v = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cv_f1[K].append(f1_v)

mean_cv_f1 = {K: float(sum(cv_f1[K]) / 5) for K in K_values}

# Write part2_knn_cv.csv
with open(os.path.join("outputs", "part2_knn_cv.csv"), "w", newline="") as f:
    w_csv = csv.writer(f)
    w_csv.writerow(["K", "mean_cv_f1"])
    for K in K_values:
        w_csv.writerow([K, f"{mean_cv_f1[K]:.4f}"])

# Best K: highest mean F1 (ties: smaller K)
knn_best_k = K_values[0]
best_f1_cv = mean_cv_f1[K_values[0]]
for K in K_values[1:]:
    if mean_cv_f1[K] > best_f1_cv:
        best_f1_cv = mean_cv_f1[K]
        knn_best_k = K

lr_results["knn_best_k"] = knn_best_k
print(f"[2.7] knn_best_k={knn_best_k}, best_cv_f1={best_f1_cv:.4f}")

# Refit with all 560 training rows as reference, evaluate on test
proba_knn_test = knn_predict_proba(X_train_s, y_train, X_test_s, knn_best_k)
preds_knn_test = (proba_knn_test * knn_best_k >= (knn_best_k + 1) / 2).astype(int)

knn_tp = int(np.sum((preds_knn_test == 1) & (y_test == 1)))
knn_fp = int(np.sum((preds_knn_test == 1) & (y_test == 0)))
knn_tn = int(np.sum((preds_knn_test == 0) & (y_test == 0)))
knn_fn = int(np.sum((preds_knn_test == 0) & (y_test == 1)))
knn_accuracy  = (knn_tp + knn_tn) / test_rows
knn_precision = knn_tp / (knn_tp + knn_fp) if (knn_tp + knn_fp) > 0 else 0.0
knn_recall    = knn_tp / (knn_tp + knn_fn) if (knn_tp + knn_fn) > 0 else 0.0
knn_f1        = (2 * knn_precision * knn_recall / (knn_precision + knn_recall)
                 if (knn_precision + knn_recall) > 0 else 0.0)

lr_results["knn_tp"]        = knn_tp
lr_results["knn_fp"]        = knn_fp
lr_results["knn_tn"]        = knn_tn
lr_results["knn_fn"]        = knn_fn
lr_results["knn_accuracy"]  = knn_accuracy
lr_results["knn_precision"] = knn_precision
lr_results["knn_recall"]    = knn_recall
lr_results["knn_f1"]        = knn_f1

print(f"[2.7] knn_f1={knn_f1:.4f}, knn_recall={knn_recall:.4f}")

# ---------------------------------------------------------------------------
# Task 2.8 — Cost-based threshold (training rows)
# ---------------------------------------------------------------------------
# cost = 50 * fp + 400 * fn
best_tau_cost = None
best_train_cost = float("inf")
for k in range(1, 20):
    tau_k = k / 20
    preds_tr = (final_probs_train_lr >= tau_k).astype(int)
    fp_tr = int(np.sum((preds_tr == 1) & (y_train == 0)))
    fn_tr = int(np.sum((preds_tr == 0) & (y_train == 1)))
    cost_tr = 50 * fp_tr + 400 * fn_tr
    if cost_tr < best_train_cost:
        best_train_cost = cost_tr
        best_tau_cost = tau_k

# Round to nearest 0.05 (the problem says cost_tau is a multiple of 0.05)
cost_tau = round(round(best_tau_cost / 0.05) * 0.05, 2)

# Evaluate at cost_tau on training rows
preds_tr_at_tau = (final_probs_train_lr >= cost_tau).astype(int)
fp_tr = int(np.sum((preds_tr_at_tau == 1) & (y_train == 0)))
fn_tr = int(np.sum((preds_tr_at_tau == 0) & (y_train == 1)))
cost_train_at_tau = 50 * fp_tr + 400 * fn_tr

# Test cost at cost_tau
preds_te_at_tau = (final_probs_test_lr >= cost_tau).astype(int)
fp_te = int(np.sum((preds_te_at_tau == 1) & (y_test == 0)))
fn_te = int(np.sum((preds_te_at_tau == 0) & (y_test == 1)))
cost_test_at_tau = 50 * fp_te + 400 * fn_te

# Test cost at 0.5
preds_te_05 = (final_probs_test_lr >= 0.5).astype(int)
fp_te05 = int(np.sum((preds_te_05 == 1) & (y_test == 0)))
fn_te05 = int(np.sum((preds_te_05 == 0) & (y_test == 1)))
cost_test_at_0_5 = 50 * fp_te05 + 400 * fn_te05

# Flag nobody: 0 predicted positive -> all positives missed
cost_test_flag_nobody = 400 * int(np.sum(y_test))

lr_results["cost_tau"]              = cost_tau
lr_results["cost_train_at_tau"]     = cost_train_at_tau
lr_results["cost_test_at_tau"]      = cost_test_at_tau
lr_results["cost_test_at_0.5"]      = cost_test_at_0_5
lr_results["cost_test_flag_nobody"] = cost_test_flag_nobody

print(f"[2.8] cost_tau={cost_tau}, cost_train={cost_train_at_tau}, cost_test={cost_test_at_tau}")

# ---------------------------------------------------------------------------
# Write outputs/part2_results.txt
# ---------------------------------------------------------------------------
int_keys_p2 = {
    "train_rows", "train_positives", "test_rows", "test_positives",
    "tau_0.5_tp", "tau_0.5_fp", "tau_0.5_tn", "tau_0.5_fn",
    "tau_0.3_tp", "tau_0.3_fp", "tau_0.3_tn", "tau_0.3_fn",
    "knn_best_k",
    "knn_tp", "knn_fp", "knn_tn", "knn_fn",
    "cost_train_at_tau", "cost_test_at_tau",
    "cost_test_at_0.5", "cost_test_flag_nobody",
}

lines = []
for key, val in lr_results.items():
    if key in int_keys_p2:
        lines.append(f"{key} = {int(val)}")
    elif key == "cost_tau":
        lines.append(f"{key} = {float(val):.2f}")
    else:
        lines.append(f"{key} = {float(val):.4f}")

with open(os.path.join("outputs", "part2_results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("[Part 2] Done. Wrote outputs/part2_results.txt, part2_thresholds.csv, part2_knn_cv.csv")

# ---------------------------------------------------------------------------
# Save computed objects for Part 4 to import
# ---------------------------------------------------------------------------
import pickle
p2_data = {
    "train_idx": train_idx,
    "test_idx": test_idx,
    "X_train_s": X_train_s,
    "y_train": y_train,
    "X_test_s": X_test_s,
    "y_test": y_test,
    "train_mean": train_mean,
    "train_sd": train_sd,
    "final_w": final_w,
    "final_b": final_b,
    "final_probs_train_lr": final_probs_train_lr,
    "final_probs_test_lr": final_probs_test_lr,
    "knn_best_k": knn_best_k,
    "proba_knn_test": proba_knn_test,
    "cost_tau": cost_tau,
    "train_rows": train_rows,
    "test_rows": test_rows,
}
with open(os.path.join("outputs", "p2_data.pkl"), "wb") as f:
    pickle.dump(p2_data, f)
print("[Part 2] Saved shared data to outputs/p2_data.pkl")
