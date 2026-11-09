"""
Part 4 — Model-choice and architecture memo
Domain: E-commerce (Flipkart)

Covers Tasks 4.1 to 4.5:
  4.1 Split purity (entropy, Gini, info gain) → part4_splits.csv
  4.2 Voting models → part4_voting.csv
  4.3 AdaBoost (2 rounds by hand) → part4_adaboost.csv
  4.4 Out-of-bag fraction
  4.5 Neural network parameter counts → part4_architecture.csv
"""

import csv
import math
import os
import pickle

import numpy as np

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Load shared data from Part 2
# ---------------------------------------------------------------------------
with open(os.path.join("outputs", "p2_data.pkl"), "rb") as f:
    p2 = pickle.load(f)

X_train_s = p2["X_train_s"]
y_train    = p2["y_train"]
X_test_s   = p2["X_test_s"]
y_test     = p2["y_test"]
final_w    = p2["final_w"]
final_b    = p2["final_b"]
knn_best_k = p2["knn_best_k"]
proba_knn_test     = p2["proba_knn_test"]
final_probs_train_lr = p2["final_probs_train_lr"]
final_probs_test_lr  = p2["final_probs_test_lr"]
cost_tau   = p2["cost_tau"]
train_rows = p2["train_rows"]
test_rows  = p2["test_rows"]
train_idx  = p2["train_idx"]

# Load original (unstandardised) predictors for the training rows
rows = []
with open(os.path.join("data", "records.csv"), newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

X1_all = np.array([float(r["customer_tenure_months"]) for r in rows])
X3_all = np.array([float(r["discount_order_share"]) for r in rows])
X6_all = np.array([float(r["cod_flag"]) for r in rows])
X7_all = np.array([float(r["plus_member_flag"]) for r in rows])
outcome_all = np.array([int(r["churned"]) for r in rows])

# Training rows (original units, for split decisions)
X1_tr = X1_all[train_idx]
X3_tr = X3_all[train_idx]
X6_tr = X6_all[train_idx]
X7_tr = X7_all[train_idx]
y_tr  = y_train

results = {}

# ---------------------------------------------------------------------------
# Task 4.1 — Which split is purest?
# ---------------------------------------------------------------------------
def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    p = float(np.sum(labels)) / n
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

def gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    p = float(np.sum(labels)) / n
    return 1.0 - (p ** 2 + (1 - p) ** 2)

n_tr = train_rows
root_entropy = entropy(y_tr)
root_gini    = gini(y_tr)

results["root_entropy"] = root_entropy
results["root_gini"]    = root_gini
print(f"[4.1] root_entropy={root_entropy:.4f}  (Checkpoint: 0.7462)")

# Four candidate splits (yes side = condition holds)
splits = {
    "S1": X6_tr == 1,   # cod_flag = 1
    "S2": X7_tr == 1,   # plus_member_flag = 1
    "S3": X3_tr > 0.5,  # discount_order_share > 0.5
    "S4": X1_tr <= 24,  # customer_tenure_months <= 24
}

splits_csv_rows = []
best_ig_split   = None
best_ig_val     = -1.0
best_gg_split   = None
best_gg_val     = -1.0

for sname, mask in splits.items():
    yes_labels = y_tr[mask]
    no_labels  = y_tr[~mask]
    n_yes = len(yes_labels)
    n_no  = len(no_labels)
    pos_yes = int(np.sum(yes_labels))
    pos_no  = int(np.sum(no_labels))

    # Weighted child entropy / gini
    H_yes = entropy(yes_labels)
    H_no  = entropy(no_labels)
    info_gain = root_entropy - (n_yes / n_tr * H_yes + n_no / n_tr * H_no)

    G_yes = gini(yes_labels)
    G_no  = gini(no_labels)
    gini_gain = root_gini - (n_yes / n_tr * G_yes + n_no / n_tr * G_no)

    splits_csv_rows.append([sname, n_yes, pos_yes, n_no, pos_no,
                             f"{info_gain:.4f}", f"{gini_gain:.4f}"])

    if info_gain > best_ig_val:
        best_ig_val   = info_gain
        best_ig_split = sname
    if gini_gain > best_gg_val:
        best_gg_val   = gini_gain
        best_gg_split = sname

with open(os.path.join("outputs", "part4_splits.csv"), "w", newline="") as f:
    w_csv = csv.writer(f)
    w_csv.writerow(["split", "n_yes", "positives_yes", "n_no", "positives_no",
                    "info_gain", "gini_gain"])
    w_csv.writerows(splits_csv_rows)

results["best_split_by_info_gain"] = best_ig_split
results["best_split_by_gini_gain"] = best_gg_split
print(f"[4.1] best_split_by_info_gain={best_ig_split}, best_split_by_gini_gain={best_gg_split}")

# ---------------------------------------------------------------------------
# Task 4.2 — Stump and voting models
# ---------------------------------------------------------------------------
# Stump: best split by info gain; predicted probability = share of positives on that side (training)
best_split_mask_tr = splits[best_ig_split]
stump_rate_yes = float(np.mean(y_tr[best_split_mask_tr]))
stump_rate_no  = float(np.mean(y_tr[~best_split_mask_tr]))
results["stump_rate_yes"] = stump_rate_yes
results["stump_rate_no"]  = stump_rate_no

# Stump predictions on test rows (need test masks in original units)
X1_te = X1_all[p2["test_idx"]]
X3_te = X3_all[p2["test_idx"]]
X6_te = X6_all[p2["test_idx"]]
X7_te = X7_all[p2["test_idx"]]

test_split_masks = {
    "S1": X6_te == 1,
    "S2": X7_te == 1,
    "S3": X3_te > 0.5,
    "S4": X1_te <= 24,
}
stump_mask_te = test_split_masks[best_ig_split]
proba_stump_test = np.where(stump_mask_te, stump_rate_yes, stump_rate_no)

# Five prediction vectors at tau=0.3
tau_vote = 0.3
preds_logistic = (final_probs_test_lr >= tau_vote).astype(int)
preds_knn      = (proba_knn_test * knn_best_k >= (knn_best_k + 1) / 2).astype(int)
preds_stump    = (proba_stump_test >= tau_vote).astype(int)

# Hard vote: majority of 3 (at least 2 of 3)
preds_hard = ((preds_logistic + preds_knn + preds_stump) >= 2).astype(int)

# Soft vote: mean probability >= 0.3
mean_proba = (final_probs_test_lr + proba_knn_test + proba_stump_test) / 3.0
preds_soft = (mean_proba >= tau_vote).astype(int)

voting_csv_rows = []
for mname, preds in [("logistic", preds_logistic), ("knn", preds_knn),
                     ("stump", preds_stump), ("hard_vote", preds_hard),
                     ("soft_vote", preds_soft)]:
    tp = int(np.sum((preds == 1) & (y_test == 1)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    tn = int(np.sum((preds == 0) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    voting_csv_rows.append([mname, tp, fp, tn, fn,
                             f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])
    print(f"[4.2] {mname}: tp={tp}, fp={fp}, tn={tn}, fn={fn}, f1={f1:.4f}")

with open(os.path.join("outputs", "part4_voting.csv"), "w", newline="") as f:
    w_csv = csv.writer(f)
    w_csv.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])
    w_csv.writerows(voting_csv_rows)

# ---------------------------------------------------------------------------
# Task 4.3 — AdaBoost (2 rounds by hand)
# ---------------------------------------------------------------------------
# Data table (1-indexed, convert to 0-indexed arrays)
y_ab  = np.array([+1, +1, +1, -1, -1, -1, +1, -1], dtype=float)
h1_ab = np.array([+1, +1, -1, -1, -1, -1, -1, -1], dtype=float)
h2_ab = np.array([-1, +1, +1, -1, +1, -1, +1, -1], dtype=float)

n_ab = 8
w_ab = np.full(n_ab, 1.0 / n_ab)  # equal weights 1/8

adaboost_rows = []

# Round 1: use h1
round1_error = float(np.sum(w_ab[h1_ab != y_ab]))
round1_alpha = 0.5 * math.log((1 - round1_error) / round1_error)
print(f"[4.3] round1_error={round1_error:.4f}  (Checkpoint: 0.2500)")
print(f"      round1_alpha={round1_alpha:.4f}")

w_unnorm_r1 = w_ab * np.exp(-round1_alpha * y_ab * h1_ab)
w_norm_r1   = w_unnorm_r1 / float(np.sum(w_unnorm_r1))

for i in range(n_ab):
    adaboost_rows.append([1, i+1,
                          f"{w_ab[i]:.4f}",
                          f"{w_unnorm_r1[i]:.4f}",
                          f"{w_norm_r1[i]:.4f}"])

# Round 2: use h2 with w_norm_r1
round2_error = float(np.sum(w_norm_r1[h2_ab != y_ab]))
round2_alpha = 0.5 * math.log((1 - round2_error) / round2_error)
print(f"[4.3] round2_error={round2_error:.4f}, round2_alpha={round2_alpha:.4f}")

w_unnorm_r2 = w_norm_r1 * np.exp(-round2_alpha * y_ab * h2_ab)
w_norm_r2   = w_unnorm_r2 / float(np.sum(w_unnorm_r2))

for i in range(n_ab):
    adaboost_rows.append([2, i+1,
                          f"{w_norm_r1[i]:.4f}",
                          f"{w_unnorm_r2[i]:.4f}",
                          f"{w_norm_r2[i]:.4f}"])

# Ensemble: sign(alpha1*h1 + alpha2*h2)
ensemble_scores = round1_alpha * h1_ab + round2_alpha * h2_ab
ensemble_preds  = np.sign(ensemble_scores)
ensemble_accuracy = float(np.mean(ensemble_preds == y_ab))

results["round1_error"]       = round1_error
results["round1_alpha"]       = round1_alpha
results["round2_error"]       = round2_error
results["round2_alpha"]       = round2_alpha
results["ensemble_accuracy"]  = ensemble_accuracy

with open(os.path.join("outputs", "part4_adaboost.csv"), "w", newline="") as f:
    w_csv = csv.writer(f)
    w_csv.writerow(["round", "point", "weight_before", "weight_unnormalised", "weight_after"])
    w_csv.writerows(adaboost_rows)

print(f"[4.3] ensemble_accuracy={ensemble_accuracy:.4f}")

# ---------------------------------------------------------------------------
# Task 4.4 — Out-of-bag fraction
# ---------------------------------------------------------------------------
oob_fraction = (1.0 - 1.0 / 560) ** 560
results["oob_fraction"] = oob_fraction
print(f"[4.4] oob_fraction={oob_fraction:.4f}")

# ---------------------------------------------------------------------------
# Task 4.5 — Neural network parameter counts
# ---------------------------------------------------------------------------
# Model A — image input (28x28x1, 10 classes)
# Option 1: MLP 784→16→16→10
mlp_hidden1 = 784 * 16 + 16   # weights + biases
mlp_hidden2 = 16  * 16 + 16
mlp_output  = 16  * 10 + 10
mlp_total   = mlp_hidden1 + mlp_hidden2 + mlp_output

# Option 2: CNN
# Conv1: 5x5x1 filters x6, stride 1, padding 2 → output (28, 28, 6)
# params: (5*5*1 + 1) * 6
cnn_conv1_params = (5 * 5 * 1 + 1) * 6
# MaxPool → (14, 14, 6)
# Conv2: 5x5x6 filters x16, stride 1, padding 0 → output = (14-5+0)/1+1 = 10, so (10,10,16)
cnn_conv2_params = (5 * 5 * 6 + 1) * 16
# MaxPool → (5, 5, 16) = flatten 400
# Dense1: 400→32
cnn_dense1_params = 400 * 32 + 32
# Dense2 (output): 32→10
cnn_dense2_params = 32 * 10 + 10
cnn_total = cnn_conv1_params + cnn_conv2_params + cnn_dense1_params + cnn_dense2_params

arch_rows = [
    ["mlp_hidden1", mlp_hidden1],
    ["mlp_hidden2", mlp_hidden2],
    ["mlp_output",  mlp_output],
    ["cnn_conv1",   cnn_conv1_params],
    ["cnn_conv2",   cnn_conv2_params],
    ["cnn_dense1",  cnn_dense1_params],
    ["cnn_dense2",  cnn_dense2_params],
]

results["mlp_total"] = mlp_total
results["cnn_total"] = cnn_total

# FC layer vs Conv layer for 128x128x3 input
fc_layer_params   = (128 * 128 * 3) * 1024 + 1024
conv_layer_params = (3 * 3 * 3 + 1) * 64

results["fc_layer_params"]   = fc_layer_params
results["conv_layer_params"] = conv_layer_params

# Model B — RNN for text (64-dim input, 128-dim hidden, 16 classes)
# W_xh: (64, 128)  W_hh: (128, 128)  b_h: (128,)  W_hy: (128, 16)  b_y: (16,)
rnn_total = 64 * 128 + 128 * 128 + 128 + 128 * 16 + 16
results["rnn_total"] = rnn_total

with open(os.path.join("outputs", "part4_architecture.csv"), "w", newline="") as f:
    w_csv = csv.writer(f)
    w_csv.writerow(["layer", "parameters"])
    w_csv.writerows(arch_rows)

print(f"[4.5] mlp_total={mlp_total}, cnn_total={cnn_total}, rnn_total={rnn_total}")

# ---------------------------------------------------------------------------
# Write outputs/part4_results.txt
# ---------------------------------------------------------------------------
int_keys_p4 = {
    "mlp_total", "cnn_total", "fc_layer_params", "conv_layer_params", "rnn_total",
}

lines = []
for key, val in results.items():
    if key in int_keys_p4:
        lines.append(f"{key} = {int(val)}")
    elif key == "best_split_by_info_gain" or key == "best_split_by_gini_gain":
        lines.append(f"{key} = {val}")
    else:
        lines.append(f"{key} = {float(val):.4f}")

with open(os.path.join("outputs", "part4_results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("[Part 4] Done. Wrote outputs/part4_results.txt, part4_splits.csv, part4_voting.csv, part4_adaboost.csv, part4_architecture.csv")
