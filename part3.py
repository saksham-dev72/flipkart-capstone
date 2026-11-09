"""
Part 3 — Segmentation and dimensionality reduction
Domain: E-commerce (Flipkart)

Covers Tasks 3.1 to 3.5:
  3.1 K-means from scratch (K = 2..6)
  3.2 Elbow rule → elbow_k (use K=3 throughout)
  3.3 Profile K=3 segments → part3_segments.csv
  3.4 Hierarchical clustering (single/complete/average) → part3_hierarchical.csv
  3.5 PCA in 7 steps → eigenvalues, loadings, projections
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

X1 = np.array([float(r["customer_tenure_months"]) for r in rows])
X2 = np.array([float(r["complaint_count_90d"]) for r in rows])
X3 = np.array([float(r["discount_order_share"]) for r in rows])
X4 = np.array([float(r["avg_order_value_inr_hundreds"]) for r in rows])
X5 = np.array([float(r["loyalty_score"]) for r in rows])
outcome = np.array([int(r["churned"]) for r in rows])
value_col = np.array([float(r["order_value_inr_hundreds"]) for r in rows])

# Continuous predictors X1..X5 (800 x 5)
Xcont = np.column_stack([X1, X2, X3, X4, X5])

# Standardise over all 800 rows
Xcont_mean = np.sum(Xcont, axis=0) / N
Xcont_sd   = np.sqrt(np.sum((Xcont - Xcont_mean) ** 2, axis=0) / (N - 1))
Xcont_s    = (Xcont - Xcont_mean) / Xcont_sd  # (800, 5)

results = {}

# ---------------------------------------------------------------------------
# Task 3.1 — K-means from scratch
# ---------------------------------------------------------------------------
def kmeans(X, K, seed_indices):
    """Run K-means with pinned starting centroids (seed_indices)."""
    centroids = X[seed_indices].copy()
    n = X.shape[0]
    labels = np.full(n, -1)
    while True:
        # Assignment step
        new_labels = np.zeros(n, dtype=int)
        for i in range(n):
            dists = np.sum((centroids - X[i]) ** 2, axis=1)
            new_labels[i] = int(np.argmin(dists))  # ties: lower cluster wins (argmin returns first)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update step
        for k in range(K):
            members = X[labels == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)
    return labels, centroids

wcss_values = {}
for K in range(2, 7):
    stride = 800 // K
    seed_idx = [i * stride for i in range(K)]
    labels, centroids = kmeans(Xcont_s, K, seed_idx)
    wcss = 0.0
    for i in range(len(labels)):
        diff = Xcont_s[i] - centroids[labels[i]]
        wcss += float(np.sum(diff ** 2))
    wcss_values[K] = wcss
    results[f"wcss_k{K}"] = wcss
    print(f"[3.1] K={K}: wcss={wcss:.4f}")

print(f"      Checkpoint wcss_k2 = 2262.6753 -> got {wcss_values[2]:.4f}")

# Store K=3 labels for later tasks
stride3 = 800 // 3
seed_idx3 = [0, stride3, 2 * stride3]
labels_k3, centroids_k3 = kmeans(Xcont_s, 3, seed_idx3)

# ---------------------------------------------------------------------------
# Task 3.2 — Elbow rule
# ---------------------------------------------------------------------------
elbow_k = None
for K in [3, 4, 5]:
    drop_before = wcss_values[K - 1] - wcss_values[K]
    drop_after  = wcss_values[K] - wcss_values[K + 1]
    if drop_after < 0.5 * drop_before:
        elbow_k = K
        break
if elbow_k is None:
    elbow_k = 5  # fallback

results["elbow_k"] = elbow_k
print(f"[3.2] elbow_k={elbow_k}")

# Always use K=3 for the rest
# ---------------------------------------------------------------------------
# Task 3.3 — Profile K=3 segments (order by ascending mean X3 in original units)
# ---------------------------------------------------------------------------
# Compute mean X3 (original) per cluster
cluster_mean_X3_orig = {}
for k in range(3):
    idx_k = np.where(labels_k3 == k)[0]
    cluster_mean_X3_orig[k] = float(np.mean(X3[idx_k]))

# Sort clusters by mean X3 ascending → A, B, C
sorted_clusters = sorted(range(3), key=lambda k: cluster_mean_X3_orig[k])
seg_map = {sorted_clusters[0]: "A", sorted_clusters[1]: "B", sorted_clusters[2]: "C"}

seg_labels = np.array([seg_map[k] for k in labels_k3])

segments_csv_rows = []
for seg in ["A", "B", "C"]:
    idx_s = np.where(seg_labels == seg)[0]
    size_s = len(idx_s)
    mX1 = float(np.mean(X1[idx_s]))
    mX2 = float(np.mean(X2[idx_s]))
    mX3 = float(np.mean(X3[idx_s]))
    mX4 = float(np.mean(X4[idx_s]))
    mX5 = float(np.mean(X5[idx_s]))
    out_rate = float(np.mean(outcome[idx_s]))
    mean_val = float(np.mean(value_col[idx_s]))
    results[f"size_segment_{seg}"] = size_s
    segments_csv_rows.append([seg, size_s,
                               f"{mX1:.4f}", f"{mX2:.4f}", f"{mX3:.4f}",
                               f"{mX4:.4f}", f"{mX5:.4f}",
                               f"{out_rate:.4f}", f"{mean_val:.4f}"])

with open(os.path.join("outputs", "part3_segments.csv"), "w", newline="") as f:
    w_csv = csv.writer(f)
    w_csv.writerow(["segment", "size", "mean_X1", "mean_X2", "mean_X3",
                    "mean_X4", "mean_X5", "outcome_rate", "mean_value"])
    w_csv.writerows(segments_csv_rows)

print(f"[3.3] Segment sizes: A={results['size_segment_A']}, B={results['size_segment_B']}, C={results['size_segment_C']}")

# ---------------------------------------------------------------------------
# Task 3.4 — Hierarchical clustering (6 cohorts from fixed table)
# ---------------------------------------------------------------------------
# Pairwise dissimilarity table (symmetric, zero diagonal)
labels_hc = ["E1", "E2", "E3", "E4", "E5", "E6"]
D_raw = [
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
]

def agglomerative(D_raw, labels_hc, linkage):
    """Run agglomerative clustering with given linkage. Returns list of merge records."""
    # clusters: list of sets of original indices
    clusters = [{i} for i in range(len(labels_hc))]
    D = [row[:] for row in D_raw]  # deep copy

    merges = []
    for merge_no in range(1, 6):
        # Find pair with smallest linkage distance
        best_dist = float("inf")
        best_i, best_j = -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compute linkage distance
                dists = []
                for a in clusters[i]:
                    for b in clusters[j]:
                        dists.append(D_raw[a][b])
                if linkage == "single":
                    d = min(dists)
                elif linkage == "complete":
                    d = max(dists)
                else:  # average
                    d = sum(dists) / len(dists)
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        # Merge clusters[best_j] into clusters[best_i]
        ci = clusters[best_i]
        cj = clusters[best_j]
        # Cluster name: member names sorted ascending joined by +
        ci_name = "+".join(sorted([labels_hc[k] for k in ci]))
        cj_name = "+".join(sorted([labels_hc[k] for k in cj]))
        # cluster_a has smaller smallest-member number
        min_ci = min(ci)
        min_cj = min(cj)
        if min_ci <= min_cj:
            cluster_a, cluster_b = ci_name, cj_name
        else:
            cluster_a, cluster_b = cj_name, ci_name

        merges.append((merge_no, cluster_a, cluster_b, best_dist))
        # Update clusters
        clusters[best_i] = ci | cj
        clusters.pop(best_j)

    return merges

hc_rows = []
hc_cut_results = {}
for linkage in ["single", "complete", "average"]:
    merges = agglomerative(D_raw, labels_hc, linkage)
    for merge_no, ca, cb, height in merges:
        hc_rows.append([linkage, merge_no, ca, cb, f"{height:.4f}"])

    # Cut: find largest gap between consecutive merge heights
    heights = [m[3] for m in merges]
    gaps = [heights[i+1] - heights[i] for i in range(4)]
    max_gap_idx = gaps.index(max(gaps))  # index of largest gap (0-based among 4 gaps)
    # number of merges below the cut = max_gap_idx + 1
    n_merges_below = max_gap_idx + 1
    n_clusters_at_cut = 6 - n_merges_below
    hc_cut_results[f"{linkage}_clusters_at_cut"] = n_clusters_at_cut
    results[f"{linkage}_clusters_at_cut"] = n_clusters_at_cut
    print(f"[3.4] {linkage}: {n_clusters_at_cut} clusters at cut, heights={[round(h,2) for h in heights]}")

with open(os.path.join("outputs", "part3_hierarchical.csv"), "w", newline="") as f:
    w_csv = csv.writer(f)
    w_csv.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
    w_csv.writerows(hc_rows)

# ---------------------------------------------------------------------------
# Task 3.5 — PCA in 7 steps
# ---------------------------------------------------------------------------
# Data: Xcont_s (800 x 5), already standardised → mean is 0 by construction

# Step 1: mean vector (all zeros after standardising)
# Step 2: data already centred (Xcont_s has mean 0)
# Step 3: covariance matrix S = (1/(n-1)) * X'X
S = (Xcont_s.T @ Xcont_s) / (N - 1)

# Step 4: eigenvalues/vectors (numpy.linalg.eigh — returns sorted ascending)
eigenvalues_raw, eigenvectors_raw = np.linalg.eigh(S)

# Step 5: sort descending
sort_order = np.argsort(eigenvalues_raw)[::-1]
eigenvalues = eigenvalues_raw[sort_order]
eigenvectors = eigenvectors_raw[:, sort_order]  # (5, 5) columns are eigenvectors

# Step 6: keep top 2 eigenvectors
# Sign convention: flip so entry of largest abs value is positive
for j in range(5):
    max_abs_idx = int(np.argmax(np.abs(eigenvectors[:, j])))
    if eigenvectors[max_abs_idx, j] < 0:
        eigenvectors[:, j] *= -1

W = eigenvectors[:, :2]  # (5, 2) top 2

# Step 7: project Z = Xcont_s @ W
Z = Xcont_s @ W  # (800, 2)

eigenvalue_sum = float(np.sum(eigenvalues))
results["eigenvalue_sum"] = eigenvalue_sum
print(f"[3.5] eigenvalue_sum={eigenvalue_sum:.4f}  (Checkpoint: 5.0000)")

for j in range(5):
    results[f"eigenvalue_{j+1}"] = float(eigenvalues[j])
    results[f"explained_ratio_{j+1}"] = float(eigenvalues[j] / eigenvalue_sum)

cum = 0.0
for j in range(5):
    cum += eigenvalues[j] / eigenvalue_sum
    results[f"cumulative_ratio_{j+1}"] = float(cum)

# Components for 80%
components_for_80pct = None
cum2 = 0.0
for j in range(5):
    cum2 += eigenvalues[j] / eigenvalue_sum
    if cum2 >= 0.8:
        components_for_80pct = j + 1
        break
results["components_for_80pct"] = components_for_80pct

# PC loadings
for j in range(5):
    results[f"pc1_loading_X{j+1}"] = float(eigenvectors[j, 0])
    results[f"pc2_loading_X{j+1}"] = float(eigenvectors[j, 1])

# Projections of first 3 data rows
for row_i in range(3):
    results[f"row{row_i+1}_pc1"] = float(Z[row_i, 0])
    results[f"row{row_i+1}_pc2"] = float(Z[row_i, 1])

# Mean PC1 per segment
for seg in ["A", "B", "C"]:
    idx_s = np.where(seg_labels == seg)[0]
    results[f"mean_pc1_segment_{seg}"] = float(np.mean(Z[idx_s, 0]))

# ---------------------------------------------------------------------------
# Write outputs/part3_results.txt
# ---------------------------------------------------------------------------
int_keys_p3 = {"elbow_k", "size_segment_A", "size_segment_B", "size_segment_C",
                "single_clusters_at_cut", "complete_clusters_at_cut", "average_clusters_at_cut",
                "components_for_80pct"}

lines = []
for key, val in results.items():
    if key in int_keys_p3:
        lines.append(f"{key} = {int(val)}")
    else:
        lines.append(f"{key} = {float(val):.4f}")

with open(os.path.join("outputs", "part3_results.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("[Part 3] Done. Wrote outputs/part3_results.txt, part3_segments.csv, part3_hierarchical.csv")
