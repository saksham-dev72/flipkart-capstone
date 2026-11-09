Domain: E-commerce

# Flipkart E-commerce Decision Intelligence — Capstone Project

**Program:** Minor in Artificial Intelligence & Data Science, IIT Mandi
**Domain:** E-commerce · **Total marks:** 100

---

## How to run

```bash
# 1. Clone the repository and install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset (run once from the repo root)
python generate_data.py
# Expected output: 800 10 170 c6f249ba693f1ebeed318d6d5e18de23

# 3. Run all four Parts in order
python part1.py
python part2.py
python part3.py
python part4.py
```

All output files will be written to the `outputs/` directory.

---

## Part 1 findings

### R1.1 — Hypothesis Test Result

The z-statistic is **z_stat = 2.8930**, which exceeds the critical value of **1.96**.
Therefore **H0 is rejected** (`reject_h0 = yes`): the data does not support the planning
assumption that the average order value is INR 800 (8.0 × 100).

A **Type I error** in this context would be concluding that the true average order value
differs from INR 800 when it actually does not — leading Flipkart to change category plans
based on a statistical fluke.

The gap between the observed mean and the assumed value is **8.9382 − 8.0 = 0.9382 units**
(roughly INR 94 per order). While statistically significant at α = 0.05 (n = 800 is large
enough to detect even modest gaps), INR 94 on typical orders in the INR 800–2600 range
is a ~5% difference — small enough that category planners may choose to retain the
assumption rather than revise every budget line for a modest uplift.

### R1.2 — Sample Size

`n_required = 1293` to estimate the mean within ±0.5 units (±INR 50) at 95% confidence.
The current sample of 800 rows falls short of this target. The planning team should note
that the present confidence interval `[8.3026, 9.5738]` is 1.27 units wide — roughly
±INR 63 — which is wider than the ±INR 50 precision they would ideally want. Collecting
at least 500 more records would bring the estimate to the required precision.

### R1.3 — Gradient Descent vs. Closed Form

`gd_slope_gap = 0.0000` (less than 1e-4). After 500 updates with learning rate η = 0.1,
gradient descent converged to the same slope as the closed-form formula because the cost
surface for simple linear regression is a convex paraboloid — there is exactly one global
minimum and no local minima. With a suitable learning rate, gradient descent always finds
it given enough steps.

### R1.4 — Funnel (Heteroscedasticity)

`resid_sd_ratio_raw = 3.7525`: the upper-half residual spread is 3.75× the lower-half
spread, confirming a strong funnel pattern — residual variance grows with fitted value.
The raw OLS model is unreliable at high order values.

`resid_sd_ratio_log = 1.0396`: after regressing ln(order value) on ln(avg order value),
the ratio drops to 1.04 — essentially flat. The log transformation successfully removes
the funnel and stabilises residual variance across the fitted range.

### R1.5 — Multiple Regression Predictors and Model Comparison

Predictors whose |coefficient| > 0.05 (excluding the constant `mr_beta_0`):
- **mr_beta_1 = 1.1576** (ln X4 — avg order value in hundreds)
- **mr_beta_3 = 0.0940** (X2 — complaint count in 90 days)
- **mr_beta_4 = 0.1067** (X3 — discount order share)
- **mr_beta_6 = 0.1518** (X6 — COD flag)

`mr_adj_r2 = 0.7403` vs. `loglog_adj_r2 = 0.6561`. The multiple regression model
(seven predictors) achieves an adjusted R² that is 0.084 higher than the simple
log-log model, suggesting the extra six predictors do earn their place in explaining
variation in log(order value).

---

## Part 2 findings

### R2.1 — Learning-Rate Effect

Ordering the three losses after 200 updates (lowest to highest):
- η = 1.0 → **0.3703** (lowest)
- η = 0.1 → **0.3776**
- η = 0.01 → **0.5091** (highest)

A learning rate that is **too small** (0.01) moves only tiny steps per update, so after
200 steps the model is still far from its optimum. A learning rate that is **too large**
can overshoot, but here η = 1.0 converges faster without diverging — suggesting the loss
surface is smooth enough in this case.

### R2.2 — Top Predictors and Odds Ratios

The three predictors with the largest absolute weight:
1. **X3 (discount_order_share)**: weight = 0.9013, odds_ratio = 2.4629
2. **X1 (customer_tenure_months)**: weight = −0.5554, odds_ratio = 0.5739
3. **X2 (complaint_count_90d)**: weight = 0.5505, odds_ratio = 1.7341

Largest odds ratio interpretation: for a customer whose discount order share rises by one
training SD, the odds of churning multiply by **2.46** (all other features held equal).
This means customers who habitually buy only on discount are roughly 2.5× more likely to
leave once discounts dry up — a key insight for Flipkart's retention team.

### R2.3 — Why Accuracy Alone is Weak

`blind_accuracy = 0.7875` (predicting "no churn" for every customer). The model at
τ = 0.5 achieves `accuracy = 0.8042` — only 1.7 percentage points above the do-nothing
baseline. With 79% of customers not churning, raw accuracy is an uninformative score
because a trivial classifier scores nearly as high.

Lowering τ from 0.5 to 0.3:
- **Recall** rose from **0.4706 → 0.7451** (+27.5 pp): the model now flags far more
  true churners.
- **Precision** fell from **0.5455 → 0.4634** (−8.2 pp): more false alarms are sent.
  This trade-off is worthwhile in a churn context where missing a churner is expensive.

### R2.4 — Logistic vs. KNN

| Model | Recall | F1 |
|---|---|---|
| Logistic (τ = 0.3) | **0.7451** | **0.5714** |
| KNN (K = 5) | 0.3333 | 0.3696 |

The logistic model at τ = 0.3 dominates KNN on both recall and F1. KNN's performance
was limited here by the class imbalance (21% churn rate) and the relatively small
training set.

### R2.5 — Recommended Model and Threshold

**Recommended: Logistic regression at τ = 0.15 (the cost-optimal threshold).**

| Metric | Value |
|---|---|
| `cost_test_at_tau` (τ = 0.15) | INR 6050 |
| `cost_test_at_0.5` | INR 11800 |
| `cost_test_flag_nobody` | INR 20400 |

The cost-optimal threshold τ = 0.15 cuts test cost from INR 11800 (at τ = 0.5) to
INR 6050 — a 49% reduction — and far below the INR 20400 cost of flagging no one.
The threshold was chosen on **training rows** rather than test rows because the test set
must remain a held-out, unbiased estimate; using it for threshold selection would
constitute a form of data leakage, inflating the apparent performance.

---

## Part 3 findings

### R3.1 — WCSS Drops and Elbow

| Drop | Value |
|---|---|
| WCSS(2) − WCSS(3) | 659.83 |
| WCSS(3) − WCSS(4) | 108.07 |
| WCSS(4) − WCSS(5) | 171.06 |
| WCSS(5) − WCSS(6) | 45.55 |

The largest drop is at K = 2 → 3. The elbow rule (smallest K in {3,4,5} where the next
drop is less than 50% of the current drop) yields `elbow_k = 3`. Note that the drops
do not decrease monotonically (K=4→5 is larger than K=3→4): the pinned starting rows
give a reproducible solution but not necessarily the global optimum for each K.

### R3.2 — Segment Profiles and Business Actions

| Segment | Size | Avg Tenure (X1) | Avg Complaints (X2) | Avg Discount Share (X3) | Churn Rate | Avg Order Value |
|---|---|---|---|---|---|---|
| **A — Loyal Veterans** | 322 | 47.5 mo | 1.95 | 22.5% | **1.9%** | INR 360 |
| **B — High-Value Actives** | 255 | 28.6 mo | 6.07 | 45.3% | 23.5% | INR 1712 |
| **C — Discount Dependents** | 223 | 14.4 mo | 4.23 | 71.2% | **46.6%** | INR 729 |

- **Segment A (Loyal Veterans):** Long-tenured, low-complaint, low discount share, almost
  zero churn. Action: reward with loyalty benefits to maintain engagement.
- **Segment B (High-Value Actives):** Mid-tenure, high complaint count, mid discount share,
  highest absolute order value. Action: prioritise complaint resolution and premium support
  to protect revenue.
- **Segment C (Discount Dependents):** New, moderate complaints, highest discount share,
  **highest churn rate (46.6%)**. This is the segment most at risk. Action: identify these
  customers early and implement progressive discount reduction with engagement incentives
  to wean them off price-only shopping.

### R3.3 — Hierarchical Clustering Cut

**Single linkage (4 clusters at cut):** {E1, E4, E5}, {E2, E3, E6}, with two singletons
still separate. Single linkage is sensitive to chaining — small pairwise distances cause
long, thin chains, so a cluster can grow large before the inter-cluster gap widens enough
to cut.

**Complete linkage (2 clusters at cut):** {E2, E3, E6} and {E1, E4, E5}. Complete linkage
uses the maximum pairwise distance between clusters, which grows quickly as clusters
merge, creating compact, well-separated groups and producing a wider final gap — leading
to fewer, rounder clusters at the same cut level.

The two rules disagree because the pairwise distances in this table have both a tight
cluster (E2, E3, E6 with distances 1.3–2.0) and a looser cluster (E1, E4, E5 with
distances 2.8–2.9). Single linkage chains them into fewer cuts, while complete linkage
creates a larger final height gap that separates into two distinct groups.

### R3.4 — PCA Findings

**Drivers of PC1** (largest absolute loadings):
- X1 (tenure) loads **+0.5317** — longer tenure pushes PC1 positive
- X3 (discount share) loads **−0.5187** — high discounters push PC1 negative
- X2 (complaints) loads **−0.4100** — more complaints → more negative PC1
- X5 (loyalty score) loads **+0.4418** — higher loyalty → more positive PC1

PC1 represents a **"loyalty axis"**: high PC1 = loyal, long-tenured, low-discount customer;
low PC1 = new, discount-heavy, complaint-prone customer.

**Components for ≥80% variance:** `components_for_80pct = 3`
(PC1 = 52.3%, PC1+PC2 = 75.9%, PC1+PC2+PC3 = 86.2%)

**Segment separation along PC1:**
- Segment A (Loyal Veterans): `mean_pc1 = 1.7773` (high — loyal, long-tenured)
- Segment B (High-Value Actives): `mean_pc1 = −0.7831` (mid — complaints pull negative)
- Segment C (Discount Dependents): `mean_pc1 = −1.6709` (low — heavy discounters, new)

The segments are well-separated along PC1, confirming the K-means grouping captures a
real loyalty gradient in the data.

---

## Part 4 findings

### R4.1 — Best Split and Overfitting

**Best split: S4 (customer_tenure_months ≤ 24)** by both information gain and Gini gain.
Tenure ≤ 24 months separates a high-churn group (45.7% positive rate on the "yes" side)
from a low-churn group (8.9% positive rate on the "no" side), achieving the largest
reduction in impurity among the four candidates.

A tree grown until every leaf is pure would perfectly memorise the training data — it
would have one leaf per record — but would fail to generalise because it would fit noise
rather than signal. One rule that prevents over-growing is **minimum samples per leaf**:
refuse to split a node whose child would have fewer than a threshold number of records.

### R4.2 — Best Model for Production

| Model | Recall | F1 |
|---|---|---|
| logistic (τ = 0.3) | **0.7451** | **0.5714** |
| soft_vote | 0.7451 | 0.5507 |
| hard_vote | 0.6863 | 0.5344 |
| stump | 0.7647 | 0.5532 |
| knn | 0.3333 | 0.3696 |

**Recommended model: Logistic regression at τ = 0.3** (or τ = 0.15 for cost optimality).
Combining models (hard vote and soft vote) did not outperform the standalone logistic
model on F1 or recall. The KNN model was the weakest contributor. Logistic regression
achieves the highest F1 (0.5714) and ties for the best recall (0.7451) while remaining
simple, interpretable, and fast.

### R4.3 — AdaBoost Round 1

After round 1, points 3 and 7 (the two misclassified by h1) each received weight 0.2500,
for a combined weight of **0.5000** (50% of the total). Round 2 uses h2 with these
upweighted misclassified points, and achieves `round2_error = 0.1667` while h1 had
`round1_error = 0.2500` — h2 is a better rule on the reweighted distribution, so
AdaBoost assigns it a **larger alpha** (0.8047 vs. 0.5493), trusting h2 more.

### R4.4 — Architecture Details

**Model A Option 2 (CNN) — layer shapes and parameters:**

| Layer | Output shape | Parameters |
|---|---|---|
| cnn_conv1 (5×5×1, 6 filters, pad 2) | 28 × 28 × 6 | 156 |
| MaxPool 2×2 | 14 × 14 × 6 | 0 |
| cnn_conv2 (5×5×6, 16 filters, pad 0) | 10 × 10 × 16 | 2416 |
| MaxPool 2×2 | 5 × 5 × 16 = 400 | 0 |
| Flatten | 400 | 0 |
| cnn_dense1 (400 → 32) | 32 | 12832 |
| cnn_dense2 (32 → 10) | 10 | 330 |
| **cnn_total** | | **15734** |

**Model B (RNN) — weight shapes:**
- W_xh: (64, 128) — input-to-hidden
- W_hh: (128, 128) — hidden-to-hidden (recurrent)
- W_hy: (128, 16) — hidden-to-output
- `rnn_total = 26768`

The parameter count does not grow when sequences get longer because the **same weight
matrices** (W_xh, W_hh, W_hy) are reused at every time step — the model has no
sequence-length-dependent parameters.

**Activation and loss choices for 10 classes:** hidden layers use **ReLU** (fast,
avoids vanishing gradient in deep networks); output layer uses **softmax** with
**cross-entropy loss** to produce a valid probability distribution over 10 classes.

A very deep plain network can stop learning in early layers because gradients are
multiplied across many layers during backpropagation, and each multiplication with
small values shrinks them exponentially — the **vanishing gradient problem**. A **skip
connection** (residual connection) adds the layer's input directly to its output,
providing a gradient highway that bypasses the multiplications and allows gradients to
flow directly to early layers without shrinking.

---

## Model-choice memo

**To:** Head of Customer Retention Operations, Flipkart
**From:** Analytics Team
**Subject:** Recommended churn-prediction model for production

After evaluating five models on 240 held-out customer records, we recommend deploying
**logistic regression at the cost-optimal threshold τ = 0.15**.

This model achieves a test F1 score of 0.5714 and recall of 0.7451 — meaning it correctly
flags approximately 74% of customers who will churn. At τ = 0.15, the total test cost is
**INR 6,050**, compared to INR 11,800 at the standard τ = 0.5 threshold and INR 20,400
if no model is used. The 49% cost reduction over the default threshold demonstrates
concrete business value.

Three supporting numbers: (1) the discount-order-share predictor has an odds ratio of
**2.46** — the single strongest churn signal; (2) the KNN alternative (K=5) achieved
only F1 = 0.3696; (3) ensemble models (soft vote F1 = 0.5507, hard vote F1 = 0.5344)
did not surpass the logistic baseline.

The threshold τ = 0.15 was chosen by minimising cost on training rows only. If used on
the test set for selection, the apparent cost would be optimistically biased.

**Key limitation:** The data is synthetic and trained on 560 records. Real churn patterns
will differ, and the model should be retrained on live data before production deployment.

Regarding deep-learning models: **Model B (RNN for resume text, 26,768 parameters)** is
not worth building before the structured model is in production. The structured logistic
model already delivers actionable predictions with far less complexity and data requirement.

---

## Limitations

1. **Synthetic data:** All 800 records were generated by a fixed random-number script.
   The relationships between features and churn are deterministic artefacts of the
   generator, not real Flipkart customer behaviour. Conclusions cannot be generalised
   to production without validating on real data.

2. **Single train/test split:** The 560/240 split is fixed and performed only once.
   Performance metrics (F1, cost) depend on which records landed in the test set.
   Cross-validated evaluation across multiple splits would give a more reliable estimate
   of true generalisation.

3. **Small positive class in test set:** Only 51 of 240 test records are churners (21%).
   This means confusion-matrix metrics fluctuate significantly with just a few
   misclassifications — a single record change can shift recall by ~2 percentage points.
   The apparent superiority of one threshold over another should be treated cautiously.

4. **No temporal structure:** A real churn model would use time-ordered data and evaluate
   on future customers, not a random held-out subset. The current split may allow implicit
   data leakage through shared temporal trends.

5. **No feature engineering or interaction terms:** The logistic regression uses raw
   features only. Real-world performance could improve substantially with interaction
   terms (e.g., high discount share × low loyalty score) or non-linear transformations.
