# PCA Validation Experiments

## Overview

### English

This document explains the PCA validation experiments implemented in
`pca_validation_experiments.py`. The goal is to check whether benign client
updates form a meaningful PCA subspace before using that subspace in
`PCASubspaceAttack`.

The validation contains three experiments:

1. Scree plot: checks whether update variation is concentrated in the first few principal components.
2. Cross-round stability: checks whether the PCA subspaces are stable across communication rounds.
3. Random-subspace baseline: checks whether PCA captures more update energy than a random subspace.

### 中文

本文說明 `pca_validation_experiments.py` 中 PCA validation 實驗的定義、公式與設定。目標是在使用
`PCASubspaceAttack` 前，先驗證 benign client updates 是否真的形成有意義的 PCA 子空間。

validation 包含三個實驗：

1. Scree plot：檢查 update 的變異是否集中在前幾個 principal components。
2. Cross-round stability：檢查 PCA 子空間在不同 communication rounds 之間是否穩定。
3. Random-subspace baseline：檢查 PCA 是否比隨機子空間捕捉更多 update energy。

## Data Matrix

### English

For each sampled communication round, the script collects only benign client
updates. If a round has `b` benign clients and the model has `d` parameters, the
update matrix is:

```text
X in R^{b x d}
```

Each row is one benign client update:

```text
x_i = local_model_i - global_model
```

Before PCA, the matrix is centered:

```text
mu = (1 / b) * sum_i x_i
X_c = X - mu
```

Here, `mu` is the mean benign update for that round.

### 中文

對每一個 sampled communication round，腳本只收集 benign client updates。若該 round 有 `b` 個
benign clients，模型有 `d` 個參數，則 update matrix 為：

```text
X in R^{b x d}
```

每一列是一個 benign client update：

```text
x_i = local_model_i - global_model
```

做 PCA 前會先將矩陣 center：

```text
mu = (1 / b) * sum_i x_i
X_c = X - mu
```

其中 `mu` 是該 round 的 benign update 平均。

## PCA Definition

### English

The script computes SVD on the centered update matrix:

```text
X_c = U S V^T
```

The principal directions are the columns of `V`. The singular values are the
entries of `S`.

Since `X_c` has `b` centered rows, the maximum meaningful rank is:

```text
rank <= min(b - 1, d)
```

For example, if a round has 8 benign clients, at most 7 principal components are
meaningful.

### 中文

腳本會對 centered update matrix 做 SVD：

```text
X_c = U S V^T
```

principal directions 是 `V` 的 columns，singular values 是 `S` 的元素。

因為 `X_c` 有 `b` 個 centered rows，有意義的最大 rank 是：

```text
rank <= min(b - 1, d)
```

例如若一個 round 有 8 個 benign clients，最多只有 7 個有意義的 principal components。

## Update Energy

### English

The total update energy is the squared Frobenius norm of the centered benign
updates:

```text
E_total = ||X_c||_F^2
        = sum_i sum_j X_c[i, j]^2
```

This measures the total variation of benign updates around their mean update.

For a k-dimensional subspace with orthonormal basis `Q in R^{d x k}`, the captured
energy is:

```text
E_Q(k) = ||X_c Q||_F^2
```

The captured energy ratio is:

```text
R_Q(k) = E_Q(k) / E_total
```

PCA chooses the top-k subspace that maximizes this ratio.

### 中文

total update energy 定義為 centered benign updates 的 Frobenius norm 平方：

```text
E_total = ||X_c||_F^2
        = sum_i sum_j X_c[i, j]^2
```

它衡量 benign updates 相對於平均 update 的總變異量。

對於一個 k 維子空間，其 orthonormal basis 為 `Q in R^{d x k}`，captured energy 為：

```text
E_Q(k) = ||X_c Q||_F^2
```

captured energy ratio 為：

```text
R_Q(k) = E_Q(k) / E_total
```

PCA 會選擇能最大化此比例的 top-k 子空間。

## Experiment 1: Scree Plot

### English

For each round, the explained variance ratio of principal component `j` is:

```text
lambda_j = S_j^2
explained_j = lambda_j / sum_l lambda_l
```

The cumulative explained variance for top-k PCs is:

```text
cumulative_k = sum_{j=1}^k explained_j
```

If `cumulative_k` quickly approaches 0.9 or 0.95, benign updates are concentrated
in a low-dimensional PCA subspace. If `cumulative_k` grows slowly, the update
variation is spread across many directions.

Output file:

```text
{config}_scree.csv
```

Columns:

```text
round
k
explained_variance_ratio
cumulative_explained_variance
```

### 中文

對每個 round，第 `j` 個 principal component 的 explained variance ratio 為：

```text
lambda_j = S_j^2
explained_j = lambda_j / sum_l lambda_l
```

top-k PCs 的 cumulative explained variance 為：

```text
cumulative_k = sum_{j=1}^k explained_j
```

若 `cumulative_k` 很快接近 0.9 或 0.95，表示 benign updates 集中在低維 PCA 子空間。若
`cumulative_k` 上升很慢，表示 update variation 分散在很多方向。

輸出檔案：

```text
{config}_scree.csv
```

欄位：

```text
round
k
explained_variance_ratio
cumulative_explained_variance
```

## Experiment 2: Cross-Round Stability

### English

This experiment checks whether the PCA subspace is stable across rounds.

Let `Q_a(k)` and `Q_b(k)` be the top-k PCA bases from two different rounds `a`
and `b`. The script computes:

```text
M = Q_a(k)^T Q_b(k)
```

Then it computes the singular values of `M`:

```text
sigma_1, ..., sigma_k
```

The subspace overlap score is:

```text
overlap(Q_a, Q_b, k) = (1 / k) * sum_i sigma_i^2
```

The score is between 0 and 1. A value of 1 means the two top-k subspaces are
identical. A value of 0 means the two subspaces are orthogonal. Higher values
mean PCA directions are more stable across rounds.

Output file:

```text
{config}_stability.csv
```

Columns:

```text
round_a
round_b
k
subspace_overlap
round_a_cumulative_explained
round_b_cumulative_explained
```

### 中文

此實驗檢查 PCA 子空間在不同 rounds 之間是否穩定。

令 `Q_a(k)` 和 `Q_b(k)` 分別為 round `a` 和 round `b` 的 top-k PCA basis。腳本會計算：

```text
M = Q_a(k)^T Q_b(k)
```

接著計算 `M` 的 singular values：

```text
sigma_1, ..., sigma_k
```

subspace overlap 分數為：

```text
overlap(Q_a, Q_b, k) = (1 / k) * sum_i sigma_i^2
```

此分數介於 0 和 1 之間。`1` 表示兩個 top-k 子空間完全相同，`0` 表示兩個子空間互相正交。數值越高，表示 PCA directions 跨 round 越穩定。

輸出檔案：

```text
{config}_stability.csv
```

欄位：

```text
round_a
round_b
k
subspace_overlap
round_a_cumulative_explained
round_b_cumulative_explained
```

## Experiment 3: Random-Subspace Baseline

### English

This experiment checks whether PCA captures more update energy than a random
k-dimensional subspace.

For each round and each `k`, the script samples random orthonormal bases:

```text
Q_random in R^{d x k}
```

For each random subspace, it computes:

```text
random_ratio_k = ||X_c Q_random||_F^2 / ||X_c||_F^2
```

The script repeats this `random_trials` times and reports the mean and standard
deviation.

If PCA cumulative explained variance is much higher than the random baseline,
PCA finds meaningful structure. If PCA is close to the random baseline, the PCA
subspace may not be informative.

Output file:

```text
{config}_random_baseline.csv
```

Columns:

```text
round
k
pca_cumulative_explained
random_cumulative_explained_mean
random_cumulative_explained_std
random_trials
```

### 中文

此實驗檢查 PCA 是否比隨機 k 維子空間捕捉更多 update energy。

對每個 round 與每個 `k`，腳本會 sample 隨機 orthonormal bases：

```text
Q_random in R^{d x k}
```

對每個 random subspace，計算：

```text
random_ratio_k = ||X_c Q_random||_F^2 / ||X_c||_F^2
```

腳本會重複 `random_trials` 次，並回報 mean 和 standard deviation。

若 PCA cumulative explained variance 明顯高於 random baseline，表示 PCA 找到了有意義的結構。若 PCA 與 random baseline 接近，表示 PCA 子空間可能沒有太多資訊量。

輸出檔案：

```text
{config}_random_baseline.csv
```

欄位：

```text
round
k
pca_cumulative_explained
random_cumulative_explained_mean
random_cumulative_explained_std
random_trials
```

## Current Script Settings

### English

The script is run as:

```bash
python pca_validation_experiments.py \
  --config analyze_pca \
  --rounds 10 \
  --max-k 10 \
  --random-trials 50 \
  --output-dir results/pca_validation
```

Arguments:

```text
--config
    Which experiment config to use.

--rounds
    Number of sampled rounds used for PCA validation.

--max-k
    Maximum k to evaluate.

--random-trials
    Number of random subspaces sampled for each k and round.

--output-dir
    Directory for validation CSV outputs.
```

### 中文

腳本執行方式：

```bash
python pca_validation_experiments.py \
  --config analyze_pca \
  --rounds 10 \
  --max-k 10 \
  --random-trials 50 \
  --output-dir results/pca_validation
```

參數：

```text
--config
    使用哪個 experiment config。

--rounds
    PCA validation 要 sample 幾個 rounds。

--max-k
    要評估的最大 k。

--random-trials
    每個 k 和 round 要 sample 幾個 random subspaces。

--output-dir
    validation CSV 的輸出資料夾。
```

## Plotting

### English

After generating the CSV files, plots can be created with:

```bash
python plot_pca_validation.py --config analyze_pca
```

Default plot outputs:

```text
results/pca_validation/plots/{config}_scree.png
results/pca_validation/plots/{config}_scree.svg
results/pca_validation/plots/{config}_stability.png
results/pca_validation/plots/{config}_stability.svg
results/pca_validation/plots/{config}_random_baseline.png
results/pca_validation/plots/{config}_random_baseline.svg
```

### 中文

產生 CSV 後，可以用以下指令畫圖：

```bash
python plot_pca_validation.py --config analyze_pca
```

預設圖檔輸出：

```text
results/pca_validation/plots/{config}_scree.png
results/pca_validation/plots/{config}_scree.svg
results/pca_validation/plots/{config}_stability.png
results/pca_validation/plots/{config}_stability.svg
results/pca_validation/plots/{config}_random_baseline.png
results/pca_validation/plots/{config}_random_baseline.svg
```

## Notes

### English

- PCA is computed per round, not across all rounds pooled together.
- Only benign updates are used for PCA validation.
- The maximum meaningful number of PCs per round is `num_benign_clients - 1`.
- Random baseline uses random orthonormal subspaces with the same dimension `k`.
- These experiments validate the PCA assumption; they do not run the PCA attack itself.

### 中文

- PCA 是 per-round 計算，不是把所有 rounds 的 updates 混在一起計算。
- PCA validation 只使用 benign updates。
- 每個 round 有意義的最大 PC 數量是 `num_benign_clients - 1`。
- random baseline 使用相同維度 `k` 的隨機 orthonormal 子空間。
- 這些實驗是在驗證 PCA 假設，不是在執行 PCA attack 本身。
