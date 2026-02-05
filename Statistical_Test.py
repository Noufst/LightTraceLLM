import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# =========================
# 1) Put per-run F2 scores here
#    Each list must have the SAME length and be aligned by run index.
# =========================

DATA = {
    "CM1_NASA": {
        "TraceLLM":       [],
        "LightLLM_MV":    [],
        "LightLLM_HE":    [],
        "Embed_HE":       [],
    },
    "EasyClinic_UC_TC": {
        "TraceLLM":       [],
        "LightLLM_MV":    [],
        "LightLLM_HE":    [],
        "Embed_HE":       [],
    },
    "EasyClinic_UC_ID": {
        "TraceLLM":       [],
        "LightLLM_MV":    [],
        "LightLLM_HE":    [],
        "Embed_HE":       [],
    },
    "CCHIT": {
        "TraceLLM":       [],
        "LightLLM_MV":    [],
        "LightLLM_HE":    [],
        "Embed_HE":       [],
    },
}

# =========================
# 2) Stats helpers
# =========================

def cliffs_delta(x, y):
    """
    Cliff's delta effect size for two independent samples.
    We use it here on paired run results as a simple, common practice.
    Returns delta in [-1, 1]. Positive means x tends to be larger than y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Efficient O(n log n) approach using sorting
    # Count how many y are less than each x, and how many y are greater than each x.
    y_sorted = np.sort(y)
    n = len(x)
    m = len(y)

    # For each xi: number of y < xi is idx_left, number of y <= xi is idx_right
    idx_left = np.searchsorted(y_sorted, x, side="left")
    idx_right = np.searchsorted(y_sorted, x, side="right")

    n_greater = (m - idx_right).sum()  # y > x
    n_less = idx_left.sum()            # y < x
    delta = (n_less - n_greater) / (n * m)
    return float(delta)

def delta_magnitude(delta):
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"

def wilcoxon_test(baseline, method):
    """
    Two-sided paired Wilcoxon signed-rank test.
    Handles edge cases where all differences are zero.
    """
    baseline = np.asarray(baseline, dtype=float)
    method = np.asarray(method, dtype=float)

    if baseline.shape != method.shape:
        raise ValueError(f"Length mismatch: baseline={len(baseline)} method={len(method)}")

    diffs = method - baseline
    if np.allclose(diffs, 0.0):
        # No difference at all across runs
        return 1.0

    # zero_method='wilcox' drops zero-differences; standard practice
    stat, p = wilcoxon(method, baseline, alternative="two-sided", zero_method="wilcox")
    return float(p)

# =========================
# 3) Run comparisons: each method vs TraceLLM
# =========================

comparisons = ["LightLLM_MV", "LightLLM_HE", "Embed_HE"]
rows = []

for dataset, scores in DATA.items():
    base = scores["TraceLLM"]

    for method_name in comparisons:
        method = scores[method_name]

        p_value = wilcoxon_test(base, method)
        d = cliffs_delta(method, base)  # positive => method tends to be higher than TraceLLM
        rows.append({
            "dataset": dataset,
            "comparison": f"{method_name} vs TraceLLM",
            "p_value": p_value,
            "cliffs_delta": d,
            "delta_mag": delta_magnitude(d),
            "mean_F2_method": float(np.mean(method)),
            "mean_F2_baseline": float(np.mean(base)),
            "mean_diff(method-baseline)": float(np.mean(np.asarray(method) - np.asarray(base))),
        })

results = pd.DataFrame(rows).sort_values(["dataset", "comparison"]).reset_index(drop=True)

# Display in console (or write to CSV)
pd.set_option("display.max_colwidth", None)
print(results.to_string(index=False))

# Optionally save:
results.to_csv("Results/wilcoxon_cliffs_delta_results.csv", index=False)