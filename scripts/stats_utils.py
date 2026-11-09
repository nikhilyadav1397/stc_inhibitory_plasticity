import numpy as np
import pandas as pd
from scipy.stats import sem, t


def ci_t(vals, alpha=0.05):
    vals = np.asarray(pd.Series(vals).dropna(), dtype=float)
    n = len(vals)

    if n < 2:
        return np.nan, np.nan

    m = np.mean(vals)
    h = sem(vals) * t.ppf(1 - alpha / 2, n - 1)
    return m - h, m + h


def cohen_dz(diff):
    diff = np.asarray(pd.Series(diff).dropna(), dtype=float)
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return np.nan
    return np.mean(diff) / np.std(diff, ddof=1)


def cohen_d_independent(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)

    pooled_sd = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))

    if pooled_sd == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd


def ci_t(vals, alpha=0.05):
    vals = np.asarray(pd.Series(vals).dropna(), dtype=float)
    n = len(vals)

    if n < 2:
        return np.nan, np.nan

    m = np.mean(vals)
    h = sem(vals) * t.ppf(1 - alpha / 2, n - 1)

    return m - h, m + h


def cohen_d_1samp(vals, popmean=0):
    vals = np.asarray(pd.Series(vals).dropna(), dtype=float)
    diff = vals - popmean

    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return np.nan

    return np.mean(diff) / np.std(diff, ddof=1)


def cohen_d_independent(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)

    pooled_sd = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))

    if pooled_sd == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled_sd


def get_posthoc_p(posthoc_df, pattern, exp, group1, group2, corrected=True):
    row = posthoc_df[
        (posthoc_df["comparison_type"] == "overlap_within_experiment_and_pattern")
        & (posthoc_df["pattern"] == pattern)
        & (posthoc_df["experiment"] == exp)
        & (posthoc_df["group1"] == group1)
        & (posthoc_df["group2"] == group2)
    ]

    if len(row) == 0:
        return np.nan

    col = "p_fdr" if corrected and "p_fdr" in row.columns else "p_uncorrected"
    return row.iloc[0][col]


def get_gain_p(gain_df, condition, metric, corrected=True):
    row = gain_df[(gain_df["condition"] == condition) & (gain_df["metric"] == metric)]

    if len(row) == 0:
        return np.nan

    col = "p_fdr" if corrected and "p_fdr" in row.columns else "p_uncorrected"
    return row.iloc[0][col]


def get_between_p(between_df, metric, corrected=True):
    row = between_df[between_df["metric"] == metric]

    if len(row) == 0:
        return np.nan

    col = "p_fdr" if corrected and "p_fdr" in row.columns else "p_uncorrected"
    return row.iloc[0][col]
