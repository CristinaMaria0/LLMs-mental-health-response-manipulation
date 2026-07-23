"""Shared statistics helpers used by both studies."""
import numpy as np
from scipy import stats as _ss


def cohen_d_paired(a, b):
    """Paired Cohen's d for the difference a - b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / (d.std(ddof=1) + 1e-9))


def paired_pvalues(a, b):
    """Paired Wilcoxon signed-rank and t-test p-values (NaN if undefined)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    try:
        wp = float(_ss.wilcoxon(a, b, zero_method="zsplit").pvalue)
    except Exception:
        wp = float("nan")
    try:
        tp = float(_ss.ttest_rel(a, b).pvalue)
    except Exception:
        tp = float("nan")
    return wp, tp


def risk_ratio(c_exp, n_exp, c_ref, n_ref):
    """Risk ratio of `exposed` vs `reference` with Haldane-Anscombe correction
    so a 0%/100% cell does not blow up the ratio."""
    return float(((c_exp + 0.5) / (n_exp + 1)) / ((c_ref + 0.5) / (n_ref + 1)))


def chi2_p(contingency):
    """p-value of a chi-square test of independence on a contingency table."""
    return float(_ss.chi2_contingency(contingency)[1])
