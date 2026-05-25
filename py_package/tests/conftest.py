import numpy as np


def nan_allclose(result, expected, rtol=1e-12, atol=1e-12):
    """Assert arrays are equal, treating NaN positions as matching."""
    result = np.asarray(result, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    nan_exp = np.isnan(expected)
    assert np.array_equal(np.isnan(result), nan_exp), (
        f"NaN mask mismatch:\n  result  = {result}\n  expected= {expected}"
    )
    if np.any(~nan_exp):
        np.testing.assert_allclose(result[~nan_exp], expected[~nan_exp], rtol=rtol, atol=atol)


def var_ref(x, k):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k + 1):i + 1]
        if len(w) >= 2:
            out[i] = np.var(w, ddof=1)
    return out


def mean_ref(x, k):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k + 1):i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) >= 1:
            out[i] = valid.mean()
    return out


def median_ref(x, k):
    return np.array([np.median(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])


def cov_ref(x, y, k):
    """Sample covariance (ddof=1) over valid pairs in each rolling window."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        xi = x[max(0, i - k + 1):i + 1]
        yi = y[max(0, i - k + 1):i + 1]
        mask = ~np.isnan(xi) & ~np.isnan(yi)
        xi, yi = xi[mask], yi[mask]
        if len(xi) >= 2:
            out[i] = np.cov(xi, yi, ddof=1)[0, 1]
    return out


def cor_ref(x, y, k):
    """Pearson correlation over valid pairs in each rolling window."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        xi = x[max(0, i - k + 1):i + 1]
        yi = y[max(0, i - k + 1):i + 1]
        mask = ~np.isnan(xi) & ~np.isnan(yi)
        xi, yi = xi[mask], yi[mask]
        if len(xi) >= 2 and np.std(xi) > 0 and np.std(yi) > 0:
            out[i] = np.corrcoef(xi, yi)[0, 1]
    return out