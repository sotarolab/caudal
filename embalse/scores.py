"""Proper scoring rules for the probabilistic streamflow output.

The fleet scores predictive draws with `crps_ensemble` (all heads emit draws, so no closed
form is needed).
"""

from __future__ import annotations

import numpy as np


def crps_ensemble(ensemble: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Empirical CRPS from an ensemble/sample-based predictive distribution (no parametric assumption).

    Uses the standard unbiased pairwise-distance estimator (e.g. Hersbach 2000):
    CRPS = E|X - y| - 0.5 * E|X - X'|, with X, X' independent draws from the ensemble.

    Parameters
    ----------
    ensemble : np.ndarray
        Shape (n_samples, n_members) -- one row per prediction, one column per ensemble member/sample.
    obs : np.ndarray
        Shape (n_samples,).

    Returns
    -------
    np.ndarray
        Per-sample CRPS, shape (n_samples,).
    """
    n_members = ensemble.shape[1]
    term1 = np.mean(np.abs(ensemble - obs[:, None]), axis=1)
    term2 = np.mean(np.abs(ensemble[:, :, None] - ensemble[:, None, :]), axis=(1, 2))
    return term1 - 0.5 * term2 * (n_members / max(n_members - 1, 1))
