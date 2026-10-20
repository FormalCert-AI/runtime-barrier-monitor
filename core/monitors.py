"""E-process monitors on the barrier values B_0, ..., B_K of a run (one run per row of B).

Each returns M_k with M_0 = 1; an alarm is raised at the first k >= 1 with M_k >= a^(K - k) / alpha.
"""
import numpy as np

from .config import ALPHA, CAP, FRACTIONS, GAMMA

B_MIN = 1e-12


def _wealth(factors):
    return np.cumprod(np.concatenate([np.ones((len(factors), 1)), factors], axis=1), axis=1)


def direct(B):
    """lambda_k = 1 / B_k, i.e. M_k = B_k / B_0."""
    return _wealth(1.0 + np.diff(B, axis=1) / np.maximum(B[:, :-1], B_MIN))


def dab(B, gamma=GAMMA, cap=CAP):
    """DAB: lambda_k = min(max(S_k / Q_k, 0), cap, 1 / B_k), S_k and Q_k discounted sums of past increments and squares."""
    R, K1 = B.shape
    lam = np.empty((R, K1 - 1))
    S, Q = np.zeros(R), np.zeros(R)
    for k in range(K1 - 1):
        lam[:, k] = np.minimum(np.minimum(np.maximum(S / (Q + B_MIN), 0.0), cap), 1.0 / np.maximum(B[:, k], B_MIN))
        d = B[:, k + 1] - B[:, k]
        S = gamma * S + d
        Q = gamma * Q + d * d
    return _wealth(1.0 + lam * np.diff(B, axis=1))


def fraction_mixture(B, fractions=FRACTIONS):
    """Equal-weight mixture of lambda_k = c / B_k, c in fractions."""
    relative = np.diff(B, axis=1) / np.maximum(B[:, :-1], B_MIN)
    return np.mean([_wealth(1.0 + c * relative) for c in fractions], axis=0)


def absolute_mixture(B, betas):
    """Equal-weight mixture of lambda_k = min(beta, 1 / B_k), beta in betas."""
    d, largest = np.diff(B, axis=1), 1.0 / np.maximum(B[:, :-1], B_MIN)
    return np.mean([_wealth(1.0 + np.minimum(beta, largest) * d) for beta in betas], axis=0)


def alarm_times(M, a=1.0, alpha=ALPHA):
    """First k >= 1 with M_k >= a^(K - k) / alpha (K = M.shape[1] - 1); K + 1 if there is none."""
    horizon = M.shape[1] - 1
    cross = M >= a ** (horizon - np.arange(horizon + 1)) / alpha
    cross[:, 0] = False
    return np.where(cross.any(axis=1), cross.argmax(axis=1), horizon + 1)
