"""Monte Carlo runs and the per-initial-state alarm rates."""
import numpy as np

from .config import (A_DEFAULT, A_VALUES, ABSOLUTE, CAP, GAMMA, HORIZONS, K_EXT, K_MAX, LOGM_BINS,
                     LOGM_RANGE, LOGM_STEPS, SENS_CAP, SENS_GAMMA)
from .monitors import absolute_mixture, alarm_times, dab, direct, fraction_mixture

MONITORS = {'direct': direct, 'dab': dab}
MIXTURES = {'fraction': fraction_mixture,
            'absolute1': lambda B: absolute_mixture(B, ABSOLUTE[0]),
            'absolute2': lambda B: absolute_mixture(B, ABSOLUTE[1])}


def rollouts(system, barrier, x0, R, rng, record=False, extension=None):
    horizon = K_MAX if extension is None else extension[1]
    X = np.repeat(np.asarray(x0, float)[None], R, 0)
    B, T, alive = np.empty((R, horizon + 1)), np.full(R, horizon + 1), np.ones(R, bool)
    B[:, 0] = barrier(X)
    path = [X] if record else None
    for k in range(horizon):
        X_next = system.step(X, k, rng if k < K_MAX else extension[0])
        X_next[~alive] = X[~alive]
        left = alive & ~system.inside(X_next)
        T[left] = k + 1
        alive &= ~left
        X = X_next
        B[:, k + 1] = np.where(alive, barrier(X), 1.0)
        if record:
            path.append(X)
    return B, T, (np.stack(path, 1) if record else None)


def _rate(M, a, T, horizon=K_MAX):
    return (alarm_times(M[:, :horizon + 1], a) <= np.minimum(T, horizon)).mean()


def evaluate(system, barrier, N, R, rng, extension_rng, run_rng=None):
    X0 = system.sample_initial(N, rng)
    rng = run_rng or rng
    n_m, n_a = len(MONITORS), len(A_VALUES)
    out = dict(x0=X0, B0=np.empty(N), n_exit=np.empty(N, int), alarm_rate=np.empty((n_m, n_a, N)),
               caught=np.zeros((n_m, n_a), int), mixture_rate=np.empty((len(MIXTURES), N)),
               logm_hist=np.zeros((n_m, len(LOGM_STEPS), LOGM_BINS), int), horizon_rate=np.empty((n_m, len(HORIZONS), N)),
               gamma_rate=np.empty((len(SENS_GAMMA), N)), cap_rate=np.empty((len(SENS_CAP), N)))
    edges = np.linspace(*LOGM_RANGE, LOGM_BINS + 1)
    for i, x0 in enumerate(X0):
        B_ext, T_ext, _ = rollouts(system, barrier, x0, R, rng, extension=(extension_rng, K_EXT))
        B, T = B_ext[:, :K_MAX + 1], np.minimum(T_ext, K_MAX + 1)
        unsafe = T <= K_MAX
        out['B0'][i], out['n_exit'][i] = B[0, 0], unsafe.sum()
        for g, gamma in enumerate(SENS_GAMMA):
            out['gamma_rate'][g, i] = _rate(dab(B, gamma, CAP), A_DEFAULT, T)
        for u, cap in enumerate(SENS_CAP):
            out['cap_rate'][u, i] = _rate(dab(B, GAMMA, cap), A_DEFAULT, T)
        for b, mixture in enumerate(MIXTURES.values()):
            out['mixture_rate'][b, i] = _rate(mixture(B), A_DEFAULT, T)
        for m, monitor in enumerate(MONITORS.values()):
            M_ext = monitor(B_ext)
            for h, horizon in enumerate(HORIZONS):
                out['horizon_rate'][m, h, i] = _rate(M_ext, 1.0, T_ext, horizon)
            M = M_ext[:, :K_MAX + 1]
            logM = np.clip(np.log10(np.maximum(M[:, list(LOGM_STEPS)], 1e-300)), *LOGM_RANGE)
            for s, k in enumerate(LOGM_STEPS):
                out['logm_hist'][m, s] += np.histogram(logM[T >= k, s], bins=edges)[0]
            for j, a in enumerate(A_VALUES):
                tau = alarm_times(M, a)
                out['alarm_rate'][m, j, i] = (tau <= np.minimum(T, K_MAX)).mean()
                out['caught'][m, j] += (unsafe & (tau < T)).sum()
    return out


def visited_states(system, n_init, R, rng):
    states, n_exit = [], 0
    for x0 in system.sample_initial(n_init, rng):
        X, alive = np.repeat(x0[None], R, 0), np.ones(R, bool)
        for k in range(K_MAX):
            states.append(X[alive])
            X_next = system.step(X, k, rng)
            X_next[~alive] = X[~alive]
            left = alive & ~system.inside(X_next)
            n_exit += left.sum()
            alive &= ~left
            X = X_next
            if not alive.any():
                break
    return np.concatenate(states), 1 - n_exit / (n_init * R)


def expected_barrier(system, barrier, X, k=0):
    total = 0
    for weight, X_next in system.successors(X, k):
        total = total + weight * np.where(system.inside(X_next), barrier(X_next), 1.0)
    return total
