"""Barriers and policies, evaluated row-wise on states X (n x dim)."""
import json

import numpy as np


def quadratic(X):
    return X[:, 0] ** 2


class PolynomialBarrier:
    def __init__(self, path):
        with open(path) as f:
            cert = json.load(f)
        self.coef = np.asarray(cert['coef'], float)
        self.expo = np.asarray(cert['expo'], int)

    def __call__(self, X):
        powers = np.arange(self.expo.max() + 1)
        monomials = np.ones((len(X), len(self.coef)))
        for j in range(X.shape[1]):
            monomials *= np.power(X[:, j][:, None], powers[None, :])[:, self.expo[:, j]]
        return monomials @ self.coef


class MLP:
    def __init__(self, weights, prefix):
        self.layers = [(weights[f'{prefix}_net.{i}.weight'], weights[f'{prefix}_net.{i}.bias']) for i in (0, 2, 4)]

    def __call__(self, X):
        (W0, b0), (W1, b1), (W2, b2) = self.layers
        h = np.tanh(X @ W0.T + b0)
        h = np.tanh(h @ W1.T + b1)
        return (h @ W2.T + b2)[:, 0]


def _weights(source):
    return np.load(source) if isinstance(source, str) else source


class NeuralBarrier:
    def __init__(self, source):
        weights = _weights(source)
        self.net = MLP(weights, 'B')
        self.floor = float(weights['B_eps']) if 'B_eps' in weights else 0.0

    def __call__(self, X):
        return self.floor + np.logaddexp(0.0, self.net(X))


class NeuralPolicy:
    def __init__(self, source, u_max=1.5):
        self.net = MLP(_weights(source), 'pi')
        self.u_max = u_max

    def __call__(self, X):
        return self.u_max * np.tanh(self.net(X))
