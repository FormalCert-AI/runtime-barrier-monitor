"""Benchmark dynamics and sampling regions."""
import numpy as np

GAUSS_NODES = 12


def _gauss_legendre(n=GAUSS_NODES):
    u, w = np.polynomial.legendre.leggauss(n)
    return u, w / 2


class System1:
    dim = 1

    def sample_initial(self, n, rng):
        return rng.uniform(-0.4, 0.4, (n, 1))

    def step(self, X, k, rng):
        xi = rng.uniform(-1, 1, (len(X), 1))
        return (0.5 + 1.5 * xi) * X * (1 - X ** 2)

    def inside(self, X):
        return np.abs(X[:, 0]) < 1


class System2:
    dim = 2

    def __init__(self, x0_radius=0.1, input_start=None, input_range=(0.0, 0.45)):
        self.x0_radius = x0_radius
        self.input_start = input_start
        self.input_range = input_range

    def sample_initial(self, n, rng):
        r = self.x0_radius * np.sqrt(rng.uniform(size=n))
        phi = rng.uniform(0, 2 * np.pi, n)
        return np.c_[r * np.cos(phi), r * np.sin(phi)]

    def input_active(self, k):
        return self.input_start is not None and k >= self.input_start

    @staticmethod
    def next_state(X, xi, b):
        return np.c_[X[:, 0] - 0.02 * X[:, 1],
                     X[:, 1] + 0.01 * ((0.8 + xi) * X[:, 0] + 10 * (X[:, 0] ** 2 - 0.21) * X[:, 1] + b)]

    def step(self, X, k, rng):
        xi = rng.uniform(-0.1, 0.1, len(X))
        b = rng.uniform(*self.input_range, len(X)) if self.input_active(k) else 0.0
        return self.next_state(X, xi, b)

    def inside(self, X):
        return (X ** 2).sum(1) <= 1

    def successors(self, X, k):
        u, w = _gauss_legendre()
        if not self.input_active(k):
            for i in range(len(u)):
                yield w[i], self.next_state(X, 0.1 * u[i], 0.0)
            return
        lo, hi = self.input_range
        for i in range(len(u)):
            for j in range(len(u)):
                yield w[i] * w[j], self.next_state(X, 0.1 * u[i], lo + (hi - lo) / 2 * (1 + u[j]))


class System3:
    dim = 2
    X0_DISK, X0_RADIUS, X0_BOX1 = (1.5, 0.0), 0.5, (-1.8, -1.4)

    def __init__(self, policy=None):
        self.policy = policy

    def sample_initial(self, n, rng):
        (cx, _), r0, box1 = self.X0_DISK, self.X0_RADIUS, self.X0_BOX1
        areas = np.array([np.pi * r0 ** 2, (box1[1] - box1[0]) * 0.2, 0.2 * 0.6])
        piece = rng.choice(3, n, p=areas / areas.sum())
        r = r0 * np.sqrt(rng.uniform(size=n))
        phi = rng.uniform(0, 2 * np.pi, n)
        X = np.stack([cx + r * np.cos(phi), r * np.sin(phi)], 1)
        m1, m2 = piece == 1, piece == 2
        X[m1] = np.stack([rng.uniform(box1[0], box1[1], m1.sum()), rng.uniform(-0.1, 0.1, m1.sum())], 1)
        X[m2] = np.stack([rng.uniform(-1.4, -1.2, m2.sum()), rng.uniform(-0.5, 0.1, m2.sum())], 1)
        return X

    @staticmethod
    def obstacle(X):
        x1, x2 = X[..., 0], X[..., 1]
        disk = (x1 + 1) ** 2 + (x2 + 1) ** 2 <= 0.16
        box1 = (x1 >= 0.4) & (x1 <= 0.6) & (x2 >= 0.1) & (x2 <= 0.5)
        box2 = (x1 >= 0.4) & (x1 <= 0.8) & (x2 >= 0.1) & (x2 <= 0.3)
        return disk | box1 | box2

    @classmethod
    def inside(cls, X):
        x1, x2 = X[..., 0], X[..., 1]
        return (x1 >= -3.5) & (x1 <= 2.0) & (x2 >= -2.0) & (x2 <= 1.0) & ~cls.obstacle(X)

    @classmethod
    def sample_safe(cls, n, rng):
        out = []
        while sum(len(o) for o in out) < n:
            X = np.stack([rng.uniform(-3.5, 2.0, 2 * n), rng.uniform(-2.0, 1.0, 2 * n)], 1)
            out.append(X[cls.inside(X)])
        return np.concatenate(out)[:n]

    @classmethod
    def sample_unsafe(cls, n, rng):
        X = np.stack([rng.uniform(-3.8, 2.3, 6 * n), rng.uniform(-2.3, 1.3, 6 * n)], 1)
        return X[~cls.inside(X)][:n]

    def mean_next(self, X):
        x1, x2 = X[:, 0], X[:, 1]
        u = self.policy(X)
        return np.stack([x1 + 0.1 * x2, x2 + 0.1 * (x1 ** 3 / 3 - x1 - x2 + u)], 1)

    def step(self, X, k, rng):
        return self.mean_next(X) + 0.01 * rng.uniform(-1, 1, X.shape)

    def successors(self, X, k):
        u, w = _gauss_legendre()
        m = self.mean_next(X)
        for i in range(len(u)):
            for j in range(len(u)):
                yield w[i] * w[j], m + 0.01 * np.array([u[i], u[j]])
