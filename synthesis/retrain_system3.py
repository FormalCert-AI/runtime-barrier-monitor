"""Policy/barrier retraining for System 3 (Section 4.4, Algorithm 1).

Fit a dynamics model to monitored episodes, then retrain with one sampled successor per state (M = 1).
Evaluate each update on fresh runs; stopping uses empirical DAB alarm rates.
"""
import time

import numpy as np
import torch

from core.config import A_DEFAULT, ALPHA, DEFAULT_INDEX, K_MAX, N_INIT, N_RUNS, SEED
from core.experiments import model
from core.models import NeuralBarrier, NeuralPolicy
from core.monitors import alarm_times, dab
from core.report import p_max, rho
from core.simulate import evaluate, expected_barrier, visited_states
from core.systems import System3, _gauss_legendre
from synthesis.common import augmented_lagrangian, load, mlp, save, weights

EPISODES_PER_STATE, DITHER, MAX_ROUNDS = 5, 0.3, 4
M, W_O, RHO, BATCH = 1, 0.1, 100.0, 4096
MODEL_ITERATIONS, MODEL_BATCH = 4000, 4096
x1_grid, x2_grid = np.linspace(-3.5, 2, 276), np.linspace(-2, 1, 151)
GRID = np.stack(np.meshgrid(x1_grid, x2_grid, indexing='ij'), -1).reshape(-1, 2)
GRID = GRID[System3.inside(GRID)]


def closed_loop(W):
    return System3(NeuralPolicy(W)), NeuralBarrier(W)


def collect(W, rng):
    """Transitions (x, u, x+) of the dithered episodes."""
    system, barrier = closed_loop(W)
    X, U, X_next, n_alarm, n_exit = [], [], [], 0, 0
    for x0 in system.sample_initial(N_INIT, rng):
        x = np.repeat(x0[None], EPISODES_PER_STATE, 0)
        alive = np.ones(EPISODES_PER_STATE, bool)
        B, xs, us = [barrier(x)], [x], []
        for k in range(K_MAX):
            u = np.clip(system.policy(x) + rng.uniform(-DITHER, DITHER, len(x)), -1.5, 1.5)
            x1, x2 = x[:, 0], x[:, 1]
            x_next = np.stack([x1 + 0.1 * x2, x2 + 0.1 * (x1 ** 3 / 3 - x1 - x2 + u)], 1) + 0.01 * rng.uniform(-1, 1, x.shape)
            us.append(u)
            xs.append(x_next)
            alive &= system.inside(x_next)
            B.append(np.where(alive, barrier(np.where(alive[:, None], x_next, x)), 1.0))
            x = np.where(alive[:, None], x_next, x)
        B, xs, us = np.stack(B, 1), np.stack(xs, 1), np.stack(us, 1)
        exit_step = np.where((B == 1.0).any(1), (B == 1.0).argmax(1), K_MAX + 1)
        tau = alarm_times(dab(B), A_DEFAULT)
        for j in range(EPISODES_PER_STATE):
            n = min(int(tau[j]), int(exit_step[j]), K_MAX)
            X.append(xs[j, :n])
            U.append(us[j, :n])
            X_next.append(xs[j, 1:n + 1])
            n_alarm += tau[j] <= min(exit_step[j], K_MAX)
            n_exit += exit_step[j] <= min(tau[j], K_MAX)
    print(f'  {sum(map(len, X))} transitions; episode cutoffs at alarm {n_alarm}, exit {n_exit}', flush=True)
    return np.concatenate(X), np.concatenate(U), np.concatenate(X_next)


def fit_model(X, U, X_next):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 2))
    inp = torch.tensor(np.c_[X, U], dtype=torch.float32)
    target = torch.tensor(X_next - X, dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(MODEL_ITERATIONS):
        idx = torch.randint(0, len(inp), (MODEL_BATCH,))
        loss = ((net(inp[idx]) - target[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        residual = (net(inp) - target).numpy()
    sigma = np.quantile(np.abs(residual), 0.999, axis=0)
    print(f'  dynamics model: RMSE {np.sqrt((residual ** 2).mean()):.2e}, '
          f'99.9% residual quantile {sigma.round(4)}', flush=True)
    return net, sigma


class LearnedSystem(System3):
    """System 3 with f_hat in place of the dynamics."""

    def __init__(self, policy, net, sigma):
        super().__init__(policy)
        self.net, self.sigma = net, sigma

    def mean_next(self, X):
        with torch.no_grad():
            d = self.net(torch.tensor(np.c_[X, self.policy(X)], dtype=torch.float32)).numpy()
        return X + d

    def step(self, X, k, rng):
        return self.mean_next(X) + self.sigma * rng.uniform(-1, 1, X.shape)

    def successors(self, X, k):
        u, w = _gauss_legendre()
        m = self.mean_next(X)
        for i in range(len(u)):
            for j in range(len(u)):
                yield w[i] * w[j], m + self.sigma * np.array([u[i], u[j]])


def resynthesise(start, net, sigma, rounds=(6000, 3000, 3000, 3000), lr=1e-3, floor=None, train_policy=True):
    torch.manual_seed(0)
    torch.set_num_threads(6)
    policy_net, barrier_net = mlp(32), mlp(64)
    load(policy_net, start, 'pi')
    load(barrier_net, start, 'B')
    if not train_policy:
        for p in policy_net.parameters():
            p.requires_grad_(False)
    offset = 0.0 if floor is None else floor
    sig = torch.tensor(sigma, dtype=torch.float32)

    def policy(x):
        return 1.5 * torch.tanh(policy_net(x)).squeeze(-1)

    def barrier(x):
        return offset + torch.nn.functional.softplus(barrier_net(x)).squeeze(-1)

    def barrier_bar(x):
        return torch.where(System3.inside(x), barrier(x), torch.ones(x.shape[:-1]))

    def step(x, u):
        return x + net(torch.cat([x, u[..., None]], -1)) + sig * (torch.rand(M, BATCH, 2) * 2 - 1)

    def numpy_closed_loop():
        W = weights(barrier_net, policy_net, floor)
        return LearnedSystem(NeuralPolicy(W), net, sigma), NeuralBarrier(W)

    system, _ = numpy_closed_loop()
    visited, _ = visited_states(system, 1000, 5, np.random.default_rng(7))
    rng = np.random.default_rng(1)
    D_X = np.concatenate([visited[rng.choice(len(visited), min(100000, len(visited)), replace=False)],
                          System3.sample_safe(100000, rng)])
    params = list(policy_net.parameters()) + list(barrier_net.parameters()) if train_policy else barrier_net.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    for rnd, iterations in enumerate(rounds):
        P, nu = torch.tensor(D_X, dtype=torch.float32), torch.zeros(len(D_X))
        for _ in range(iterations):
            idx = torch.randint(0, len(P), (BATCH,))
            x = P[idx]
            with torch.set_grad_enabled(train_policy):
                x_next = step(x[None].expand(M, -1, -1), policy(x)[None].expand(M, -1))
            g = barrier_bar(x_next).mean(0) - barrier(x)
            x0 = torch.tensor(system.sample_initial(1024, rng), dtype=torch.float32)
            loss = W_O * barrier(x0).mean() + augmented_lagrangian(g, nu[idx], RHO)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                nu[idx] = torch.clamp(nu[idx] + RHO * g.detach(), min=0)
        system, B = numpy_closed_loop()
        visited_now, stay_fraction = visited_states(system, 200, 5, np.random.default_rng(100 + rnd))
        reached = visited_now[::5]
        d_grid = expected_barrier(system, B, GRID) - B(GRID)
        d_reached = expected_barrier(system, B, reached) - B(reached)
        print(f'    round {rnd}: model in S through K_max {100 * stay_fraction:.2f}% | '
              f'grid: max {d_grid.max():+.1e}', flush=True)
        bad = np.concatenate([GRID[d_grid > 1e-5], reached[d_reached > 1e-5]])
        if len(bad) == 0 or rnd == len(rounds) - 1:
            break
        new = [np.repeat(bad, 5, 0) + rng.normal(0, 0.005, (5 * len(bad), 2))]
        if train_policy:
            new.append(visited_now[rng.choice(len(visited_now), min(50000, len(visited_now)), replace=False)])
        D_X = np.concatenate([D_X] + new)
    return weights(barrier_net, policy_net, floor)


if __name__ == '__main__':
    start = time.time()
    W = dict(np.load(model('system3.npz')))
    data = []
    for r in range(MAX_ROUNDS):
        print(f'round {r + 1} ({time.time() - start:.0f} s)', flush=True)
        data.append(collect(W, np.random.default_rng([SEED, 10 + r])))
        X, U, X_next = (np.concatenate(parts) for parts in zip(*data))
        net, sigma = fit_model(X, U, X_next)
        joint = resynthesise(W, net, sigma)
        W = resynthesise(joint, net, sigma, rounds=(3000,), lr=5e-4, floor=0.01, train_policy=False)
        path = save('system3_retrained', W)
        res = evaluate(*closed_loop(dict(np.load(path))), N_INIT, N_RUNS, np.random.default_rng(SEED),
                       np.random.default_rng([SEED, 1]), np.random.default_rng([SEED, 2]))
        p = res['alarm_rate'][1, DEFAULT_INDEX]
        print(f'  real system: DAB rho {rho(p):.2f}%, p_max {p_max(p):.2f}% ({time.time() - start:.0f} s)', flush=True)
        if rho(p) == 0 and p_max(p) <= 100 * ALPHA:
            break
