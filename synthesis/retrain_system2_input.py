"""Barrier retraining for System 2+input (Section 4.4, Algorithm 1).

Use the first 5 runs per initial state, truncated at the default DAB alarm, exit or K_max.
Each observed transition imposes B_bar(x+) <= B(x) (M = 1), with an augmented Lagrangian.
"""
import time

import numpy as np
import torch

from core.config import A_DEFAULT, K_MAX, N_INIT, N_RUNS, SEED
from core.experiments import EXPERIMENTS
from core.monitors import alarm_times, dab
from core.simulate import rollouts
from synthesis.common import augmented_lagrangian, mlp, save, weights

EPISODES_PER_STATE, FLOOR, ITERATIONS, W_O, RHO, BATCH = 5, 0.01, 12000, 0.1, 100.0, 4096

start = time.time()
system, barrier = EXPERIMENTS['system2_input']()
rng = np.random.default_rng(SEED)
X0 = system.sample_initial(N_INIT, rng)
X, X_next = [], []
for x0 in X0:
    B, T, path = rollouts(system, barrier, x0, N_RUNS, rng, record=True)
    tau = alarm_times(dab(B), A_DEFAULT)
    for j in range(EPISODES_PER_STATE):
        n = min(int(T[j]), int(tau[j]), K_MAX)
        X.append(path[j, :n])
        X_next.append(path[j, 1:n + 1])
X, X_next = np.concatenate(X), np.concatenate(X_next)
print(f'{len(X)} transitions from {EPISODES_PER_STATE * N_INIT} episodes '
      f'({time.time() - start:.0f} s)', flush=True)

torch.manual_seed(0)
torch.set_num_threads(6)
net = mlp(64)
inside = torch.from_numpy(system.inside(X_next))
P, P_next = torch.tensor(X, dtype=torch.float32), torch.tensor(X_next, dtype=torch.float32)
objective = torch.tensor(np.concatenate([X0, X[np.random.default_rng(1).choice(len(X), 50000, replace=False)]]),
                         dtype=torch.float32)
nu = torch.zeros(len(P))


def B_net(x):
    return FLOOR + torch.nn.functional.softplus(net(x)).squeeze(-1)


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
for it in range(ITERATIONS + 1):
    idx = torch.randint(0, len(P), (BATCH,))
    g = torch.where(inside[idx], B_net(P_next[idx]), torch.ones(BATCH)) - B_net(P[idx])
    loss = W_O * B_net(objective[torch.randint(0, len(objective), (1024,))]).mean() + augmented_lagrangian(g, nu[idx], RHO)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        nu[idx] = torch.clamp(nu[idx] + RHO * g.detach(), min=0)
    if it % 3000 == 0:
        with torch.no_grad():
            g_all = torch.cat([torch.where(inside[s], B_net(P_next[s]), torch.ones(len(s))) - B_net(P[s])
                               for s in torch.split(torch.arange(len(P)), 200000)])
        print(f'it {it:5d}: increasing transitions {100 * (g_all > 0).float().mean():.2f}%, mean B(x0) '
              f'{B_net(torch.tensor(X0, dtype=torch.float32)).mean():.4f} ({time.time() - start:.0f} s)', flush=True)

save('system2_input_retrained', weights(net, floor=FLOOR))
