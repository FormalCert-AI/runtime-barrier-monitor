"""Joint policy/barrier training for System 3 (Appendix C, M = 1)."""
import numpy as np
import torch

from core.systems import System3
from synthesis.common import mlp, save, weights

SEED, ITERATIONS = 2, 1000
W_DEC, W_UNSAFE, W_INIT = 10.0, 1.0, 1.0
B_INIT = 0.2

torch.manual_seed(SEED)
torch.set_num_threads(4)
rng = np.random.default_rng(SEED)

policy_net, barrier_net = mlp(32), mlp(64)


def policy(x):
    return 1.5 * torch.tanh(policy_net(x)).squeeze(-1)


def barrier(x):
    return torch.nn.functional.softplus(barrier_net(x)).squeeze(-1)


def barrier_bar(x):
    inside = torch.from_numpy(System3.inside(x.detach().numpy()))
    return torch.where(inside, barrier(x), torch.ones(len(x)))


def step(x):
    x1, x2 = x[:, 0], x[:, 1]
    mean = torch.stack([x1 + 0.1 * x2, x2 + 0.1 * (x1 ** 3 / 3 - x1 - x2 + policy(x))], 1)
    return mean + 0.01 * (torch.rand_like(mean) * 2 - 1)


system = System3()
optimizer = torch.optim.Adam(list(policy_net.parameters()) + list(barrier_net.parameters()), lr=1e-3)
for it in range(ITERATIONS + 1):
    xs = torch.tensor(system.sample_safe(4096, rng), dtype=torch.float32)
    xu = torch.tensor(system.sample_unsafe(1024, rng), dtype=torch.float32)
    x0 = torch.tensor(system.sample_initial(1024, rng), dtype=torch.float32)
    l_dec = torch.relu(barrier_bar(step(xs)) - barrier(xs)).mean()
    l_unsafe = torch.relu(1 - barrier(xu)).mean()
    l_init = torch.relu(barrier(x0) - B_INIT).mean()
    loss = W_DEC * l_dec + W_UNSAFE * l_unsafe + W_INIT * l_init
    if it > 0:                                # iteration 0 only draws samples
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if it % 200 == 0:
        print(f'it {it:4d}  L_dec {l_dec.item():.2e}  L_unsafe {l_unsafe.item():.2e}  L_init {l_init.item():.2e}', flush=True)

W = weights(barrier_net, policy_net)
save('system3', W)
