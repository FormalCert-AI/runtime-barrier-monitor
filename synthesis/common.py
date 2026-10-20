"""Shared pieces of the training scripts."""
import os

import numpy as np
import torch
import torch.nn as nn

from core.config import RESULTS

OUT = os.path.join(RESULTS, 'synthesis')


def mlp(hidden):
    """2 -> hidden -> hidden -> 1, tanh."""
    return nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def load(net, weights, prefix):
    net.load_state_dict({k.split('.', 1)[1]: torch.tensor(weights[k]) for k in weights if k.startswith(f'{prefix}_net.')})


def weights(barrier_net=None, policy_net=None, floor=None):
    """Arrays in the format of models/*.npz: 'B_net.*', 'pi_net.*', 'B_eps'."""
    out = {}
    for prefix, net in (('B', barrier_net), ('pi', policy_net)):
        if net is not None:
            out.update({f'{prefix}_net.{k}': v.detach().numpy() for k, v in net.state_dict().items()})
    if floor is not None:
        out['B_eps'] = np.array(floor)
    return out


def augmented_lagrangian(g, nu, rho):
    """Penalty of g <= 0 with multipliers nu: mean of (max(nu + rho g, 0)^2 - nu^2) / (2 rho)."""
    return ((torch.clamp(nu + rho * g, min=0) ** 2 - nu ** 2) / (2 * rho)).mean()


def save(name, W):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{name}.npz')
    np.savez(path, **W)
    print(f'saved {path}')
    return path
