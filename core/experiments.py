"""Experiments of the paper. Each is simulated once and stored in results/<name>.npz."""
import os

import numpy as np

from . import config
from .models import NeuralBarrier, NeuralPolicy, PolynomialBarrier, quadratic
from .simulate import evaluate
from .systems import System1, System2, System3


def model(file):
    return os.path.join(config.MODELS, file)


def _system2_barrier():
    return PolynomialBarrier(model('system2_sos.json'))


def _system3(file):
    return System3(NeuralPolicy(model(file))), NeuralBarrier(model(file))


EXPERIMENTS = {
    # RQ1: valid certificates
    'system1': lambda: (System1(), quadratic),
    'system2': lambda: (System2(x0_radius=0.1), _system2_barrier()),
    # RQ2: invalid certificates
    'system2_input': lambda: (System2(x0_radius=0.35, input_start=100), _system2_barrier()),
    'system3': lambda: _system3('system3.npz'),
    # Section 4.4: after retraining
    'system2_input_retrained': lambda: (System2(x0_radius=0.35, input_start=100),
                                        NeuralBarrier(model('system2_input_retrained.npz'))),
    'system3_retrained': lambda: _system3('system3_retrained.npz'),
}
FRESH_RUNS = {'system2_input_retrained', 'system3_retrained'}   # new noise, not the training episodes


def run(name, N=config.N_INIT, R=config.N_RUNS, seed=config.SEED):
    """Results of one experiment; simulated if not stored yet."""
    path = os.path.join(config.RESULTS, f'{name}.npz')
    settings = dict(N=N, R=R, seed=seed, kmax_ln_a=np.array(config.KMAX_LN_A), horizons=np.array(config.HORIZONS),
                    sens_gamma=np.array(config.SENS_GAMMA), sens_cap=np.array(config.SENS_CAP))
    if os.path.exists(path):
        results = dict(np.load(path))
        if all(key in results and np.array_equal(results[key], value) for key, value in settings.items()):
            return results
    system, barrier = EXPERIMENTS[name]()
    rng, extension_rng = np.random.default_rng(seed), np.random.default_rng([seed, 1])
    run_rng = np.random.default_rng([seed, 2]) if name in FRESH_RUNS else None
    results = {**evaluate(system, barrier, N, R, rng, extension_rng, run_rng), **settings}
    os.makedirs(config.RESULTS, exist_ok=True)
    np.savez(path, **results)
    return results
