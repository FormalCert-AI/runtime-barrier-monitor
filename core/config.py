"""Settings of all experiments."""
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'models')
RESULTS = os.path.join(ROOT, 'results')

K_MAX = 400
ALPHA = 0.05
N_INIT = 1000                      # initial states
N_RUNS = 1000                      # runs per initial state
SEED = 2026
VERSION = 2                        # bump when the simulation changes, so cached results are recomputed

GAMMA = 0.92                       # DAB memory
CAP = 100.0                        # DAB cap U

# alarm threshold a^(K_max - k) / alpha with a = exp(c / K_max), c = K_max ln a
KMAX_LN_A = (0.0, 1.0, 2.0, 4.0, 8.0)
DEFAULT_KMAX_LN_A = 4.0
A_VALUES = tuple(math.exp(c / K_MAX) for c in KMAX_LN_A)
A_DEFAULT = math.exp(DEFAULT_KMAX_LN_A / K_MAX)
DEFAULT_INDEX = KMAX_LN_A.index(DEFAULT_KMAX_LN_A)

# mixtures of fixed bets (appendix)
FRACTIONS = (0.2, 0.5, 0.75, 1.0)
ABSOLUTE = ((1.0, 10.0, 20.0, 100.0), (10.0, 30.0, 60.0, 100.0))

# sensitivity (appendix), one setting varied at a time
K_EXT = 800
HORIZONS = (100, 200, 300, 400, 500, 600, 700, 800)
SENS_GAMMA = (0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.98, 1.0)
SENS_CAP = (1.0, 2.0, 5.0, 13.0, 20.0, 50.0, 100.0, 200.0, 1000.0)

# histogram of log10 M_k at these steps (appendix figure)
LOGM_STEPS = (100, 150, 200, 250, 300, 350, 400)
LOGM_RANGE, LOGM_BINS = (-2.0, 3.0), 250
