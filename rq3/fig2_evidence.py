"""Figure 2 (RQ3): example DAB evidence paths, a = 1 and K_max ln a = 4."""
import numpy as np
from matplotlib.lines import Line2D

from core.config import A_DEFAULT, ALPHA, DEFAULT_KMAX_LN_A, K_MAX, SEED
from core.experiments import EXPERIMENTS
from core.monitors import alarm_times, dab
from core.simulate import rollouts
from core.systems import System3
from plotting.style import DAB, WIDTH, legend_below, plt, save

N_SHOWN = 2
steps = np.arange(1, K_MAX + 1)


def thresholds(ax, xmax):
    x = np.arange(0, xmax + 1)
    ax.axhline(np.log(1 / ALPHA), color='#333333', lw=0.8, ls='--', zorder=1)
    ax.plot(x, np.log(1 / ALPHA) + DEFAULT_KMAX_LN_A * (1 - x / K_MAX), color='#333333', lw=0.9, zorder=1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-2, 8)
    ax.set_xticks(np.linspace(0, xmax, 5 if xmax == K_MAX else 4))
    ax.set_xlabel('time step $k$')
    ax.minorticks_off()


def panel(ax, B, T, shown, title, xmax=K_MAX):
    """ln M_k of the runs `shown`; hollow marker: alarm with a = 1, filled: alarm with the default."""
    M = dab(B)
    lnM, tau1, tau = np.log(np.maximum(M[:, 1:], 1e-300)), alarm_times(M), alarm_times(M, A_DEFAULT)
    thresholds(ax, xmax)
    for i in shown:
        n = min(int(T[i]), K_MAX)
        ax.plot(steps[:n], lnM[i, :n], color=DAB, lw=1.0, zorder=2)
        for k, face in ((int(tau1[i]), dict(mfc='white', mec=DAB, mew=1.1)), (int(tau[i]), dict(mfc=DAB, mec='white'))):
            if k <= n:
                ax.plot(k, lnM[i, k - 1], marker='o', ms=5.5, ls='', zorder=4, **face)
    ax.set_title(title, fontsize=8)


rng = np.random.default_rng([SEED, 5])
fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.1), gridspec_kw=dict(wspace=0.22, left=0.06, right=0.99, top=0.9,
                                                                      bottom=0.3))

# (a) System 1, x0 = 0.1: runs with a DAB alarm at a = 1 and none with the default
system, barrier = EXPERIMENTS['system1']()
B, T, _ = rollouts(system, barrier, np.array([0.1]), 20000, rng)
M = dab(B)
candidates = np.flatnonzero((alarm_times(M) <= K_MAX) & (alarm_times(M, A_DEFAULT) > K_MAX))
panel(axes[0], B, T, rng.choice(candidates, N_SHOWN, replace=False), '(a) System1 (valid), first 30 steps', xmax=30)

# (b) System 2+input: earliest and median default alarm
system, barrier = EXPERIMENTS['system2_input']()
B, T, _ = rollouts(system, barrier, np.array([0.137, -0.145]), 300, rng)
tau = alarm_times(dab(B), A_DEFAULT)
candidates = np.flatnonzero(tau <= K_MAX)
panel(axes[1], B, T, candidates[np.argsort(tau[candidates])][[0, len(candidates) // 2]], '(b) System2 + input (invalid)')

# (c) System 3: earliest and median default alarm before the exit
x0 = System3().sample_initial(300, np.random.default_rng(11))[181]
system, barrier = EXPERIMENTS['system3']()
B, T, _ = rollouts(system, barrier, x0, 300, rng)
tau = alarm_times(dab(B), A_DEFAULT)
candidates = np.flatnonzero((tau <= K_MAX) & (tau < T))
panel(axes[2], B, T, candidates[np.argsort(tau[candidates])][[0, len(candidates) // 2]], '(c) System3 (invalid)')

axes[0].set_ylabel(r'DAB evidence $\ln M_k$')
legend_below(fig, [Line2D([], [], color=DAB, lw=1.0, label='DAB evidence of one run'),
                   Line2D([], [], ls='', marker='o', ms=5, mfc='white', mec=DAB, mew=1.1, label='alarm, $a=1$'),
                   Line2D([], [], ls='', marker='o', ms=5, color=DAB, mec='white', label='alarm, default'),
                   Line2D([], [], color='#333333', lw=0.8, ls='--', label=r'threshold, $a=1$'),
                   Line2D([], [], color='#333333', lw=0.9,
                          label=f'threshold, $K_{{\\max}}\\ln a={DEFAULT_KMAX_LN_A:g}$ (default)')])
save(fig, 'fig_evidence')
