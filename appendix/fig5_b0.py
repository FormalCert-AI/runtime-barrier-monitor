"""Figure 5: initial barrier values and alarm rates, grouped into 8 equal-size bins by B(x0)."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from core.config import ALPHA, DEFAULT_KMAX_LN_A, KMAX_LN_A
from core.experiments import run
from plotting.style import DAB, DIRECT, WIDTH, legend_below, plt, save

plt.rcParams['axes.titlepad'] = 4
RULES = ((0.0, '-', dict()), (DEFAULT_KMAX_LN_A, '--', dict(mfc='white')))   # K_max ln a, line style, marker face
SYSTEMS = {  # experiment: label, panel title, colour and line style in (a), x-ticks
    'system1': ('System1', '(b) System1 (valid)', '#9a9a9a', (0, (4, 1.5)), [1e-3, 1e-2, 1e-1]),
    'system2_input': ('System2 + input', '(c) System2 + input (invalid)', '#5a5a5a', (0, (6, 1.5, 1.5, 1.5)), [1e-3, 1e-2, 1e-1]),
    'system3': ('System3', '(d) System3 (invalid)', '#222222', '-', [0.05, 0.1, 0.2]),
}
N_GROUPS = 8


def grouped(b0, p):
    out = []
    for group in np.array_split(np.argsort(b0), N_GROUPS):
        v = 100 * p[group]
        out.append((np.median(b0[group]), v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))))
    return np.array(out)


results = {name: run(name) for name in SYSTEMS}
fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.0), gridspec_kw=dict(wspace=0.34, left=0.06, right=0.99, top=0.86, bottom=0.3))

grid = np.linspace(-5.2, 0.2, 400)
for name, (label, _, color, style, _) in SYSTEMS.items():
    density = gaussian_kde(np.log10(np.clip(results[name]['B0'], 1e-6, None)), bw_method=0.15)(grid)
    axes[0].fill_between(10 ** grid, density, color=color, alpha=0.18, lw=0)
    axes[0].plot(10 ** grid, density, color=color, lw=1.4, ls=style, label=label)
axes[0].set_xscale('log')
axes[0].set_xlim(10 ** -5.2, 10 ** 0.2)
axes[0].set_ylim(0, 2.4)
axes[0].set_xticks([1e-4, 1e-2, 1])
axes[0].set_xticklabels(['0.0001', '0.01', '1'])
axes[0].set_xlabel('$B(x_0)$')
axes[0].set_ylabel('density')
axes[0].set_title('(a) Distribution of $B(x_0)$', loc='center', fontsize=8)
axes[0].legend(loc='upper left', fontsize=7, handlelength=1.8)

for ax, (name, (_, title, _, _, ticks)) in zip(axes[1:], SYSTEMS.items()):
    for m, color, marker in ((1, DAB, 'o'), (0, DIRECT, '^')):
        for c, ls, face in RULES:
            g = grouped(results[name]['B0'], results[name]['alarm_rate'][m, list(KMAX_LN_A).index(c)])
            if c == 0.0:
                ax.fill_between(g[:, 0], g[:, 1] - g[:, 2], g[:, 1] + g[:, 2], color=color, alpha=0.2, lw=0)
            ax.plot(g[:, 0], g[:, 1], color=color, lw=1.3, ls=ls, marker=marker, ms=5, mec=color if face else 'white',
                    mew=0.8, **face)
    valid = name == 'system1'
    ax.set_xscale('log')
    ax.set_xlabel('$B(x_0)$')
    ax.set_title(title, loc='center', fontsize=8)
    ax.set_ylim(-0.25 if valid else -4, 6 if valid else 104)
    ax.axhline(100 * ALPHA, color='#333333', lw=0.8, ls='--')
    ax.text(0.03, 100 * ALPHA, r'$\alpha$', transform=ax.get_yaxis_transform(), fontsize=8, va='bottom', ha='left')
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
axes[1].set_ylabel(r'alarm rate $\hat p_i$ (%)')
for ax in axes:
    ax.minorticks_off()
legend_below(fig, [Line2D([], [], color=DIRECT, lw=1.3, marker='^', ms=5, mec='white', label=r'Direct ($B_k/B_0$)'),
                   Line2D([], [], color=DAB, lw=1.3, marker='o', ms=5, mec='white', label='DAB'),
                   Line2D([], [], color='#555555', lw=1.3, ls='-', label=r'$K_{\max}\ln a=0$ ($a=1$)'),
                   Line2D([], [], color='#555555', lw=1.3, ls='--', label=f'$K_{{\\max}}\\ln a={DEFAULT_KMAX_LN_A:g}$ (default)'),
                   Patch(color='#999999', alpha=0.35, label='95% CI ($a=1$)')], columnspacing=1.2)
save(fig, 'fig_b0')
