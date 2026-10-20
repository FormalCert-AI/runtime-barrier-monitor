"""Violins of alarm rates, with a separate row below the axis for p_hat_i = 0."""
import numpy as np

from core.config import ALPHA

BINS = 50


def smooth(counts, width=2.0):
    x = np.arange(-4 * int(width), 4 * int(width) + 1)
    kernel = np.exp(-x ** 2 / (2 * width ** 2))
    return np.convolve(counts, kernel / kernel.sum(), mode='same')


def zero_row(top=100):
    return -0.09 * top, -0.02 * top


def _span(x, width, side):
    return (x - width if side in ('both', 'left') else x), (x + width if side in ('both', 'right') else x)


def draw(ax, x, centers, density, color, half_width, side='both'):
    left, right = _span(x, half_width * density / density.max(), side)
    ax.fill_betweenx(centers, left, right, color=color, lw=0, zorder=2)


def alarm_rates(ax, x, p, color, half_width, side='both', top=100):
    zero = np.mean(p == 0)
    if zero < 1:
        edges = np.linspace(0, top, BINS + 1)
        counts, _ = np.histogram(np.minimum(p[p > 0], top), bins=edges)
        density = smooth(counts.astype(float))
        occupied = np.flatnonzero(counts)
        density[:occupied[0]] = 0
        density[occupied[-1] + 1:] = 0
        draw(ax, x, (edges[:-1] + edges[1:]) / 2, density, color, half_width * (1 - zero), side)
    if zero > 0:
        left, right = _span(x, half_width * zero, side)
        ax.fill_between([left, right], *zero_row(top), color=color, lw=0, zorder=2)


def alarm_axis(ax, label_alpha=True, top=100):
    low, high = zero_row(top)
    margin = 0.01 * top
    ax.axhspan(low - margin, high + margin, color='#f6f6f6', zorder=0)
    ax.axhline(100 * ALPHA, color='#333333', lw=0.8, ls='--', zorder=4)
    if label_alpha:
        ax.text(0.03, 100 * ALPHA, r'$\alpha$', transform=ax.get_yaxis_transform(), fontsize=8, va='bottom', ha='left',
                zorder=5)
    ax.set_ylim(low - margin, top + 2 * margin)
    ticks = np.linspace(0, top, 6)[1:]
    ax.set_yticks([(low + high) / 2, *ticks])
    ax.set_yticklabels(['0'] + [f'{t:g}' for t in ticks])
