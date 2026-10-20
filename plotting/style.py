"""Figure style: serif fonts, DAB in blue, Direct in orange."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from core.config import RESULTS  # noqa: E402

DAB, DIRECT = '#2a78d6', '#eb6834'
DAB_FILL, DIRECT_FILL = '#aac8ee', '#f7c7a9'
ALARM_BG = '#f4e6e6'
WIDTH = 7.0                          # page width (inches)
LEGEND_GAP = 0.06                    # inches between the lowest axis label and the legend

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'axes.grid': True, 'grid.color': '#ececec', 'grid.linewidth': 0.5,
    'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 300, 'axes.titlepad': 3})


def legend_below(fig, handles, ncol=None, gap=LEGEND_GAP, **kwargs):
    """One legend row centred under the panels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lowest = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes) / fig.dpi
    y = (lowest - gap) / fig.get_figheight()
    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
                      handlelength=kwargs.pop('handlelength', 1.6), columnspacing=kwargs.pop('columnspacing', 1.4),
                      handletextpad=0.5, borderaxespad=0, **kwargs)


def save(fig, name):
    """results/figures/<name>.pdf and .png"""
    folder = os.path.join(RESULTS, 'figures')
    os.makedirs(folder, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(folder, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
