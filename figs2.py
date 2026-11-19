"""Figures for Section VI (journal version). Usage: python3 figs2.py [a|b|all]"""
import sys, os, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results2')
OUT = os.path.join(HERE, 'figures'); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': 0.6,
    'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.5,
    'lines.linewidth': 1.6, 'lines.markersize': 4.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# fixed color, marker and line style per method (identity never by color alone)
STY = {
    'tinybev':         dict(c='#2a78d6', m='o', ls='-',  lab='Tiny-BEV-Cache'),
    'tinybev_h':       dict(c='#184f95', m='s', ls='--', lab='Tiny-BEV-Cache-H'),
    'tinybev_route':   dict(c='#1baf7a', m='D', ls='-',  lab='Route look-ahead'),
    'tinybev_noinf':   dict(c='#86b6ef', m='o', ls=':',  lab='Deadline-blind'),
    'agnostic':        dict(c='#008300', m='v', ls='-.', lab='Direction-agnostic'),
    'noncoop':         dict(c='#eb6834', m='^', ls='-',  lab='Non-cooperative'),
    'static':          dict(c='#e87ba4', m='x', ls='-.', lab='Static'),
    'coop_lfu':        dict(c='#1baf7a', m='D', ls='--', lab='Cooperative LFU'),
    'oracle':          dict(c='#52514e', m='*', ls=':',  lab='Full-information'),
    'cmarl':           dict(c='#6250d6', m='P', ls='-',  lab='C-MARL'),
    'lru_ft':          dict(c='#eda100', m='s', ls='-',  lab='Fixed tier'),
    'energy_agnostic': dict(c='#e34948', m='X', ls='--', lab='Max tier'),
}


def load(name):
    R = json.load(open(os.path.join(RES, name + '.json')))
    D = defaultdict(list)
    for e, m, x, s, r in R:
        D[(e, m, json.dumps(x))].append(r)
    return D


def stat(D, e, m, x, key, scale=1.0):
    rs = D.get((e, m, json.dumps(x)), [])
    v = np.array([r[key] for r in rs]) * scale
    if len(v) == 0:
        return np.nan, np.nan
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def line(ax, xs, D, e, m, key, scale=1.0, xmap=None, xplot=None, label=True):
    st = STY[m]
    mu, ci = zip(*[stat(D, e, m, xmap(x) if xmap else x, key, scale) for x in xs])
    xp = xplot if xplot is not None else xs
    ax.errorbar(xp, mu, yerr=ci, color=st['c'], marker=st['m'], ls=st['ls'], capsize=2, elinewidth=0.7,
                markeredgewidth=0.8, label=st['lab'] if label else None,
                markerfacecolor='white' if st['m'] in 'oDs^v' else st['c'])


def top_legend(fig, axes, ncol):
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.4)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(HERE, name + '.png'), dpi=160, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', name)


def fig_budget(D):
    xs = [0.06, 0.1, 0.2, 0.4, 0.8]
    ms = ['tinybev', 'cmarl', 'lru_ft', 'energy_agnostic']
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for m in ms:
        line(ax[0], xs, D, 'budget', m, 'util')
        line(ax[1], xs, D, 'budget', m, 'lat_mean', 1e-3, label=False)
        line(ax[2], xs, D, 'budget', m, 'energy_ratio_max', label=False)
    ax[2].axhline(1.0, color='#0b0b0b', lw=0.8, ls='-', zorder=0)
    ax[2].set_yscale('log'); ax[2].set_yticks([0.3, 0.5, 1, 2, 5, 10]); ax[2].set_yticklabels(['0.3', '0.5', '1', '2', '5', '10'])
    ylab = ['Average utility per request', 'Mean latency (s)', r'Worst vehicle $\bar E_n^{\rm used}/\bar E_n$']
    for a, yl, t in zip(ax, ylab, '(a) (b) (c)'.split()):
        a.set_xscale('log'); a.set_xticks(xs); a.set_xticklabels(['0.06', '0.1', '0.2', '0.4', '0.8'])
        a.minorticks_off(); a.set_xlabel(r'Budget $\bar E_n/(a_n E_{\max})$'); a.set_ylabel(yl)
        a.set_xlabel(a.get_xlabel() + '\n' + t)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.6)
    top_legend(fig, ax, 4)
    save(fig, 'fig_budget')


def fig_bh(D):
    rb = [50, 100, 200, 500, 1000]
    ms = ['tinybev', 'noncoop', 'coop_lfu', 'oracle']
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for m in ms:
        line(ax[0], rb, D, 'bh_dens', m, 'lat_mean', 1e-3, xmap=lambda r: [r, 400])
        line(ax[1], rb, D, 'bh_dens', m, 'deadline', 100, xmap=lambda r: [r, 400], label=False)
        line(ax[2], [100, 200, 400], D, 'bh_dens', m, 'backhaul', 1, xmap=lambda n: [200, n], label=False)
    for a in ax[:2]:
        a.set_xscale('log'); a.set_xticks(rb); a.set_xticklabels([str(r) for r in rb]); a.minorticks_off()
        a.set_xlabel(r'Backhaul rate $R_{\rm bh}$ (Mb/s), $N=400$')
    ax[2].set_xticks([100, 200, 400]); ax[2].set_xlabel(r'Equipped vehicles $N$, $R_{\rm bh}=200$ Mb/s')
    ax[0].set_ylabel('Mean latency (s)'); ax[1].set_ylabel('Deadlines met (%)')
    ax[2].set_ylabel('Backhaul load (MB/s)')
    for a, t in zip(ax, '(a) (b) (c)'.split()):
        a.set_xlabel(a.get_xlabel() + '\n' + t)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.6)
    top_legend(fig, ax, 4)
    save(fig, 'fig_bh')


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if part in ('a', 'all'):
        D = load('sweeps_a')
        fig_budget(D); fig_bh(D)


TBL = ['1–2', '2–3', '3–5', '5–8', '8–13', '>13']


def bins(D, e, m, x, key):
    rs = D[(e, m, json.dumps(x))]
    num = np.array([r['bins'][key] for r in rs]).sum(0); den = np.array([r['bins']['n'] for r in rs]).sum(0)
    return num / np.maximum(den, 1)


def fig_follower(D):
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.45))
    xb = np.arange(6)
    series = [('tinybev', 'h7', 'tinybev'), ('tinybev_noinf', 'noinf-h7', 'tinybev'), ('cmarl', 'h7', 'cmarl'),
              ('lru_ft', 'h7', 'lru_ft'), ('energy_agnostic', 'h7', 'energy_agnostic')]
    for sty, x, m in series:
        st = STY[sty]
        kw = dict(color=st['c'], marker=st['m'], ls=st['ls'], markeredgewidth=0.8,
                  markerfacecolor='white' if st['m'] in 'oDs^v' else st['c'])
        ax[0].plot(xb, bins(D, 'hw', m, x, 'tier'), label=st['lab'], **kw)
        ax[1].plot(xb, 100 * bins(D, 'hw', m, x, 'dl'), **kw)
    for a in ax[:2]:
        a.set_xticks(xb); a.set_xticklabels(TBL); a.set_xlabel(r'Remaining contact time $\tau_n(t)$ (s)')
    ax[0].set_ylabel('Mean tier $k_n(t)$'); ax[0].set_yticks([1, 2, 3, 4])
    ax[1].set_ylabel('Deadlines met (%)')
    Vs = [0.25, 0.5, 1, 2, 5, 10, 20]
    for m in ('tinybev', 'tinybev_h'):
        st = STY[m]
        zx = [stat(D, 'V', m, v, 'Zbar')[0] for v in Vs]; uy = [stat(D, 'V', m, v, 'util')[0] for v in Vs]
        ax[2].plot(zx, uy, color=st['c'], marker=st['m'], ls=st['ls'], markeredgewidth=0.8, markerfacecolor='white',
                   label=st['lab'] if m == 'tinybev_h' else None)
        if m == 'tinybev':
            for v, a_, b_ in zip(Vs, zx, uy):
                if v in (0.25, 1, 5, 20):
                    ax[2].annotate(f'$V={v:g}$', (a_, b_), textcoords='offset points', xytext=(9, -5) if v == 0.25 else (4, -10), fontsize=7,
                                   color='#52514e')
    ax[2].set_xscale('log'); ax[2].set_xticks([0.3, 1, 3, 10]); ax[2].set_xticklabels(['0.3', '1', '3', '10'])
    ax[2].minorticks_off(); ax[2].set_xlabel(r'Average backlog $\frac{1}{T}\sum_t\bar Z(t)$')
    ax[2].set_ylabel('Average utility per request')
    for a, t in zip(ax, '(a) (b) (c)'.split()):
        a.set_xlabel(a.get_xlabel() + '\n' + t)
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.4)
    h0, l0 = ax[0].get_legend_handles_labels(); h2, l2 = ax[2].get_legend_handles_labels()
    fig.legend(h0 + h2, l0 + l2, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.4)
    save(fig, 'fig_follower')


def fig_leader(D):
    xs = [0.05, 0.1, 0.15, 0.2, 0.3]
    ms = ['tinybev', 'noncoop', 'static', 'coop_lfu', 'oracle', 'cmarl', 'lru_ft']
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.45))
    for m in ms:
        line(ax[0], xs, D, 'cache', m, 'hit', 100, xplot=[100 * x for x in xs])
        line(ax[1], xs, D, 'cache', m, 'backhaul', xplot=[100 * x for x in xs], label=False)
    for a in ax[:2]:
        a.set_xticks([5, 10, 15, 20, 30]); a.set_xlabel(r'Cache size $C_m$ (% of map-based volume)')
    ax[0].set_ylabel('Hit ratio (%)'); ax[1].set_ylabel('Backhaul load (MB/s)')
    Ts = [60, 150, 300, 600, 900]; st = STY['tinybev']
    ag = [stat(D, 'tref', 'tinybev', T, 'age_mean')[0] for T in Ts]
    bh = [stat(D, 'tref', 'tinybev', T, 'backhaul')[0] for T in Ts]
    ax[2].plot(ag, bh, color=st['c'], marker=st['m'], ls=st['ls'], markerfacecolor='white', markeredgewidth=0.8)
    off = {60: (6, -3), 150: (6, 4), 300: (-6, -14), 600: (-14, 7), 900: (-24, -14)}
    for T, a_, b_ in zip(Ts, ag, bh):
        ax[2].annotate(rf'$T_{{\rm ref}}={T}$ s', (a_, b_), textcoords='offset points', xytext=off[T], fontsize=7,
                       color='#52514e')
    ax[2].set_xlabel('Mean age of a served cached asset (s)'); ax[2].set_ylabel('Backhaul load (MB/s)')
    ax[2].set_xlim(0, 360); ax[2].set_ylim(10.9, 14.1)
    for a, t in zip(ax, '(a) (b) (c)'.split()):
        a.set_xlabel(a.get_xlabel() + '\n' + t)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.4)
    top_legend(fig, ax, 4)
    save(fig, 'fig_leader')


if __name__ == '__main__' and (len(sys.argv) > 1 and sys.argv[1] in ('b', 'all')):
    D = load('sweeps_b')
    fig_follower(D); fig_leader(D)
