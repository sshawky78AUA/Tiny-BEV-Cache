"""Revised Section VI figures: denser sweeps, 20 seeds, shaded 95% confidence bands.
Usage: python3 figs3.py budget"""
import sys, os, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from figs2 import STY, OUT, HERE   # same colors, markers, fonts, and output folder

RES = os.path.join(HERE, 'results4')
FILES = {'bh': (['bh1', 'bh2', 'bh3'], 'bh'), 'dens': (['dens'], 'bh'), 'cache': (['cacheA', 'cacheB'], 'cache'),
         'hw': (['hwtref'], 'hw'), 'tref': (['hwtref'], 'tref')}


def load(name):
    D = defaultdict(list)
    files, ex = FILES.get(name, ([name], None))
    for fn in files:
        for line in open(os.path.join(RES, fn + '.jsonl')):
            e, m, x, s, r = json.loads(line)
            if ex is None or e == ex:
                D[(m, json.dumps(x))].append(r)
    return D


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return v.mean(), 0.0
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def band(ax, xs, mu, hw, m, label=True, every=1):
    st = STY[m]
    mu, hw = np.asarray(mu), np.asarray(hw)
    ax.fill_between(xs, mu - hw, mu + hw, color=st['c'], alpha=0.18, lw=0)
    ax.plot(xs, mu, color=st['c'], ls=st['ls'], lw=1.5, label=st['lab'] if label else None,
            marker=st['m'], markevery=every, markersize=4, markeredgewidth=0.8,
            markerfacecolor='white' if st['m'] in 'oDs^v' else st['c'])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(HERE, name + '.png'), dpi=160, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('wrote', name)


def fig_budget():
    D = load('budget')
    xs = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for m in ('tinybev', 'cmarl', 'lru_ft', 'energy_agnostic'):
        for i, (key, sc) in enumerate((('util', 1), ('lat_mean', 1e-3), ('energy_ratio_max', 1))):
            mu, hw = [], []
            for x in xs:
                if m == 'energy_agnostic':
                    rs = D[(m, json.dumps(0.2))]
                    f = 0.2 / x if key == 'energy_ratio_max' else 1.0   # budget-independent policy
                else:
                    rs = D[(m, json.dumps(x))]; f = 1.0
                a, b = ci([r[key] * sc * f for r in rs]); mu.append(a); hw.append(b)
            band(ax[i], xs, mu, hw, m, label=(i == 0), every=2)
    ax[2].axhline(1.0, color='#0b0b0b', lw=0.8, zorder=0)
    ax[2].set_yscale('log'); ax[2].set_yticks([0.2, 0.5, 1, 2, 5, 10])
    ax[2].set_yticklabels(['0.2', '0.5', '1', '2', '5', '10']); ax[2].minorticks_off()
    ylab = ['Average utility per request', 'Mean latency (s)', r'Worst vehicle $\bar E_n^{\rm used}/\bar E_n$']
    for a, yl, t in zip(ax, ylab, '(a) (b) (c)'.split()):
        a.set_xscale('log'); a.set_xticks([0.06, 0.1, 0.2, 0.5, 1.0])
        a.set_xticklabels(['0.06', '0.1', '0.2', '0.5', '1']); a.minorticks_off()
        a.set_xlim(0.055, 1.08)
        a.set_xlabel(r'Budget $\bar E_n/(a_n E_{\max})$' + '\n' + t); a.set_ylabel(yl)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.6)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
               columnspacing=1.4)
    save(fig, 'fig_budget4')


def series(D, m, xs, key, sc=1.0, tag=lambda x: x):
    out = [ci([r[key] * sc for r in D[(m, json.dumps(tag(x)))]]) for x in xs]
    return [a for a, _ in out], [b for _, b in out]


def top_legend(fig, handles, labels, ncol, rect_top=0.9):
    fig.tight_layout(rect=(0, 0, 1, rect_top), w_pad=1.4)
    fig.legend(handles, labels, loc='upper center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.4)


def fig_bh():
    D = load('bh')
    for k, v in load('dens').items():
        D[k] += v
    from experiments3 import RBH, NS
    ms = ('tinybev', 'noncoop', 'coop_lfu', 'oracle')
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for m in ms:
        mu, hw = series(D, m, RBH, 'lat_mean', 1e-3, tag=lambda r: [r, 400]); band(ax[0], RBH, mu, hw, m, every=2)
        mu, hw = series(D, m, RBH, 'deadline', 100, tag=lambda r: [r, 400]); band(ax[1], RBH, mu, hw, m, False, 2)
        mu, hw = series(D, m, NS, 'backhaul', 1, tag=lambda n: [200, n]); band(ax[2], NS, mu, hw, m, False)
    for a in ax[:2]:
        a.set_xscale('log'); a.set_xticks([25, 50, 100, 200, 500, 1000, 2000])
        a.set_xticklabels(['25', '50', '100', '200', '500', '1k', '2k']); a.minorticks_off()
        a.set_xlabel(r'Backhaul rate $R_{\rm bh}$ (Mb/s), $N=400$')
    ax[2].set_xticks(NS); ax[2].set_xlabel(r'Equipped vehicles $N$, $R_{\rm bh}=200$ Mb/s')
    ax[0].set_ylabel('Mean latency (s)'); ax[1].set_ylabel('Deadlines met (%)'); ax[2].set_ylabel('Backhaul load (MB/s)')
    for a, t in zip(ax, '(a) (b) (c)'.split()):
        a.set_xlabel(a.get_xlabel() + '\n' + t)
    h, l = ax[0].get_legend_handles_labels(); top_legend(fig, h, l, 4)
    save(fig, 'fig_bh4')


def fig_leader():
    D = load('cache'); Dt = load('tref')
    xs = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]; xp = [100 * x for x in xs]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.45))
    for m in ('tinybev', 'noncoop', 'static', 'coop_lfu', 'oracle', 'cmarl', 'lru_ft'):
        mu, hw = series(D, m, xs, 'hit', 100); band(ax[0], xp, mu, hw, m, every=2)
        mu, hw = series(D, m, xs, 'backhaul'); band(ax[1], xp, mu, hw, m, False, 2)
    for a in ax[:2]:
        a.set_xticks([0, 10, 20, 30, 40, 50]); a.set_xlabel(r'Cache size $C_m$ (% of map-based volume)')
    ax[0].set_ylabel('Hit ratio (%)'); ax[1].set_ylabel('Backhaul load (MB/s)')
    Ts = [30, 60, 100, 150, 200, 300, 450, 600, 900, 1200, 1800]
    ag, _ = series(Dt, 'tinybev', Ts, 'age_mean'); bh, bw = series(Dt, 'tinybev', Ts, 'backhaul')
    st = STY['tinybev']
    ax[2].errorbar(ag, bh, yerr=bw, color=st['c'], marker='o', markerfacecolor='white', markersize=4, lw=1.5,
                   capsize=0, elinewidth=0.8)
    for T, a_, b_ in zip(Ts, ag, bh):
        if T in (30, 60, 150, 300, 1800):
            off = (-2, 7) if T == 1800 else (5, 4)
            ax[2].annotate(rf'${T}$ s', (a_, b_), textcoords='offset points', xytext=off, fontsize=7, color='#52514e',
                           ha='center' if T == 1800 else 'left')
    ax[2].set_xscale('log'); ax[2].set_xticks([10, 30, 100, 300, 1000]); ax[2].set_xticklabels(['10', '30', '100', '300', '1000'])
    ax[2].minorticks_off()
    ax[2].set_xlabel(r'Mean age of a served cached asset (s), labels: $T_{\rm ref}$'); ax[2].set_ylabel('Backhaul load (MB/s)')
    for a, t in zip(ax, '(a) (b) (c)'.split()):
        a.set_xlabel(a.get_xlabel() + '\n' + t)
    h, l = ax[0].get_legend_handles_labels(); top_legend(fig, h, l, 4, 0.84)
    save(fig, 'fig_leader4')


def fig_follower():
    from experiments3 import FINE_TB
    D = load('hw'); Dv = load('V')
    edges = np.array(FINE_TB[:-1]); cen = list((edges[:-1] + edges[1:]) / 2) + [15.0]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.45))
    for sty, m, x in (('tinybev', 'tinybev', 'h7'), ('tinybev_noinf', 'tinybev', 'noinf-h7'), ('cmarl', 'cmarl', 'h7'),
                      ('lru_ft', 'lru_ft', 'h7'), ('energy_agnostic', 'energy_agnostic', 'h7')):
        rs = D[(m, json.dumps(x))]
        n = np.array([r['bins']['n'] for r in rs], float)
        for i, key, sc in ((0, 'tier', 1), (1, 'dl', 100)):
            per = np.array([r['bins'][key] for r in rs]) / np.where(n > 0, n, np.nan) * sc   # per-seed bin means
            k_ = np.sum(n > 0, 0); mu = np.nanmean(per, 0)
            hw = stats.t.ppf(0.975, np.maximum(k_ - 1, 1)) * np.nanstd(per, 0, ddof=1) / np.sqrt(np.maximum(k_, 1))
            band(ax[i], cen, mu, hw, sty, label=(i == 0), every=2)
    for a in ax[:2]:
        a.set_xticks([1, 3, 5, 7, 9, 11, 13, 15]); a.set_xticklabels(['1', '3', '5', '7', '9', '11', '13', r'$>$13'])
        a.set_xlabel(r'Remaining contact time $\tau_n(t)$ (s)')
    ax[0].set_ylabel('Mean tier $k_n(t)$'); ax[0].set_yticks([1, 2, 3, 4]); ax[1].set_ylabel('Deadlines met (%)')
    Vs = [0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 20, 50]
    for m in ('tinybev', 'tinybev_h'):
        st = STY[m]
        zx, _ = series(Dv, m, Vs, 'Zbar'); uy, uw = series(Dv, m, Vs, 'util')
        ax[2].errorbar(zx, uy, yerr=uw, color=st['c'], ls=st['ls'], marker=st['m'], markerfacecolor='white',
                       markersize=4, lw=1.5, capsize=0, elinewidth=0.8, label=st['lab'] if m == 'tinybev_h' else None)
        if m == 'tinybev':
            for v, a_, b_ in zip(Vs, zx, uy):
                if v in (0.1, 1, 10, 50):
                    off, ha = ((-4, -10), 'right') if v == 0.1 else ((5, -11), 'left')
                    ax[2].annotate(f'$V={v:g}$', (a_, b_), textcoords='offset points', xytext=off, fontsize=7,
                                   color='#52514e', ha=ha)
    ax[2].set_xscale('log'); ax[2].set_xticks([0.1, 0.3, 1, 3, 10, 30]); ax[2].set_xticklabels(['0.1', '0.3', '1', '3', '10', '30'])
    ax[2].minorticks_off(); ax[2].set_xlabel(r'Average backlog $\frac{1}{T}\sum_t\bar Z(t)$')
    ax[2].set_ylabel('Average utility per request')
    for a, t in zip(ax, '(a) (b) (c)'.split()):
        a.set_xlabel(a.get_xlabel() + '\n' + t)
    h0, l0 = ax[0].get_legend_handles_labels(); h2, l2 = ax[2].get_legend_handles_labels()
    top_legend(fig, h0 + h2, l0 + l2, 3, 0.86)
    save(fig, 'fig_follower4')


if __name__ == '__main__':
    F = {'budget': fig_budget, 'bh': fig_bh, 'leader': fig_leader, 'follower': fig_follower}
    for k in (sys.argv[1:] or F):
        F[k]()
