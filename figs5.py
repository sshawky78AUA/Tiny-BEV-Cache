"""Figures of the revised Section VI: paired gains over Non-cooperative, utility-energy trade-off,
latency CDFs, and time-domain behaviour.  Usage: python3 figs5.py follower budget bh leader time"""
import sys, os, json, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from figs2 import STY, OUT, HERE
from figs4 import load as load4, ci, band, series, top_legend, FILES
from experiments3 import FINE_TB, RBH, NS

RES = os.path.join(HERE, 'results4')
plt.rcParams.update({'legend.fontsize': 7.5})
GRAY = '#8a8984'


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(HERE, name + '.png'), dpi=160, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig); print('wrote', name)


def loads(name):
    """{(method, x-json): {seed: record}} to form paired differences."""
    D = defaultdict(dict)
    files, ex = FILES.get(name, ([name], None))
    for fn in files:
        for line in open(os.path.join(RES, fn + '.jsonl')):
            e, m, x, s, r = json.loads(line)
            if ex is None or e == ex:
                D[(m, json.dumps(x))][s] = r
    return D


def paired(D, m, ref, xs, key, sc=1.0, tag=lambda x: x, rel=False):
    mu, hw = [], []
    for x in xs:
        A, B = D[(m, json.dumps(tag(x)))], D[(ref, json.dumps(tag(x)))]
        ss = sorted(set(A) & set(B))
        d = np.array([(A[s][key] - B[s][key]) / (B[s][key] if rel else 1.0) for s in ss]) * sc
        a, b = ci(d); mu.append(a); hw.append(b)
    return mu, hw


def label_panels(ax):
    for a, t in zip(ax, '(a) (b) (c) (d)'.split()):
        a.set_xlabel(a.get_xlabel() + '\n' + t)


# ------------------------------------------------------------------ Fig. 2: deadline-aware tiers
def fig_follower():
    D = load4('hw'); Dv = load4('V')
    edges = np.array(FINE_TB[:-1]); cen = list((edges[:-1] + edges[1:]) / 2) + [15.0]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3))
    for sty, m, x in (('tinybev', 'tinybev', 'h7'), ('tinybev_noinf', 'tinybev', 'noinf-h7'), ('cmarl', 'cmarl', 'h7'),
                      ('lru_ft', 'lru_ft', 'h7'), ('energy_agnostic', 'energy_agnostic', 'h7')):
        rs = D[(m, json.dumps(x))]
        n = np.array([r['bins']['n'] for r in rs], float)
        for i, key, f in ((0, 'tier', lambda v: v), (1, 'dl', lambda v: 100 * (1 - v))):
            per = f(np.array([r['bins'][key] for r in rs]) / np.where(n > 0, n, np.nan))
            k_ = np.sum(n > 0, 0); mu = np.nanmean(per, 0)
            hw = stats.t.ppf(0.975, np.maximum(k_ - 1, 1)) * np.nanstd(per, 0, ddof=1) / np.sqrt(np.maximum(k_, 1))
            if i == 1:
                hw = np.where(mu < 0.03, 0, np.minimum(hw, mu * 0.9)); mu = np.maximum(mu, 0.03)
            band(ax[i], cen, mu, hw, sty, label=(i == 0), every=2 if i == 0 else 1)
    for a in ax[:2]:
        a.set_xticks([1, 3, 5, 7, 9, 11, 13, 15]); a.set_xticklabels(['1', '3', '5', '7', '9', '11', '13', r'$>$13'])
        a.set_xlabel(r'Remaining contact time $\tau_n(t)$ (s)')
    ax[0].set_ylabel('Mean tier $k_n(t)$'); ax[0].set_yticks([1, 2, 3, 4])
    ax[1].set_yscale('log'); ax[1].set_ylim(0.025, 150); ax[1].set_xlim(0.8, 7.5)
    ax[1].set_xticks([1, 2, 3, 4, 5, 6, 7]); ax[1].set_xticklabels(['1', '2', '3', '4', '5', '6', '7'])
    ax[1].set_yticks([0.1, 1, 10, 100]); ax[1].set_yticklabels(['$\\leq$0.1', '1', '10', '100']); ax[1].minorticks_off()
    ax[1].set_ylabel('Deadlines missed (%)')
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
    label_panels(ax)
    h0, l0 = ax[0].get_legend_handles_labels(); h2, l2 = ax[2].get_legend_handles_labels()
    top_legend(fig, h0 + h2, l0 + l2, 3, 0.86)
    save(fig, 'fig_follower5')


# ------------------------------------------------------------------ Fig. 3: energy budgets
def fig_budget():
    D = load4('budget')
    for k_, v_ in load4('budgetH').items():
        D[k_] += [r for r in v_]
    xs = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for m in ('tinybev', 'tinybev_h', 'cmarl', 'lru_ft'):
        st = STY[m]
        ex = [np.mean([r['energy_ratio'] * x for r in D[(m, json.dumps(x))]]) for x in xs]
        ut = [ci([r['util'] for r in D[(m, json.dumps(x))]]) for x in xs]
        over = [np.mean([r['energy_ratio'] for r in D[(m, json.dumps(x))]]) > 1.01 for x in xs]
        o_ = np.argsort(ex); ex = list(np.array(ex)[o_]); ut = [ut[i] for i in o_]; over = [over[i] for i in o_]
        ax[0].plot(ex, [u for u, _ in ut], color=st['c'], ls=st['ls'], lw=1.5, label=st['lab'])
        for e_, (u, _), o in zip(ex, ut, over):
            ax[0].plot(e_, u, st['m'], color=st['c'], ms=4, mew=0.8,
                       mfc=(st['c'] if o else 'white') if st['m'] in 'oDs^v' else st['c'])
            if o:
                ax[0].plot(e_, u, 'o', ms=8, mfc='none', mec='#e34948', mew=0.8)
        mu, hw = series(D, m, xs, 'success', 100); band(ax[1], xs, mu, hw, m, label=False, every=2)
        mu, hw = series(D, m, xs, 'energy_ratio_max'); band(ax[2], xs, mu, hw, m, label=False, every=2)
    st = STY['energy_agnostic']; ra = D[('energy_agnostic', json.dumps(0.2))]
    ea = np.mean([r['energy_ratio'] * 0.2 for r in ra]); ua = np.mean([r['util'] for r in ra])
    ax[0].plot([ea], [ua], st['m'], color=st['c'], ms=5, label=st['lab'])
    ax[0].annotate('misses 21% of\ndeadlines', (ea, ua), textcoords='offset points', xytext=(0, -24), fontsize=6.5,
                   color='#52514e', ha='center')
    ax[0].plot([], [], 'o', ms=8, mfc='none', mec='#e34948', mew=0.8, label='Over budget')
    mu, hw = series(D, 'energy_agnostic', [0.2] * len(xs), 'success', 100, tag=lambda x: 0.2)
    band(ax[1], xs, mu, hw, 'energy_agnostic', label=False, every=2)
    mu = [np.mean([r['energy_ratio_max'] for r in ra]) * 0.2 / x for x in xs]
    band(ax[2], xs, mu, np.zeros(len(xs)), 'energy_agnostic', label=False, every=2)
    ax[0].set_xscale('log'); ax[0].set_xticks([0.05, 0.1, 0.2, 0.5, 1]); ax[0].set_xticklabels(['0.05', '0.1', '0.2', '0.5', '1'])
    ax[0].minorticks_off(); ax[0].set_xlabel(r'Energy used $\bar E_n^{\rm used}/(a_n E_{\max})$')
    ax[0].set_ylabel('Average utility per request')
    ax[1].set_ylabel('Success (%)'); ax[2].set_ylabel(r'Worst vehicle $\bar E_n^{\rm used}/\bar E_n$')
    ax[2].axhline(1.0, color='#0b0b0b', lw=0.8, zorder=0)
    ax[2].set_yscale('log'); ax[2].set_yticks([0.2, 0.5, 1, 2, 5, 10])
    ax[2].set_yticklabels(['0.2', '0.5', '1', '2', '5', '10'])
    for a in ax[1:]:
        a.set_xscale('log'); a.set_xticks([0.06, 0.1, 0.2, 0.5, 1.0]); a.set_xticklabels(['0.06', '0.1', '0.2', '0.5', '1'])
        a.minorticks_off(); a.set_xlim(0.055, 1.08); a.set_xlabel(r'Budget $\bar E_n/(a_n E_{\max})$')
    ax[2].minorticks_off()
    label_panels(ax)
    h, l = ax[0].get_legend_handles_labels(); top_legend(fig, h, l, 6)
    save(fig, 'fig_budget5')


# ------------------------------------------------------------------ Fig. 4: backhaul and density
def lat_cdf(ax, N, R, ms):
    for m in ms:
        fs = sorted(glob.glob(os.path.join(RES, 'lat', f'{m}_{N}_{R}_*.npy')))
        if not fs:
            continue
        v = np.concatenate([np.load(f) for f in fs]) / 1000
        q = np.linspace(0, 1, 400); xq = np.quantile(v, q)
        st = STY[m]
        ax.plot(xq, 100 * q, color=st['c'], ls=st['ls'], lw=1.4)
        qi = [0.5, 0.9, 0.99]; ax.plot(np.quantile(v, qi), [100 * a for a in qi], st['m'], color=st['c'], ms=4, mew=0.8,
                                        mfc='white' if st['m'] in 'oDs^v' else st['c'])
    ax.set_xlabel(f'Latency (s), $R_{{\\rm bh}}={R}$ Mb/s'); ax.set_ylabel('CDF (%)')


def lat_bars(ax, N, R, ms, legend=False):
    comp_c = ['#2a78d6', '#86b6ef', '#eb6834', '#1baf7a']
    comp_l = ['Inference $d_k$', 'Access link', 'Backhaul', 'Neighbor (X2)']
    xs = np.arange(len(ms))
    for i, m in enumerate(ms):
        fs = sorted(glob.glob(os.path.join(RES, 'latc', f'{m}_{N}_{R}_*.npy')))
        C = np.array([np.load(f) for f in fs])            # seeds x 4 (s)
        mu = C.mean(0); bot = 0.0
        for c in range(4):
            ax.bar(i, mu[c], bottom=bot, color=comp_c[c], width=0.62, edgecolor='white', lw=0.5,
                   label=comp_l[c] if (legend and i == 0) else None)
            bot += mu[c]
        tot = C.sum(1); h = stats.t.ppf(0.975, len(tot) - 1) * tot.std(ddof=1) / np.sqrt(len(tot))
        ax.errorbar(i, bot, yerr=h, color='#0b0b0b', capsize=2, lw=0.8)
        ax.text(i, bot + h + 0.01 * bot, f'{bot:.2f}', ha='center', va='bottom', fontsize=6.3, color='#0b0b0b')
    ax.set_xticks(xs); ax.set_xticklabels([{'tinybev': 'Tiny-BEV', 'noncoop': 'Non-coop.', 'coop_lfu': 'Coop. LFU',
                                            'oracle': 'Full-info.'}[m] for m in ms], fontsize=6.6, rotation=28, ha='right', rotation_mode='anchor')
    ax.set_xlabel(f'$R_{{\\rm bh}}={R}$ Mb/s, $N={N}$'); ax.set_ylabel('Mean latency (s)')
    ax.grid(axis='x', visible=False)


def fig_bh():
    D = loads('bh')
    for k, v in loads('dens').items():
        D[k].update(v)
    ms = ('tinybev', 'coop_lfu', 'oracle')
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15, 1]))
    for m in ms:
        mu, hw = paired(D, m, 'noncoop', RBH, 'lat_mean', -100, tag=lambda r: [r, 400], rel=True)
        band(ax[0], RBH, mu, hw, m, every=2)
        mu, hw = paired(D, m, 'noncoop', NS, 'backhaul', -100, tag=lambda n: [200, n], rel=True)
        band(ax[3], NS, mu, hw, m, label=False)
    for a in (ax[0], ax[3]):
        a.axhline(0, color='#0b0b0b', lw=0.7, zorder=0)
    ax[0].set_xscale('log'); ax[0].set_xticks([25, 100, 500, 2000]); ax[0].set_xticklabels(['25', '100', '500', '2k'])
    ax[0].minorticks_off(); ax[0].set_xlabel(r'$R_{\rm bh}$ (Mb/s), $N=400$')
    ax[0].set_ylabel('Latency reduction vs.\nNon-cooperative (%)')
    ms4 = ('tinybev', 'noncoop', 'coop_lfu', 'oracle')
    lat_bars(ax[1], 400, 200, ms4, legend=True); ax[1].set_ylim(0, 1.5)
    lat_bars(ax[2], 400, 50, ms4); ax[2].set_ylim(0, 3.5)
    ax[3].set_xticks([50, 200, 300, 400]); ax[3].set_xlabel(r'$N$, $R_{\rm bh}=200$ Mb/s')
    ax[3].set_ylabel('Backhaul saved vs.\nNon-cooperative (%)')
    label_panels(ax)
    h = [plt.Line2D([], [], color=STY[m]['c'], ls=STY[m]['ls'], marker=STY[m]['m'], ms=4, mfc='white' if STY[m]['m'] in 'oDs^v' else STY[m]['c'])
         for m in ('tinybev', 'coop_lfu', 'oracle')]
    hb, lb = ax[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.2)
    fig.legend(h + hb, [STY[m]['lab'] for m in ('tinybev', 'coop_lfu', 'oracle')] + lb, loc='upper center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.2)
    save(fig, 'fig_bh5')


# ------------------------------------------------------------------ Fig. 5: cache size and freshness
def fig_leader():
    D = loads('cache'); Dt = load4('tref')
    xs = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]; xp = [100 * x for x in xs]
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3))
    for m in ('tinybev', 'coop_lfu', 'oracle', 'static', 'cmarl'):
        mu, hw = paired(D, m, 'noncoop', xs, 'hit', 100); band(ax[0], xp, mu, hw, m, every=2)
        mu, hw = paired(D, m, 'noncoop', xs, 'backhaul', -1); band(ax[1], xp, mu, hw, m, False, 2)
    for a in ax[:2]:
        a.axhline(0, color='#0b0b0b', lw=0.7, zorder=0)
        a.set_xticks([0, 10, 20, 30, 40, 50]); a.set_xlabel(r'Cache size $C_m$ (% of map-based volume)')
    ax[0].set_ylabel('Hit-ratio gain over\nNon-cooperative (points)')
    ax[1].set_ylabel('Backhaul saved vs.\nNon-cooperative (MB/s)')
    Ts = [30, 60, 100, 150, 200, 300, 450, 600, 900, 1200, 1800]
    ag, _ = series(Dt, 'tinybev', Ts, 'age_mean'); bh, bw = series(Dt, 'tinybev', Ts, 'backhaul')
    TM = defaultdict(list)
    fn = os.path.join(RES, 'tref_more.jsonl')
    if os.path.exists(fn):
        for l in open(fn):
            m_, s_, T_, r_ = json.loads(l); TM[(m_, T_)].append(r_)
    for m in ('noncoop', 'coop_lfu', 'oracle'):
        T2 = sorted({T for (mm_, T) in TM if mm_ == m})
        if not T2:
            continue
        st = STY[m]
        a2 = [np.mean([r['age_mean'] for r in TM[(m, T)]]) for T in T2]
        b2 = [ci([r['backhaul'] for r in TM[(m, T)]]) for T in T2]
        ax[2].errorbar(a2, [x for x, _ in b2], yerr=[y for _, y in b2], color=st['c'], ls=st['ls'], marker=st['m'],
                       markerfacecolor='white' if st['m'] in 'oDs^v' else st['c'], markersize=4, lw=1.3, capsize=0, elinewidth=0.8)
    st = STY['tinybev']
    ax[2].errorbar(ag, bh, yerr=bw, color=st['c'], marker='o', markerfacecolor='white', markersize=4, lw=1.5,
                   capsize=0, elinewidth=0.8)
    for T, a_, b_ in zip(Ts, ag, bh):
        if T in (30, 60, 150, 300, 1800):
            off = (-2, -11) if T == 1800 else (-6, -3)
            ax[2].annotate(rf'${T}$ s', (a_, b_), textcoords='offset points', xytext=off, fontsize=6.5, color='#52514e',
                           ha='center' if T == 1800 else 'right')
    ax[2].set_xscale('log'); ax[2].set_xticks([10, 30, 100, 300, 1000]); ax[2].set_xticklabels(['10', '30', '100', '300', '1000'])
    ax[2].minorticks_off()
    ax[2].set_xlabel(r'Mean age of a served asset (s); labels: $T_{\rm ref}$'); ax[2].set_ylabel('Backhaul load (MB/s)')
    label_panels(ax)
    h, l = ax[0].get_legend_handles_labels()
    st = STY['noncoop']
    h.insert(1, plt.Line2D([], [], color=st['c'], ls=st['ls'], marker=st['m'], ms=4, mfc='white')); l.insert(1, st['lab'] + ' (reference)')
    top_legend(fig, h, l, 6, 0.86)
    save(fig, 'fig_leader5')


# ------------------------------------------------------------------ Fig. 6: time domain
def fig_time():
    from bevsim2 import Params
    P = Params()
    fs = sorted(glob.glob(os.path.join(RES, 'ts', 'tinybev_*.npz')))
    z = np.load(os.path.join(RES, 'ts', 'tinybev_1.npz'))
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))
    # (a): one vehicle, slack tau_n - tau_min against the inference time d_k of each chosen tier
    tau, k, Z, zone = z['tau'], z['k'], z['Z'], z['zone']
    en = np.array(P.energy); dk = np.array(P.d)
    n, T0, W = 116, 3198, 52; tt = np.arange(T0, T0 + W)   # seed 1: an approach, a red light, a slow restart
    sl = np.where(zone[tt, n], np.minimum(tau[tt, n], 60) - P.tau_min, np.nan)
    ax[0].step(tt - T0, sl, where='post', color=GRAY, lw=1.0, label=r'$\tau_n(t)-\tau_{\min}$')
    kk = k[tt, n]
    for kv, mk in zip(range(4), 'o^sD'):
        sel = kk == kv
        ax[0].plot((tt - T0)[sel], np.full(sel.sum(), dk[kv]), mk, color=STY['tinybev']['c'], ms=4, mfc='white',
                   mew=0.9, label=f'$d_{kv + 1}$')
    ax[0].set_yscale('log'); ax[0].set_ylim(0.08, 200); ax[0].set_yticks([0.1, 0.3, 1, 3, 10])
    ax[0].set_yticklabels(['0.1', '0.3', '1', '3', '10']); ax[0].minorticks_off()
    ax[0].axvspan(13.5, 42, color='#e4e3df', alpha=0.6, lw=0, zorder=0)
    ax[0].text(28, 0.55, 'stopped at a red light', ha='center', fontsize=6.5, color='#52514e')
    ax[0].set_ylabel('Time (s)'); ax[0].set_xlabel('Time (s), one vehicle')
    ax[0].legend(loc='upper left', ncol=5, fontsize=6.5, frameon=True, framealpha=0.9, handlelength=1.2,
                 columnspacing=0.6, borderpad=0.3, handletextpad=0.3, bbox_to_anchor=(0.0, 1.0))
    # (b): running average energy over budget, every vehicle of every seed
    allr = []
    for f in fs:
        d = np.load(f); kk = d['k'].astype(int); Zs = d['Z']
        E = np.where(kk >= 0, en[np.maximum(kk, 0)], 0.0)
        # budget from the queue update Z(t+1) = Z(t) + E(t) - Ebp whenever Z(t+1) > 0
        Ebp = np.full(E.shape[1], np.nan)
        for j in range(E.shape[1]):
            pos = np.where(Zs[1:, j] > 1e-9)[0]
            if len(pos):
                t_ = pos[0]; Ebp[j] = Zs[t_, j] + E[t_ + 1, j] - Zs[t_ + 1, j] if False else Zs[t_, j] + E[t_ + 1, j] - Zs[t_ + 1, j]
        Eb = Ebp / (1 - P.theta)
        run = np.cumsum(E, 0) / np.arange(1, E.shape[0] + 1)[:, None] / Eb[None]
        allr.append(run[:, np.isfinite(Eb)])
    R = np.concatenate(allr, 1); tm = np.arange(1, R.shape[0] + 1) / 60
    rng = np.random.default_rng(1)
    for j in rng.choice(R.shape[1], 25, replace=False):
        ax[1].plot(tm, R[:, j], color='#b9c9e0', lw=0.6)
    ax[1].plot(tm, np.percentile(R, 99, 1), color=STY['tinybev_h']['c'], ls='--', lw=1.3, label='99th percentile')
    ax[1].plot(tm, R.mean(1), color=STY['tinybev']['c'], lw=1.6, label='Mean')
    ax[1].axhline(1.0, color='#0b0b0b', lw=0.8); ax[1].set_ylim(0, 2.5)
    ax[1].set_xlabel('Time (min)'); ax[1].set_ylabel(r'Running $\bar E_n^{\rm used}(t)/\bar E_n$')
    ax[1].legend(loc='upper right', fontsize=6.5, frameon=True, framealpha=0.9, handlelength=1.6)
    # (c): hit ratio over time around demand shifts, 60-s windows, mean over seeds
    Wn = 60
    for m in ('tinybev', 'noncoop', 'oracle', 'static'):
        H = []
        for f in sorted(glob.glob(os.path.join(RES, 'ts', f'{m}_*.npz'))):
            d = np.load(f); h = d['hit'].reshape(-1, Wn).sum(1); r = d['req'].reshape(-1, Wn).sum(1)
            H.append(100 * h / np.maximum(r, 1))
        H = np.array(H); tw = (np.arange(H.shape[1]) + 0.5) * Wn
        mu = H.mean(0); hw = stats.t.ppf(0.975, len(H) - 1) * H.std(0, ddof=1) / np.sqrt(len(H))
        band(ax[2], tw / 60, mu, hw, m, every=10)
    for s_ in (1500, 3000, 4500):
        ax[2].axvline(s_ / 60, color='#52514e', lw=0.7, ls=':', zorder=0)
    ax[2].axvspan(0, P.warm / 60, color='#e4e3df', alpha=0.6, lw=0, zorder=0)
    ax[2].set_xlabel('Time (min), shifts dotted'); ax[2].set_ylabel('Hit ratio (%), 60-s windows')
    label_panels(ax)
    h, l = ax[2].get_legend_handles_labels(); top_legend(fig, h, l, 4)
    save(fig, 'fig_time5')


if __name__ == '__main__':
    F = {'follower': fig_follower, 'budget': fig_budget, 'bh': fig_bh, 'leader': fig_leader, 'time': fig_time}
    for k in (sys.argv[1:] or F):
        F[k]()
