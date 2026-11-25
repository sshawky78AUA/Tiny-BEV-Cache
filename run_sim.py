"""Run Tiny-BEV-Cache simulations with your own parameters.

Examples
--------
  python run_sim.py                                   # Tiny-BEV-Cache, default scenario, seed 0
  python run_sim.py --method noncoop --seeds 0 1 2    # another method, three seeds
  python run_sim.py --set N=400 Rbh_mbps=50 budget=0.1
  python run_sim.py --sweep cache_frac=0.05,0.15,0.3 --methods tinybev noncoop --seeds 0 1
  python run_sim.py --list-params                     # every parameter and its default
  python run_sim.py --list-methods

Every run prints the metrics of Section VI (mean over seeds, with 95% confidence
half-widths when more than one seed is given) and can save them to a CSV file.
The first run of a (seed, N, N_bg, T) combination calls SUMO to build the vehicle
traces (about 30 s); later runs reuse them from traces/.
"""
import argparse, ast, csv, itertools, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bevsim2 import Params, run          # noqa: E402
from mobility import simulate            # noqa: E402
from experiments2 import cfg             # noqa: E402  (tuned settings from results2/tuned.json)

# method name -> (simulator method, extra options, description). The tuned settings of the
# paper (results2/tuned.json) are added for each method; --set overrides them.
TUNED_KEY = {'deadline_blind': 'tinybev', 'oracle_hit': 'oracle', 'cmarl_coop': 'cmarl',
             'fixed_tier': 'lru_ft', 'max_tier': 'energy_agnostic'}
METHODS = {
    'tinybev':         ('tinybev', {}, 'Tiny-BEV-Cache (soft logit rule, nu = Params.nu)'),
    'tinybev_h':       ('tinybev', {'hard': True}, 'Tiny-BEV-Cache-H (near-deterministic rule)'),
    'tinybev_route':   ('tinybev_route', {}, 'Route look-ahead placement'),
    'agnostic':        ('agnostic', {}, 'Direction-agnostic demand'),
    'noncoop':         ('noncoop', {}, 'Non-cooperative caching'),
    'static':          ('static', {}, 'Static placement'),
    'coop_lfu':        ('coop_lfu', {}, 'Cooperative LFU'),
    'oracle':          ('oracle', {}, 'Full-information placement'),
    'oracle_hit':      ('oracle', {'leader': 'oracle_hit'}, 'Full information, hit maximization (benchmark)'),
    'deadline_blind':  ('tinybev', {}, 'Deadline-blind (no tier sets; sets infer_aware=False)'),
    'cmarl':           ('cmarl', {}, 'C-MARL with tier masks'),
    'cmarl_coop':      ('cmarl', {'leader': 'coop_lfu'}, 'C-MARL with Cooperative LFU caching'),
    'fixed_tier':      ('lru_ft', {}, 'Fixed tier with LRU caches'),
    'max_tier':        ('energy_agnostic', {}, 'Max tier (ignores the energy budget)'),
}

# (key in the simulator output, column header, description, scale)
METRICS = [('util', 'utility', 'utility per active request', 1), ('hit', 'hit%', 'hit ratio (%)', 100),
           ('sem', 'sem_hit%', 'semantic hit ratio (%)', 100), ('success', 'success%', 'correct and on time (%)', 100),
           ('deadline', 'deadline%', 'deadlines met (%)', 100), ('backhaul', 'bh_MB/s', 'backhaul load (MB/s)', 1),
           ('lat_mean', 'lat_s', 'mean latency (s)', 1e-3), ('lat_p95', 'lat95_s', '95th-percentile latency (s)', 1e-3),
           ('energy_ratio', 'E/Eb', 'energy used / budget, mean vehicle', 1),
           ('energy_ratio_max', 'E/Eb_max', 'energy used / budget, worst vehicle', 1)]

HELP = {
    'N': 'equipped vehicles', 'N_bg': 'background vehicles (queues only)', 'T': 'slots (1 s each)',
    'warm': 'warm-up slots excluded from metrics', 'zone': 'request zone radius (m)', 'tau_min': 'min delivery time (s)',
    'S': 'sectors (assets) per RSU', 'zipf': 'Zipf exponent of intents', 'beta': 'share of errors on an adjacent sector',
    'b_lo': 'min asset size (MB)', 'b_hi': 'max asset size (MB)', 'live_rsu': 'share of RSUs with sensors',
    'live_frac': 'share of live sectors at those RSUs', 'cache_frac': 'cache size (share of map volume)',
    'a_lo': 'min activity probability', 'a_hi': 'max activity probability', 'hw': "classifier hardware: 'h7' or 'n6'",
    'lam1': 'weight of accuracy', 'lam2': 'weight of delivery quality', 'Rbh': 'backhaul rate per RSU (MB/s); or use Rbh_mbps',
    'Rx2': 'RSU-RSU link (MB/s)', 'budget': 'energy budget / (a_n E_max)', 'V': 'Lyapunov trade-off V', 'nu': 'entropy weight nu',
    'theta': 'budget tightening theta', 'T_ref': 'refresh period (s)', 'Tc': 'placement period (s)', 'W': 'look-ahead window (s)',
    'share': 'route pairs shared (1 = next RSU)', 'regime': 'slots between demand shifts', 'shift_frac': 'share of demand that shifts',
    'rho': 'correlated intents', 'infer_aware': 'use deadline tier sets', 'deconv': 'deconvolve reported intents',
    'task_reward': 'quality counts only if correct', 'busy': 'no request while MCU busy', 'H_ins': 'insertion horizon T_h (s); None = T_ref',
    'tau_err': 'relative error of deadline estimate', 'fifo': 'FIFO backhaul queue',
}


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    if 'Rbh_mbps' in out:
        out['Rbh'] = out.pop('Rbh_mbps') / 8
    return out


def one(args):
    meth, seed, pkw = args
    m, opt, _ = METHODS[meth]
    tp, to = cfg(TUNED_KEY.get(meth, meth))
    pkw = dict(tp, **pkw)
    if meth == 'deadline_blind':
        pkw = dict(pkw, infer_aware=False)
    o = run(Params(**pkw), seed, m, **dict(to, **opt))
    return meth, seed, {k: float(o[k]) for k, _, _, _ in METRICS if k in o}


def summary(vals):
    v = np.array(vals)
    if len(v) < 2:
        return v.mean(), 0.0
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', default=None, help='one method (default tinybev)')
    ap.add_argument('--methods', nargs='+', default=None, help='several methods')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--set', nargs='+', metavar='KEY=VALUE', help='override parameters, e.g. N=400 budget=0.1')
    ap.add_argument('--sweep', metavar='KEY=V1,V2,...', help='vary one parameter over a list of values')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--csv', help='save per-run results to this CSV file')
    ap.add_argument('--list-params', action='store_true')
    ap.add_argument('--list-methods', action='store_true')
    a = ap.parse_args()

    if a.list_params:
        P = Params()
        for k, v in vars(P).items():
            if k in ('M', 'acc', 'd', 'energy'):
                continue
            print(f'  {k:<12} = {v!r:<12} {HELP.get(k, "")}')
        print('\n  Rbh_mbps     (alias)      backhaul rate per RSU in Mb/s; sets Rbh = value / 8')
        return
    if a.list_methods:
        for k, (_, _, d) in METHODS.items():
            print(f'  {k:<15} {d}')
        return

    methods = a.methods or [a.method or 'tinybev']
    for m in methods:
        if m not in METHODS:
            sys.exit(f'unknown method {m}; see --list-methods')
    base = parse_kv(a.set)
    bad = [k for k in base if not hasattr(Params(), k)]
    if bad:
        sys.exit(f'unknown parameter(s) {bad}; see --list-params')
    points = [({}, '')]
    if a.sweep:
        k, vs = a.sweep.split('=', 1)
        points = [(parse_kv([f'{k}={v}']), f'{k}={v}') for v in vs.split(',')]

    jobs, keys = [], []
    for (extra, tag), m in itertools.product(points, methods):
        pk = dict(base, **extra)
        for s in a.seeds:
            jobs.append((m, s, pk)); keys.append(tag)
    t0 = time.time()
    # build each SUMO trace once, before the parallel runs share it
    for tr in sorted({(s, Params(**pk).N, Params(**pk).N_bg, Params(**pk).T) for _, s, pk in jobs}):
        simulate(*tr)
    with Pool(min(a.workers, len(jobs))) as p:
        res = p.map(one, jobs, chunksize=1)
    print(f'{len(jobs)} runs in {time.time() - t0:.0f} s\n')

    rows = []
    for tag, (m, s, r) in zip(keys, res):
        rows.append(dict(point=tag, method=m, seed=s, **r))
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print('saved', a.csv)

    head = f"{'point':<18}{'method':<15}" + ''.join(f'{h:>16}' for _, h, _, _ in METRICS)
    print(head)
    for (extra, tag), m in itertools.product(points, methods):
        sel = [r for r in rows if r['point'] == tag and r['method'] == m]
        cells = []
        for k, _, _, sc in METRICS:
            mu, hw = summary([r[k] * sc for r in sel if k in r])
            cells.append(f'{mu:9.3f}' + (f'±{hw:.3f}' if hw else '      '))
        print(f'{tag:<18}{m:<15}' + ''.join(f'{c:>16}' for c in cells))
    print('\nColumns: ' + '; '.join(f'{h} = {d}' for _, h, d, _ in METRICS))


if __name__ == '__main__':
    main()
