"""Experiment suite for the journal version (Section VI).

Stage 'traces': SUMO traces for all seeds and densities used below.
Stage 'tune'  : per-method tuning on seeds 100-102 (never used for evaluation).
Stage 'main'  : 30 seeds, all methods, default scenario.
Stage 'sweeps': budget, backhaul x density, cache size, disclosure, T_ref, V, hardware, robustness.
Results are appended to results2/<stage>.json as (exp, method, x, seed, metrics).
"""
import sys, os, json, itertools
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevsim2 import Params, run
from mobility import simulate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results2')
os.makedirs(OUT, exist_ok=True)
KEYS = ['util', 'hit', 'hit_map', 'correct', 'sem', 'deadline', 'backhaul', 'lat_mean', 'lat_p95',
        'energy_ratio_max', 'energy_ratio', 'Zbar', 'nb_frac', 'age_mean', 'bh_miss', 'bh_ref', 'bh_ins']
TUNE_SEEDS = (100, 101, 102)
EVAL = list(range(30))
SWEEP = list(range(10))


def best_cfg():
    f = os.path.join(OUT, 'tuned.json')
    return json.load(open(f)) if os.path.exists(f) else {}


METHODS = ['tinybev', 'tinybev_h', 'tinybev_route', 'agnostic', 'noncoop', 'static', 'coop_lfu', 'oracle',
           'cmarl', 'lru_ft', 'energy_agnostic']


def job(a):
    exp, meth, x, seed, pkw, okw = a
    m = 'tinybev' if meth == 'tinybev_h' else meth
    if meth == 'tinybev_h':
        okw = dict(okw, hard=True)
    o = run(Params(**pkw), seed, m, **okw)
    rec = {k: float(o[k]) for k in KEYS}
    rec['bins'] = o['bins']
    return exp, meth, x, seed, rec


def cfg(meth, **extra):
    """parameters for a method: tuned values merged with the experiment's own."""
    T = best_cfg().get(meth, {})
    pkw = {k: v for k, v in T.items() if k in ('gamma', 'gamma_L')}
    okw = {k: v for k, v in T.items() if k in ('lr_q', 'eps0', 'alpha_mu', 'eta_scale')}
    pkw.update(extra)
    return pkw, okw


def go(jobs, name):
    with Pool(int(os.environ.get('WORKERS', os.cpu_count()))) as p:
        out = p.map(job, jobs, chunksize=1)
    json.dump(out, open(os.path.join(OUT, name + '.json'), 'w'))
    print(name, len(out), 'runs')


def tr(args):
    simulate(*args)


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'traces':
        need = [(s, 200, 400, 6000) for s in list(TUNE_SEEDS) + EVAL]
        need += [(s, n, 400, 6000) for s in SWEEP for n in (100, 400)]
        with Pool(int(os.environ.get('WORKERS', os.cpu_count()))) as p:
            p.map(tr, need, chunksize=1)
        print('traces done')
    elif stage == 'tune':
        jobs = []
        for s in TUNE_SEEDS:
            for g in (0.005, 0.01, 0.02, 0.05):
                jobs.append(('tune', 'agnostic', dict(gamma=g), s, dict(gamma=g), {}))
                jobs.append(('tune', 'coop_lfu', dict(gamma=g), s, dict(gamma=g), {}))
                for gl in (0.1, 0.2, 0.5):
                    for m in ('tinybev', 'tinybev_route', 'noncoop', 'static'):
                        jobs.append(('tune', m, dict(gamma=g, gamma_L=gl), s, dict(gamma=g, gamma_L=gl), {}))
            for lr, e0, du, es in itertools.product((0.2, 0.5, 1.0), (0.2, 0.5), (0.05, 0.2, 1.0), (10.0, 20.0, 40.0)):
                k = dict(lr_q=lr, eps0=e0, alpha_mu=du, eta_scale=es)
                jobs.append(('tune', 'cmarl', k, s, {}, k))
        go(jobs, 'tune')
        R = json.load(open(os.path.join(OUT, 'tune.json')))
        best = {}
        for m in set(r[1] for r in R):
            cand = {}
            for e, mm, x, s, rec in R:
                if mm != m:
                    continue
                cand.setdefault(json.dumps(x, sort_keys=True), []).append(rec)
            def score(recs):
                ok = all(r['energy_ratio_max'] <= 1.0 for r in recs)
                if m == 'cmarl':                     # learning baseline: highest utility within budget
                    return (ok, np.mean([r['util'] for r in recs]))
                return (True, -np.mean([r['backhaul'] for r in recs]))   # leaders: lowest backhaul
            bx = max(cand, key=lambda c: score(cand[c]))
            best[m] = json.loads(bx)
        for m in ('tinybev', 'tinybev_h', 'energy_agnostic', 'oracle'):
            best.setdefault(m, best.get('tinybev', {}))
        best['tinybev_h'] = best['tinybev']; best['energy_agnostic'] = best['tinybev']; best['oracle'] = best['tinybev']
        best['lru_ft'] = {}
        json.dump(best, open(os.path.join(OUT, 'tuned.json'), 'w'), indent=1)
        print(best)
    elif stage == 'main':
        jobs = [('main', m, 0, s, *cfg(m)) for m in METHODS for s in EVAL]
        go(jobs, 'main')
    elif stage == 'sweeps':
        J = []
        for x in (0.06, 0.1, 0.2, 0.4, 0.8):                       # energy budget
            for m in ('tinybev', 'cmarl', 'lru_ft', 'energy_agnostic'):
                J += [('budget', m, x, s, *cfg(m, budget=x)) for s in SWEEP]
        for rb, n in itertools.product((50, 100, 200, 500, 1000), (100, 200, 400)):   # backhaul x density
            for m in ('tinybev', 'noncoop', 'coop_lfu', 'oracle'):
                J += [('bh_dens', m, [rb, n], s, *cfg(m, Rbh=rb / 8, N=n)) for s in SWEEP]
        for x in (0.05, 0.1, 0.15, 0.2, 0.3):                      # cache size
            for m in ('tinybev', 'agnostic', 'noncoop', 'static', 'coop_lfu', 'oracle', 'cmarl', 'lru_ft'):
                J += [('cache', m, x, s, *cfg(m, cache_frac=x)) for s in SWEEP]
        for x in (60, 150, 300, 600, 900):                         # freshness
            J += [('tref', 'tinybev', x, s, *cfg('tinybev', T_ref=x)) for s in SWEEP]
        for x in (0.25, 0.5, 1, 2, 5, 10, 20):                     # Lyapunov V
            for m in ('tinybev', 'tinybev_h'):
                J += [('V', m, x, s, *cfg(m, V=x)) for s in SWEEP]
        for hw in ('h7', 'n6'):                                    # hardware class
            for m in ('tinybev', 'energy_agnostic', 'lru_ft', 'cmarl'):
                J += [('hw', m, hw, s, *cfg(m, hw=hw)) for s in SWEEP]
            J += [('hw', 'tinybev_noinf', hw, s, *cfg('tinybev', hw=hw, infer_aware=False)) for s in SWEEP]
        for x in (0.0, 0.25, 0.5):                                 # correlated intents
            for m in ('tinybev', 'agnostic', 'coop_lfu', 'oracle'):
                J += [('rho', m, x, s, *cfg(m, rho=x)) for s in SWEEP]
        for x in (0, 1, 2, 3, 5, 99):                              # route disclosure (0 = history only)
            m = 'tinybev' if x == 0 else 'tinybev_route'
            J += [('share', m, x, s, *cfg(m, share=max(x, 1))) for s in SWEEP]
        J = [(e, ('tinybev' if m == 'tinybev_noinf' else m), x, s, p, o) if m != 'tinybev_noinf' else (e, 'tinybev', 'noinf-' + str(x), s, p, o)
             for (e, m, x, s, p, o) in J]
        part = sys.argv[2] if len(sys.argv) > 2 else 'all'
        if part == 'a':
            J = [j for j in J if j[0] in ('budget', 'bh_dens')]
        elif part == 'b':
            J = [j for j in J if j[0] not in ('budget', 'bh_dens')]
        go(J, 'sweeps_' + part)
