"""Revision after the internal review: task reward, intent deconvolution, busy vehicles, masked C-MARL.

Usage: python3 experiments4.py main | checks | <sweep>   (sweep as in experiments3: budget hw V cache tref dens bh)
One run per line in results4/<stage>.jsonl, so an interrupted stage resumes where it stopped.
"""
import sys, os, json
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevsim2 import Params, run
from experiments2 import cfg, METHODS, EVAL
import experiments3 as E3

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results4')
os.makedirs(OUT, exist_ok=True)
KEYS = ['util', 'hit', 'hit_map', 'correct', 'sem', 'success', 'deadline', 'backhaul', 'lat_mean', 'lat_p95',
        'energy_ratio_max', 'energy_ratio', 'Zbar', 'nb_frac', 'age_mean', 'bh_miss', 'bh_ref', 'bh_ins',
        'brgap_mean', 'brgap_p99', 'kappa_max', 'kappa_p99', 'act_per_slot']
ALIAS = {'tinybev_h': ('tinybev', {'hard': True}), 'tinybev_noinf': ('tinybev', {}),
         'cmarl_coop': ('cmarl', {'leader': 'coop_lfu'}), 'cmarl_nomask': ('cmarl', {'cmarl_mask': False})}
CHK = list(range(10))


def job(a):
    exp, meth, x, seed, pkw, okw = a
    m, extra = ALIAS.get(meth, (meth, {}))
    if meth == 'tinybev_noinf':
        pkw = dict(pkw, infer_aware=False)
    o = run(Params(**pkw), seed, m, **dict(okw, **extra))
    rec = {k: float(o[k]) for k in KEYS if k in o}
    rec['bins'] = o['bins']
    return exp, meth, x, seed, rec


def trace_job(seed):
    """Followers compared on one common placement trace (recorded from Tiny-BEV-Cache)."""
    pk, ok = cfg('tinybev')
    base = run(Params(**pk), seed, 'tinybev', record_x=True, **ok)
    tr = base.pop('xtrace')
    out = [('trace', 'tinybev', 0, seed, {k: float(base[k]) for k in KEYS if k in base})]
    for meth in ('tinybev_noinf', 'cmarl', 'cmarl_nomask', 'lru_ft', 'energy_agnostic'):
        m, extra = ALIAS.get(meth, (meth, {}))
        p2, o2 = cfg(m)
        if meth == 'tinybev_noinf':
            p2 = dict(p2, infer_aware=False)
        o = run(Params(**p2), seed, m, replay_x=tr, leader='hist', **dict(o2, **extra))
        out.append(('trace', meth, 0, seed, {k: float(o[k]) for k in KEYS if k in o}))
    return out


def jobs_main():
    J = [('main', m, 0, s, *cfg(m)) for m in METHODS for s in EVAL]
    J += [('main', 'tinybev_noinf', 0, s, *cfg('tinybev')) for s in EVAL]
    pc, _ = cfg('coop_lfu'); _, oc = cfg('cmarl')
    J += [('main', 'cmarl_coop', 0, s, pc, oc) for s in EVAL]
    J += [('main', 'cmarl_nomask', 0, s, *cfg('cmarl')) for s in EVAL]
    return J


def jobs_checks():
    J = []
    pk, ok = cfg('tinybev')
    J += [('kappa', 'tinybev', 'default', s, pk, dict(ok, kappa=True, brgap=True)) for s in CHK]
    for name, extra in (('N400_R50', dict(N=400, Rbh=50 / 8)), ('N400_R200', dict(N=400)), ('nu0.05', dict(nu=0.05))):
        p, o = cfg('tinybev', **extra)
        J += [('kappa', 'tinybev', name, s, p, dict(o, kappa=True, brgap=True)) for s in CHK[:5]]
    for e in (0.1, 0.2, 0.3):
        J += [('tau_err', 'tinybev', e, s, *cfg('tinybev', tau_err=e)) for s in CHK]
    for rb in (25, 50, 200):
        for m in ('tinybev', 'noncoop'):
            for f in (False, True):
                J += [('fifo', m, [rb, f], s, *cfg(m, N=400, Rbh=rb / 8, fifo=f)) for s in CHK]
    for be in (0.2, 0.4, 0.8, 1.0):
        for m in ('tinybev', 'noncoop'):
            J += [('beta', m, be, s, *cfg(m, beta=be)) for s in CHK]
    for m in ('tinybev', 'noncoop', 'oracle', 'coop_lfu'):
        J += [('rho', m, 0.5, s, *cfg(m, rho=0.5)) for s in CHK]
    for sh in (1, 2, 4, 99):
        p, o = cfg('tinybev_route')
        J += [('share', 'tinybev_route', sh, s, p, dict(o, share=sh)) for s in CHK]
    return J


def jobs_tune():
    import itertools
    J = []
    for s in (100, 101, 102):
        for lr, e0, du, es in itertools.product((0.2, 0.5, 1.0), (0.2, 0.5), (0.05, 0.2, 1.0), (10.0, 20.0, 40.0)):
            k = dict(lr_q=lr, eps0=e0, alpha_mu=du, eta_scale=es)
            J.append(('tune', 'cmarl', [lr, e0, du, es], s, {}, k))
    return J


SWEEPS = dict(E3.SWEEPS, main=jobs_main, checks=jobs_checks, tune=jobs_tune)
_c = E3.jobs_cache
SWEEPS['cacheA'] = lambda: [j for j in _c() if j[1] in ('tinybev', 'noncoop', 'static', 'coop_lfu')]
SWEEPS['cacheB'] = lambda: [j for j in _c() if j[1] in ('oracle', 'cmarl', 'lru_ft')]
_b = E3.jobs_bh
SWEEPS['bh1'] = lambda: [j for j in _b() if j[2][0] <= 100]
SWEEPS['bh2'] = lambda: [j for j in _b() if 100 < j[2][0] <= 500]
SWEEPS['bh3'] = lambda: [j for j in _b() if j[2][0] > 500]
SWEEPS['budgetH'] = lambda: [('budget', 'tinybev_h', x, s, *E3.cfg('tinybev', budget=x)) for x in [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0] for s in range(10)]
SWEEPS['hwtref'] = lambda: E3.jobs_hw() + E3.jobs_tref()


def key(e, m, x, s):
    return json.dumps([e, m, x, s])


def run_one(a):
    e, m, x, s, rec = job(a)
    return [[e, m, x, s, rec]]


if __name__ == '__main__':
    name = sys.argv[1]
    path = os.path.join(OUT, name + '.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            e, m, x, s, _ = json.loads(line)
            done.add(key(e, m, x, s))
    if name == 'trace':
        todo = [s for s in CHK if key('trace', 'tinybev', 0, s) not in done]
        fn, J = trace_job, todo
    else:
        J = [j for j in SWEEPS[name]() if key(*j[:4]) not in done]
        fn = run_one
    print(name, len(J), 'to do,', len(done), 'done', flush=True)
    with Pool(2) as p, open(path, 'a') as f:
        for rows in p.imap_unordered(fn, J, chunksize=1):
            for row in rows:
                f.write(json.dumps(row) + '\n')
            f.flush()
    print(name, 'finished', flush=True)
