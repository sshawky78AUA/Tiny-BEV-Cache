"""Denser sweeps with 20 seeds (revision of the Section VI figures).

Usage: python3 experiments3.py traces | <sweep>, sweep in: budget bh dens cache tref V hw
Results are appended one run per line to results3/<sweep>.jsonl, so an interrupted
stage resumes where it stopped.
"""
import sys, os, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments2 import job, cfg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results3')
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(20))


def jobs_budget():
    xs = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
    J = []
    for m in ('tinybev', 'cmarl', 'lru_ft'):
        J += [('budget', m, x, s, *cfg(m, budget=x)) for x in xs for s in SEEDS]
    # Max tier ignores the budget (its tiers and placement do not depend on it), so one run
    # per seed gives every budget point; figs3.py rescales its energy ratio by 0.2 / x.
    J += [('budget', 'energy_agnostic', 0.2, s, *cfg('energy_agnostic', budget=0.2)) for s in SEEDS]
    return J


FINE_TB = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 13.0, 1e9]
RBH = [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]     # Mb/s
NS = [50, 100, 150, 200, 300, 400]


def jobs_bh():
    J = []
    for m in ('tinybev', 'noncoop', 'coop_lfu', 'oracle'):
        J += [('bh', m, [rb, 400], s, *cfg(m, Rbh=rb / 8, N=400)) for rb in RBH for s in SEEDS]
    return J


def jobs_dens():
    J = []
    for m in ('tinybev', 'noncoop', 'coop_lfu', 'oracle'):
        J += [('bh', m, [200, n], s, *cfg(m, Rbh=200 / 8, N=n)) for n in NS if n != 400 for s in SEEDS]
    return J


def jobs_cache():
    xs = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    J = []
    for m in ('tinybev', 'noncoop', 'static', 'coop_lfu', 'oracle', 'cmarl', 'lru_ft'):
        J += [('cache', m, x, s, *cfg(m, cache_frac=x)) for x in xs for s in SEEDS]
    return J


def jobs_tref():
    xs = [30, 60, 100, 150, 200, 300, 450, 600, 900, 1200, 1800]
    return [('tref', 'tinybev', x, s, *cfg('tinybev', T_ref=x)) for x in xs for s in SEEDS]


def jobs_V():
    xs = [0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 20, 50]
    J = []
    for m in ('tinybev', 'tinybev_h'):
        J += [('V', m, x, s, *cfg(m, V=x)) for x in xs for s in SEEDS]
    return J


def jobs_hw():
    J = []
    for hw in ('h7', 'n6'):
        for m in ('tinybev', 'energy_agnostic', 'lru_ft', 'cmarl'):
            J += [('hw', m, hw, s, *cfg(m, hw=hw, TB=FINE_TB)) for s in SEEDS]
        J += [('hw', 'tinybev', 'noinf-' + hw, s, *cfg('tinybev', hw=hw, infer_aware=False, TB=FINE_TB)) for s in SEEDS]
    return J


SWEEPS = {'budget': jobs_budget, 'bh': jobs_bh, 'dens': jobs_dens, 'cache': jobs_cache, 'tref': jobs_tref,
          'V': jobs_V, 'hw': jobs_hw}



def key(e, m, x, s):
    return json.dumps([e, m, x, s])


def run_one(a):
    e, m, x, s, rec = job(a)
    return key(e, m, x, s), [e, m, x, s, rec]


def traces():
    from mobility import simulate
    need = [(s, n, 400, 6000) for s in SEEDS for n in NS]
    with Pool(2) as p:
        p.starmap(simulate, need, chunksize=1)
    print('traces done', flush=True)


if __name__ == '__main__':
    name = sys.argv[1]
    if name == 'traces':
        traces(); sys.exit()
    path = os.path.join(OUT, name + '.jsonl')
    done = set()
    if os.path.exists(path):
        for line in open(path):
            e, m, x, s, _ = json.loads(line)
            done.add(key(e, m, x, s))
    J = [j for j in SWEEPS[name]() if key(*j[:4]) not in done]
    print(name, len(J), 'runs to do,', len(done), 'done', flush=True)
    with Pool(2) as p, open(path, 'a') as f:
        for k, row in p.imap_unordered(run_one, J, chunksize=1):
            f.write(json.dumps(row) + '\n'); f.flush()
    print(name, 'finished', flush=True)
