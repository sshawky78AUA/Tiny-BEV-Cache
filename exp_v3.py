"""Runs for the revised paper (v3). Stages:
  perfect  - perfect classifiers (all tiers accuracy 1) vs the measured tiers, to check the inference-limit bound
  scarce   - backhaul 75/100/150 Mb/s x N 300/400, where caching matters, with tail latency
One run per line in results5/<stage>.jsonl; resumable. WORKERS limits parallel processes."""
import sys, os, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevsim2 import Params, run
from experiments2 import cfg
from mobility import simulate
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results5')
KEYS = ('util', 'hit', 'sem', 'correct', 'success', 'deadline', 'backhaul', 'lat_mean', 'lat_p95', 'lat_p99',
        'energy_ratio', 'energy_ratio_max', 'bh_ins', 'bh_miss', 'T_h')
ALIAS = {'tinybev_h': ('tinybev', {'hard': True})}
TH = 75


def job(a):
    stage, meth, x, seed, pkw = a
    m, extra = ALIAS.get(meth, (meth, {}))
    tp, to = cfg(meth)
    o = run(Params(**dict(tp, **pkw)), seed, m, **dict(to, **extra))
    return stage, meth, x, seed, {k: float(o[k]) for k in KEYS if k in o}


def jobs(stage):
    S = list(range(10))
    if stage == 'perfect':
        J = []
        for meth in ('tinybev_h', 'noncoop'):
            for s in S:
                J.append((stage, meth, 'tiers', s, dict(H_ins=TH)))
                J.append((stage, meth, 'perfect', s, dict(H_ins=TH, acc_override=[1.0, 1.0, 1.0, 1.0])))
        for bud in (0.06, 0.2, 1.0):
            for s in S:
                J.append((stage, 'tinybev_h', f'budget{bud}', s, dict(H_ins=TH, budget=bud)))
        return J
    if stage in ('dair', 'dair_scarce', 'dair5', 'dair5_scarce'):
        fn = 'tiers_intent5_sim.json' if stage.startswith('dair5') else 'tiers_intent_sim.json'
        T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realdata', fn)))
        real = dict(H_ins=TH, acc_override=T['acc'], Mk_override=T['Mk'], d_override=T['d_h7'])
        cases = {'dair': [('base', {})], 'dair_scarce': [([r, 400], dict(Rbh=r / 8, N=400)) for r in (75, 150)]}[stage.replace('dair5', 'dair')]
        return [(stage, meth, x, s, dict(real, **kw)) for x, kw in cases
                for meth in (('tinybev_h', 'noncoop', 'coop_lfu', 'oracle', 'agnostic')
                             + (('tinybev', 'cmarl', 'lru_ft', 'energy_agnostic') if stage.startswith('dair5') else ()))
                for s in S]
    if stage == 'scarce':
        return [(stage, meth, [r, n], s, dict(H_ins=TH, Rbh=r / 8, N=n))
                for r in (75, 100, 150) for n in (300, 400)
                for meth in ('tinybev_h', 'noncoop', 'coop_lfu', 'oracle') for s in S]
    raise SystemExit('unknown stage')


if __name__ == '__main__':
    st = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, st + '.jsonl')
    done = set()
    if os.path.exists(fn):
        for l in open(fn):
            try:
                a = json.loads(l); done.add((a[1], json.dumps(a[2]), a[3]))
            except Exception:
                pass
    J = [j for j in jobs(st) if (j[1], json.dumps(j[2]), j[3]) not in done]
    for tr in sorted({(j[3], Params(**j[4]).N, Params(**j[4]).N_bg, Params(**j[4]).T) for j in J}):
        simulate(*tr)
    print(st, len(J), 'to do', flush=True)
    with Pool(int(os.environ.get('WORKERS', os.cpu_count()))) as p, open(fn, 'a') as f:
        for r in p.imap_unordered(job, J):
            f.write(json.dumps(r) + '\n'); f.flush()
    print('DONE', st)
