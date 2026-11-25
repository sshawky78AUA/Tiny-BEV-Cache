"""Placement benchmarks with full information: hit maximization and backhaul minimization (insertion horizon family)."""
import sys, os, json
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevsim2 import Params, run
from experiments2 import cfg
FN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results4', 'bench.jsonl')
KEYS = ('util', 'hit', 'sem', 'success', 'deadline', 'backhaul', 'lat_mean', 'bh_ins', 'bh_ref', 'bh_miss', 'age_mean')


def job(a):
    tag, seed = a
    if tag == 'fi_hit':
        pk, ok = cfg('oracle'); o = run(Params(**pk), seed, 'oracle', leader='oracle_hit', **ok)
    else:
        H = int(tag.split('_H')[1]); pk, ok = cfg('oracle'); o = run(Params(**dict(pk, H_ins=H)), seed, 'oracle', **ok)
    return tag, seed, {k: float(o[k]) for k in KEYS if k in o}


if __name__ == '__main__':
    tags = ['fi_hit'] + [f'fi_H{H}' for H in (10, 15, 30, 75, 150, 600, 1200, 2400, 6000)]
    done = set()
    if os.path.exists(FN):
        for l in open(FN):
            t_, s_, _ = json.loads(l); done.add((t_, s_))
    J = [(t, s) for s in range(10) for t in tags if (t, s) not in done]
    print(len(J), 'to do', flush=True)
    with Pool(int(os.environ.get('WORKERS', os.cpu_count()))) as p, open(FN, 'a') as f:
        for r in p.imap_unordered(job, J):
            f.write(json.dumps(r) + '\n'); f.flush(); print(r[0], r[1], round(r[2]['hit'], 3), round(r[2]['backhaul'], 2), flush=True)
    print('DONE')
