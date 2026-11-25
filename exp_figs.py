"""Extra runs for the time-domain and latency-CDF figures (saved as .npz in results4/)."""
import sys, os
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bevsim2 import Params, run
from experiments2 import cfg
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results4')
TSM = ['tinybev', 'noncoop', 'oracle', 'coop_lfu', 'static', 'cmarl']
LATM = ['tinybev', 'noncoop', 'coop_lfu', 'oracle']


def ts_job(a):
    meth, seed = a
    pk, ok = cfg(meth)
    o = run(Params(**pk), seed, meth, record_ts=True, **ok)
    ts = o['ts']
    keep = dict(req=ts['req'], hit=ts['hit'], succ=ts['succ'], bh=ts['bh'])
    if meth == 'tinybev':
        keep.update(Z=ts['Z'], tau=ts['tau'], k=ts['k'], zone=ts['zone'], u=ts['u'])
    np.savez_compressed(os.path.join(OUT, 'ts', f'{meth}_{seed}.npz'), **keep)
    return meth, seed, o['hit']


def tref_job(a):
    meth, seed, T = a
    pk, ok = cfg(meth)
    o = run(Params(**dict(pk, T_ref=T)), seed, meth, **ok)
    return meth, seed, T, {k: float(o[k]) for k in ('hit', 'backhaul', 'age_mean', 'util', 'success')}


def lat_job(a):
    meth, seed, N, R = a
    pk, ok = cfg(meth)
    o = run(Params(**dict(pk, N=N, Rbh=R / 8)), seed, meth, record_lat=True, **ok)
    np.save(os.path.join(OUT, 'lat', f'{meth}_{N}_{R}_{seed}.npy'), o['lat'])
    np.save(os.path.join(OUT, 'latc', f'{meth}_{N}_{R}_{seed}.npy'), o['lat_comp'].mean(0))
    return meth, seed, N, R, o['lat_mean']


if __name__ == '__main__':
    st = sys.argv[1]
    os.makedirs(os.path.join(OUT, st), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'latc'), exist_ok=True)
    if st == 'tref':
        TR = [30, 60, 100, 150, 300, 600, 1800]
        J = [(m, s, T) for s in range(10) for T in TR for m in ('noncoop', 'coop_lfu', 'oracle')]
        done = set()
        fn = os.path.join(OUT, 'tref_more.jsonl')
        if os.path.exists(fn):
            import json
            for l in open(fn):
                try:
                    m_, s_, T_, _ = json.loads(l); done.add((m_, s_, T_))
                except Exception:
                    pass
        J = [j for j in J if j not in done]
        print('tref', len(J), 'to do', flush=True)
        import json
        with Pool(int(os.environ.get('WORKERS', os.cpu_count()))) as p, open(fn, 'a') as fo:
            for r in p.imap_unordered(tref_job, J):
                fo.write(json.dumps(r) + '\n'); fo.flush()
        print('DONE tref'); sys.exit()
    if st == 'ts':
        J = [(m, s) for s in range(10) for m in TSM if not os.path.exists(os.path.join(OUT, 'ts', f'{m}_{s}.npz'))]
        f = ts_job
    else:
        J = [(m, s, N, R) for s in range(10) for (N, R) in ((400, 50), (400, 200)) for m in LATM
             if not os.path.exists(os.path.join(OUT, 'latc', f'{m}_{N}_{R}_{s}.npy'))]
        f = lat_job
    print(st, len(J), 'to do', flush=True)
    with Pool(int(os.environ.get('WORKERS', os.cpu_count()))) as p:
        for r in p.imap_unordered(f, J):
            print(*r, flush=True)
    print('DONE', st)
