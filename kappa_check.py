"""Evaluate the contraction modulus kappa of Theorem 2 along simulated trajectories."""
import json, os
from multiprocessing import Pool
import numpy as np
from bevsim2 import Params, run
from experiments2 import cfg, OUT

def job(a):
    name, seed, pkw = a
    o = run(Params(**pkw), seed, 'tinybev', kappa=True)
    return name, seed, o['kappa_max'], o['kappa_p99'], o['kappa_mean']

if __name__ == '__main__':
    cases = {'default': {}, 'N400_R50': dict(N=400, Rbh=50 / 8), 'N400_R200': dict(N=400), 'nu0.05': dict(nu=0.05)}
    J = [(c, s, cfg('tinybev', **kw)[0]) for c, kw in cases.items() for s in range(5)]
    with Pool(2) as p:
        R = p.map(job, J, chunksize=1)
    json.dump(R, open(os.path.join(OUT, 'kappa.json'), 'w'))
    for c in cases:
        r = np.array([x[2:] for x in R if x[0] == c])
        print(c, 'max %.3f  p99 %.3f  mean %.4f' % (r[:, 0].max(), r[:, 1].mean(), r[:, 2].mean()))
