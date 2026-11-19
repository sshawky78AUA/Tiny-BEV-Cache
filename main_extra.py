"""Extra rows for Table II: deadline-blind ablation and C-MARL with cooperative placement (30 seeds)."""
import os, json
from multiprocessing import Pool
from experiments2 import job, cfg, EVAL, OUT

if __name__ == '__main__':
    J = []
    p, o = cfg('tinybev', infer_aware=False)
    J += [('main', 'tinybev_noinf', 0, s, p, o) for s in EVAL]
    pc, _ = cfg('coop_lfu'); _, oc = cfg('cmarl')
    J += [('main', 'cmarl_coop', 0, s, pc, dict(oc, leader='coop_lfu')) for s in EVAL]
    J = [(e, m, x, s, p, o) for (e, m, x, s, p, o) in J]
    def fix(a):
        e, m, x, s, p, o = a
        return (e, {'tinybev_noinf': 'tinybev', 'cmarl_coop': 'cmarl'}[m], x, s, p, o), m
    with Pool(2) as pool:
        out = pool.map(job, [fix(a)[0] for a in J], chunksize=1)
    out = [(e, fix(a)[1], x, s, r) for (e, m, x, s, r), a in zip(out, J)]
    json.dump(out, open(os.path.join(OUT, 'main_extra.json'), 'w'))
    print('main_extra', len(out))
