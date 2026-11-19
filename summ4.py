import json, sys, numpy as np, collections
from scipy import stats
def load(fn, ex=None):
    D=collections.defaultdict(list)
    for l in open(f'results4/{fn}.jsonl'):
        e,m,x,s,r=json.loads(l)
        if ex is None or e==ex: D[(e,m,json.dumps(x))].append(r)
    return D
def hw(v):
    v=np.asarray(v,float); return stats.t.ppf(0.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0
K=['util','hit','sem','success','deadline','backhaul','lat_mean','lat_p95','energy_ratio_max','brgap_mean','brgap_p99','kappa_max','kappa_p99','nb_frac','bh_ins','correct']
fn=sys.argv[1]
for k_,rs in sorted(load(fn).items()):
    out=[]
    for k in K:
        v=[r[k] for r in rs if k in r and r[k]==r[k]]
        if v: out.append(f"{k}={np.mean(v):.4g}±{hw(v):.2g}")
    print(k_, len(rs), ' '.join(out))
