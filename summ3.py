import json, numpy as np, collections, sys
from experiments3 import FINE_TB
def load(n):
    D=collections.defaultdict(list)
    for l in open(f'results3/{n}.jsonl'):
        e,m,x,s,r=json.loads(l); D[(m,json.dumps(x))].append(r)
    return D
def mean(rs,k): return np.mean([r[k] for r in rs])
which=sys.argv[1]
if which=='hw':
    D=load('hw')
    for (m,x),rs in sorted(D.items()):
        n=np.sum([r['bins']['n'] for r in rs],0); t=np.sum([r['bins']['tier'] for r in rs],0); d=np.sum([r['bins']['dl'] for r in rs],0)
        print(m,x,len(rs),'util %.3f dl %.4f lat %.3f Emax %.3f Eavg %.3f hit %.3f sem %.3f'%(mean(rs,'util'),mean(rs,'deadline'),mean(rs,'lat_mean')/1e3,np.max([r['energy_ratio_max'] for r in rs]),mean(rs,'energy_ratio_max'),mean(rs,'hit'),mean(rs,'sem')))
        print('   tier',np.round(t/np.maximum(n,1),2)); print('   dl  ',np.round(100*d/np.maximum(n,1),1)); print('   n',n.astype(int))
    print('edges',FINE_TB)
elif which=='V':
    D=load('V')
    for (m,x),rs in sorted(D.items(),key=lambda a:(a[0][0],float(a[0][1]))):
        print(m,x,len(rs),'util %.3f Z %.2f Emax %.3f Eavg %.3f'%(mean(rs,'util'),mean(rs,'Zbar'),np.mean([r['energy_ratio_max'] for r in rs]),mean(rs,'energy_ratio')))
else:
    D=load(which)
    for (m,x),rs in sorted(D.items(),key=lambda a:(a[0][0],json.loads(a[0][1]))):
        print(m,x,len(rs),'util %.3f hit %.3f sem %.3f dl %.4f bh %.2f lat %.3f Emax %.3f age %.0f'%(mean(rs,'util'),mean(rs,'hit'),mean(rs,'sem'),mean(rs,'deadline'),mean(rs,'backhaul'),mean(rs,'lat_mean')/1e3,np.mean([r['energy_ratio_max'] for r in rs]),mean(rs,'age_mean')))
