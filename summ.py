import json,collections,numpy as np,sys
for f in sys.argv[1:]:
  R=collections.defaultdict(list)
  for l in open(f):
    s,m,c,seed,r=json.loads(l); R[(str(c),m)].append(r)
  keys=['util','hit','sem','success','deadline','backhaul','lat_mean','lat_p99','energy_ratio','energy_ratio_max']
  for (c,m),L in sorted(R.items()):
    print(f.split('/')[-1][:12],c,m.ljust(10),len(L),' '.join(f'{k}={np.mean([x[k] for x in L]):.4g}±{1.96*np.std([x[k] for x in L],ddof=1)/np.sqrt(len(L)):.2g}' for k in keys))
