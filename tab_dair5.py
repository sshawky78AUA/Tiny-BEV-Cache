import json, collections, numpy as np
def load(f):
    R = collections.defaultdict(list)
    for l in open(f):
        s, m, c, seed, r = json.loads(l); R[(str(c), m)].append(r)
    return {k: {q: np.mean([x[q] for x in v]) for q in v[0]} for k, v in R.items()}
B = load('results5/dair5.jsonl'); S = load('results5/dair5_scarce.jsonl')
names = [('tinybev_h', 'Tiny-BEV-Cache-H'), ('tinybev', 'Tiny-BEV-Cache'), ('agnostic', 'Direction-agnostic'),
         ('coop_lfu', 'Cooperative LFU'), ('oracle', 'Full-information'), ('noncoop', 'Non-cooperative'),
         ('cmarl', 'C-MARL'), ('lru_ft', 'Fixed tier'), ('energy_agnostic', 'Max tier')]
for m, n in names:
    b = B[('base', m)]; s = S[('[75, 400]', m)]
    print(f"{n:20s} & {b['util']:.3f} & {100*b['hit']:.1f} & {100*b['success']:.1f} & {b['backhaul']:.2f} & {b['lat_p99']/1e3:.2f} & {b['energy_ratio']:.2f} & {s['util']:.3f} & {100*s['deadline']:.1f} & {s['backhaul']:.2f} & {s['lat_p99']/1e3:.2f} & {s['energy_ratio']:.2f} \\\\")
