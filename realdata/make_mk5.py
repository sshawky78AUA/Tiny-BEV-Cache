"""Five-tier simulator input: tier 0 is the physics predictor (constant turn rate and speed,
a few dozen operations, 0.1 ms budgeted on the MCU), tiers 1-4 the trained CNNs."""
import json, numpy as np
from cross_eval import ctrv
from make_mk import profile, S
T = json.load(open('tiers_intent_sim.json'))
y = np.load('bev_intent.npz')['y'].astype(int); M = np.load('intent40_motion.npy')
C = np.zeros((S, S)); np.add.at(C, (y, ctrv(M)), 1)
p = profile(C)
T['widths'] = [0] + T['widths']; T['acc'] = [float(p[0])] + T['acc']; T['profile'] = [p.tolist()] + T['profile']
T['Mk'] = [[[p[(q - o) % S] for q in range(S)] for o in range(S)]] + T['Mk']
T['d_h7'] = [1e-4] + T['d_h7']; T['macs'] = [0] + T['macs']; T['params'] = [0] + T['params']
d = np.abs(ctrv(M) - y); d = np.minimum(d, S - d); T['within1'] = [float((d <= 1).mean())] + T['within1']
json.dump(T, open('tiers_intent5_sim.json', 'w'))
print([round(a, 3) for a in T['acc']], [round(x * 1e3, 2) for x in T['d_h7']])
