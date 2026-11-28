"""Cross-dataset test of the all-frames tier models, plus a physics baseline (constant turn rate
and speed: the 25 m path point follows the current curvature yaw_rate/speed)."""
import json, numpy as np, torch
from train_tiers import Net, predict, S
D = dict(dair=(np.load('bev_intent.npz')['X'], np.load('bev_intent.npz')['y'].astype(int), np.load('intent40_motion.npy').astype(np.float32)),
         nusc=(np.load('nusc_intent_fov.npz')['X'], np.load('nusc_intent_fov.npz')['y'].astype(int), np.load('nusc_intent_fov_motion.npy').astype(np.float32)))
pre = dict(dair='tiers_intent', nusc='tiers_nusc')
def score(p, y):
    d = np.abs(p - y); d = np.minimum(d, S - d); t = y != 0
    return dict(top1=round(float((d == 0).mean()), 3), within1=round(float((d <= 1).mean()), 3),
                turn=round(float((d[t] == 0).mean()), 3), straight=round(float((d[~t] == 0).mean()), 3))
def ctrv(M, sign=1, L=25.0):
    v = np.maximum(M[:, 12], 0.5); k = sign * M[:, 13] / v; s = L
    x = np.where(np.abs(k) < 1e-4, s, np.sin(k * s) / np.where(k == 0, 1, k))
    yy = np.where(np.abs(k) < 1e-4, 0, (1 - np.cos(k * s)) / np.where(k == 0, 1, k))
    b = np.degrees(np.arctan2(yy, x))
    return (((b + 180 / S) % 360) // (360 / S)).astype(int) % S
if __name__ == "__main__":
    out = {}
    for ds, (X, y, M) in D.items():
        out[f'{ds} majority'] = score(np.zeros_like(y), y)
        for sg in (1, -1):
            out[f'{ds} ctrv sign{sg}'] = score(ctrv(M, sg), y)
        for src in D:
            if src == ds: continue
            for w in (4, 8, 16, 32):
                net = Net(w); net.load_state_dict(torch.load(f'{pre[src]}_model_w{w}.pt')); 
                out[f'{src}->{ds} w{w}'] = score(predict(net, X, M), y)
        for w in (4, 8, 16, 32):
            out[f'{ds} in-domain CV w{w}'] = score(np.load(f'{pre[ds]}_pred_w{w}.npy'), y)
    for k, v in out.items(): print(f'{k:24s}', v)
    json.dump(out, open('cross_eval.json', 'w'), indent=1)
