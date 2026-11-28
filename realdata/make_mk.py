"""Turn the measured DAIR-V2X-C confusion matrices into simulator inputs.

The simulator's RSU catalogue has S sectors per approach with their own popularity, so the
measured matrix is used through its angular error profile: p_k(delta) = P(predicted sector is
true sector + delta) pooled over all test frames, which keeps the measured accuracy and how
errors spread to neighbouring sectors. Mk[o, (o + delta) % S] = p_k(delta) (circulant).

Tier latency on the STM32H747 comes from each tier's MAC count and the time per MAC of ST's
MobileNetV2 int8 benchmarks on that board (3.5 to 5.6 ns/MAC, the slower figure for small nets).

usage: python3 make_mk.py tiers_intent > writes tiers_intent_sim.json
"""
import sys, json
import numpy as np

S = 40


def profile(C):
    p = np.zeros(S)
    for o in range(S):
        for q in range(S):
            p[(q - o) % S] += C[o, q]
    return p / p.sum()


def main(prefix):
    summ = json.load(open(f'{prefix}_summary.json'))
    ws = sorted(int(w) for w in summ)
    Mk, acc, prof, d_h7 = [], [], [], []
    for w in ws:
        C = np.load(f'{prefix}_conf_w{w}.npy')
        p = profile(C); prof.append(p.tolist()); acc.append(float(p[0]))
        Mk.append([[p[(q - o) % S] for q in range(S)] for o in range(S)])
        m = summ[str(w)]['macs']
        nspm = np.interp(np.log(m), np.log([20.2e6, 582.2e6]), [5.61, 3.50])   # ST H747 calibration
        d_h7.append(float(m * nspm * 1e-9))
    out = dict(widths=ws, acc=acc, profile=prof, Mk=Mk, d_h7=d_h7,
               macs=[summ[str(w)]['macs'] for w in ws], params=[summ[str(w)]['params'] for w in ws],
               within1=[summ[str(w)]['within1'] for w in ws])
    json.dump(out, open(f'{prefix}_sim.json', 'w'))
    for i, w in enumerate(ws):
        print(f'w={w:2d}  MACs {out["macs"][i]/1e6:6.1f}M  params {out["params"][i]/1e3:6.1f}k  '
              f'top1 {acc[i]:.3f}  within1 {out["within1"][i]:.3f}  H747 latency {d_h7[i]*1e3:6.1f} ms')


if __name__ == '__main__':
    main(sys.argv[1])
