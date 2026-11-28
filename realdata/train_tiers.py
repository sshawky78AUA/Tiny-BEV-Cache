"""Train four on-board model tiers (same CNN, width w) on the DAIR-V2X-C BEV sector task.

5-fold cross-validation grouped by recording sequence (batch_id), so no sequence appears in both
training and test. Test predictions of all folds are pooled into one confusion matrix per tier.

usage: python3 train_tiers.py bev.npz out_prefix [epochs=40] [motion.npy]
"""
import sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(2)
S = 40
import os
WIDTHS = [int(w) for w in os.environ.get('WIDTHS', '4,8,16,32').split(',')]


class Net(nn.Module):
    def __init__(self, w, cin=4, nmot=14):
        super().__init__()
        self.nmot = nmot
        ch = [cin, w, 2 * w, 4 * w, 8 * w]
        L = []
        for a, b in zip(ch[:-1], ch[1:]):
            L += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                  nn.Conv2d(b, b, 3, padding=1, groups=b, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        self.f = nn.Sequential(*L)
        self.mot = nn.Sequential(nn.Linear(nmot, 2 * w), nn.ReLU(inplace=True)) if nmot else None
        self.head = nn.Linear(ch[-1] * 16 + (2 * w if nmot else 0), S)   # keep 4x4 layout: direction matters

    def forward(self, x, m=None):
        z = self.f(x).flatten(1)
        if self.mot is not None:
            z = torch.cat([z, self.mot(m)], 1)
        return self.head(z)


def macs(net, shape=(1, 4, 64, 64)):
    tot = 0
    def hook(m, i, o):
        nonlocal tot
        if isinstance(m, nn.Conv2d):
            tot += o.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            tot += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    net.eval(); net(torch.zeros(shape), torch.zeros(1, net.nmot) if net.nmot else None); [h.remove() for h in hs]
    return tot


def folds(grp, k=5, seed=0):
    g = np.unique(grp); rng = np.random.default_rng(seed); rng.shuffle(g)
    # balance fold sizes greedily by frame count
    cnt = {x: (grp == x).sum() for x in g}; bins = [[] for _ in range(k)]; sz = np.zeros(k)
    for x in sorted(g, key=lambda x: -cnt[x]):
        j = int(np.argmin(sz)); bins[j].append(x); sz[j] += cnt[x]
    return [np.isin(grp, b) for b in bins]


def train(Xtr, ytr, Xva, yva, w, epochs, seed, Mtr=None, Mva=None):
    torch.manual_seed(seed)
    net = Net(w, nmot=0 if Mtr is None else Mtr.shape[1]); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * int(np.ceil(len(ytr) / 64)))
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
    Mt = None if Mtr is None else torch.tensor(Mtr)
    best, state = -1, None
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(ytr))
        for b in range(0, len(ytr), 64):
            i = perm[b:b + 64]
            x = Xtr[i]; m = None if Mt is None else Mt[i]
            if np.random.rand() < 0.5:      # mirror left/right: sector s -> (S - 1 - s)
                x = x.flip(3).clone(); x[:, 3] = -x[:, 3]; t = S - 1 - ytr[i]
                if m is not None:
                    m = mirror_motion(m)
            else:
                t = ytr[i]
            loss = F.cross_entropy(net(x, m), t, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = (predict(net, Xva, Mva) == yva).mean()
        if acc > best: best, state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return net


def mirror_motion(m):
    # motion = [x(6), y(6), speed, yaw rate]; mirroring negates lateral position and yaw rate
    m = m.clone(); m[:, 6:12] = -m[:, 6:12]; m[:, 13] = -m[:, 13]
    return m


@torch.no_grad()
def predict(net, X, M=None):
    net.eval()
    out = []
    for b in range(0, len(X), 512):
        m = None if M is None else torch.tensor(M[b:b + 512])
        out.append(net(torch.tensor(X[b:b + 512]), m).argmax(1))
    return torch.cat(out).numpy()


def main(npz, out, epochs=40, motion=None):
    D = np.load(npz); X, y, grp = D['X'], D['y'].astype(int), D['grp']
    Mo = None if motion is None else np.load(motion).astype(np.float32)
    F_ = folds(grp)
    res = {}
    for w in WIDTHS:
        t0 = time.time(); C = np.zeros((S, S), int); pred = np.full(len(y), -1)
        for k, te in enumerate(F_):
            tr = ~te
            # hold out one training sequence group as validation for early stopping
            vg = folds(grp[tr], 5, seed=k + 1)[0]
            idx = np.where(tr)[0]
            sub = lambda a, j: None if a is None else a[j]
            net = train(X[idx[~vg]], y[idx[~vg]], X[idx[vg]], y[idx[vg]], w, epochs, seed=k,
                        Mtr=sub(Mo, idx[~vg]), Mva=sub(Mo, idx[vg]))
            pred[te] = predict(net, X[te], sub(Mo, te))
        np.add.at(C, (y, pred), 1)
        n = Net(w, nmot=0 if Mo is None else Mo.shape[1])
        d = np.abs(pred - y); d = np.minimum(d, S - d)
        res[w] = dict(width=w, params=int(sum(p.numel() for p in n.parameters())), macs=int(macs(n)),
                      top1=float((d == 0).mean()), within1=float((d <= 1).mean()),
                      adj_share=float(((d == 1).sum()) / max(1, (d > 0).sum())), secs=round(time.time() - t0))
        np.save(f'{out}_conf_w{w}.npy', C); np.save(f'{out}_pred_w{w}.npy', pred)
        # one model on all frames (a held-out sequence group for early stopping) for cross-dataset tests
        vg = folds(grp, 5, seed=99)[0]
        sub = lambda a, j: None if a is None else a[j]
        net = train(X[~vg], y[~vg], X[vg], y[vg], w, epochs, seed=99, Mtr=sub(Mo, ~vg), Mva=sub(Mo, vg))
        torch.save(net.state_dict(), f'{out}_model_w{w}.pt')
        print(json.dumps(res[w]), flush=True)
        json.dump({str(k): v for k, v in res.items()}, open(f'{out}_partial_{w}.json', 'w'))
    fn = f'{out}_summary.json'
    prev = json.load(open(fn)) if os.path.exists(fn) else {}
    prev.update({str(k): v for k, v in res.items()})
    json.dump(prev, open(fn, 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 40, a[4] if len(a) > 4 else None)
