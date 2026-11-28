"""Motion-only baseline: gradient-boosted trees on the 14-element motion vector, same grouped folds."""
import numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from train_tiers import folds, S
out = {}
for ds, npz, mot in (('dair', 'bev_intent.npz', 'intent40_motion.npy'), ('nusc', 'nusc_intent_fov.npz', 'nusc_intent_fov_motion.npy')):
    D = np.load(npz, allow_pickle=True); y = D['y'].astype(int); g = D['grp']; M = np.load(mot)
    pred = np.zeros_like(y)
    for te in folds(g):
        c = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, early_stopping=False, max_leaf_nodes=15, random_state=0)
        c.fit(M[~te], y[~te]); pred[te] = c.predict(M[te])
    d = np.abs(pred - y); d = np.minimum(d, S - d); t = y != 0
    out[ds] = dict(top1=float((d == 0).mean()), within1=float((d <= 1).mean()), turn=float((d[t] == 0).mean()))
    print(ds, {k: round(100 * v, 1) for k, v in out[ds].items()})
json.dump(out, open('motion_only.json', 'w'))
