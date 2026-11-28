"""Post-training int8 quantization (per-tensor weights and activations, PyTorch FX, x86 backend)
of the all-frames DAIR models; reports how many predictions change relative to the float model."""
import numpy as np, torch, copy
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from train_tiers import Net
torch.set_num_threads(1)
D = np.load('bev_intent.npz'); X = torch.tensor(D['X']); y = D['y'].astype(int)
M = torch.tensor(np.load('intent40_motion.npy').astype(np.float32))
rng = np.random.default_rng(0); cal = rng.choice(len(y), 512, replace=False)
for w in (4, 8, 16, 32):
    net = Net(w); net.load_state_dict(torch.load(f'tiers_intent_model_w{w}.pt')); net.eval()
    q = prepare_fx(copy.deepcopy(net), get_default_qconfig_mapping('x86'), (X[:2], M[:2]))
    with torch.no_grad():
        q(X[cal], M[cal]); q = convert_fx(q)
        pf = torch.cat([net(X[i:i+512], M[i:i+512]).argmax(1) for i in range(0, len(y), 512)]).numpy()
        pq = torch.cat([q(X[i:i+512], M[i:i+512]).argmax(1) for i in range(0, len(y), 512)]).numpy()
    print(f'w={w}: int8 changes {100*(pf!=pq).mean():.2f}% of predictions; top-1 float {100*(pf==y).mean():.1f} int8 {100*(pq==y).mean():.1f} (in-sample)')
