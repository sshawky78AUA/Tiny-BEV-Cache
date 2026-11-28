"""Tiny-BEV-Cache simulator, version 2 (journal setup).

Mobility : SUMO traces on the 3GPP TR 37.885 urban grid (mobility.py).
Channel  : TR 37.885 urban V2V path loss with the RSU as a UE-type RSU at 6 GHz,
           LOS/NLOSv states, 20 MHz shared by the requests served by an RSU in a slot.
TinyML   : MobileNetV2 tiers measured on STM32H747 (ST model zoo): latency d_k, ImageNet
           top-1 accuracy as a stand-in; energy proportional to latency at constant power.
Model    : Sections III-V of the paper (no skip tier: an active vehicle always runs a tier).
"""
import numpy as np
from mobility import simulate, edge_tables

DIRS = 4
_TABLES = None


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = edge_tables()
    return _TABLES


HW = {  # MobileNetV2, int8, ImageNet, ST model zoo
    'h7': dict(acc=[0.4394, 0.547, 0.6797, 0.7146], d=[0.11343, 0.32176, 1.11827, 2.03556]),
    'n6': dict(acc=[0.4394, 0.547, 0.6797, 0.7146], d=[0.00583, 0.00868, 0.02045, 0.03474]),
}


class Params:
    def __init__(self, **kw):
        self.G = 5; self.N = 200; self.N_bg = 400; self.T = 6000; self.warm = 1000
        self.zone = 150.0; self.tau_min = 1.0
        self.S = 40; self.zipf = 1.0; self.beta = 0.6
        self.b_lo = 0.25; self.b_hi = 2.5                 # MB, compressed BEV features
        self.live_rsu = 0.4; self.live_frac = 0.25         # equipped RSUs, sensor sectors there
        self.cache_frac = 0.15
        self.a_lo = 0.05; self.a_hi = 0.15
        self.hw = 'h7'
        self.lam1 = 1.0; self.lam2 = 1.0
        self.Rbh = 200.0 / 8                                # MB/s
        self.Rx2 = 1000.0 / 8
        self.fc = 6.0; self.BW = 20e6; self.Ptx = 23.0; self.NF = 9.0; self.h_rsu = 5.0; self.h_ue = 1.6
        self.Gant = 3.0 + 3.0                               # dBi, vehicle + RSU (TR 36.885)
        self.se_max = 7.4                                   # b/s/Hz
        self.budget = 0.2                                   # E_bar_n = budget * a_n * E_max
        self.V = 2.0; self.nu = 0.2; self.theta = 0.03
        self.gamma = 0.02; self.gamma_d = 0.1; self.gamma_L = 0.2
        self.T_ref = 300; self.Tc = 30; self.W = 20.0; self.u_est = 8.0
        self.share = 99                                     # pairs shared (1 = next RSU only)
        self.regime = 1500; self.shift_frac = 0.3
        self.rho = 0.0                                      # correlated intents: share of queued vehicles that need a queue-hidden sector
        self.infer_aware = True                             # follower uses only the feasible tiers K_n(t)
        self.deconv = True                                  # follower estimates true intents from reported labels
        self.task_reward = True                             # delivery counts only for the correct asset
        self.busy = True                                    # a vehicle issues no request while its MCU is busy
        self.H_ins = None                                   # placement horizon W (slots); None = T_ref
        self.tau_err = 0.0                                  # relative error of the vehicle's own deadline estimate
        self.fifo = False                                   # explicit FIFO backhaul queue instead of batch drain
        for k, v in kw.items():
            setattr(self, k, v)
        self.M = self.G * self.G
        h = HW[self.hw]
        self.acc = np.array(h['acc']); self.d = np.array(h['d'])
        if getattr(self, 'acc_override', None) is not None:   # e.g. measured accuracies of trained tiers
            self.acc = np.array(self.acc_override, float)
        if getattr(self, 'd_override', None) is not None:     # tier latencies of the trained tiers (s)
            self.d = np.array(self.d_override, float)
        self.energy = self.d / self.d[-1]                   # e_k / E_max


def pathloss(d2, rng, fc, h_rsu, h_ue):
    """TR 37.885 urban V2V (UE-type RSU): LOS w.p. min(1,1.05 exp(-0.0114 d)), else NLOSv."""
    d3 = np.sqrt(d2 ** 2 + (h_rsu - h_ue) ** 2)
    pl = 38.77 + 16.7 * np.log10(d3) + 18.2 * np.log10(fc)
    los = rng.random(d2.shape) < np.minimum(1.0, 1.05 * np.exp(-0.0114 * d2))
    extra = np.maximum(0.0, rng.normal(5.0 + np.maximum(0, 15 * np.log10(d3) - 41), 4.0))
    return pl + np.where(los, 0.0, extra) + rng.normal(0, 3.0, d2.shape)


class World:
    def __init__(self, P, seed):
        self.P = P
        rng = np.random.default_rng(seed + 12345)
        self.rng = rng
        M, S = P.M, P.S
        self.b = rng.uniform(P.b_lo, P.b_hi, (M, S))
        self.live = np.zeros((M, S), bool)
        eq = rng.random(M) < P.live_rsu
        for m in np.where(eq)[0]:
            self.live[m, rng.choice(S, int(round(P.live_frac * S)), replace=False)] = True
        self.C = P.cache_frac * (self.b * ~self.live).sum(1)
        to_m, from_m, dirn = tables()
        self.to_m, self.from_m, self.dirn = to_m, from_m, dirn
        self.nbr = []
        for m in range(M):
            r, c = divmod(m, P.G); L = []
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < P.G and 0 <= cc < P.G:
                    L.append(rr * P.G + cc)
            self.nbr.append(L)
        z = 1.0 / np.arange(1, S + 1) ** P.zipf; z /= z.sum()
        self.q = np.zeros((M, DIRS, S))
        for m in range(M):
            for v in range(DIRS):
                self.q[m, v, rng.permutation(S)] = z
        self.a = rng.uniform(P.a_lo, P.a_hi, P.N)
        self.Mk = []
        for A in P.acc:
            Mk = np.full((S, S), (1 - A) * (1 - P.beta) / (S - 3))
            for o in range(S):
                Mk[o, [(o - 1) % S, (o + 1) % S]] = (1 - A) * P.beta / 2
                Mk[o, o] = A
            self.Mk.append(Mk)
        self.Mk = np.array(self.Mk)
        if getattr(P, 'Mk_override', None) is not None:      # measured confusion matrices, K x S x S
            self.Mk = np.array(P.Mk_override, float)
        self.qhid = np.array([[rng.choice(S, 3, replace=False) for v in range(DIRS)] for m in range(M)])
        tr = np.load(simulate(seed, P.N, P.N_bg, P.T))
        self.E, self.X, self.U, self.R, self.elen = tr['E'], tr['X'], tr['U'], tr['R'], tr['elen']
        self.ptr = np.zeros(P.N, int)

    def shift_needs(self):
        P, rng = self.P, self.rng
        for m in np.where(rng.random(P.M) < P.shift_frac)[0]:
            for v in range(DIRS):
                self.q[m, v] = self.q[m, v][rng.permutation(P.S)]

    def state(self, t):
        P = self.P
        e = self.E[t].astype(int)
        valid = e >= 0
        e = np.where(valid, e, 0)
        for i in np.where(valid)[0]:                         # advance route pointers
            R = self.R[i]; p = self.ptr[i]
            while p < len(R) - 1 and R[p] != e[i]:
                p += 1
            self.ptr[i] = p
        m = self.to_m[e]; v = self.dirn[e]
        dist = self.elen[e] - self.X[t]
        u = self.U[t]
        tau = np.minimum(dist / np.maximum(u, 0.5), 60.0)
        return valid, m, v, dist, u, tau

    def lookahead(self, t, share):
        """Lambda[m,v]: expected requests per slot within W seconds from the shared pairs."""
        P = self.P
        Lam = np.zeros((P.M, DIRS))
        zt = P.zone / P.u_est
        e = self.E[t].astype(int)
        for i in range(P.N):
            if e[i] < 0:
                continue
            R = self.R[i]; p = self.ptr[i]
            t_arr = (self.elen[e[i]] - self.X[t, i]) / P.u_est
            for j in range(share):
                if p + j >= len(R):
                    break
                ed = int(R[p + j])
                if j > 0:
                    t_arr += self.elen[ed] / P.u_est
                if t_arr - zt > P.W:
                    break
                Lam[self.to_m[ed], self.dirn[ed]] += self.a[i] * zt
        return Lam / P.W


def coop_greedy(D, b, C, nbr, T_ref, live, coop=True, y_now=None, H=None, x_old=None):
    M, S = D.shape
    dens = D - 1.0 / T_ref
    if y_now is not None:
        dens = dens - (~y_now) / H
    dens = np.where(live, -np.inf, dens)
    order = np.argsort(-dens, axis=None)
    free = C.copy()
    x = np.zeros((M, M, S), bool)
    for idx in order:
        if dens.flat[idx] <= 0:
            break
        m, o = divmod(idx, S)
        bo = b[m, o]
        cands = [m] + (nbr[m] if coop else [])
        best = None
        if x_old is not None:                                # keep the current host: moving costs a new insertion
            held = [h for h in cands if x_old[h, m, o] and free[h] >= bo]
            if held:
                x[held[0], m, o] = True; free[held[0]] -= bo
                continue
        for h in cands:
            if free[h] >= bo and (best is None or h == m or (best != m and free[h] > free[best])):
                best = h
                if h == m:
                    break
        if best is not None:
            x[best, m, o] = True; free[best] -= bo
    return x


class LRU:
    def __init__(self, cap):
        self.cap = cap; self.d = {}; self.sz = {}; self.used = 0.0; self.t = 0

    def access(self, o, size):
        """returns (hit, fetched volume for insertion)."""
        self.t += 1
        if o in self.d:
            self.d[o] = self.t; return True, 0.0
        if size > self.cap:
            return False, 0.0
        while self.used + size > self.cap:
            old = min(self.d, key=self.d.get); self.used -= self.sz.pop(old); del self.d[old]
        self.d[o] = self.t; self.sz[o] = size; self.used += size
        return False, 0.0          # the miss fetch already brings the asset; no extra volume


def run(P, seed, method='tinybev', **opt):
    """method: tinybev | agnostic | noncoop | static | cmarl | lru_ft | energy_agnostic"""
    W = World(P, seed)
    rng = np.random.default_rng(seed + 999)
    M, S, N, K = P.M, P.S, P.N, len(P.acc)
    acc, en, dk = P.acc, P.energy, P.d
    b, live = W.b, W.live
    nu = 1e-4 if opt.get('hard') else P.nu
    leader = {'tinybev': 'hist', 'tinybev_route': 'route', 'agnostic': 'agn', 'noncoop': 'hist_nc',
              'static': 'static', 'cmarl': 'lfu', 'lru_ft': 'lru', 'energy_agnostic': 'hist',
              'oracle': 'oracle', 'coop_lfu': 'coop_lfu'}[method]
    leader = opt.get('leader', leader)
    pi = np.full(K, 1.0 / K)                                 # tier mix (oracle leader)
    follower = {'cmarl': 'cmarl', 'lru_ft': 'fixed', 'energy_agnostic': 'maxacc'}.get(method, 'logit')
    share = opt.get('share', P.share)
    Qd = np.full((M, DIRS, S), 1.0 / S); Qm = np.full((M, S), 1.0 / S); rate = np.zeros(M)
    rate_mv = np.zeros((M, DIRS))                             # EMA of requests per slot per (m, v)
    lfu = np.zeros((M, S))
    x = np.zeros((M, M, S), bool); y = np.zeros((M, S), bool)
    lrus = [LRU(W.C[m]) for m in range(M)] if leader == 'lru' else None
    d_hat = 0.0; Lam_bar = None
    Z = np.zeros(N)
    busy_until = np.zeros(N); Bq = 0.0; GAP = []; xtrace = []
    Mdiag = np.stack([np.diag(W.Mk[kk]) for kk in range(K)])     # K x S: P(report o | intent o)
    auto_h = (P.H_ins == 'auto')                           # learn T_h from how long inserted assets stay
    Hins = P.T_ref if (auto_h or not P.H_ins) else P.H_ins
    t_in = None; res_ema = None
    replay = opt.get('replay_x')
    Ebar = P.budget * W.a * en[-1]
    Ebp = (1 - P.theta) * Ebar
    k_fixed = np.array([max([k for k in range(K) if W.a[n] * en[k] <= Ebp[n]] or [0]) for n in range(N)])
    nb = 5; Qtab = np.zeros((N, nb, K)); mu = np.zeros(N)
    cm_lr = opt.get('lr_q', 0.5); cm_eps0 = opt.get('eps0', 0.5); cm_dual = opt.get('alpha_mu', 0.2)
    eta_scale = opt.get('eta_scale', 20.0)
    E_cum = np.zeros(N); Zsum = 0.0; KAP = []
    L = dict(r=[], act=0, hit=0, correct=0, sem=0, dl=0, bh=[], lat=[], nbh=0, miss=[], ref=[], ins=[],
             age=[], live=0, succ=0)
    t_fetch = np.zeros((M, S)); insert_vol = 0.0
    TB = np.array(getattr(P, 'TB', None) or [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 1e9]); nbin = len(TB) - 1
    BIN = dict(n=np.zeros(nbin), dl=np.zeros(nbin), correct=np.zeros(nbin), tier=np.zeros(nbin), util=np.zeros(nbin))
    noise_dbm = -174 + 10 * np.log10(P.BW) + P.NF
    rec_ts = opt.get('record_ts')
    if rec_ts:
        TS = dict(req=np.zeros(P.T), hit=np.zeros(P.T), succ=np.zeros(P.T), bh=np.zeros(P.T),
                  Z=np.zeros((P.T, N), np.float32), tau=np.full((P.T, N), np.nan, np.float32),
                  k=np.full((P.T, N), -1, np.int8), zone=np.zeros((P.T, N), bool), u=np.zeros((P.T, N), np.float32))
    for t in range(P.T):
        if t > 0 and t % P.regime == 0:
            W.shift_needs()
        valid, mm, vv, dist, u, tau = W.state(t)
        if P.tau_err > 0:                                   # the vehicle's own estimate of its deadline
            tau_h = tau * (1 + P.tau_err * (2 * rng.random(N) - 1))
        else:
            tau_h = tau
        inzone = valid & (dist <= P.zone) & (tau_h >= P.tau_min + dk[0])
        if P.busy:
            inzone &= (t >= busy_until)
        active = inzone & (rng.random(N) < W.a)
        # true-intent estimates: deconvolve the reported-label distribution with the mean confusion matrix
        Mbar = np.einsum('k,kso->so', pi, W.Mk)
        Minv = np.linalg.inv(Mbar) if P.deconv else None
        qcache = {}

        def qhat(m, v):
            if (m, v) not in qcache:
                if P.deconv:
                    qh = np.maximum(Qd[m, v] @ Minv, 0); qh /= max(qh.sum(), 1e-12)
                else:
                    qh = Qd[m, v]
                qcache[(m, v)] = qh
            return qcache[(m, v)]
        # ---------------- leader
        if t % P.Tc == 0 and leader != 'lru':
            newx = None
            if leader in ('hist', 'hist_nc') or (leader == 'static' and t <= P.warm):
                D = np.einsum('mv,mvs->ms', rate_mv, Qd)
                newx = coop_greedy(D, b, W.C, W.nbr, P.T_ref, live, coop=(leader != 'hist_nc'), y_now=y, H=Hins, x_old=x)
            elif leader == 'route':
                Lam = W.lookahead(t, share)
                Lam_bar = Lam if Lam_bar is None else (1 - P.gamma_L) * Lam_bar + P.gamma_L * Lam
                D = np.einsum('mv,mvs->ms', Lam_bar, Qd)
                newx = coop_greedy(D, b, W.C, W.nbr, P.T_ref, live, y_now=y, H=Hins, x_old=x)
            elif leader == 'agn':
                newx = coop_greedy(rate[:, None] * Qm, b, W.C, W.nbr, P.T_ref, live, y_now=y, H=Hins, x_old=x)
            elif leader == 'oracle':
                Lam = W.lookahead(t, 99)
                Rq = np.einsum('mvs,ksr,k->mvr', W.q, W.Mk, pi)  # true distribution of the reported asset
                newx = coop_greedy(np.einsum('mv,mvr->mr', Lam, Rq), b, W.C, W.nbr, P.T_ref, live, y_now=y, H=Hins, x_old=x)
            elif leader == 'oracle_hit':                    # full information, expected hits per byte, no cost terms
                Lam = W.lookahead(t, 99)
                Rq = np.einsum('mvs,ksr,k->mvr', W.q, W.Mk, pi)
                Dh = np.einsum('mv,mvr->mr', Lam, Rq) / np.maximum(b, 1e-12)
                newx = coop_greedy(Dh, b, W.C, W.nbr, 1e18, live, x_old=x)
            elif leader == 'coop_lfu':
                D = lfu / max(lfu.sum(), 1e-9) * max(rate.sum(), 1e-9)
                newx = coop_greedy(D, b, W.C, W.nbr, P.T_ref, live)
            elif leader == 'lfu':
                D = lfu / max(lfu.sum(), 1e-9) * max(rate.sum(), 1e-9)
                newx = coop_greedy(D, b, W.C, W.nbr, P.T_ref, live, coop=False)
            if replay is not None:
                newx = replay[min(t // P.Tc, len(replay) - 1)]
            if newx is not None:
                xtrace.append(newx)
                ynew = newx.any(0)
                insert_vol = float((b[None] * (newx & ~x)).sum())
                t_fetch[ynew & ~y] = t
                if auto_h:
                    if t_in is None:
                        t_in = np.where(newx, t, -1).astype(float)
                    else:
                        ev = x & ~newx
                        if ev.any():
                            r = float((t - t_in[ev]).mean())
                            res_ema = r if res_ema is None else 0.9 * res_ema + 0.1 * r
                            Hins = float(np.clip(res_ema, P.Tc, P.T_ref))
                        t_in[newx & ~x] = t
                x = newx; y = ynew
        if leader == 'lru':
            y = np.zeros((M, S), bool)
            for m in range(M):
                for o in lrus[m].d:
                    y[m, o] = True
        refresh = float((b * y).sum()) / P.T_ref
        # ---------------- followers
        k = np.full(N, -1)
        idx = np.where(active)[0]
        if opt.get('kappa') and t >= P.warm and t % 100 == 0:
            J = np.where(inzone)[0]
            if len(J) > 1:
                Qj = np.stack([qhat(mm[j], vv[j]) for j in J])
                Rj = np.einsum('js,kso->jko', Qj, W.Mk)                         # |J| x K x S reported
                Cj = Qj[:, None, :] * Mdiag[None] if P.task_reward else Rj      # correct and reported
                miss = np.stack([~(y[mm[j]] | live[mm[j]]) for j in J]).astype(float)
                bj = np.stack([b[mm[j]] for j in J]); aj = W.a[J]
                for i, n in enumerate(J):
                    en_ = 1.0 / (P.Rbh * tau[n])
                    wbar = float((Cj[i] @ (miss[i] * np.exp(-en_ * bj[i]))).max())
                    cbar = np.einsum('jks,js->jk', Rj, miss * (1 - np.exp(-en_ * bj))).max(1)
                    term = aj * cbar; term[i] = 0.0
                    KAP.append(P.lam2 * wbar / (4 * P.nu) * term.sum())
        gap_slot = opt.get('brgap') and t >= P.warm and t % 50 == 0 and follower == 'logit'
        loopset = np.where(inzone)[0] if gap_slot else idx
        pinfo = {}
        if follower == 'logit':
            for n in loopset:
                m, v = mm[n], vv[n]
                qh = qhat(m, v)
                rk = qh[None] * Mdiag if P.task_reward else qh @ W.Mk   # K x S
                yl = y[m] | live[m]
                h = rk @ yl
                feas = dk <= tau_h[n] - P.tau_min if P.infer_aware else np.ones(K, bool)
                eps = 1.0 / (P.Rbh * tau_h[n])
                w = rk @ ((~yl) * np.exp(-eps * b[m]))
                phi = h + w * np.exp(-eps * d_hat)
                util = P.lam1 * acc + P.lam2 * phi - Z[n] * en / P.V
                s = np.where(feas, util / nu, -np.inf); s -= s.max(); p = np.exp(s); p /= p.sum()
                if gap_slot:
                    pinfo[n] = (p, h, w, feas, eps, rk, yl)
                if active[n]:
                    k[n] = rng.choice(K, p=p)
            if gap_slot and len(pinfo) > 1:
                # best-response gap of the implemented rule in the exact frozen-state game
                J = list(pinfo); eta_c = refresh + insert_vol
                for n in J:
                    p_n, h_n, w_n, fe_n, eps_n, _, _ = pinfo[n]
                    G = np.exp(-eps_n * eta_c)
                    for j in J:
                        if j == n:
                            continue
                        p_j, _, _, _, _, rk_j, yl_j = pinfo[j]
                        mj = mm[j]
                        Rj_ = qhat(mj, vv[j]) @ W.Mk
                        c_j = Rj_ @ ((~yl_j) * (1 - np.exp(-eps_n * b[mj])))
                        G *= 1 - W.a[j] * float(p_j @ c_j)
                    u_ex = P.lam1 * acc + P.lam2 * (h_n + w_n * G) - Z[n] * en / P.V
                    uf = np.where(fe_n, u_ex, -np.inf)
                    best = nu * np.log(np.sum(np.exp((uf - uf.max()) / nu))) + uf.max()
                    pz = p_n[p_n > 0]
                    cur = float(p_n @ np.where(fe_n, u_ex, 0)) - nu * float(np.sum(pz * np.log(pz)))
                    GAP.append((best - cur) / (P.lam1 + P.lam2))
        elif follower == 'cmarl':
            sb = min(int(d_hat / eta_scale * nb), nb - 1)
            epsx = max(0.02, cm_eps0 / np.sqrt(1 + t / 100))
            for n in idx:
                fe = np.where(dk <= tau_h[n] - P.tau_min)[0] if opt.get('cmarl_mask', True) else np.arange(K)
                if len(fe) == 0:
                    fe = np.array([0])
                k[n] = int(rng.choice(fe)) if rng.random() < epsx else int(fe[Qtab[n, sb, fe].argmax()])
        elif follower == 'maxacc':
            k[idx] = K - 1
        elif follower == 'fixed':
            k[idx] = k_fixed[idx]
        En = np.zeros(N); En[idx] = en[k[idx]]
        if P.busy and len(idx):
            busy_until[idx] = t + dk[k[idx]]
        if len(idx):
            pi = 0.99 * pi + 0.01 * np.bincount(k[idx], minlength=K) / len(idx)
        # ---------------- requests
        rep = np.zeros(N, int); ost = np.zeros(N, int); Hn = np.zeros(N); fromnb = np.zeros(N, bool)
        islive = np.zeros(N, bool)
        miss_vol = 0.0
        for n in idx:
            m, v = mm[n], vv[n]
            if P.rho > 0 and u[n] < 0.5 and rng.random() < P.rho:
                o = int(rng.choice(W.qhid[m, v]))            # occlusion caused by the queue ahead
            else:
                o = rng.choice(S, p=W.q[m, v])
            ost[n] = o
            o2 = rng.choice(S, p=W.Mk[k[n], o]); rep[n] = o2
            if live[m, o2]:
                Hn[n] = 1.0; islive[n] = True
            elif leader == 'lru':
                hit, _ = lrus[m].access(o2, b[m, o2])
                if not hit and o2 in lrus[m].d:
                    t_fetch[m, o2] = t
                Hn[n] = float(hit)
                if not hit:
                    miss_vol += b[m, o2]
            else:
                Hn[n] = float(y[m, o2]); fromnb[n] = y[m, o2] and not x[m, m, o2]
                if not y[m, o2]:
                    miss_vol += b[m, o2]
        eta = miss_vol + refresh + insert_vol
        if P.fifo:                                          # backlog carried over from earlier slots
            Bq = max(Bq - P.Rbh, 0.0) + eta
            eta_d = Bq
        else:
            eta_d = eta
        if t >= P.warm:
            L['miss'].append(miss_vol); L['ref'].append(refresh); L['ins'].append(insert_vol)
        insert_vol = 0.0
        # access rates: path loss, bandwidth shared by the requests at the same RSU
        if len(idx):
            pl = pathloss(np.maximum(dist[idx], 5.0), rng, P.fc, P.h_rsu, P.h_ue)
            snr = 10 ** ((P.Ptx + P.Gant - pl - noise_dbm) / 10)
            se = np.minimum(np.log2(1 + snr), P.se_max)
            share_bw = np.bincount(mm[idx], minlength=M)[mm[idx]]
            Ra = P.BW * se / share_bw / 8e6                  # MB/s
        r = np.zeros(N)
        for j, n in enumerate(idx):
            ok = dk[k[n]] <= tau[n] - P.tau_min                 # asset can still arrive in time
            cor = (rep[n] == ost[n]) if P.task_reward else True
            phi = (Hn[n] + (1 - Hn[n]) * np.exp(-eta_d / (P.Rbh * tau[n]))) if (ok and cor) else 0.0
            r[n] = P.lam1 * acc[k[n]] + P.lam2 * phi
        # ---------------- learning and queues
        for n in idx:
            m, v, o2 = mm[n], vv[n], rep[n]
            Qd[m, v] *= (1 - P.gamma); Qd[m, v, o2] += P.gamma
            Qm[m] *= (1 - P.gamma); Qm[m, o2] += P.gamma
            lfu[m, o2] += 1
        lfu *= 0.999
        cnt = np.bincount(mm[idx], minlength=M) if len(idx) else np.zeros(M)
        rate = 0.99 * rate + 0.01 * cnt
        cmv = np.zeros((M, DIRS))
        if len(idx):
            np.add.at(cmv, (mm[idx], vv[idx]), 1.0)
        rate_mv = 0.99 * rate_mv + 0.01 * cmv
        d_hat = (1 - P.gamma_d) * d_hat + P.gamma_d * eta
        Z = np.maximum(Z + En - Ebp, 0)
        E_cum += En
        if rec_ts:
            TS['Z'][t] = Z; TS['k'][t] = k; TS['zone'][t] = valid & (dist <= P.zone)
            TS['tau'][t] = np.where(valid, np.minimum(tau, 1e3), np.nan); TS['u'][t] = np.where(valid, u, 0)
            TS['bh'][t] = eta; TS['req'][t] = len(idx); TS['hit'][t] = Hn[idx].sum()
        if follower == 'cmarl':
            sb2 = min(int(d_hat / eta_scale * nb), nb - 1)
            rew = r - mu * En
            Qtab[idx, sb2, k[idx]] += cm_lr * (rew[idx] - Qtab[idx, sb2, k[idx]])
            mu = np.maximum(mu + cm_dual * (En - Ebar), 0)
        # ---------------- metrics
        if t >= P.warm:
            L['act'] += len(idx); L['r'].append(r.sum()); L['bh'].append(eta); Zsum += Z.mean()
            for j, n in enumerate(idx):
                m = mm[n]
                lat = dk[k[n]] + b[m, rep[n]] / max(Ra[j], 1e-3) + (1 - Hn[n]) * eta_d / P.Rbh \
                    + (b[m, rep[n]] / P.Rx2 if fromnb[n] else 0.0)
                L['lat'].append(lat); L['dl'] += lat <= tau[n]
                if opt.get('record_lat'):
                    L.setdefault('comp', []).append((dk[k[n]], b[m, rep[n]] / max(Ra[j], 1e-3),
                                                     (1 - Hn[n]) * eta_d / P.Rbh, b[m, rep[n]] / P.Rx2 if fromnb[n] else 0.0))
                L['succ'] += (rep[n] == ost[n]) and lat <= tau[n]
                if rec_ts:
                    TS['succ'][t] += (rep[n] == ost[n]) and lat <= tau[n]
                L['hit'] += Hn[n]; L['nbh'] += fromnb[n]; L['live'] += islive[n]
                if Hn[n] and not islive[n]:
                    L['age'].append((t - t_fetch[m, rep[n]]) % P.T_ref)
                L['correct'] += rep[n] == ost[n]; L['sem'] += (rep[n] == ost[n]) * Hn[n]
                bi = min(np.searchsorted(TB, tau[n], side='right') - 1, nbin - 1)
                BIN['n'][bi] += 1; BIN['dl'][bi] += lat <= tau[n]; BIN['correct'][bi] += rep[n] == ost[n]
                BIN['tier'][bi] += k[n] + 1; BIN['util'][bi] += r[n]
    Tm = P.T - P.warm
    act = max(L['act'], 1); lat = np.array(L['lat']) * 1000
    nonlive = max(L['act'] - L['live'], 1)
    res_h = Hins
    res = dict(util=float(np.sum(L['r']) / act), hit=L['hit'] / act,
                hit_map=(L['hit'] - L['live']) / nonlive, correct=L['correct'] / act, sem=L['sem'] / act,
                deadline=L['dl'] / act, backhaul=float(np.mean(L['bh'])),
                lat_mean=float(lat.mean()) if len(lat) else np.nan,
                lat_p95=float(np.percentile(lat, 95)) if len(lat) else np.nan,
                lat_p99=float(np.percentile(lat, 99)) if len(lat) else np.nan,
                energy_ratio_max=float(np.max(E_cum / P.T / Ebar)), energy_ratio=float(np.mean(E_cum / P.T / Ebar)),
                Zbar=Zsum / Tm, nb_frac=L['nbh'] / max(L['hit'], 1), age_mean=float(np.mean(L['age'])) if L['age'] else np.nan,
                bh_miss=float(np.mean(L['miss'])), bh_ref=float(np.mean(L['ref'])), bh_ins=float(np.mean(L['ins'])),
                act_per_slot=act / Tm, kappa_max=float(max(KAP)) if KAP else np.nan,
                success=L['succ'] / act,
                brgap_mean=float(np.mean(GAP)) if GAP else np.nan, brgap_p99=float(np.percentile(GAP, 99)) if GAP else np.nan,
                kappa_p99=float(np.percentile(KAP, 99)) if KAP else np.nan, kappa_mean=float(np.mean(KAP)) if KAP else np.nan, bins={kk: vv.tolist() for kk, vv in BIN.items()})
    res['T_h'] = float(res_h)
    if rec_ts:
        res['ts'] = TS
    if opt.get('record_lat'):
        res['lat'] = lat.astype(np.float32)
        res['lat_comp'] = np.array(L.get('comp', []), np.float32)
    if opt.get('record_x'):
        res['xtrace'] = xtrace
    return res
