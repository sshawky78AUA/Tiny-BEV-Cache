"""Intent-based need: the sector of the road the car is about to drive into.

Ego poses come from the vehicle-side calibration chain (LiDAR -> NovAtel -> world) at every frame.
Frames are split into continuous segments (gap < 0.5 s). For a frame at time t, the future path is
the ego positions over (t, t + Tmax]; the need point is the first one at least L metres of path
length away. Its bearing in the current LiDAR frame (x forward) gives the needed sector.
Frames whose car does not cover L metres within Tmax (stopped or very slow) are dropped.

Inputs for the classifier are also written: the car's past motion over the last Hpast seconds
(positions in the current frame, resampled at 0.2 s) and its speed and yaw rate.

Usefulness check with the cooperative labels: how often the needed sector contains an object only
the RSU sees, versus a random sector.

usage: python3 intent_sectors.py <root> <out.csv> <out_motion.npy> [S=40] [L=25] [Tmax=8]
"""
import json, os, sys, csv, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dair_sectors import load, world_to_lidar

HPAST, DT = 1.0, 0.2


def pose(root, v):
    ln = load(os.path.join(root, 'vehicle-side', v['calib_lidar_to_novatel_path']))['transform']
    nw = load(os.path.join(root, 'vehicle-side', v['calib_novatel_to_world_path']))
    Rln, tln = np.array(ln['rotation']), np.array(ln['translation']).reshape(3)
    Rnw, tnw = np.array(nw['rotation']), np.array(nw['translation']).reshape(3)
    return Rnw @ Rln, Rnw @ tln + tnw          # LiDAR frame -> world


def main(root, out, out_mot, S=40, L=25.0, Tmax=8.0):
    V = load(os.path.join(root, 'vehicle-side', 'data_info.json'))
    by = collections.defaultdict(list)
    for v in V:
        by[v['batch_id']].append(v)
    info = {}
    for b, lst in by.items():
        lst.sort(key=lambda v: int(v['image_timestamp']))
        t = np.array([int(v['image_timestamp']) for v in lst]) / 1e6
        P = [pose(root, v) for v in lst]
        seg = np.concatenate([[0], np.cumsum(np.diff(t) > 0.5)])
        for i, v in enumerate(lst):
            fid = os.path.basename(v['image_path']).split('.')[0]
            R, p0 = P[i]
            j = [k for k in range(len(lst)) if seg[k] == seg[i]]
            fut = [k for k in j if t[i] < t[k] <= t[i] + Tmax]
            past = [k for k in j if t[i] - HPAST - 0.05 <= t[k] <= t[i]]
            loc = lambda k: (P[k][1] - p0) @ R          # world -> current LiDAR frame
            need = None
            if fut:
                pts = np.array([np.zeros(3)] + [loc(k) for k in fut])
                s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
                if s[-1] >= L:
                    q = pts[np.argmax(s >= L)]
                    need = float(np.degrees(np.arctan2(q[1], q[0])) % 360)
            # past motion resampled to fixed times
            tp = np.array([t[k] - t[i] for k in past]); pp = np.array([loc(k)[:2] for k in past])
            grid = np.arange(-HPAST, 1e-6, DT)
            if len(past) >= 2 and tp.min() <= -HPAST + 0.15:
                mx = np.interp(grid, tp, pp[:, 0]); my = np.interp(grid, tp, pp[:, 1])
                yaw = [np.arctan2(*(P[k][0][:, 0] @ R)[[1, 0]]) for k in past]
                yr = (yaw[-1] - yaw[0]) / max(1e-3, tp[-1] - tp[0])
                spd = np.hypot(mx[-1] - mx[-2], my[-1] - my[-2]) / DT
                mot = np.r_[mx, my, spd, yr].astype(np.float32)
            else:
                mot = None
            info[fid] = (need, mot)
    coop = load(os.path.join(root, 'cooperative', 'data_info.json'))
    vinf = {os.path.basename(v['image_path']).split('.')[0]: v for v in V}
    rows, M = [], []; n_slow = n_nomot = 0; useful = rnd = 0.0
    for c in coop:
        vid = os.path.basename(c['vehicle_image_path']).split('.')[0]
        need, mot = info.get(vid, (None, None))
        if need is None:
            n_slow += 1; continue
        if mot is None:
            n_nomot += 1; continue
        sec = int(((need + 180 / S) % 360) // (360 / S)) % S     # sector 0 centred on straight ahead
        v = vinf[vid]; T = world_to_lidar(root, v)
        cw = load(os.path.join(root, c['cooperative_label_path']))
        vl = load(os.path.join(root, 'vehicle-side', v['label_lidar_std_path']))
        Pc = T(np.array([[o['3d_location'][k] for k in 'xyz'] for o in cw]).reshape(-1, 3))
        Pv = np.array([[o['3d_location'][k] for k in 'xyz'] for o in vl]).reshape(-1, 3)
        dmin = np.linalg.norm(Pc[:, None, :2] - Pv[None, :, :2], axis=2).min(1) if len(Pv) else np.full(len(Pc), 9e9)
        rg = np.hypot(Pc[:, 0], Pc[:, 1]); hid = (dmin > 2) & (rg < 80) & (rg > 3)
        hs = set((np.degrees(np.arctan2(Pc[hid, 1], Pc[hid, 0])) % 360 // (360 / S)).astype(int).tolist())
        useful += sec in hs; rnd += len(hs) / S
        rows.append(dict(frame=vid, sector=sec, bearing=round(need, 2), batch=v['batch_id'],
                         rsu_obj_in_sector=int(sec in hs)))
        M.append(mot)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.save(out_mot, np.array(M))
    n = len(rows)
    print(f'cooperative frames {len(coop)}  labelled {n}  dropped slow/stopped {n_slow}  no past motion {n_nomot}')
    print(f'P(needed sector holds an RSU-only object) {useful / n:.3f}   random sector {rnd / n:.3f}')
    h = np.bincount([r['sector'] for r in rows], minlength=S)
    print('sector histogram', h.tolist(), ' majority share', round(h.max() / n, 3))


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], a[3], *(int(a[4]),) if len(a) > 4 else ())
