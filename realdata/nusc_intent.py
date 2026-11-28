"""nuScenes version of the intent task (same definition as intent_sectors.py + build_bev.py).

Ego poses: every LIDAR_TOP sample_data record (keyframes and 20 Hz sweeps). For each keyframe the
needed sector is the bearing, in the ego frame (x forward, y left), of the first future ego position
at least L metres of path away within Tmax; frames that do not cover L are dropped. Past motion: ego
positions over the last 1 s resampled every 0.2 s, plus speed and yaw rate (same 14-vector).
BEV input: annotations with at least one LiDAR point, moved to the ego frame and rasterised like
DAIR (vehicles / vulnerable road users / heading cos / sin). With fov=True only objects whose bearing
lies inside the DAIR vehicle-label field of view (-48.2 .. 56.0 deg) are kept, so inputs match DAIR.

usage: python3 nusc_intent.py <v1.0-trainval dir> <out prefix> [S=40] [L=25] [Tmax=8]
writes <prefix>_fov.npz and <prefix>_360.npz with X, y, grp (scene), mot
"""
import sys, os, json, collections
import numpy as np
import ijson
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bev import footprint

FOV = (-48.2, 56.0)
G, R = 64, 80.0
HPAST, DT = 1.0, 0.2


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def stream(path, keep):
    with open(path, 'rb') as f:
        for rec in ijson.items(f, 'item', use_float=True):
            r = keep(rec)
            if r is not None:
                yield r


def main(d, out, S=40, L=25.0, Tmax=8.0):
    J = lambda n: json.load(open(os.path.join(d, n)))
    sensor = {s['token']: s['channel'] for s in J('sensor.json')}
    lidar_cs = {c['token'] for c in J('calibrated_sensor.json') if sensor[c['sensor_token']] == 'LIDAR_TOP'}
    samples = {s['token']: s for s in J('sample.json')}
    scene_of = {s['token']: s['scene_token'] for s in samples.values()}
    cat = {c['token']: c['name'] for c in J('category.json')}
    inst_cat = {i['token']: cat[i['category_token']] for i in J('instance.json')}
    print('loading LIDAR_TOP sample_data ...', flush=True)
    sd = list(stream(os.path.join(d, 'sample_data.json'),
                     lambda r: (r['sample_token'], r['ego_pose_token'], r['timestamp'], r['is_key_frame'])
                     if r['calibrated_sensor_token'] in lidar_cs else None))
    need = {x[1] for x in sd}
    print(len(sd), 'lidar records; loading ego poses ...', flush=True)
    ego = dict(stream(os.path.join(d, 'ego_pose.json'),
                      lambda r: (r['token'], (r['translation'][0], r['translation'][1], yaw_of(r['rotation'])))
                      if r['token'] in need else None))
    # per-scene time series of ego poses
    ser = collections.defaultdict(list)
    key = {}
    for st, et, ts, kf in sd:
        sc = scene_of[st]
        x, y, yaw = ego[et]
        ser[sc].append((ts / 1e6, x, y, yaw))
        if kf:
            key[st] = (ts / 1e6, x, y, yaw)
    for sc in ser:
        ser[sc] = np.array(sorted(ser[sc]))
    print('loading annotations ...', flush=True)
    ann = collections.defaultdict(list)
    for st, c, t, s, q in stream(os.path.join(d, 'sample_annotation.json'),
                                 lambda r: (r['sample_token'], inst_cat[r['instance_token']], r['translation'],
                                            r['size'], r['rotation']) if r['num_lidar_pts'] > 0 else None):
        if c.startswith('vehicle.') and c not in ('vehicle.bicycle', 'vehicle.motorcycle'):
            ch = 0
        elif c.startswith('human.') or c in ('vehicle.bicycle', 'vehicle.motorcycle'):
            ch = 1
        else:
            continue
        ann[st].append((ch, t[0], t[1], s[1], s[0], yaw_of(q)))     # size = [w, l, h]
    cell = 2 * R / G
    res = {m: dict(X=[], y=[], grp=[], mot=[]) for m in ('fov', '360')}
    nslow = nmot = 0
    for st, (t0, x0, y0, h0) in key.items():
        A = ser[scene_of[st]]
        c, s = np.cos(h0), np.sin(h0)
        to_ego = lambda X, Y: ((X - x0) * c + (Y - y0) * s, -(X - x0) * s + (Y - y0) * c)
        fut = A[(A[:, 0] > t0) & (A[:, 0] <= t0 + Tmax)]
        if not len(fut):
            nslow += 1; continue
        fx, fy = to_ego(fut[:, 1], fut[:, 2])
        pts = np.c_[np.r_[0, fx], np.r_[0, fy]]
        sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        if sl[-1] < L:
            nslow += 1; continue
        q = pts[np.argmax(sl >= L)]
        sec = int(((np.degrees(np.arctan2(q[1], q[0])) + 180 / S) % 360) // (360 / S)) % S   # sector 0 centred on straight ahead
        past = A[(A[:, 0] >= t0 - HPAST - 0.05) & (A[:, 0] <= t0)]
        if len(past) < 2 or past[0, 0] > t0 - HPAST + 0.15:
            nmot += 1; continue
        px, py = to_ego(past[:, 1], past[:, 2]); tp = past[:, 0] - t0
        grid = np.arange(-HPAST, 1e-6, DT)
        mx, my = np.interp(grid, tp, px), np.interp(grid, tp, py)
        yr = (np.unwrap(past[:, 3])[-1] - np.unwrap(past[:, 3])[0]) / max(1e-3, tp[-1] - tp[0])
        spd = np.hypot(mx[-1] - mx[-2], my[-1] - my[-2]) / DT
        mot = np.r_[mx, my, spd, yr].astype(np.float32)
        for mode in res:
            Xg = np.zeros((4, G, G), np.float32)
            for ch, ax, ay, w, l, ya in ann.get(st, []):
                ex, ey = to_ego(ax, ay)
                b = np.degrees(np.arctan2(ey, ex))
                if mode == 'fov' and not (FOV[0] < b < FOV[1]):
                    continue
                if np.hypot(ex, ey) < 3.0:
                    continue
                xs, ys = footprint(ex, ey, l, w, ya - h0)
                ci = ((R - xs) / cell).astype(int); cj = ((R - ys) / cell).astype(int)
                ok = (ci >= 0) & (ci < G) & (cj >= 0) & (cj < G)
                Xg[ch, ci[ok], cj[ok]] = 1.0
                if ch == 0:
                    Xg[2, ci[ok], cj[ok]] = np.cos(ya - h0); Xg[3, ci[ok], cj[ok]] = np.sin(ya - h0)
            r = res[mode]
            r['X'].append(Xg); r['y'].append(sec); r['grp'].append(scene_of[st]); r['mot'].append(mot)
    for mode, r in res.items():
        np.savez_compressed(f'{out}_{mode}.npz', X=np.array(r['X']), y=np.array(r['y'], np.int16),
                            grp=np.array(r['grp']), mot=np.array(r['mot']))
    y = np.array(res['fov']['y'])
    print(f'keyframes {len(key)}  labelled {len(y)}  dropped slow {nslow}  no past motion {nmot}')
    h = np.bincount(y, minlength=S); print('sector histogram', h.tolist(), 'majority share', round(h.max() / len(y), 3))


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2])
