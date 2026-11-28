"""Rasterise each labelled frame's own (vehicle-side) LiDAR detections into a BEV grid.

Input to the on-board classifier = what the car itself perceives; target = sector of the nearest
object only the RSU sees (from dair_sectors.py). Grid: G x G cells covering +-R m around the ego
car, x forward. Channels: 0 vehicles (car/van/bus/truck), 1 vulnerable road users, 2 heading cos,
3 heading sin (written on the vehicle footprint cells).

usage: python3 build_bev.py <root> <sectors.csv> <out.npz> [G=64] [R=80]
"""
import json, os, sys, csv
import numpy as np

VEH = {'car', 'van', 'bus', 'truck'}


def footprint(x, y, l, w, yaw, n=5):
    u, v = np.meshgrid(np.linspace(-l / 2, l / 2, n), np.linspace(-w / 2, w / 2, n))
    c, s = np.cos(yaw), np.sin(yaw)
    return (x + c * u - s * v).ravel(), (y + s * u + c * v).ravel()


def main(root, sec_csv, out, G=64, R=80.0):
    vinfo = {os.path.basename(v['image_path']).split('.')[0]: v
             for v in json.load(open(os.path.join(root, 'vehicle-side', 'data_info.json')))}
    rows = list(csv.DictReader(open(sec_csv)))
    X = np.zeros((len(rows), 4, G, G), np.float32)
    y = np.zeros(len(rows), np.int16); grp = []; loc = []
    cell = 2 * R / G
    for i, r in enumerate(rows):
        v = vinfo[r['frame']]
        y[i] = int(r['sector']); grp.append(v['batch_id']); loc.append(v.get('intersection_loc') or 'na')
        for o in json.load(open(os.path.join(root, 'vehicle-side', v['label_lidar_std_path']))):
            p = o['3d_location']; d = o['3d_dimensions']
            xs, ys = footprint(p['x'], p['y'], d['l'], d['w'], o['rotation'])
            ci = ((R - xs) / cell).astype(int); cj = ((R - ys) / cell).astype(int)   # row = forward, col = left
            ok = (ci >= 0) & (ci < G) & (cj >= 0) & (cj < G)
            ch = 0 if o['type'].lower() in VEH else 1
            X[i, ch, ci[ok], cj[ok]] = 1.0
            if ch == 0:
                X[i, 2, ci[ok], cj[ok]] = np.cos(o['rotation']); X[i, 3, ci[ok], cj[ok]] = np.sin(o['rotation'])
    np.savez_compressed(out, X=X, y=y, grp=np.array(grp), loc=np.array(loc),
                        frame=np.array([r['frame'] for r in rows]))
    print(out, X.shape, 'occupied cells/frame', float((X[:, :2].sum((1, 2, 3))).mean()))


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 64, float(a[5]) if len(a) > 5 else 80.0)
