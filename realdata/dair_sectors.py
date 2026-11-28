"""Build occlusion-sector labels from DAIR-V2X-C.

For every cooperative frame we take the fused world-frame labels (cooperative/label_world),
move them into the vehicle LiDAR frame, and drop every object the vehicle already annotated
in its own LiDAR label. What is left is seen only from the roadside unit. The label of the
frame is the angular sector (S equal bins of bearing around the ego car) of the nearest such
hidden object: the region the vehicle would need to fetch from the RSU.

usage: python3 dair_sectors.py <root of cooperative-vehicle-infrastructure> <out.csv> [S] [Rmax]
"""
import json, os, sys, csv
import numpy as np


def load(p):
    with open(p) as f:
        return json.load(f)


def world_to_lidar(root, vinfo):
    ln = load(os.path.join(root, 'vehicle-side', vinfo['calib_lidar_to_novatel_path']))['transform']
    nw = load(os.path.join(root, 'vehicle-side', vinfo['calib_novatel_to_world_path']))
    Rln, tln = np.array(ln['rotation']), np.array(ln['translation']).reshape(3)
    Rnw, tnw = np.array(nw['rotation']), np.array(nw['translation']).reshape(3)
    # p_w = Rnw (Rln p_l + tln) + tnw  ->  p_l = Rln^T (Rnw^T (p_w - tnw) - tln)
    return lambda P: ((P - tnw) @ Rnw - tln) @ Rln


def main(root, out, S=40, Rmax=80.0, match=2.0, Rego=3.0):
    coop = load(os.path.join(root, 'cooperative', 'data_info.json'))
    vinf = {os.path.basename(v['image_path']).split('.')[0]: v
            for v in load(os.path.join(root, 'vehicle-side', 'data_info.json'))}
    rows, stats = [], dict(frames=0, no_hidden=0, chk=[])
    for c in coop:
        vid = os.path.basename(c['vehicle_image_path']).split('.')[0]
        v = vinf.get(vid)
        if v is None:
            continue
        stats['frames'] += 1
        T = world_to_lidar(root, v)
        cw = load(os.path.join(root, c['cooperative_label_path']))
        vl = load(os.path.join(root, 'vehicle-side', v['label_lidar_std_path']))
        Pc = T(np.array([[o['3d_location'][k] for k in 'xyz'] for o in cw]).reshape(-1, 3))
        Pv = np.array([[o['3d_location'][k] for k in 'xyz'] for o in vl]).reshape(-1, 3)
        if len(Pv) and len(Pc):
            d = np.linalg.norm(Pc[:, None, :2] - Pv[None, :, :2], axis=2)
            stats['chk'].append(np.median(d.min(0)))   # every vehicle-labelled object should reappear in the fused label
            hidden = d.min(1) > match
        else:
            hidden = np.ones(len(Pc), bool)
        rng = np.linalg.norm(Pc[:, :2], axis=1)
        hidden &= (rng < Rmax) & (rng > Rego)        # drop the ego car itself, which the RSU also labels
        if not hidden.any():
            stats['no_hidden'] += 1
            continue
        i = np.where(hidden)[0][np.argmin(rng[hidden])]
        bearing = np.arctan2(Pc[i, 1], Pc[i, 0]) % (2 * np.pi)
        sec = int(bearing / (2 * np.pi) * S) % S
        rows.append(dict(frame=vid, image=os.path.join('vehicle-side', v['image_path']),
                         sector=sec, n_hidden=int(hidden.sum()), range_m=round(float(rng[i]), 2),
                         x=round(float(Pc[i, 0]), 2), y=round(float(Pc[i, 1]), 2)))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    chk = np.array(stats['chk'])
    print(f"frames {stats['frames']}  labelled {len(rows)}  without hidden object {stats['no_hidden']}")
    print(f"registration check: median distance from vehicle-labelled objects to the fused label {np.median(chk):.2f} m")
    h = np.bincount([r['sector'] for r in rows], minlength=S)
    print('sector histogram', h.tolist())


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 40, float(a[4]) if len(a) > 4 else 80.0)
