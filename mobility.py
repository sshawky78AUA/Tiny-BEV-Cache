"""SUMO mobility on the 3GPP TR 37.885 urban grid.

Builds the grid (433 m x 250 m blocks, 2 lanes per direction, 60 km/h, fixed-time
signals), generates long random-walk routes with the TR 37.885 turning probabilities
(straight 0.5, left 0.25, right 0.25), runs SUMO, and stores a compact per-second trace
of the equipped vehicles (those running Tiny-BEV-Cache). Background vehicles create
queues at the signals but do not request assets.
"""
import os, subprocess, gzip, json
import numpy as np
import sumo, sumolib
import xml.etree.ElementTree as ET

SUMO_HOME = sumo.SUMO_HOME
BIN = os.path.join(SUMO_HOME, 'bin')
HERE = os.path.dirname(os.path.abspath(__file__))
G = 5
LETTERS = 'ABCDE'


def build_net(path=os.path.join(HERE, 'grid.net.xml'), cycle=90):
    if os.path.exists(path):
        return path
    subprocess.run([os.path.join(BIN, 'netgenerate'), '--grid', '--grid.x-number', str(G),
                    '--grid.y-number', str(G), '--grid.x-length', '433', '--grid.y-length', '250',
                    '--default.lanenumber', '2', '--default.speed', '16.67',
                    '--tls.guess', 'true', '--tls.default-type', 'static',
                    '--tls.cycle.time', str(cycle), '--no-turnarounds', 'true',
                    '-o', path], check=True, capture_output=True)
    return path


def node_index(nid):
    """RSU index m = x + G*y for node id like 'B3' (x letter, y digit)."""
    return LETTERS.index(nid[0]) + G * int(nid[1:])


def direction(n_from, n_to):
    """approach direction of travel: 0 east, 1 west, 2 north, 3 south."""
    xf, yf = LETTERS.index(n_from[0]), int(n_from[1:])
    xt, yt = LETTERS.index(n_to[0]), int(n_to[1:])
    if yt == yf:
        return 0 if xt > xf else 1
    return 2 if yt > yf else 3


def random_walk(net, rng, n_edges):
    edges = net.getEdges()
    e = edges[rng.integers(len(edges))]
    route = [e]
    while len(route) < n_edges:
        cur = route[-1]
        outs = [o for o in cur.getToNode().getOutgoing() if o.getToNode() != cur.getFromNode()]
        d0 = direction(cur.getFromNode().getID(), cur.getToNode().getID())
        w = []
        for o in outs:
            d1 = direction(o.getFromNode().getID(), o.getToNode().getID())
            w.append(0.5 if d1 == d0 else 0.25)       # TR 37.885: straight 0.5, left/right 0.25
        w = np.array(w) / sum(w)
        route.append(outs[rng.choice(len(outs), p=w)])
    return [x.getID() for x in route]


def make_routes(net, seed, n_eq, n_bg, T, path):
    rng = np.random.default_rng(seed)
    n_edges = int(T * 17 / 233) + 20                  # long enough for the whole run
    lines = ['<routes>',
             '<vType id="eq" accel="2.6" decel="4.5" sigma="0.5" length="5" speedFactor="normc(1,0.1,0.6,1.3)">'
             '<param key="has.fcd.device" value="true"/></vType>',
             '<vType id="bg" accel="2.6" decel="4.5" sigma="0.5" length="5" speedFactor="normc(1,0.1,0.6,1.3)"/>']
    veh = []
    for i in range(n_eq + n_bg):
        typ = 'eq' if i < n_eq else 'bg'
        dep = float(rng.uniform(0, 300))
        veh.append((dep, i, typ, random_walk(net, rng, n_edges)))
    veh.sort()
    routes = {}
    for dep, i, typ, r in veh:
        lines.append(f'<vehicle id="{typ}{i}" type="{typ}" depart="{dep:.1f}" departLane="best" departSpeed="max">'
                     f'<route edges="{" ".join(r)}"/></vehicle>')
        if typ == 'eq':
            routes[f'{typ}{i}'] = r
    lines.append('</routes>')
    open(path, 'w').write('\n'.join(lines))
    return routes


def simulate(seed, n_eq=200, n_bg=600, T=6000, out_dir=os.path.join(HERE, 'traces')):
    """Return path of the compact trace npz for (seed, n_eq, n_bg, T)."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f's{seed}_n{n_eq}_b{n_bg}_T{T}'
    npz = os.path.join(out_dir, tag + '.npz')
    if os.path.exists(npz):
        return npz
    netp = build_net()
    net = sumolib.net.readNet(netp)
    rou = os.path.join(out_dir, tag + '.rou.xml')
    fcd = os.path.join(out_dir, tag + '.fcd.xml')
    routes = make_routes(net, seed, n_eq, n_bg, T, rou)
    subprocess.run([os.path.join(BIN, 'sumo'), '-n', netp, '-r', rou, '--begin', '0', '--end', str(T),
                    '--step-length', '1', '--fcd-output', fcd, '--device.fcd.explicit', '',
                    '--fcd-output.attributes', 'lane,pos,speed', '--seed', str(seed),
                    '--no-step-log', 'true', '--no-warnings', 'true', '--time-to-teleport', '120'],
                   check=True, capture_output=True)
    # parse: per equipped vehicle and second -> edge index, pos, speed
    eid = {e.getID(): j for j, e in enumerate(net.getEdges())}
    elen = np.array([e.getLength() for e in net.getEdges()])
    names = sorted(routes, key=lambda s: int(s[2:]))
    vid = {v: i for i, v in enumerate(names)}
    E = np.full((T, len(names)), -1, np.int16); X = np.zeros((T, len(names)), np.float32)
    U = np.zeros((T, len(names)), np.float32)
    for ev, el in ET.iterparse(fcd, events=('end',)):
        if el.tag == 'timestep':
            t = int(float(el.get('time')))
            if t < T:
                for v in el:
                    i = vid.get(v.get('id'))
                    if i is None:
                        continue
                    lane = v.get('lane')
                    if lane.startswith(':'):       # inside a junction: keep previous edge
                        continue
                    E[t, i] = eid[lane.rsplit('_', 1)[0]]; X[t, i] = float(v.get('pos')); U[t, i] = float(v.get('speed'))
            el.clear()
    # carry junction-internal steps forward from the previous second
    for t in range(1, T):
        miss = E[t] < 0
        E[t, miss] = E[t - 1, miss]; X[t, miss] = X[t - 1, miss]; U[t, miss] = U[t - 1, miss]
    R = np.array([[eid[e] for e in routes[v]] for v in names], np.int16)
    np.savez_compressed(npz, E=E, X=X, U=U, R=R, elen=elen)
    os.remove(fcd)
    return npz


def edge_tables(netp=None):
    net = sumolib.net.readNet(netp or build_net())
    ed = net.getEdges()
    to_m = np.array([node_index(e.getToNode().getID()) for e in ed])
    from_m = np.array([node_index(e.getFromNode().getID()) for e in ed])
    dirn = np.array([direction(e.getFromNode().getID(), e.getToNode().getID()) for e in ed])
    return to_m, from_m, dirn


if __name__ == '__main__':
    import time
    t0 = time.time(); p = simulate(0, n_eq=200, n_bg=600, T=6000); print(p, time.time() - t0)
