import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
E = json.load(open('cross_eval.json')); T = json.load(open('tiers_intent_sim.json'))
lat = [0.1] + [d * 1e3 for d in T['d_h7']]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
for a, ds, name in ((ax[0], 'dair', 'DAIR-V2X-C'), (ax[1], 'nusc', 'nuScenes')):
    keys = [f'{ds} ctrv sign1'] + [f'{ds} in-domain CV w{w}' for w in (4, 8, 16, 32)]
    for met, mk, lab in (('top1', 'o-', 'all frames'), ('turn', 's--', 'turning frames'), ('within1', '^:', 'within one sector')):
        v = [100 * E[k][met] for k in keys]
        a.plot(lat, v, mk, ms=4, lw=1.2, label=lab)
    a.axhline(100 * E[f'{ds} majority']['top1'], color='gray', lw=0.8)
    a.text(0.12, 100 * E[f'{ds} majority']['top1'] + 1.5, 'always straight', color='gray', fontsize=7)
    a.set_xscale('log'); a.set_xlabel('estimated STM32H747 latency (ms)'); a.set_title(f'({"ab"[ds == "nusc"]}) {name}', fontsize=9)
    a.set_xticks(lat); a.set_xticklabels(['phys.', '10', '30', '99', '312'])
    a.set_ylim(0, 100); a.grid(alpha=0.3)
ax[0].set_ylabel('accuracy (%)'); ax[0].legend(fontsize=7, loc='center right', bbox_to_anchor=(1.0, 0.38))
plt.tight_layout(); plt.savefig('../paper/Tiny-BEV-Cache_VC_latex/fig_tiers.pdf'); plt.savefig('/tmp/claude-0/-home-claude/45f53616-7253-5a94-ba04-43bc668a5521/scratchpad/fig_tiers.png', dpi=110)
