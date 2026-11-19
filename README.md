# Tiny-BEV-Cache simulation code

Simulation code for the paper *Tiny-BEV-Cache: Deadline-Aware Inference and Cooperative BEV Caching* (S. Hosny and J. Tadrous, under review). It produces every result and figure in Section VI. Requires Python 3.9 or newer, NumPy, SciPy and Matplotlib; the SUMO traces are generated with `experiments2.py traces` if they are not present.

The results in the current paper come from `experiments4.py` and `results4/`
(see "Revised results" below). The `results2/` and `results3/` folders hold the
runs of the earlier draft, before the review changes, and are kept for reference.

Python 3.10+, with `numpy`, `matplotlib`, and `eclipse-sumo` (1.28, which provides
`sumo` and `sumolib`): `pip install numpy matplotlib eclipse-sumo`.

## Files

| File | Purpose |
|---|---|
| `mobility.py` | Builds the 3GPP TR 37.885 urban grid (5x5, 433x250 m blocks, 2 lanes, 60 km/h, 90 s fixed-time signals) with SUMO `netgenerate`, generates random-walk routes with the TR 37.885 turning probabilities, runs SUMO, and stores per-second traces in `traces/`. |
| `bevsim2.py` | The simulator: TR 37.885 path loss, assets, intents and confusion matrices, Tiny-BEV-Cache (leader placement and logit follower), and all baselines. `run(Params(...), seed, method, **opt)` returns the metrics of one run. |
| `experiments2.py` | Stages `traces`, `tune`, `main`, `sweeps a`, `sweeps b`; writes `results2/<stage>.json`. |
| `main_extra.py` | Deadline-blind and C-MARL-Coop rows of Table II (`results2/main_extra.json`). |
| `kappa_check.py` | Contraction modulus kappa of Theorem 2 along simulated trajectories (`results2/kappa.json`). |
| `figs2.py` | Earlier 10-seed versions of Figs. 2-5 (`python3 figs2.py all`); also holds the shared colors and markers. |
| `experiments3.py` | The 20-seed sweeps behind Figs. 2-5 of the paper: stages `traces`, `budget`, `hw`, `V`, `cache`, `tref`, `dens`, `bh`. One run per line in `results3/<stage>.jsonl`, so an interrupted stage resumes where it stopped. |
| `run3.sh` | Runs all stages of `experiments3.py` in order. |
| `figs3.py` | Figs. 2-5 of the paper with shaded 95% confidence bands (`python3 figs3.py budget bh leader follower`). |
| `summ3.py` | Prints the per-point means quoted in Section VI (`python3 summ3.py hw`, `V`, `budget`, `cache`, `tref`, `bh`, `dens`). |
| `grid.net.xml` | The generated SUMO network. |

Method names in the code: `tinybev` (Tiny-BEV-Cache), `tinybev_h` (hard rule),
`tinybev_route` (route look-ahead), `agnostic` (direction-agnostic), `noncoop`,
`static`, `coop_lfu`, `oracle`, `cmarl` (C-MARL; with `leader='coop_lfu'` it is
C-MARL-Coop), `lru_ft` (fixed tier), `energy_agnostic` (max tier). Deadline-blind is
`tinybev` with `Params(infer_aware=False)`.

## Reproducing the results

```
python3 experiments2.py traces     # SUMO traces for all seeds (about 360 MB)
python3 experiments2.py tune       # tuning on seeds 100-102
python3 experiments2.py main       # Table II, 30 seeds
python3 main_extra.py              # two extra rows of Table II
python3 experiments2.py sweeps a   # budget, backhaul x density
python3 experiments2.py sweeps b   # cache, T_ref, V, hardware, correlated intents, route sharing
python3 kappa_check.py             # kappa of Theorem 2
./run3.sh                          # 20-seed sweeps for Figs. 2-5 (about 9 h on two cores)
python3 figs3.py budget bh leader follower   # figures of the paper
```

Each stage uses two processes (`Pool(2)`); change the pool size in the scripts to use
more cores. On two cores, `main` takes about 25 min and each sweep stage about 75 min.
The `results2/` and `results3/` folders hold the results used in the paper, so `figs3.py`
runs without re-simulating. Table II, the kappa values, correlated intents, and route
sharing come from `results2/`; Figs. 2-5 come from `results3/`.

## Revised results (current paper)

After the internal review, `bevsim2.py` gained the task reward (a request earns
delivery quality only if its label is correct and its tier meets the deadline),
deconvolved intent estimates, busy vehicles, per-host insertion counting, the
placement horizon `H_ins`, masked C-MARL, the best-response gap, and the optional
FIFO backhaul queue, contact-time error, correlated intents, and confusion share.
New `Params` fields default to the settings of the paper.

| File | Purpose |
|---|---|
| `experiments4.py` | All runs of the current paper. Stages: `main` (Table II, 30 seeds), `trace` (common placement trace, 10 seeds), `checks` (kappa and best-response gap, contact-time error, FIFO queue, confusion share, correlated intents, route sharing), `tune` (C-MARL grid on seeds 100-102), and the 20-seed sweeps `budget`, `hwtref`, `V`, `cacheA`, `cacheB`, `dens`, `bh1`, `bh2`, `bh3`, and `budgetH` (Tiny-BEV-Cache-H over the budgets, 10 seeds). One run per line in `results4/<stage>.jsonl`; stages resume where they stopped. |
| `run4.sh` | Runs the stages given as arguments in order, e.g. `./run4.sh main trace checks`. |
| `summ4.py` | Means and 95% t-intervals per method and point: `python3 summ4.py main`. |
| `exp_figs.py` | Extra runs for the figures: `ts` (per-slot traces, 10 seeds), `lat` (latency samples and mean latency components with N=400 at 50 and 200 Mb/s, 10 seeds; `results4/lat/`, `results4/latc/`), and `tref` (T_ref sweep for Non-cooperative, Cooperative LFU and Full-information, 10 seeds; `results4/tref_more.jsonl`). |
| `exp_bench.py` | Full-information placement benchmarks over 10 seeds: hit maximization without cost terms (`fi_hit`) and backhaul minimization over insertion horizons T_h from 10 to 6000 s (`fi_H<T_h>`); results in `results4/bench.jsonl`. |
| `figs5.py` | Figs. 2-6 of the paper: `python3 figs5.py time follower budget bh leader` (PDF to `figures/`, PNG here). `figs4.py` holds the earlier versions and shared helpers. |

```
python3 experiments2.py traces            # SUMO traces, if not present
./run4.sh tune main trace checks          # Table II, common trace, robustness checks
./run4.sh budget budgetH hwtref V cacheA cacheB dens bh1 bh2 bh3   # sweeps (about 16 h on two cores)
python3 exp_figs.py ts; python3 exp_figs.py lat; python3 exp_figs.py tref
python3 exp_bench.py
python3 figs5.py time follower budget bh leader
```

C-MARL was re-tuned under the task reward; the grid chose the same settings except
`eta_scale` (40 instead of 20), with a utility difference below 0.001 and one tuning
seed over budget, so the original settings were kept.

The per-slot traces (`results4/ts/`, about 90 MB) and latency samples (`results4/lat/`, about 36 MB)
are not included in the archive to keep it small. They are deterministic by seed:
`python3 exp_figs.py ts` and `python3 exp_figs.py lat` regenerate them (about 1.5 h on two cores).
