# Tiny-BEV-Cache simulation code

Simulation code for the paper *Tiny-BEV-Cache: Deadline-Aware Inference and Cooperative BEV Caching* (S. Hosny and J. Tadrous, under review). The simulator runs the full system: SUMO vehicle mobility on the 3GPP TR 37.885 urban grid, the vehicles' TinyML tier choice with Lyapunov energy control, cooperative RSU caching, the access and backhaul links, and all baselines. It produces every table and figure of Section VI, and you can rerun any of them with your own parameters.

## Install

Python 3.10 or newer.

```
pip install -r requirements.txt
```

This installs NumPy, SciPy, Matplotlib and SUMO (`eclipse-sumo` and `sumolib` from PyPI), so no separate SUMO installation is needed. The first run of each seed and vehicle count calls SUMO to build the vehicle traces (about 30 s) and stores them in `traces/`; later runs reuse them.

## Run your own simulations

`run_sim.py` runs any method with any parameter values and prints the metrics of Section VI: utility per active request, hit and semantic hit ratio, success (correct and on time), deadlines met, backhaul load, mean and 95th-percentile latency, and energy used over budget. With several seeds it adds 95% confidence half-widths.

```
python run_sim.py                                     # Tiny-BEV-Cache, default scenario, seed 0
python run_sim.py --methods tinybev noncoop cmarl --seeds 0 1 2 3 4
python run_sim.py --set N=400 Rbh_mbps=50 budget=0.1  # change any parameters
python run_sim.py --sweep cache_frac=0.05,0.15,0.3,0.5 --methods tinybev noncoop oracle --seeds 0 1 2
python run_sim.py --sweep V=0.5,2,10 --csv v_sweep.csv   # also save every run to CSV
python run_sim.py --list-params                       # all parameters, defaults and meaning
python run_sim.py --list-methods                      # all methods
```

The default scenario is the one in the paper (200 equipped vehicles, 200 Mb/s backhaul per RSU, STM32H747 tiers, budget 0.2, V = 2, nu = 0.2), and each method uses the tuned settings of the paper (`results2/tuned.json`), so `python run_sim.py --methods tinybev cmarl --seeds 0` reproduces the seed-0 values stored in `results4/main.jsonl` exactly. A run of the full 6,000-s horizon takes about 30 s per seed and method on one core; runs go in parallel over all cores (`--workers` to limit). Use `--set T=2000 warm=500` for quick tests.

Parameters you are likely to change:

| Parameter | Default | Meaning |
|---|---|---|
| `N` | 200 | equipped vehicles |
| `Rbh` / `Rbh_mbps` | 25 MB/s / 200 Mb/s | backhaul rate per RSU |
| `cache_frac` | 0.15 | cache size, as a share of the map-based volume |
| `budget` | 0.2 | energy budget, as a share of a_n E_max |
| `hw` | `'h7'` | classifier hardware: `'h7'` (STM32H747) or `'n6'` (STM32N6) |
| `V`, `nu`, `theta` | 2, 0.2, 0.03 | Lyapunov trade-off, entropy weight, budget tightening |
| `T_ref` | 300 | refresh period of cached assets (s) |
| `share` | 99 | route pairs a vehicle shares (1 = next intersection only) |
| `rho`, `beta`, `tau_err` | 0, 0.6, 0 | correlated intents, error structure, deadline-estimate error |
| `fifo` | False | FIFO backhaul queue instead of the per-slot batch model |

In your own Python code:

```python
from bevsim2 import Params, run
from experiments2 import cfg                 # tuned settings of the paper
pk, opt = cfg('tinybev')
out = run(Params(**dict(pk, N=400, budget=0.1)), seed=0, method='tinybev', **opt)
print(out['util'], out['hit'], out['success'], out['backhaul'])
```

## Reproducing the paper

The results of the paper are stored in `results4/` (and `results2/tuned.json`), so the figures can be redrawn without simulating. To rerun everything:

```
./run4.sh main trace checks                      # Table II, common placement trace, robustness checks
./run4.sh budget budgetH hwtref V cacheA cacheB dens bh1 bh2 bh3   # 20-seed sweeps
python3 exp_figs.py ts; python3 exp_figs.py lat; python3 exp_figs.py tref
python3 exp_bench.py                             # full-information benchmarks
python3 figs5.py time follower budget bh leader  # Figs. 2-6 (PDF to figures/, PNG here)
python3 summ4.py main                            # means and 95% intervals of any stage
```

All of it takes about 35 CPU-hours; the scripts use every core (set `WORKERS=n` to limit) and resume where they stopped, since each run is appended to `results4/<stage>.jsonl`.

| Paper item | Stage(s) | Plot |
|---|---|---|
| Table II | `main`, `trace` | `summ4.py main` |
| Fig. 2 (behavior over time) | `exp_figs.py ts` | `figs5.py time` |
| Fig. 3 (model selection under deadlines) | `hwtref`, `V` | `figs5.py follower` |
| Fig. 4 (energy budgets) | `budget`, `budgetH` | `figs5.py budget` |
| Fig. 5 (backhaul rate and density) | `bh1`-`bh3`, `dens`, `exp_figs.py lat` | `figs5.py bh` |
| Fig. 6 (cache size and freshness) | `cacheA`, `cacheB`, `hwtref`, `exp_figs.py tref` | `figs5.py leader` |
| Robustness and kappa checks | `checks` | `summ4.py checks` |
| Full-information benchmarks | `exp_bench.py` | (printed values in `results4/bench.jsonl`) |

The per-slot traces (`results4/ts/`) and latency samples (`results4/lat/`) are not in the repository because of their size (about 125 MB); they are deterministic by seed and `exp_figs.py` regenerates them.

## Main files

| File | Purpose |
|---|---|
| `bevsim2.py` | The simulator. `Params` holds every parameter; `run(Params(...), seed, method, **opt)` returns the metrics of one run. |
| `mobility.py` | Builds the TR 37.885 grid with SUMO, generates routes with the TR 37.885 turning probabilities, runs SUMO, and stores per-second traces. |
| `run_sim.py` | Command-line runner for your own parameters and sweeps. |
| `experiments4.py`, `run4.sh` | All runs of the paper. |
| `exp_figs.py`, `exp_bench.py` | Extra runs for Figs. 2, 5 and 6 and the full-information benchmarks. |
| `figs5.py`, `summ4.py` | Figures and summary tables. |
| `experiments2.py`, `experiments3.py`, `figs2.py`, `figs4.py` | Shared settings and plotting helpers used by the scripts above (they also hold the earlier draft's runs). |
| `grid.net.xml` | The generated SUMO network. |

C-MARL was re-tuned under the task reward of the current paper; the grid chose the same settings except `eta_scale` (40 instead of 20), with a utility difference below 0.001 and one tuning seed over budget, so the original settings were kept.

## Earlier draft

`results2/`, `results3/` and the scripts below produced the draft before the internal review, and are kept for reference.

### Files of the earlier draft

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

### Reproducing the earlier draft

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

On two cores, `main` takes about 25 min and each sweep stage about 75 min.
The `results2/` and `results3/` folders hold the results used in the paper, so `figs3.py`
runs without re-simulating. Table II, the kappa values, correlated intents, and route
sharing come from `results2/`; Figs. 2-5 come from `results3/`.


## Real-data study (DAIR-V2X-C and nuScenes)

`realdata/` holds the label, input and training pipelines for the classifier tiers trained on DAIR-V2X-C and nuScenes, the physics and motion-only baselines, and the simulator inputs built from the measured confusion matrices (`realdata/tiers_intent5_sim.json`). `python3 exp_v3.py dair5` and `python3 exp_v3.py dair5_scarce` rerun the simulations with these tiers; `python3 tab_dair5.py` prints the results table. See `realdata/README.md`.
