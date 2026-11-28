# Classifier tiers trained on DAIR-V2X-C and nuScenes

Code for the real-data study of the paper (section "Classifier Tiers Trained on Real Driving Data").
The datasets are not included; download them from their providers.

| Step | Script | Output |
|---|---|---|
| Needed-sector labels and past motion, DAIR-V2X-C | `intent_sectors.py <DAIR-V2X-C root> intent40.csv intent40_motion.npy` | labels, 14-element motion vectors |
| BEV input grids, DAIR-V2X-C | `build_bev.py` (uses `intent40.csv`) | `bev_intent.npz` |
| Labels, motion and BEV grids, nuScenes | `nusc_intent.py <v1.0-trainval dir> nusc_intent` | `nusc_intent_fov.npz`, `nusc_intent_360.npz` |
| Train the four tiers, 5-fold CV grouped by sequence/scene | `run_queue.sh d1 d2 n1 n2 n3` (calls `train_tiers.py`) | confusion matrices, predictions, all-frames models, `*_summary.json` |
| Physics baseline and cross-dataset test | `cross_eval.py` | `cross_eval.json` |
| Motion-only baseline | `motion_only.py` | `motion_only.json` |
| int8 quantization check | `quant_check.py` | printed |
| Simulator input (confusion profiles, STM32H747 latency estimates) | `make_mk.py tiers_intent`, then `make_mk5.py` | `tiers_intent_sim.json`, `tiers_intent5_sim.json` |
| Simulations with the measured tiers | `python3 exp_v3.py dair5`, `python3 exp_v3.py dair5_scarce` (repository root) | `results5/*.jsonl` |

`tiers_intent_sim.json` and `tiers_intent5_sim.json` are included, so the simulations run without the datasets.
`dair_sectors.py` holds the calibration helpers that `intent_sectors.py` uses.
