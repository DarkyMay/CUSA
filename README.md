# CUSA — Severity rating as measurement



The repository contains the measurement chain, the calibration and the referral rule of the paper (`core/`), and the scripts that produced every table and figure (`experiments/`). It is research code: paths to data and weights are constants at the top of each script and must be edited for a new machine (see *Data layout*).

## Layout

```
core/
  pavement_codes.py        standard mapping: JTG 5210 / ASTM D6433 thresholds and reductions,
                           skeleton + distance-transform width profile, Euler-characteristic gate
  geometry_layer.py        width profile, reductions, pixel-to-millimetre scale, tilt bound (Eq. 6)
  closeloop_sutcrack_v2.py mask generation with SAM3 (text prompt "crack", semantic head, logit 0)
                           and end-to-end width measurement on SUT-Crack; writes the per-image CSV
                           that every severity experiment reads
  dump_masks.py            writes the SAM3 masks of every inference protocol to disk
  p3_prior_shift.py        prior correction of the calibrated probability to a survey's
                           exceedance rate (Eq. 13)
experiments/               one script per table or section (see table below)
scripts/                   shell chains used to run the experiments overnight
draw/                      figure scripts
logs/                      the logged outputs the tables were transcribed from
```

## Which script produced what

| Paper | Script |
|---|---|
| Table 4, Section 4.2.1 (segmentation baselines) | `train_baselines.py`, `train_sam3_v3_textonly.py`, `train_sam3_lora.py`, `e8_hybrid_segmentor_baseline.py`, `h_hs_prep.py` |
| Table 5, Section 4.2.2 (width collapse, mitigations) | `e5e7_timing_collapse.py`, `e1_collapse_mitigation.py`, `e1b_subepoch.py`; calibration on the checkpoints: `p_width_dependent.py` (P2) |
| Section 4.2.3 (prompt sweep) | `d_prompt_sweep.py` |
| Figure 3, Section 4.3.1 (sensitivity to the protocol) | `e2e3_sensitivity_rank.py`, `e6_yolo_mask_source.py` |
| Table 6, Figure 4, Section 4.3.2 (granularity model) | `probe_width_mechanism.py`, `e6_yolo_mask_source.py`, `j_external_datasets.py` (UAV-CrackX) |
| Table 7, Figure 5, Section 4.4 (calibration) | `e1_calibration_reanalysis.py` |
| Table 8 (end-to-end comparison) | `e10_endtoend_comparison.py` |
| Table 9 (thresholds, reductions, estimators) | `b_estimators_gating.py`; three-parameter test: `p_width_dependent.py` (P1) |
| Table 10 (referral threshold, prevalence) | `a_csv_analyses.py` (A1), `p3_prior_shift.py` |
| Section 4.4, porting and interactions | `a_csv_analyses.py` (A2, A3) |
| Section 4.4, tilt bound (reprojection, distortion) | `e4_scale_sensitivity.py`, `c_tilt_simulation.py` |
| Table 11, Section 4.5 (applicability gate) | `b_estimators_gating.py` (B2), `g_rdd_gate.py` |
| Tables 12–13, Figure 6, Section 4.6 (field check) | `f_field_tilt.py`, `k_click_width.py` (manual width picking, run locally) |
| Section 4.6 (public datasets) | `j_external_datasets.py` |
| Section 4.1 (data separation) | `a_csv_analyses.py` (A4) |

## Requirements

Python 3.10+, PyTorch 2.11, `transformers` 5.14 (for SAM3), and the packages in `requirements.txt`. All experiments were run on one RTX 4090 with bfloat16 autocast; the calibration and analysis scripts (`a_*`, `b_*`, `p*`, `e1_calibration_reanalysis.py`, `e10_*`) need no GPU and read the CSVs produced by `closeloop_sutcrack_v2.py` and `dump_masks.py`.

```
pip install -r requirements.txt
```

## Weights and data

Not included; obtain from their sources and place as below (or edit the path constants at the top of each script).

| Item | Source |
|---|---|
| SAM3 | Meta AI, via `transformers` (`Sam3Model`, `Sam3Processor`); weights under `sam3_weights/` |
| SUT-Crack | Segmentation masks, crack lengths and acquisition metadata; `SUT-Crack/1-Segmentation/{Original_Image,Ground_Truth}` |
| dataset_400, DeepCrack, NHA12D, Crack500 | as published; see Section 4.1 |
| RDD2022 | China_Drone, China_MotorBike, Japan subsets |
| CRKWH100, CrackLS315 | Zou et al., 2019 |
| UAV-CrackX | Shan et al., 2025 (`UAV-CrackX-Datasets`) |
| Hybrid-Segmentor | authors' repository and released weights |
| YOLO26-seg-s weights | from our earlier work (DuEx-Crack) |

Default layout assumed by the scripts:

```
/root/autodl-tmp/
  proj_bridge/            this repository (core/ and experiments/ flattened)
  proj_bridge/results/    CSV and TXT outputs
  sam3_weights/
  SUT-Crack/
  dataset_400_clean/  DeepCrack-master/  NHA12D/  CRACK500/  RDD2022/  ext/  field/
```

## Running the chain on one image

```python
import numpy as np, pavement_codes as pc
# mask: binary numpy array from SAM3 (see closeloop_sutcrack_v2.seg)
st = pc.measure(mask, mm_per_px=0.2219)       # width profile, reductions, gate
w  = st["w_p95_mm"]                            # reduction prescribed by ASTM D6433
# calibrated probability, Eq. 11, with (b, sigma) from e1_calibration_reanalysis.py
from scipy.stats import norm
P = norm.cdf((np.log(w) - b - np.log(10.0)) / sigma)
# prior correction to a survey exceedance rate eta1 from the calibration rate eta0, Eq. 13
odds = P / (1 - P) * (eta1 / (1 - eta1)) / (eta0 / (1 - eta0)); P = odds / (1 + odds)
```

The complete procedure is Algorithm 1 of the paper; `closeloop_sutcrack_v2.py` runs it over SUT-Crack and `e1_calibration_reanalysis.py` fits `(b, sigma)` and evaluates the referral rule.

## Reproducing a table

Each experiment script tees its console output to `results/<name>_<timestamp>.txt`; the files in `logs/` are the runs the paper reports. Run order: `closeloop_sutcrack_v2.py` → `dump_masks.py` → the analysis scripts; `scripts/run_overnight.sh` and `scripts/run_gpu_chain.sh` give the order used for the revision experiments.

## Licence

MIT (see `LICENSE`).
