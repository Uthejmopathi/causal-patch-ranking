# A2P5 – A Causality-Guided Program Repair Framework with Counterfactual Regression Verification for AI-Generated Code

Application: repair of AI-generated blockchain smart contracts (P2P energy trading), Track 6, ICE2CPT 2026.
Author: Uthej Mopathi.
Proposed method: multi-task FT-Transformer for regression-risk ranking of candidate patches, with a counterfactual
regression-verification gate. Baselines: test-pass-only, random selection, LightGBM, MLP.

## Contents
- `data/` – synthetic dataset `A2P5_contract_patch_candidates.csv` (13,650 candidate patches, 2,500 bugs) and its README
- `code/` – `run_all.py` (entire pipeline), `fig1_flowchart.dot`, `requirements.txt`, `README.md`
- `results/` – all figures (JPG), all tables (CSV), `metrics.json`, run log

License: MIT

The data are fully synthetic (generator seed 2051); no real contracts or vulnerabilities are included.

## Reproduce
```
pip install -r code/requirements.txt   # plus graphviz for Fig. 1
cd code && python3 run_all.py           # writes ../results and ../figures
```
