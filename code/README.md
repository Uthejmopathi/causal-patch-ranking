# A2P5 code – multi-task FT-Transformer for regression-risk ranking of AI-generated smart-contract patches

## Run
```
pip install -r requirements.txt      # plus graphviz (`dot`) on the PATH
python3 run_all.py                    # reads ../data/A2P5_contract_patch_candidates.csv
```
Everything is seeded (fold split seed 2026, model seed = fold index, bootstrap seed 7). PyTorch and LightGBM are pinned
to one thread. Runtime was about 45 min on a shared 2-core CPU (the FT-Transformer fits dominate). Each FT-Transformer fit is
cached in ../results/_cache (predictions + fit time), so an interrupted run resumes; delete that folder for a clean run.

## What run_all.py does
1. Pre-processing, fitted on the training fold only: clip coverage to [0, 1], set gross outliers (|x - median| > 6 MAD) of
   diff size, gas delta and slice size to missing, impute with per-strategy medians, log1p of diff size and slice size, and five
   label-free bug-context features (number of candidates, candidate value minus the bug's median for CF divergence,
   slice, fan-out and diff size). `fixes_target_bug`, `introduces_regression` and `hidden_regression` are never used as features.
2. 5-fold GroupKFold by `bug_id` (no bug appears in both train and test).
3. Proposed: multi-task FT-Transformer (feature tokenizer + [CLS] + 2 pre-norm transformer layers, d = 32, 4 heads) with a
   regression-risk head and a fix head, loss BCE_r + 0.5 BCE_f, AdamW, cosine LR, early stopping on an inner bug-level validation split.
   Patches are ranked by s = P(fix) (1 - P(regression)); a gate tau decides between auto-merge and human audit.
4. Baselines: test-pass-only (first candidate that passes the existing tests), random selection, LightGBM and MLP (both
   trained on the same features for both targets and ranked with the same score).
5. Paired bootstrap over bugs for top-1 success, ablation (3 folds), architecture sensitivity (2 folds), risk-coverage
   curve over tau, strategy rates with bootstrap CIs, permutation importance, all figures.

## Outputs (../results; figures also as 300-dpi PNG in ../figures)
| file | content |
|---|---|
| table_dataset_summary.csv | dataset statistics (Table I) |
| main_comparison_per_fold.csv, table_main_comparison.csv | all metrics per method (per fold; mean ± std) |
| table_significance.csv | paired bootstrap of top-1 success, FT-Transformer vs each baseline |
| ablation_per_fold.csv, table_ablation.csv | ablation variants |
| sensitivity_arch_per_fold.csv, table_sensitivity_arch.csv | token dimension / depth sensitivity |
| risk_coverage.csv | coverage vs correctness of auto-merged patches for gate tau |
| strategy_rates.csv | fix / regression / hidden-regression rate per repair strategy with 95% CI |
| feature_importance_ftt.csv | permutation importance (drop in PR-AUC), fold 0 |
| metrics.json | headline numbers used in the paper |
| fig1_flowchart.dot/.jpg, fig2–fig7 *.jpg | figures |
| run_log.txt | console log |
