#!/usr/bin/env python3
"""A2P5 - Causality-guided patch selection with counterfactual regression verification for AI-generated
smart contracts. Proposed: multi-task FT-Transformer (feature-tokenizer transformer) that ranks candidate patches
by regression risk and fix likelihood. Baselines: test-pass-only, random selection, LightGBM, MLP.

Runs everything (preprocessing, 5-fold grouped CV, baselines, ablation, sensitivity, statistics, figures).
Reads ../data/A2P5_contract_patch_candidates.csv, writes ../results and ../figures.
"""
import os, json, time, subprocess, shutil
import numpy as np, pandas as pd
import torch, torch.nn as nn
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "A2P5_contract_patch_candidates.csv")
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEED = 2026
T0 = time.time()

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 100,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})
COL = {"FT-Transformer (proposed)": "#c0392b", "LightGBM": "#2c7fb8", "MLP": "#7fcdbb",
       "Test-pass-only": "#888888", "Random": "#bbbbbb"}


def savefig(fig, name):
    p = os.path.join(FIG, name + ".png")
    fig.savefig(p, dpi=300); plt.close(fig)
    Image.open(p).convert("RGB").save(os.path.join(RES, name + ".jpg"), quality=92)


# ----------------------------------------------------------------------------------------------------------------
# 1. Data and preprocessing
# ----------------------------------------------------------------------------------------------------------------
df = pd.read_csv(DATA)
raw_missing = df.isna().sum()
TARGETS = ["fixes_target_bug", "introduces_regression", "hidden_regression"]
# reported_ts, bug_id, patch_id: identifiers only; fixes/introduces/hidden: oracle outcomes (never features)
CAT = ["contract_type", "bug_type", "repair_strategy"]
NUM_BASE = ["contract_functions", "contract_state_vars", "external_calls", "existing_tests", "existing_test_coverage",
            "patch_lines_changed", "functions_touched", "state_vars_written", "causal_slice_size", "dependency_fanout",
            "counterfactual_trace_divergence", "gas_delta_pct", "modifiers_changed", "passes_existing_tests"]
CTX = ["n_candidates", "cfd_rel", "slice_rel", "fanout_rel", "lines_rel"]   # bug-context (label-free) features
CAUSAL = ["causal_slice_size", "dependency_fanout", "slice_rel", "fanout_rel"]
CF = ["counterfactual_trace_divergence", "cfd_rel"]

n_out = {}


def preprocess(tr, te):
    """Fit outlier limits and medians on the training fold only, apply to both."""
    tr = tr.copy(); te = te.copy()
    for d in (tr, te):
        d["existing_test_coverage"] = d["existing_test_coverage"].clip(0, 1)
    for c in ["patch_lines_changed", "gas_delta_pct", "causal_slice_size"]:
        med = tr[c].median(); mad = (tr[c] - med).abs().median() * 1.4826 + 1e-9
        lo, hi = med - 6 * mad, med + 6 * mad
        for d in (tr, te):
            m = (d[c] < lo) | (d[c] > hi)
            n_out[c] = n_out.get(c, 0) + int(m.sum()) if d is te else n_out.get(c, 0)
            d.loc[m, c] = np.nan            # treat gross outliers as missing, then impute
    for c in ["counterfactual_trace_divergence", "gas_delta_pct", "causal_slice_size", "patch_lines_changed"]:
        med = tr.groupby("repair_strategy")[c].median()
        for d in (tr, te):
            d[c] = d[c].fillna(d["repair_strategy"].map(med)).fillna(tr[c].median())
    for d in (tr, te):
        g = d.groupby("bug_id")
        d["n_candidates"] = g["patch_id"].transform("count")
        d["cfd_rel"] = d["counterfactual_trace_divergence"] - g["counterfactual_trace_divergence"].transform("median")
        d["slice_rel"] = d["causal_slice_size"] - g["causal_slice_size"].transform("median")
        d["fanout_rel"] = d["dependency_fanout"] - g["dependency_fanout"].transform("median")
        d["lines_rel"] = d["patch_lines_changed"] - g["patch_lines_changed"].transform("median")
        for c in ["patch_lines_changed", "causal_slice_size"]:
            d[c] = np.log1p(d[c])
    return tr, te


# ----------------------------------------------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------------------------------------------
class FTTransformer(nn.Module):
    """Feature tokenizer: numeric x_j -> x_j W_j + b_j, categorical -> embedding; [CLS] token; pre-norm encoder;
    two heads on [CLS]: regression risk and fix likelihood."""
    def __init__(self, n_num, cards, d=32, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_num, d) * 0.1); self.b = nn.Parameter(torch.zeros(n_num, d))
        self.emb = nn.ModuleList([nn.Embedding(k, d) for k in cards])
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=drop, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head_reg = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.head_fix = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, xn, xc):
        tok = xn.unsqueeze(-1) * self.W + self.b
        if len(self.emb):
            tok = torch.cat([tok, torch.stack([e(xc[:, i]) for i, e in enumerate(self.emb)], 1)], 1)
        tok = torch.cat([self.cls.expand(len(xn), -1, -1), tok], 1)
        h = self.norm(self.enc(tok)[:, 0])
        return self.head_reg(h).squeeze(-1), self.head_fix(h).squeeze(-1)


def encode(tr, te, num, cat):
    sc = StandardScaler().fit(tr[num])
    Xn_tr = sc.transform(tr[num]).astype(np.float32); Xn_te = sc.transform(te[num]).astype(np.float32)
    maps = [{v: i for i, v in enumerate(sorted(tr[c].unique()))} for c in cat]
    Xc_tr = np.stack([tr[c].map(m).values for c, m in zip(cat, maps)], 1).astype(np.int64) if cat else np.zeros((len(tr), 0), np.int64)
    Xc_te = np.stack([te[c].map(m).fillna(0).values for c, m in zip(cat, maps)], 1).astype(np.int64) if cat else np.zeros((len(te), 0), np.int64)
    return Xn_tr, Xc_tr, Xn_te, Xc_te, [len(m) for m in maps]


CACHE = os.path.join(RES, "_cache"); os.makedirs(CACHE, exist_ok=True)
FIT_SECONDS = [0.0]   # wall-clock of the last FT-Transformer fit (stored in the cache, so re-runs report the original time)


def fit_ftt(tr, te, num, cat, seed, d=32, layers=2, epochs=30, lam=0.5, curve=None, keep_model=False):
    """Cached wrapper: identical inputs give identical (seeded) outputs, so a re-run after an interruption resumes."""
    import hashlib
    key = hashlib.md5(repr((sorted(te.index[:50].tolist()), len(te), num, cat, seed, d, layers, epochs, lam)).encode()).hexdigest()[:16]
    f = os.path.join(CACHE, key + ".npz")
    if os.path.exists(f) and not keep_model and "sec" in np.load(f).files:
        z = np.load(f)
        if curve is not None: curve.extend([tuple(r) for r in z["curve"]])
        FIT_SECONDS[0] = float(z["sec"])
        return z["pr"], z["pf"], None, None
    c = []; t0 = time.time()
    out = _fit_ftt(tr, te, num, cat, seed, d, layers, epochs, lam, c)
    FIT_SECONDS[0] = time.time() - t0
    np.savez(f, pr=out[0], pf=out[1], curve=np.array(c), sec=FIT_SECONDS[0])
    if curve is not None: curve.extend(c)
    return out


def _fit_ftt(tr, te, num, cat, seed, d=32, layers=2, epochs=30, lam=0.5, curve=None):
    torch.manual_seed(seed); np.random.seed(seed)
    Xn_tr, Xc_tr, Xn_te, Xc_te, cards = encode(tr, te, num, cat)
    yr = tr["introduces_regression"].values.astype(np.float32); yf = tr["fixes_target_bug"].values.astype(np.float32)
    # inner validation split by bug for early stopping
    bugs = tr["bug_id"].unique(); rng = np.random.default_rng(seed); vb = set(rng.choice(bugs, len(bugs) // 8, replace=False))
    vm = tr["bug_id"].isin(vb).values; tm = ~vm
    m = FTTransformer(len(num), cards, d=d, layers=layers)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss()
    T = lambda a: torch.from_numpy(a)
    xn, xc, r, f = T(Xn_tr[tm]), T(Xc_tr[tm]), T(yr[tm]), T(yf[tm])
    vxn, vxc, vr, vf = T(Xn_tr[vm]), T(Xc_tr[vm]), T(yr[vm]), T(yf[vm])
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(xn)); tl = 0
        for i in range(0, len(xn), 512):
            idx = perm[i:i + 512]
            lr_, lf_ = m(xn[idx], xc[idx])
            loss = bce(lr_, r[idx]) + lam * bce(lf_, f[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        sched.step(); m.eval()
        with torch.no_grad():
            a, b = m(vxn, vxc); vl = (bce(a, vr) + lam * bce(b, vf)).item()
        if curve is not None: curve.append((ep + 1, tl / len(xn), vl))
        if vl < best - 1e-4: best, wait = vl, 0; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 6: break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        a, b = m(T(Xn_te), T(Xc_te))
    return torch.sigmoid(a).numpy(), torch.sigmoid(b).numpy(), m, (Xn_te, Xc_te)


def onehot(tr, te, num, cat):
    A = pd.get_dummies(pd.concat([tr[num + cat], te[num + cat]]), columns=cat, dtype=float)
    return A.iloc[:len(tr)].values, A.iloc[len(tr):].values


def fit_lgb(tr, te, num, cat, seed):
    out = []
    Xtr = tr[num + cat].copy(); Xte = te[num + cat].copy()
    for c in cat:
        Xtr[c] = Xtr[c].astype("category"); Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    for y in ["introduces_regression", "fixes_target_bug"]:
        g = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                               n_jobs=1, verbose=-1).fit(Xtr, tr[y])
        out.append(g.predict_proba(Xte)[:, 1])
    return out[0], out[1]


def fit_mlp(tr, te, num, cat, seed):
    Xtr, Xte = onehot(tr, te, num, cat)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    out = []
    for y in ["introduces_regression", "fixes_target_bug"]:
        m = MLPClassifier((64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                          n_iter_no_change=10, random_state=seed).fit(Xtr, tr[y])
        out.append(m.predict_proba(Xte)[:, 1])
    return out[0], out[1]


# ----------------------------------------------------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------------------------------------------------
def select(te, score, rng):
    """Pick one candidate per bug: argmax score, ties broken at random."""
    s = score + rng.random(len(score)) * 1e-9
    t = te.assign(_s=s)
    idx = t.groupby("bug_id")["_s"].idxmax()
    return t.loc[idx]


def evaluate(te, p_reg, p_fix, score, rng):
    y = te["introduces_regression"].values
    out = {}
    if p_reg is not None:
        out["roc_auc"] = roc_auc_score(y, p_reg); out["pr_auc"] = average_precision_score(y, p_reg)
        out["f1"] = f1_score(y, p_reg >= 0.5); out["brier"] = brier_score_loss(y, p_reg)
        # hidden regressions: among patches that pass the existing tests, flag top-risk ones at 0.5
        pm = te["passes_existing_tests"].values == 1
        h = te["hidden_regression"].values
        out["hidden_recall"] = float(((p_reg >= 0.5) & (h == 1)).sum() / max(h.sum(), 1))
        out["hidden_pr_auc"] = average_precision_score(h[pm], p_reg[pm])
    sel = select(te, score, rng)
    ok = (sel["fixes_target_bug"] == 1) & (sel["introduces_regression"] == 0)
    out["top1_success"] = float(ok.mean())
    out["sel_regression"] = float(sel["introduces_regression"].mean())
    out["sel_hidden_regression"] = float(sel["hidden_regression"].mean())
    out["sel_fix"] = float(sel["fixes_target_bug"].mean())
    out["oracle_top1"] = float(te.assign(ok=(te.fixes_target_bug == 1) & (te.introduces_regression == 0)).groupby("bug_id").ok.any().mean())
    return out, sel


def rank_score(p_reg, p_fix):
    return p_fix * (1 - p_reg)        # Eq. (8): expected probability of a correct, non-regressing patch


# ----------------------------------------------------------------------------------------------------------------
# 4. Main 5-fold grouped CV
# ----------------------------------------------------------------------------------------------------------------
NUM = NUM_BASE + CTX
gkf = GroupKFold(n_splits=5)
order = np.random.default_rng(SEED).permutation(df["bug_id"].unique())
grp = df["bug_id"].map({b: i for i, b in enumerate(order)}).values
folds = list(gkf.split(df, groups=grp))

rows, oof, curves, sels = [], {}, {}, {}
for k, (tri, tei) in enumerate(folds):
    tr, te = preprocess(df.iloc[tri], df.iloc[tei])
    rng = np.random.default_rng(SEED + k)
    preds = {}
    curve = []
    t = time.time(); pr, pf, model, Xte_enc = fit_ftt(tr, te, NUM, CAT, seed=k, curve=curve, keep_model=(k == 0))
    preds["FT-Transformer (proposed)"] = (pr, pf, FIT_SECONDS[0]); curves[k] = curve
    if k == 0: ftt0, te0, tr0 = model, te, tr
    t = time.time(); pr, pf = fit_lgb(tr, te, NUM, CAT, seed=k); preds["LightGBM"] = (pr, pf, time.time() - t)
    t = time.time(); pr, pf = fit_mlp(tr, te, NUM, CAT, seed=k); preds["MLP"] = (pr, pf, time.time() - t)
    for name, (pr, pf, sec) in preds.items():
        met, sel = evaluate(te, pr, pf, rank_score(pr, pf), rng)
        rows.append(dict(method=name, fold=k, train_s=sec, **met)); sels.setdefault(name, []).append(sel)
        o = oof.setdefault(name, {"idx": [], "reg": [], "fix": []})
        o["idx"].append(te.index.values); o["reg"].append(pr); o["fix"].append(pf)
    # test-pass-only: pick first candidate (in proposal order) that passes existing tests
    s_tp = te["passes_existing_tests"].values * 10 - te["patch_id"].str[-1].astype(int).values * 0.1
    met, sel = evaluate(te, None, None, s_tp, rng)
    met["roc_auc"] = roc_auc_score(te["introduces_regression"], 1 - te["passes_existing_tests"])
    met["pr_auc"] = average_precision_score(te["introduces_regression"], 1 - te["passes_existing_tests"])
    met["f1"] = f1_score(te["introduces_regression"], 1 - te["passes_existing_tests"])
    met["hidden_recall"] = 0.0
    rows.append(dict(method="Test-pass-only", fold=k, train_s=0.0, **met)); sels.setdefault("Test-pass-only", []).append(sel)
    met, sel = evaluate(te, None, None, rng.random(len(te)), rng)
    rows.append(dict(method="Random", fold=k, train_s=0.0, **met)); sels.setdefault("Random", []).append(sel)
    print(f"fold {k} done  {time.time() - T0:.0f}s", flush=True)

main = pd.DataFrame(rows); main.to_csv(os.path.join(RES, "main_comparison_per_fold.csv"), index=False)
METHODS = ["FT-Transformer (proposed)", "LightGBM", "MLP", "Test-pass-only", "Random"]
mcols = ["roc_auc", "pr_auc", "f1", "hidden_recall", "top1_success", "sel_regression", "sel_hidden_regression", "sel_fix", "train_s"]
agg = main.groupby("method")[mcols].agg(["mean", "std"]).reindex(METHODS)
tab = pd.DataFrame({c: [f"{agg.loc[m, (c, 'mean')]:.3f} ± {agg.loc[m, (c, 'std')]:.3f}" for m in METHODS] for c in mcols}, index=METHODS)
tab.to_csv(os.path.join(RES, "table_main_comparison.csv"))
print(tab.to_string())

# ----------------------------------------------------------------------------------------------------------------
# 5. Paired bootstrap over bugs (pooled out-of-fold selections)
# ----------------------------------------------------------------------------------------------------------------
def ok_by_bug(name):
    s = pd.concat(sels[name]); return s.set_index("bug_id").eval("fixes_target_bug == 1 and introduces_regression == 0").astype(float)

base = ok_by_bug("FT-Transformer (proposed)")
sig = []
brng = np.random.default_rng(7)
B = 2000
for other in METHODS[1:]:
    o = ok_by_bug(other).reindex(base.index)
    diff = (base - o).values
    boots = np.array([diff[brng.integers(0, len(diff), len(diff))].mean() for _ in range(B)])
    sig.append(dict(comparison=f"FT-T vs {other}", delta_top1=diff.mean(), ci_low=np.percentile(boots, 2.5),
                    ci_high=np.percentile(boots, 97.5), p_two_sided=min(1, 2 * min((boots <= 0).mean(), (boots >= 0).mean())),
                    ftt_only_ok=int((diff > 0).sum()), other_only_ok=int((diff < 0).sum())))
sig = pd.DataFrame(sig); sig.to_csv(os.path.join(RES, "table_significance.csv"), index=False); print(sig.to_string())

# ----------------------------------------------------------------------------------------------------------------
# 6. Ablation (FT-Transformer, 5 folds)
# ----------------------------------------------------------------------------------------------------------------
ABL = {"Full model": dict(num=NUM, cat=CAT, lam=0.5, score="joint"),
       "w/o counterfactual divergence": dict(num=[c for c in NUM if c not in CF], cat=CAT, lam=0.5, score="joint"),
       "w/o causal-slice features": dict(num=[c for c in NUM if c not in CAUSAL], cat=CAT, lam=0.5, score="joint"),
       "w/o CF + causal features": dict(num=[c for c in NUM if c not in CF + CAUSAL], cat=CAT, lam=0.5, score="joint"),
       "Single-task (lambda = 0)": dict(num=NUM, cat=CAT, lam=0.0, score="joint"),
       "w/o bug-context features": dict(num=NUM_BASE, cat=CAT, lam=0.5, score="joint"),
       "Risk-only ranking (no fix head)": dict(num=NUM, cat=CAT, lam=0.5, score="risk")}
acols = ["roc_auc", "pr_auc", "hidden_recall", "hidden_pr_auc", "top1_success", "sel_hidden_regression"]
arows = []
ABL_FOLDS = 3
for k, (tri, tei) in enumerate(folds[:ABL_FOLDS]):
    tr, te = preprocess(df.iloc[tri], df.iloc[tei]); rng = np.random.default_rng(SEED + k)
    for name, cfg in ABL.items():
        if name == "Full model":
            r = main[(main.method == "FT-Transformer (proposed)") & (main.fold == k)].iloc[0]
            arows.append(dict(variant=name, fold=k, **{c: r[c] for c in acols})); continue
        if name == "Risk-only ranking (no fix head)":
            pr = oof["FT-Transformer (proposed)"]["reg"][k]; pf = oof["FT-Transformer (proposed)"]["fix"][k]
            met, _ = evaluate(te, pr, pf, 1 - pr, rng)
        else:
            pr, pf, _, _ = fit_ftt(tr, te, cfg["num"], cfg["cat"], seed=k, lam=cfg["lam"])
            sc = 1 - pr if cfg["lam"] == 0 else rank_score(pr, pf)   # lambda=0: fix head untrained, rank by risk only
            met, _ = evaluate(te, pr, pf, sc, rng)
        arows.append(dict(variant=name, fold=k, **met))
    print(f"ablation fold {k} done  {time.time() - T0:.0f}s", flush=True)
abl = pd.DataFrame(arows); abl.to_csv(os.path.join(RES, "ablation_per_fold.csv"), index=False)
acols = ["roc_auc", "pr_auc", "hidden_recall", "hidden_pr_auc", "top1_success", "sel_hidden_regression"]
aagg = abl.groupby("variant")[acols].agg(["mean", "std"]).reindex(list(ABL))
atab = pd.DataFrame({c: [f"{aagg.loc[v, (c, 'mean')]:.3f} ± {aagg.loc[v, (c, 'std')]:.3f}" for v in ABL] for c in acols}, index=list(ABL))
atab.to_csv(os.path.join(RES, "table_ablation.csv")); print(atab.to_string())

# ----------------------------------------------------------------------------------------------------------------
# 7. Sensitivity: token dim x depth (folds 0-2) and verification threshold tau (risk-coverage)
# ----------------------------------------------------------------------------------------------------------------
srows = []
SENS_FOLDS = 2
for d, L in [(16, 2), (32, 2), (64, 2), (32, 1), (32, 3)]:
    for k in range(SENS_FOLDS):
        tri, tei = folds[k]; tr, te = preprocess(df.iloc[tri], df.iloc[tei])
        if d == 32 and L == 2:
            r = main[(main.method == "FT-Transformer (proposed)") & (main.fold == k)].iloc[0]
            srows.append(dict(d=d, layers=L, fold=k, pr_auc=r.pr_auc, top1_success=r.top1_success)); continue
        pr, pf, _, _ = fit_ftt(tr, te, NUM, CAT, seed=k, d=d, layers=L)
        met, _ = evaluate(te, pr, pf, rank_score(pr, pf), np.random.default_rng(SEED + k))
        srows.append(dict(d=d, layers=L, fold=k, pr_auc=met["pr_auc"], top1_success=met["top1_success"]))
    print(f"sensitivity d={d} L={L} done  {time.time() - T0:.0f}s", flush=True)
sens = pd.DataFrame(srows); sens.to_csv(os.path.join(RES, "sensitivity_arch_per_fold.csv"), index=False)
sens_t = sens.groupby(["d", "layers"])[["pr_auc", "top1_success"]].mean().reset_index()
sens_t.to_csv(os.path.join(RES, "table_sensitivity_arch.csv"), index=False)

# verification gate: accept selected patch only if p_reg <= tau, else escalate to human review
cov_rows = []
for name in ["FT-Transformer (proposed)", "LightGBM", "MLP"]:
    idx = np.concatenate(oof[name]["idx"]); pr = np.concatenate(oof[name]["reg"]); pf = np.concatenate(oof[name]["fix"])
    te = df.loc[idx].assign(p_reg=pr, p_fix=pf, s=rank_score(pr, pf))
    sel = te.loc[te.groupby("bug_id")["s"].idxmax()]
    for tau in np.round(np.arange(0.05, 1.001, 0.05), 2):
        acc = sel[sel.p_reg <= tau]
        cov_rows.append(dict(method=name, tau=tau, coverage=len(acc) / len(sel),
                             success_accepted=((acc.fixes_target_bug == 1) & (acc.introduces_regression == 0)).mean() if len(acc) else np.nan,
                             hidden_reg_accepted=acc.hidden_regression.mean() if len(acc) else np.nan,
                             shipped_hidden_per_100bugs=100 * acc.hidden_regression.sum() / len(sel)))
cov = pd.DataFrame(cov_rows); cov.to_csv(os.path.join(RES, "risk_coverage.csv"), index=False)

# ----------------------------------------------------------------------------------------------------------------
# 8. Strategy comparison (bootstrap CIs) and feature importance (permutation on fold 0)
# ----------------------------------------------------------------------------------------------------------------
st = []
for s, g in df.groupby("repair_strategy"):
    for col in ["fixes_target_bug", "introduces_regression", "hidden_regression"]:
        v = g[col].values; bs = [v[brng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
        st.append(dict(strategy=s, outcome=col, rate=v.mean(), ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5), n=len(v)))
st = pd.DataFrame(st); st.to_csv(os.path.join(RES, "strategy_rates.csv"), index=False)

Xn_te, Xc_te = encode(tr0, te0, NUM, CAT)[2:4]
yr0 = te0["introduces_regression"].values
with torch.no_grad():
    base_auc = average_precision_score(yr0, torch.sigmoid(ftt0(torch.from_numpy(Xn_te), torch.from_numpy(Xc_te))[0]).numpy())
imp = []
prng = np.random.default_rng(11)
for j, c in enumerate(NUM + CAT):
    drops = []
    for _ in range(5):
        xn, xc = Xn_te.copy(), Xc_te.copy()
        if j < len(NUM): xn[:, j] = xn[prng.permutation(len(xn)), j]
        else: xc[:, j - len(NUM)] = xc[prng.permutation(len(xc)), j - len(NUM)]
        with torch.no_grad():
            p = torch.sigmoid(ftt0(torch.from_numpy(xn), torch.from_numpy(xc))[0]).numpy()
        drops.append(base_auc - average_precision_score(yr0, p))
    imp.append(dict(feature=c, pr_auc_drop=np.mean(drops), std=np.std(drops)))
imp = pd.DataFrame(imp).sort_values("pr_auc_drop", ascending=False); imp.to_csv(os.path.join(RES, "feature_importance_ftt.csv"), index=False)

# ----------------------------------------------------------------------------------------------------------------
# 9. Dataset summary table
# ----------------------------------------------------------------------------------------------------------------
summ = [("Bugs (groups) / candidate patches", f"{df.bug_id.nunique():,} / {len(df):,}"),
        ("Candidates per bug (min–max, mean)", f"{df.groupby('bug_id').size().min()}–{df.groupby('bug_id').size().max()}, {df.groupby('bug_id').size().mean():.2f}"),
        ("Contract modules / bug classes / strategies", f"{df.contract_type.nunique()} / {df.bug_type.nunique()} / {df.repair_strategy.nunique()}"),
        ("Numeric + categorical input features", f"{len(NUM_BASE)} + {len(CAT)} (+{len(CTX)} bug-context)"),
        ("Patches passing existing tests", f"{df.passes_existing_tests.mean() * 100:.1f}%"),
        ("Fix target bug", f"{df.fixes_target_bug.mean() * 100:.1f}%"),
        ("Introduce regression (main target)", f"{df.introduces_regression.mean() * 100:.1f}%"),
        ("Hidden regression (passes tests)", f"{df.hidden_regression.mean() * 100:.1f}%"),
        ("Regressions hidden from tests", f"{df.hidden_regression.sum() / df.introduces_regression.sum() * 100:.1f}%"),
        ("Bugs with ≥1 correct patch (oracle)", f"{df.assign(ok=(df.fixes_target_bug == 1) & (df.introduces_regression == 0)).groupby('bug_id').ok.any().mean() * 100:.1f}%"),
        ("Missing: slice / CF div. / gas", f"{raw_missing['causal_slice_size']} / {raw_missing['counterfactual_trace_divergence']} / {raw_missing['gas_delta_pct']}")]
pd.DataFrame(summ, columns=["Item", "Value"]).to_csv(os.path.join(RES, "table_dataset_summary.csv"), index=False)

# ----------------------------------------------------------------------------------------------------------------
# 10. Figures
# ----------------------------------------------------------------------------------------------------------------
# Fig 2: training curves
fig, ax = plt.subplots(figsize=(3.3, 2.1))
for k, c in curves.items():
    c = np.array(c)
    ax.plot(c[:, 0], c[:, 1], color="#c0392b", alpha=0.35 + 0.1 * k, lw=1, label="train" if k == 0 else None)
    ax.plot(c[:, 0], c[:, 2], color="#2c7fb8", ls="--", alpha=0.35 + 0.1 * k, lw=1, label="validation" if k == 0 else None)
ax.set_xlabel("Epoch"); ax.set_ylabel(r"Multi-task loss $\mathcal{L}$, Eq. (6)"); ax.legend(title="5 folds")
savefig(fig, "fig2_training_curves")

# Fig 3: PR curves for regression (pooled out-of-fold)
from sklearn.metrics import precision_recall_curve
fig, ax = plt.subplots(figsize=(3.3, 2.4))
y_all = None
for name in ["FT-Transformer (proposed)", "LightGBM", "MLP"]:
    idx = np.concatenate(oof[name]["idx"]); pr = np.concatenate(oof[name]["reg"]); y = df.loc[idx, "introduces_regression"].values
    P, R, _ = precision_recall_curve(y, pr)
    ax.plot(R, P, color=COL[name], lw=1.3 if "proposed" in name else 1, label=f"{name.replace(' (proposed)', '')} (AP={average_precision_score(y, pr):.3f})")
    y_all = y
yt = df.loc[idx]
tp = (yt.passes_existing_tests == 0)
prec_tp = yt.introduces_regression[tp].mean(); rec_tp = yt.introduces_regression[tp].sum() / yt.introduces_regression.sum()
ax.scatter([rec_tp], [prec_tp], marker="s", color="#555555", zorder=5, s=18, label="Test-pass-only (fails tests)")
ax.axhline(y_all.mean(), color="k", ls=":", lw=0.8, label=f"Prevalence ({y_all.mean():.3f})")
ax.set_xlabel("Recall of regressing patches"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="lower left", fontsize=6.3)
savefig(fig, "fig3_pr_curves")

# Fig 4: top-1 success and selected hidden-regression rate per method
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.4))
for ax, met, lab in [(axs[0], "top1_success", "Top-1 correct patch rate"), (axs[1], "sel_hidden_regression", "Hidden regressions shipped")]:
    m = [agg.loc[x, (met, "mean")] for x in METHODS]; s = [agg.loc[x, (met, "std")] for x in METHODS]
    ax.bar(range(len(METHODS)), m, yerr=s, color=[COL[x] for x in METHODS], capsize=2, edgecolor="k", lw=0.4)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(["FT-T", "LGBM", "MLP", "Test-pass", "Random"], rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel(lab + " (fraction)", fontsize=7)
    for i, v in enumerate(m): ax.text(i, v + max(s) + 0.01, f"{v:.3f}", ha="center", fontsize=5.5, rotation=90)
axs[0].axhline(agg.loc[METHODS[0], ("oracle_top1", "mean")] if ("oracle_top1", "mean") in agg else main.oracle_top1.mean(), color="k", ls=":", lw=0.8)
axs[0].set_ylim(0, 1.15); axs[1].set_ylim(0, max(agg[("sel_hidden_regression", "mean")]) * 1.5)
fig.tight_layout(w_pad=0.8)
savefig(fig, "fig4_selection_comparison")

# Fig 5: risk-coverage curve for the verification gate
fig, ax = plt.subplots(figsize=(3.3, 2.3))
for name in ["FT-Transformer (proposed)", "LightGBM", "MLP"]:
    c = cov[cov.method == name]
    ax.plot(c.coverage, c.success_accepted, marker="o", ms=2.5, color=COL[name], lw=1.3 if "proposed" in name else 1, label=name.replace(" (proposed)", ""))
c = cov[(cov.method == "FT-Transformer (proposed)") & (cov.tau.isin([0.1, 0.2, 0.3, 0.5]))]
for _, r in c.iterrows(): ax.annotate(f"τ={r.tau:.1f}", (r.coverage, r.success_accepted), textcoords="offset points", xytext=(4, -9), fontsize=6)
ax.set_xlabel("Coverage (fraction of bugs auto-repaired)"); ax.set_ylabel("Correct-patch rate of accepted")
ax.legend(loc="lower left")
savefig(fig, "fig5_risk_coverage")

# Fig 6: permutation importance
fig, ax = plt.subplots(figsize=(3.3, 2.5))
top = imp.head(12).iloc[::-1]
ax.barh(top.feature.str.replace("counterfactual_trace_divergence", "cf_trace_divergence"), top.pr_auc_drop, xerr=top["std"],
        color=["#c0392b" if f in CF + CAUSAL else "#999999" for f in top.feature], edgecolor="k", lw=0.3, capsize=1.5)
ax.set_xlabel("Drop in regression PR-AUC when permuted"); ax.tick_params(axis="y", labelsize=6.5)
savefig(fig, "fig6_feature_importance")

# Fig 7: sensitivity (token dim d and depth L) + lambda/tau handled elsewhere
fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
for ax, sub, xcol, lab in [(axs[0], sens_t[sens_t.layers == 2], "d", "Token dimension d (L = 2)"),
                           (axs[1], sens_t[sens_t.d == 32], "layers", "Encoder layers L (d = 32)")]:
    ax.plot(sub[xcol].astype(str), sub.top1_success, marker="o", color="#c0392b", lw=1.2, label="Top-1 success")
    ax.plot(sub[xcol].astype(str), sub.pr_auc, marker="s", color="#2c7fb8", lw=1.2, label="Regression PR-AUC")
    ax.set_xlabel(lab, fontsize=7)
axs[0].set_ylabel("Score (mean of 2 folds)"); axs[0].set_ylim(0.75, 0.91)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.55, 1.04))
fig.tight_layout(w_pad=0.5, rect=(0, 0, 1, 0.92))
savefig(fig, "fig7_sensitivity")

# Fig 1 flowchart via graphviz
dot = os.path.join(HERE, "fig1_flowchart.dot")
subprocess.run(["dot", "-Tpng", "-Gdpi=300", dot, "-o", os.path.join(FIG, "fig1_flowchart.png")], check=True)
Image.open(os.path.join(FIG, "fig1_flowchart.png")).convert("RGB").save(os.path.join(RES, "fig1_flowchart.jpg"), quality=92)
shutil.copy(dot, os.path.join(RES, "fig1_flowchart.dot"))

# ----------------------------------------------------------------------------------------------------------------
# 11. metrics.json
# ----------------------------------------------------------------------------------------------------------------
def mm(m, c): return [round(float(agg.loc[m, (c, "mean")]), 4), round(float(agg.loc[m, (c, "std")]), 4)]
cf = cov[cov.method == "FT-Transformer (proposed)"].set_index("tau")
metrics = dict(
    main={m: {c: mm(m, c) for c in mcols} for m in METHODS},
    oracle_top1=round(float(main.oracle_top1.mean()), 4),
    significance=sig.round(4).to_dict(orient="records"),
    ablation={v: {c: [round(float(aagg.loc[v, (c, 'mean')]), 4), round(float(aagg.loc[v, (c, 'std')]), 4)] for c in acols} for v in ABL},
    sensitivity_arch=sens_t.round(4).to_dict(orient="records"),
    risk_coverage_ftt=cf.loc[[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]].round(4).reset_index().to_dict(orient="records"),
    feature_importance_top=imp.head(8).round(4).to_dict(orient="records"),
    epochs_used=[len(c) for c in curves.values()],
    outliers_flagged_test_folds=n_out, raw_missing={k: int(v) for k, v in raw_missing.items() if v},
    runtime_s=round(time.time() - T0, 1))
json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=1)
print(json.dumps(metrics["main"]["FT-Transformer (proposed)"], indent=0))
print(f"total runtime {time.time() - T0:.0f}s")
