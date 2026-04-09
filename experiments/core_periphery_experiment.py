"""
Sandbox Experiment: Core vs Periphery as Signal Carriers
=========================================================
Tests whether the tight squamous core (largest k=9 spectral cluster)
carries most of the classification signal vs the peripheral genes.

SANDBOX CONSTRAINTS:
- Reads from data/processed/ only — no writes back
- No imports from src/ — classifier logic reimplemented locally
- No modifications to any existing .npy feature files
- Results printed to console only
- Pipeline files identical before and after running this script
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, "data", "processed")

# ── Load shared inputs (read-only) ─────────────────────────────────────────
print("Loading data...")
go_matrix = np.load(os.path.join(PROC, "go_similarity_matrix.npy"))
with open(os.path.join(PROC, "similarity_genes.json")) as f:
    sim_genes = json.load(f)          # 84 genes ordered to match matrix rows/cols

train_df = pd.read_csv(os.path.join(PROC, "train.csv"), index_col=0)
val_df   = pd.read_csv(os.path.join(PROC, "val.csv"),   index_col=0)

y_train = train_df["cancer_type"].values
y_val   = val_df["cancer_type"].values
X_train = train_df.drop(columns=["cancer_type"])
X_val   = val_df.drop(columns=["cancer_type"])

print(f"  Train: {X_train.shape}  Val: {X_val.shape}")
print(f"  GO matrix: {go_matrix.shape}  GO genes: {len(sim_genes)}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — k=9 spectral clustering → identify core (largest cluster)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: Spectral clustering at k=9 → core / periphery split")
print("="*60)

mat = np.clip(go_matrix, 0, 1)
np.fill_diagonal(mat, 1)

sc = SpectralClustering(
    n_clusters=9,
    affinity="precomputed",
    random_state=42,
    assign_labels="kmeans"
)
labels = sc.fit_predict(mat)

modules = {}
for i, lbl in enumerate(labels):
    modules.setdefault(lbl, []).append(sim_genes[i])

# Largest cluster = core
core_module_id = max(modules, key=lambda k: len(modules[k]))
core_genes     = modules[core_module_id]
periph_genes   = [g for mid, genes in modules.items()
                  if mid != core_module_id for g in genes]

sizes = sorted(len(v) for v in modules.values())
print(f"\nModule sizes (k=9): {sizes}")
print(f"Core module size  : {len(core_genes)} genes")
print(f"Periphery size    : {len(periph_genes)} genes")

print(f"\nCore genes ({len(core_genes)}):")
for g in sorted(core_genes):
    print(f"  {g}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature extraction (expression-only, RF-compatible)
# Same logic as pipeline: PCA fitted on train + raw gene values
# RF config: 100 trees, max_depth=10  (matches expression-only baseline)
# ══════════════════════════════════════════════════════════════════════════

def extract_expr_features(X_tr, X_vl, gene_subset, n_pca=20, n_top_var=30):
    """
    Fit PCA on train, extract PCA + top-variance raw values.
    All fitting strictly on train subset — no leakage.
    gene_subset: list of gene names to restrict to (must exist in X columns)
    """
    cols = [g for g in gene_subset if g in X_tr.columns]

    X_tr_sub = X_tr[cols]
    X_vl_sub = X_vl[cols]

    # PCA
    n_comp = min(n_pca, len(cols), X_tr_sub.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    tr_pca = pca.fit_transform(X_tr_sub)
    vl_pca = pca.transform(X_vl_sub)

    # Top-variance genes (fitted on train only)
    top_var = X_tr_sub.var().nlargest(min(n_top_var, len(cols))).index.tolist()
    tr_top = X_tr_sub[top_var].values
    vl_top = X_vl_sub[top_var].values

    tr_feat = np.hstack([tr_pca, tr_top])
    vl_feat = np.hstack([vl_pca, vl_top])
    return tr_feat, vl_feat, cols


def run_cv(X_tr, X_vl, y_tr, y_vl, n_folds=5, n_trees=100, max_depth=10):
    """
    5-fold CV on combined train+val, then also report held-out val accuracy.
    Returns (cv_mean, cv_std, val_acc).
    """
    X_all = np.vstack([X_tr, X_vl])
    y_all = np.concatenate([y_tr, y_vl])

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    y_tr_enc = le.transform(y_tr)
    y_vl_enc = le.transform(y_vl)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    for tr_idx, vl_idx in skf.split(X_all, y_enc):
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                    random_state=42)
        rf.fit(X_all[tr_idx], y_enc[tr_idx])
        cv_scores.append(accuracy_score(y_enc[vl_idx], rf.predict(X_all[vl_idx])))

    # Held-out val accuracy (train on full train set)
    rf_final = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                      random_state=42)
    rf_final.fit(X_tr, y_tr_enc)
    val_acc = accuracy_score(y_vl_enc, rf_final.predict(X_vl))

    return np.mean(cv_scores), np.std(cv_scores), val_acc


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Run all three classifiers
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: Running classifiers")
print("="*60)

# ── Classifier 1: Core-only ────────────────────────────────────────────────
print(f"\n[1/3] Core-only ({len(core_genes)} genes)...")
tr_core, vl_core, core_cols = extract_expr_features(X_train, X_val, core_genes)
print(f"      Feature dim: {tr_core.shape[1]}")
cv_core, std_core, val_core = run_cv(tr_core, vl_core, y_train, y_val)
print(f"      CV: {cv_core:.3f} ± {std_core:.3f}   Val: {val_core:.3f}")

# ── Classifier 2: Periphery-only ──────────────────────────────────────────
print(f"\n[2/3] Periphery-only ({len(periph_genes)} genes)...")
tr_peri, vl_peri, peri_cols = extract_expr_features(X_train, X_val, periph_genes)
print(f"      Feature dim: {tr_peri.shape[1]}")
cv_peri, std_peri, val_peri = run_cv(tr_peri, vl_peri, y_train, y_val)
print(f"      CV: {cv_peri:.3f} ± {std_peri:.3f}   Val: {val_peri:.3f}")

# ── Classifier 3: Full DEG baseline (all GO-annotated genes + rest of DEGs)
# Use all columns in train.csv (all 100 DEGs) to match existing baseline
print(f"\n[3/3] Full DEG set ({X_train.shape[1]} genes)...")
tr_full, vl_full, _ = extract_expr_features(
    X_train, X_val, X_train.columns.tolist(), n_pca=20, n_top_var=30
)
print(f"      Feature dim: {tr_full.shape[1]}")
cv_full, std_full, val_full = run_cv(tr_full, vl_full, y_train, y_val)
print(f"      CV: {cv_full:.3f} ± {std_full:.3f}   Val: {val_full:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\n{'Classifier':<30} {'Genes':>6} {'CV Acc':>9} {'CV Std':>8} {'Val Acc':>9}")
print("-" * 65)
print(f"{'Full DEG set':<30} {X_train.shape[1]:>6} {cv_full:>8.1%}  {std_full:>7.1%}  {val_full:>8.1%}")
print(f"{'Core-only (largest cluster)':<30} {len(core_cols):>6} {cv_core:>8.1%}  {std_core:>7.1%}  {val_core:>8.1%}")
print(f"{'Periphery-only (rest)':<30} {len(peri_cols):>6} {cv_peri:>8.1%}  {std_peri:>7.1%}  {val_peri:>8.1%}")

print(f"\nCore vs Full DEG gap   : {cv_core - cv_full:+.1%}")
print(f"Periphery vs Full gap  : {cv_peri - cv_full:+.1%}")
print(f"Core vs Periphery gap  : {cv_core - cv_peri:+.1%}")


# ══════════════════════════════════════════════════════════════════════════
# INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

CORE_CLOSE  = cv_core  >= cv_full - 0.03   # within 3pp of full set
PERI_LOWER  = cv_peri  <  cv_full - 0.03   # >3pp below full set
CORE_BEATS_PERI = cv_core > cv_peri + 0.01 # core meaningfully beats periphery

print(f"\nCore within 3pp of full set : {CORE_CLOSE}  "
      f"({cv_core:.1%} vs {cv_full:.1%}, gap={cv_core-cv_full:+.1%})")
print(f"Periphery >3pp below full   : {PERI_LOWER}  "
      f"({cv_peri:.1%} vs {cv_full:.1%}, gap={cv_peri-cv_full:+.1%})")
print(f"Core meaningfully > peri    : {CORE_BEATS_PERI}  "
      f"({cv_core:.1%} vs {cv_peri:.1%}, gap={cv_core-cv_peri:+.1%})")

if CORE_CLOSE and PERI_LOWER and CORE_BEATS_PERI:
    verdict = (
        "SIGNAL IS CONCENTRATED IN THE CORE. The squamous differentiation "
        "gene cluster carries most of the LUAD/LUSC classification signal. "
        "The periphery genes add noise rather than useful signal — removing "
        "them does not hurt performance."
    )
elif CORE_CLOSE and not PERI_LOWER:
    verdict = (
        "SIGNAL IS DISTRIBUTED. Core alone matches full-set performance, "
        "but periphery is also competitive. Both gene groups carry signal — "
        "the core/periphery split does not map cleanly onto a signal/noise split."
    )
elif not CORE_CLOSE and PERI_LOWER:
    verdict = (
        "SIGNAL REQUIRES BOTH GROUPS. Core alone underperforms the full set "
        "and periphery is also weak — the classification requires the combination "
        "of both functional and peripheral expression signals."
    )
else:
    verdict = (
        "MIXED RESULT. Inspect CV values directly — no clean biological "
        "interpretation emerges from the core/periphery split."
    )

print(f"\nVerdict: {verdict}")
print("\n[Sandbox complete — no pipeline files were modified]")
