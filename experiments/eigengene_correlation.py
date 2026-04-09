"""
Sandbox Experiment: Eigengene Correlation — Within-Cancer-Type
==============================================================
Tests whether Cluster 1 (lineage identity module) eigengene
correlates with Clusters 4, 6, 7 (terminal effector modules)
within LUSC and LUAD samples separately.

This removes the binary label confounder and tests whether
the upstream-downstream regulatory story is supported by data.

SANDBOX: read-only on data/processed/, no pipeline imports, no writes.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
go_matrix = np.load(os.path.join(PROC, "go_similarity_matrix.npy"))
with open(os.path.join(PROC, "similarity_genes.json")) as f:
    sim_genes = json.load(f)

train_df = pd.read_csv(os.path.join(PROC, "train.csv"), index_col=0)
val_df   = pd.read_csv(os.path.join(PROC, "val.csv"),   index_col=0)
test_df  = pd.read_csv(os.path.join(PROC, "test.csv"),  index_col=0)

# Combine all splits for maximum statistical power
all_df = pd.concat([train_df, val_df, test_df])
y_all  = all_df["cancer_type"].values
X_all  = all_df.drop(columns=["cancer_type"])

lusc_mask = y_all == "LUSC"
luad_mask = y_all == "LUAD"
print(f"  Total samples: {len(y_all)}  LUSC: {lusc_mask.sum()}  LUAD: {luad_mask.sum()}")


# ── k=9 spectral clustering ────────────────────────────────────────────────
print("\nFitting k=9 spectral clustering...")
mat = np.clip(go_matrix, 0, 1)
np.fill_diagonal(mat, 1)
sc = SpectralClustering(n_clusters=9, affinity="precomputed",
                        random_state=42, assign_labels="kmeans")
labels = sc.fit_predict(mat)

modules = {}
for i, lbl in enumerate(labels):
    modules.setdefault(lbl, []).append(sim_genes[i])

sorted_mods = sorted(modules.items(), key=lambda x: len(x[1]), reverse=True)

# Label clusters by biological identity
cluster_names = {
    0: "Lineage Identity (TFs + core keratins)",   # size 29
    1: "Calcium/S100 proteins",                     # size 10
    2: "Ion channels & transporters",               # size 10
    3: "Cornification (SPRR/LCE)",                  # size 9
    4: "Serine proteases & MAGE",                   # size 8
    5: "Desmosomes (DSC/DSG)",                      # size 5
    6: "Type II keratins",                          # size 5
    7: "Mixed effectors",                           # size 4
    8: "TMPRSS proteases",                          # size 4
}

print(f"\nClusters identified:")
for rank, (mid, gene_list) in enumerate(sorted_mods):
    print(f"  Cluster {rank+1} (n={len(gene_list)}): {cluster_names[rank]}")
    print(f"    {', '.join(sorted(gene_list))}")


# ── Eigengene computation ──────────────────────────────────────────────────
def compute_eigengene(X, gene_list):
    """
    First principal component of the cluster's expression matrix.
    Sign is flipped if needed so eigengene correlates positively
    with mean expression (standard convention).
    """
    cols = [g for g in gene_list if g in X.columns]
    if len(cols) < 2:
        return None, cols
    sub = X[cols].values
    pca = PCA(n_components=1)
    eg  = pca.fit_transform(sub).flatten()
    # Flip sign if anticorrelated with mean
    if np.corrcoef(eg, sub.mean(axis=1))[0, 1] < 0:
        eg = -eg
    return eg, cols


# ── Full-dataset correlations (naive / confounded) ─────────────────────────
print("\n" + "="*60)
print("NAIVE CORRELATIONS (all samples — confounded by cancer type)")
print("="*60)

eigengenes_all = {}
for rank, (mid, gene_list) in enumerate(sorted_mods):
    eg, cols = compute_eigengene(X_all, gene_list)
    eigengenes_all[rank] = eg

cluster1_eg = eigengenes_all[0]
print(f"\nCluster 1 (Lineage Identity) vs others — all {len(y_all)} samples:")
print(f"{'Cluster':<40} {'Pearson r':>10} {'p-value':>12}")
print("-" * 64)
for rank in range(1, 9):
    eg = eigengenes_all[rank]
    if eg is None:
        continue
    r, p = pearsonr(cluster1_eg, eg)
    print(f"  vs Cluster {rank+1} ({cluster_names[rank][:30]:<30}) {r:>9.3f}  {p:>11.2e}")


# ── Within-cancer-type correlations (the real test) ───────────────────────
for cancer, mask in [("LUSC", lusc_mask), ("LUAD", luad_mask)]:
    print(f"\n{'='*60}")
    print(f"WITHIN-{cancer} CORRELATIONS (n={mask.sum()} samples)")
    print(f"{'='*60}")
    print("(Label confounder removed — tests true co-movement)\n")

    X_sub = X_all[mask]

    eigengenes = {}
    for rank, (mid, gene_list) in enumerate(sorted_mods):
        eg, cols = compute_eigengene(X_sub, gene_list)
        eigengenes[rank] = eg

    c1_eg = eigengenes[0]

    print(f"Cluster 1 (Lineage Identity) vs others — within {cancer}:")
    print(f"{'Cluster':<42} {'Pearson r':>9} {'Spearman r':>11} {'p-value':>11} {'Interpretation'}")
    print("-" * 100)

    for rank in range(1, 9):
        eg = eigengenes[rank]
        if eg is None:
            continue
        r_p, p_p   = pearsonr(c1_eg, eg)
        r_s, p_s   = spearmanr(c1_eg, eg)

        if abs(r_p) >= 0.5 and p_p < 0.01:
            interp = "STRONG co-movement — supports upstream-downstream"
        elif abs(r_p) >= 0.3 and p_p < 0.05:
            interp = "Moderate co-movement"
        elif p_p >= 0.05:
            interp = "No significant correlation — independent modules"
        else:
            interp = "Weak"

        print(f"  vs C{rank+1} {cluster_names[rank][:32]:<32}  {r_p:>8.3f}  {r_s:>10.3f}  {p_p:>10.2e}  {interp}")


# ── Cross-cluster correlations within LUSC ────────────────────────────────
print(f"\n{'='*60}")
print("EFFECTOR-TO-EFFECTOR CORRELATIONS within LUSC")
print("(Tests if effector clusters are co-regulated or independent)")
print("="*60)

X_lusc = X_all[lusc_mask]
effector_ranks = [3, 5, 6]   # Cornification, Desmosomes, Type II keratins
effector_egs   = {}
for rank in effector_ranks:
    eg, _ = compute_eigengene(X_lusc, sorted_mods[rank][1])
    effector_egs[rank] = eg

pairs = [(3,5), (3,6), (5,6)]
names = {3: "Cornification", 5: "Desmosomes", 6: "Type II keratins"}
print(f"\n{'Pair':<45} {'Pearson r':>9} {'p-value':>11}")
print("-" * 68)
for a, b in pairs:
    r, p = pearsonr(effector_egs[a], effector_egs[b])
    print(f"  {names[a]} vs {names[b]:<28}  {r:>8.3f}  {p:>10.2e}")


# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print("="*60)

X_lusc = X_all[lusc_mask]
egs_lusc = {}
for rank, (mid, gene_list) in enumerate(sorted_mods):
    eg, _ = compute_eigengene(X_lusc, gene_list)
    egs_lusc[rank] = eg

c1 = egs_lusc[0]
strong, moderate, none_ = [], [], []
for rank in range(1, 9):
    r, p = pearsonr(c1, egs_lusc[rank])
    if abs(r) >= 0.5 and p < 0.01:
        strong.append(f"C{rank+1} ({cluster_names[rank].split('(')[0].strip()})")
    elif abs(r) >= 0.3 and p < 0.05:
        moderate.append(f"C{rank+1} ({cluster_names[rank].split('(')[0].strip()})")
    else:
        none_.append(f"C{rank+1} ({cluster_names[rank].split('(')[0].strip()})")

print(f"\nWithin LUSC — Cluster 1 co-movement:")
print(f"  Strong (r≥0.5, p<0.01) : {strong if strong else 'none'}")
print(f"  Moderate (r≥0.3)       : {moderate if moderate else 'none'}")
print(f"  Independent            : {none_ if none_ else 'none'}")

if strong:
    print(f"\nVerdict: UPSTREAM-DOWNSTREAM STORY IS SUPPORTED for {len(strong)} cluster(s).")
    print(f"  Cluster 1 (lineage identity) co-moves with terminal effector modules")
    print(f"  within LUSC — consistent with TF-driven differentiation program.")
elif moderate:
    print(f"\nVerdict: PARTIAL SUPPORT — moderate co-movement with some effectors.")
    print(f"  Regulatory relationship exists but weaker than expected.")
else:
    print(f"\nVerdict: NOT SUPPORTED — Cluster 1 is independent of effector clusters")
    print(f"  within LUSC. The control tower framing is not backed by data.")
    print(f"  Clusters are parallel LUSC markers, not hierarchically related.")

print("\n[Sandbox complete — no pipeline files modified]")
