import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_data():
    df = pd.read_csv(os.path.join(RAW_DIR, "lung_cancer_combined.csv"), index_col=0)
    labels = df['cancer_type']
    expression = df.drop(columns=['cancer_type'])
    return expression, labels


def compute_deg(log2_expression, labels, fc_threshold=1.0, pval_threshold=0.01):
    """Compute DEG analysis on log2-transformed expression data."""
    luad_samples = log2_expression[labels == 'LUAD']
    lusc_samples = log2_expression[labels == 'LUSC']

    results = []
    for gene in log2_expression.columns:
        luad_vals = luad_samples[gene].values
        lusc_vals = lusc_samples[gene].values

        log2fc = np.mean(luad_vals) - np.mean(lusc_vals)
        _, pval = stats.ttest_ind(luad_vals, lusc_vals, equal_var=False)

        if np.isnan(pval):
            pval = 1.0

        results.append({
            'gene': gene,
            'log2fc': log2fc,
            'pvalue': pval,
            'abs_log2fc': abs(log2fc)
        })

    deg_df = pd.DataFrame(results)
    valid_pvals = deg_df['pvalue'].clip(0, 1)
    deg_df['pvalue_adj'] = stats.false_discovery_control(valid_pvals)

    significant = deg_df[
        (deg_df['abs_log2fc'] >= fc_threshold) &
        (deg_df['pvalue_adj'] < pval_threshold)
    ].sort_values('abs_log2fc', ascending=False)

    return deg_df, significant


def create_splits(expression, labels, test_size=0.15, val_size=0.15):
    """Stratified split into train/val/test on raw (un-normalized) data."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        expression, labels, test_size=test_size, stratify=labels, random_state=42
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
    )

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading data...")
    expression, labels = load_data()
    print(f"Loaded {len(expression)} samples, {len(expression.columns)} genes")
    print(f"LUAD: {sum(labels == 'LUAD')}, LUSC: {sum(labels == 'LUSC')}")

    # Step 1: Split first (stratified) on raw expression — no leakage
    print("\nCreating train/val/test splits (before any normalization)...")
    splits = create_splits(expression, labels)
    X_train, y_train = splits['train']
    X_val,   y_val   = splits['val']
    X_test,  y_test  = splits['test']
    print(f"  train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Step 2: Log2 transform all splits
    log2_train = np.log2(X_train + 1)
    log2_val   = np.log2(X_val   + 1)
    log2_test  = np.log2(X_test  + 1)

    # Step 3: DEG only on training set
    print("\nComputing differential expression (train only, on log2 data)...")
    deg_all, deg_significant = compute_deg(log2_train, y_train)
    print(f"  Total genes analyzed: {len(deg_all)}")
    print(f"  Significant DEGs: {len(deg_significant)}")

    top_genes = deg_significant.head(100)['gene'].tolist()
    if len(top_genes) < 100:
        print(f"  Warning: Only {len(top_genes)} genes passed threshold, using all")
    print(f"  Selected top {len(top_genes)} genes")

    deg_all.to_csv(os.path.join(PROCESSED_DIR, "deg_analysis.csv"), index=False)
    deg_significant.to_csv(os.path.join(PROCESSED_DIR, "deg_significant.csv"), index=False)

    # Step 4: Fit z-score parameters from train only
    print("\nNormalizing (z-score params from train only)...")
    train_mean = log2_train[top_genes].mean()
    train_std  = log2_train[top_genes].std().replace(0, 1)

    # Step 5: Apply same train params to normalize all splits
    norm_train = (log2_train[top_genes] - train_mean) / train_std
    norm_val   = (log2_val[top_genes]   - train_mean) / train_std
    norm_test  = (log2_test[top_genes]  - train_mean) / train_std

    # Step 6: Save outputs
    for split_name, (X_norm, y) in [
        ('train', (norm_train, y_train)),
        ('val',   (norm_val,   y_val)),
        ('test',  (norm_test,  y_test)),
    ]:
        df = X_norm.copy()
        df.insert(0, 'cancer_type', y)
        df.to_csv(os.path.join(PROCESSED_DIR, f"{split_name}.csv"))
        print(f"  Saved {split_name}.csv ({len(df)} samples)")

    with open(os.path.join(PROCESSED_DIR, "selected_genes.txt"), 'w') as f:
        f.write('\n'.join(top_genes))

    print("\nPreprocessing complete (leak-free).")
    return splits, top_genes


if __name__ == "__main__":
    main()
